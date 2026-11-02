#!/usr/bin/env python3
"""Speculative decoding speed against the cached greedy decoder, one prompt at a time.

Workloads:
- json: the fine-tuned target (seed-44 adapter merged) answers validation rows with the short JSON, stop tokens on.
  Drafts: Qwen2.5-0.5B-Instruct base and with the draft adapter merged in.
- text: the untuned 1.5B target writes a short free-text explanation for the same rows (up to --text-tokens new tokens).
  The draft is the untuned 0.5B model, so draft and target are both untuned and the output is longer than the JSON.

Drafts are chosen with --drafts: base and lora (models; lora only in the json workload) and lookup (prompt lookup, no model).
--tag is appended to the results folder name so a run with other drafts does not overwrite an earlier report.

Per cell (workload, draft, K) the same prompts run through decode_cached and decode_speculative, --repeats times after one
discarded warm-up per path. Reported per cell: median over repeats of the sum over prompts of prefill, decode and total
time, tokens per second, speedup over greedy, acceptance rate, tokens per round, forward counts, time inside the draft and
the verification passes, and peak extra memory (GPU allocator peak above the resident models, or peak RSS on CPU).
Token equality with greedy is recorded for every prompt. Results go to results/speculative/bench_<device>_<dtype>/report.json,
rewritten after each cell.
"""
import argparse
import gc
import json

import numpy as np
import torch

from trialmatch.data.prompt import chat_prompt, row_prompt
from trialmatch.decoding.experiment import PeakRSS, fetch_files, left_pad, load_model_for, load_tokenizer, validation_rows
from trialmatch.decoding.greedy import decode_cached
from trialmatch.decoding.prompt_lookup import PromptLookupDraft
from trialmatch.decoding.speculative import ModelDraft, decode_speculative
from trialmatch.runtime import check_versions, environment, results_dir, write_json


def text_prompt(row):
    user = (
        f"Patient note (numbered sentences):\n{row.note}\n\nTrial criterion ({row.criterion_type}): {row.criterion_text}\n\n"
        "In a short paragraph, explain whether this patient meets the criterion and cite the numbers of the relevant sentences."
    )
    return chat_prompt(user)


def measured(run, device):
    """Call run() and return (result, peak extra memory in bytes)."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        result = run()
        return result, torch.cuda.max_memory_allocated() - before
    with PeakRSS() as rss:
        result = run()
    return result, rss.peak


def stats(values):
    low, median, high = np.percentile(values, [25, 50, 75])
    return {"median": float(median), "iqr": float(high - low), "runs": list(values)}


def total_time(result):
    return result.prefill_seconds + result.decode_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--workloads", nargs="+", default=["json", "text"], choices=["json", "text"])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--drafts", nargs="+", default=["base", "lora"], choices=["base", "lora", "lookup"])
    parser.add_argument("--tag", default="")
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-tokens", type=int, default=96)
    parser.add_argument("--text-tokens", type=int, default=128)
    parser.add_argument("--fetch", action="append", default=[], help="<folder>/<file>=<url>, repeatable")
    parser.add_argument("--adapter-root", default="adapters")
    parser.add_argument("--target-adapter", default="target")
    parser.add_argument("--draft-adapter", default="draft")
    args = parser.parse_args()

    check_versions()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = results_dir(f"speculative/bench_{device}_{args.dtype}{args.tag}")
    fetch_files(args.fetch, args.adapter_root)
    tokenizer = load_tokenizer()
    pad_id = tokenizer.pad_token_id
    stops = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    rows = validation_rows()[:args.rows]
    report = {
        "device": device, "dtype": args.dtype, "rows": len(rows), "repeats": args.repeats, "threads": torch.get_num_threads(),
        "environment": environment(), "workloads": {},
    }

    def inputs(prompts):
        return [left_pad([torch.tensor(tokenizer(p).input_ids)], pad_id, device) for p in prompts]

    for workload in args.workloads:
        base_target = workload == "text"
        target = load_model_for("target", args.dtype, device, None if base_target else f"{args.adapter_root}/{args.target_adapter}")
        drafts = {}
        if "base" in args.drafts:
            drafts["base"] = load_model_for("draft", args.dtype, device)
        if "lora" in args.drafts and not base_target:
            drafts["lora"] = load_model_for("draft", args.dtype, device, f"{args.adapter_root}/{args.draft_adapter}")
        for model in (target, *drafts.values()):
            model.generation_config.eos_token_id = None
        prompts = [row_prompt(r) for r in rows] if workload == "json" else [text_prompt(r) for r in rows]
        batch = inputs(prompts)
        new_tokens = args.json_tokens if workload == "json" else args.text_tokens
        names = [name for name in args.drafts if name == "lookup" or name in drafts]
        cells = [(name, k) for name in names for k in args.ks]

        def greedy(i):
            ids, mask = batch[i]
            return measured(lambda: decode_cached(target, ids, mask, new_tokens, stops, pad_id), device)

        def speculative(i, name, k):
            ids, mask = batch[i]
            draft = PromptLookupDraft() if name == "lookup" else ModelDraft(drafts[name], target.config)
            return measured(lambda: decode_speculative(target, draft, ids, mask, new_tokens, k, stops), device)

        greedy(0)  # Warm-up: kernels, allocator and the draft's shapes.
        for name, k in cells[:1] + cells[-1:]:
            speculative(0, name, k)

        greedy_runs, spec_runs, equal, texts = [], {cell: [] for cell in cells}, {cell: [] for cell in cells}, []
        for repeat in range(args.repeats):
            runs = [greedy(i) for i in range(len(batch))]
            greedy_runs.append(runs)
            if repeat == 0:
                texts = [tokenizer.decode(r.tokens[0], skip_special_tokens=True) for r, _ in runs]
            for cell in cells:
                spec = [speculative(i, *cell) for i in range(len(batch))]
                spec_runs[cell].append(spec)
                if repeat == 0:
                    equal[cell] = [bool(torch.equal(s.tokens, g.tokens)) for (s, _), (g, _) in zip(spec, runs)]
            print(workload, "repeat", repeat, "done", flush=True)

        def sums(runs, f):
            return [sum(f(r) for r, _ in rep) for rep in runs]

        g_total, g_decode = sums(greedy_runs, total_time), sums(greedy_runs, lambda r: r.decode_seconds)
        g_tokens = sum(r.tokens.numel() for r, _ in greedy_runs[0])
        entry = {
            "new_tokens_cap": new_tokens, "greedy_tokens": g_tokens,
            "greedy_valid_json": sum(_is_json(t) for t in texts) if workload == "json" else None,
            "greedy": {
                "total_seconds": stats(g_total), "decode_seconds": stats(g_decode), "prefill_seconds": stats(sums(greedy_runs, lambda r: r.prefill_seconds)),
                "tokens_per_second": stats([g_tokens / t for t in g_total]), "peak_extra_bytes": max(p for rep in greedy_runs for _, p in rep),
            },
            "cells": [],
        }
        for (name, k) in cells:
            reps = spec_runs[(name, k)]
            s_total, s_decode = sums(reps, total_time), sums(reps, lambda r: r.decode_seconds)
            first = [r for r, _ in reps[0]]
            proposed, accepted = sum(r.proposed for r in first), sum(r.accepted for r in first)
            rounds = sum(len(r.rounds) for r in first)
            s_tokens = sum(r.tokens.numel() for r in first)
            entry["cells"].append({
                "draft": name, "k": k, "tokens": s_tokens, "prompts_equal": sum(equal[(name, k)]), "prompts": len(batch),
                "total_seconds": stats(s_total), "decode_seconds": stats(s_decode), "tokens_per_second": stats([s_tokens / t for t in s_total]),
                "speedup_total": float(np.median(g_total) / np.median(s_total)), "speedup_decode": float(np.median(g_decode) / np.median(s_decode)),
                "acceptance_rate": accepted / proposed if proposed else None, "tokens_per_round": sum(r.tokens.numel() - 1 for r in first) / rounds if rounds else None,
                "rounds": rounds, "empty_rounds": sum(1 for r in first for x in r.rounds if x["k"] == 0), "target_forwards": sum(r.target_forwards for r in first), "draft_forwards": sum(r.draft_forwards for r in first),
                "draft_seconds": stats(sums(reps, lambda r: r.draft_seconds)), "verify_seconds": stats(sums(reps, lambda r: r.verify_seconds)),
                "peak_extra_bytes": max(p for rep in reps for _, p in rep),
            })
            report["workloads"][workload] = entry
            write_json(out / "report.json", report)
            print(workload, name, "k", k, "speedup", round(entry["cells"][-1]["speedup_total"], 2), "accept", entry["cells"][-1]["acceptance_rate"],
                  "equal", entry["cells"][-1]["prompts_equal"], "/", len(batch), flush=True)
        report["workloads"][workload] = entry
        (out / f"greedy_outputs_{workload}.json").write_text(json.dumps(texts, indent=1))
        write_json(out / "report.json", report)
        del target, drafts
        torch.cuda.empty_cache()


def _is_json(text):
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
