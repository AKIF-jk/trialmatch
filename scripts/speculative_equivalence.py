#!/usr/bin/env python3
"""Speculative decoding against greedy decoding on the real models: do the tokens match?

The target is Qwen2.5-1.5B-Instruct with the seed-44 adapter merged in. The draft is Qwen2.5-0.5B-Instruct, base or with
the draft adapter merged in. For each draft, K and input, the speculative decoder runs next to decode_cached (the Phase 3
greedy path) and the two are compared on tokens and on target logits. Inputs are validation rows with real stop tokens and
token-pool prompts with 64 forced new tokens. In float32 the library's assisted generation runs too, as a cross-check only.

Float32 is the equality result. Fp16 is reported next to it, because fp16 greedy paths can flip near ties
(docs/STATUS.md, Phase 3 and 4). A mismatch is recorded, not raised. Give each dtype its own run: a float32 run with
logits kept is slow. Results go to results/speculative/equivalence/{records,report}_<dtype>.
"""
import argparse
import json

import torch

from trialmatch.decoding.experiment import (
    STARTS, fetch_files, left_pad, load_model_for, load_tokenizer, token_pool, validation_prompts,
)
from trialmatch.decoding.greedy import compare, decode_cached
from trialmatch.decoding.speculative import ModelDraft, decode_speculative
from trialmatch.eval.generate import GREEDY
from trialmatch.runtime import check_versions, environment, results_dir, write_json


def assisted(target, draft, ids, mask, new_tokens, stop_ids, pad_id):
    with torch.inference_mode():
        out = target.generate(
            input_ids=ids, attention_mask=mask, assistant_model=draft, max_new_tokens=new_tokens, eos_token_id=stop_ids or None,
            pad_token_id=pad_id, **GREEDY,
        )
    tokens = out[0, ids.shape[1]:]
    return tokens[tokens != pad_id] if stop_ids else tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", nargs="+", default=["float32"])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--fetch", action="append", default=[], help="<folder>/<file>=<url>, repeatable")
    parser.add_argument("--adapter-root", default="adapters")
    parser.add_argument("--target-adapter", default="target")
    parser.add_argument("--draft-adapter", default="draft")
    args = parser.parse_args()

    check_versions()
    out = results_dir("speculative/equivalence")
    write_json(out / "environment.json", environment())
    fetch_files(args.fetch, args.adapter_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    pad_id = tokenizer.pad_token_id
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    prompts = validation_prompts()[:args.rows]
    pool = token_pool(tokenizer, validation_prompts())
    cases = [("validation_row", {"row": i}, left_pad([torch.tensor(tokenizer(p).input_ids)], pad_id, device), 96, stop_ids) for i, p in enumerate(prompts)]
    for start in STARTS:
        for length in (64, 256):
            cases.append(("pool_prompt", {"start": start, "prompt": length}, left_pad([pool[start:start + length]], pad_id, device), 64, ()))

    for dtype in args.dtypes:
        records = []
        path = out / f"records_{dtype}.jsonl"
        target = load_model_for("target", dtype, device, f"{args.adapter_root}/{args.target_adapter}")
        drafts = {
            "base": load_model_for("draft", dtype, device),
            "lora": load_model_for("draft", dtype, device, f"{args.adapter_root}/{args.draft_adapter}"),
        }
        for model in (target, *drafts.values()):
            model.generation_config.eos_token_id = None  # Stops are passed explicitly, so a no-stop case runs to its full length.
        baseline = [decode_cached(target, ids, mask, n, stops, pad_id, keep_logits=True) for _, _, (ids, mask), n, stops in cases]
        for name, draft_model in drafts.items():
            for k in args.ks:
                for i, (kind, label, (ids, mask), n, stops) in enumerate(cases):
                    result = decode_speculative(target, ModelDraft(draft_model, target.config), ids, mask, n, k, stops, keep_logits=True)
                    record = {"kind": kind, "dtype": dtype, "draft": name, "k": k, **label, **compare(baseline[i], result), **result.summary()}
                    if dtype == "float32":
                        library = assisted(target, draft_model, ids, mask, n, list(stops), pad_id)
                        record["assisted_equal_to_greedy"] = library.tolist() == baseline[i].tokens[0].tolist()
                    records.append(record)
                    print(json.dumps({key: record[key] for key in ("kind", "draft", "k", "tokens_equal", "acceptance_rate", "rounds")}), flush=True)
                    path.write_text("".join(json.dumps(r) + "\n" for r in records))

        group = lambda key: sorted({r[key] for r in records})
        summary = {
            "dtype": dtype, "device": device, "n_records": len(records), "cases": len(cases),
            "tokens_equal": all(r["tokens_equal"] for r in records),
            "tokens_equal_by_draft_and_k": {f"{d}/k{k}": all(r["tokens_equal"] for r in records if r["draft"] == d and r["k"] == k) for d in group("draft") for k in group("k")},
            "max_logit_diff": max(r["max_logit_diff"] or 0 for r in records),
            "min_top2_margin": min(r["min_top2_margin"] for r in records if r["min_top2_margin"] is not None),
            "assisted_all_equal": all(r["assisted_equal_to_greedy"] for r in records) if dtype == "float32" else None,
            "mismatches": [r for r in records if not r["tokens_equal"]],
        }
        write_json(out / f"report_{dtype}.json", summary)
        print({key: value for key, value in summary.items() if key != "mismatches"}, "mismatches", len(summary["mismatches"]), flush=True)
        del target, drafts, baseline
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
