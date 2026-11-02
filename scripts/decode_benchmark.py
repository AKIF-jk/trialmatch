#!/usr/bin/env python3
"""Benchmark the uncached reference against the cached decoder on the same model and inputs.

Each cell runs one discarded warm-up per path, then alternates the paths for --repeats timed runs, and reports
median and IQR. Stop tokens are off, so every run makes exactly the requested number of tokens. GPU memory is
the allocator peak with the model loaded. CPU memory is peak RSS, sampled every 10 ms because ru_maxrss cannot be reset.
Results go to results/decoding/benchmark_<device>_<dtype>, rewritten after each cell.
"""
import argparse
import gc

import numpy as np
import torch

from trialmatch.decoding.experiment import PeakRSS, left_pad, load_model, load_tokenizer, token_pool, validation_prompts
from trialmatch.decoding.greedy import decode_cached, decode_uncached
from trialmatch.runtime import TARGET_REVISION, check_versions, environment, results_dir, write_json

PATHS = {"uncached": decode_uncached, "cached": decode_cached}
GRIDS = {"cuda": ([64, 256, 1024], [16, 64, 256], 8), "cpu": ([64, 256], [16, 64], 0)}


def timed_run(decode, model, ids, mask, new_tokens, pad_id, device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = decode(model, ids, mask, new_tokens, (), pad_id)
        peak = torch.cuda.max_memory_allocated()
    else:
        with PeakRSS() as rss:
            result = decode(model, ids, mask, new_tokens, (), pad_id)
        peak = rss.peak
    total = result.prefill_seconds + result.decode_seconds
    made = result.tokens.numel()
    metrics = {
        "prefill_seconds": result.prefill_seconds, "decode_seconds": result.decode_seconds, "total_seconds": total,
        "decode_tokens_per_second": ids.shape[0] * (new_tokens - 1) / result.decode_seconds, "total_tokens_per_second": made / total,
        "peak_memory_gb": peak / 2**30,
    }
    return metrics, result


def summarize(runs):
    stats = {}
    for name in runs[0]:
        values = [run[name] for run in runs]
        low, median, high = np.percentile(values, [25, 50, 75])
        stats[name] = {"median": float(median), "iqr": float(high - low), "runs": values}
    return stats


def run_cell(model, ids, mask, new_tokens, pad_id, device, repeats, label):
    for decode in PATHS.values():
        timed_run(decode, model, ids, mask, new_tokens, pad_id, device)
    runs = {name: [] for name in PATHS}
    tokens, equal, cache_bytes = None, True, None
    for repeat in range(repeats):
        for name in (list(PATHS) if repeat % 2 == 0 else list(PATHS)[::-1]):
            metrics, result = timed_run(PATHS[name], model, ids, mask, new_tokens, pad_id, device)
            runs[name].append(metrics)
            tokens = result.tokens if tokens is None else tokens
            equal = equal and bool((tokens == result.tokens).all())
            cache_bytes = result.cache_bytes if name == "cached" else cache_bytes
    cell = {**label, "batch": ids.shape[0], "width": ids.shape[1], "repeats": repeats, "tokens_equal": equal, "cache_bytes": cache_bytes, "paths": {name: summarize(r) for name, r in runs.items()}}
    medians = {name: cell["paths"][name]["total_seconds"]["median"] for name in PATHS}
    print(label, "equal", equal, "total s", {k: round(v, 3) for k, v in medians.items()}, "speedup", round(medians["uncached"] / medians["cached"], 2), flush=True)
    return cell


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_lengths, output_lengths, batch = GRIDS[device]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float16" if device == "cuda" else "float32")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=prompt_lengths)
    parser.add_argument("--output-lengths", type=int, nargs="+", default=output_lengths)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", type=int, default=batch, help="rows in the extra left-padded validation cell, 0 to skip")
    args = parser.parse_args()

    check_versions()
    out = results_dir(f"decoding/benchmark_{device}_{args.dtype}")
    tokenizer = load_tokenizer()
    pad_id = tokenizer.pad_token_id
    prompts = validation_prompts()
    pool = token_pool(tokenizer, prompts)
    model = load_model(args.dtype, device)
    report = {
        "revision": TARGET_REVISION, "device": device, "dtype": args.dtype, "attn_implementation": model.config._attn_implementation,
        "threads": torch.get_num_threads(), "environment": environment(), "cells": [],
    }
    for length in args.prompt_lengths:
        for new_tokens in args.output_lengths:
            ids, mask = left_pad([pool[:length]], pad_id, device)
            report["cells"].append(run_cell(model, ids, mask, new_tokens, pad_id, device, args.repeats, {"kind": "single", "prompt": length, "output": new_tokens}))
            write_json(out / "report.json", report)
    if args.batch:
        ids, mask = left_pad([torch.tensor(tokenizer(p).input_ids) for p in prompts[:args.batch]], pad_id, device)
        report["cells"].append(run_cell(model, ids, mask, 64, pad_id, device, args.repeats, {"kind": "padded_rows", "prompt": ids.shape[1], "output": 64}))
    report["all_tokens_equal"] = all(cell["tokens_equal"] for cell in report["cells"])
    write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
