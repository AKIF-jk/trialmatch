#!/usr/bin/env python3
"""Speed and memory of reference, tiled and F.scaled_dot_product_attention at the Qwen2.5-1.5B shape.

Shape: 12 query heads, 2 KV heads, head dim 128, batch 1. Two cases per sequence length T: prefill (T queries,
causal) and decode (1 query over T cached keys). Warm-up, synchronized clocks, median and IQR of --repeats runs.
Memory is the CUDA allocator peak above the inputs. Error is the maximum difference from the float64 reference.
Results go to results/attention/bench_<device>_<dtype>, rewritten after each cell.
"""
import argparse
import gc
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from trialmatch.attention.cases import make_inputs
from trialmatch.attention.reference import reference_attention
from trialmatch.attention.tiled import tiled_attention
from trialmatch.runtime import environment, results_dir, write_json

HEADS, KV_HEADS, DIM = 12, 2, 128
TILES = (64, 256, 1024)


def sdpa(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal and q.shape[2] > 1, enable_gqa=True)


def backends(causal):
    found = {"reference": lambda q, k, v: reference_attention(q, k, v, causal=causal)}
    for tile in TILES:
        found[f"tiled_{tile}"] = lambda q, k, v, tile=tile: tiled_attention(q, k, v, causal=causal, tile=tile)
    found["sdpa"] = lambda q, k, v: sdpa(q, k, v, causal)
    return found


def measure(fn, args, device, repeats):
    fn(*args)  # warm-up
    times, peak = [], 0
    for _ in range(repeats):
        gc.collect()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        start = time.perf_counter()
        out = fn(*args)
        if device == "cuda":
            torch.cuda.synchronize()
            peak = max(peak, torch.cuda.max_memory_allocated() - base)
        times.append(time.perf_counter() - start)
    low, mid, high = np.percentile(times, [25, 50, 75])
    return out, {"median_ms": mid * 1e3, "iqr_ms": (high - low) * 1e3, "peak_extra_mb": peak / 2**20 if device == "cuda" else None}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float16" if device == "cuda" else "float32")
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 2048, 4096])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    out_dir = results_dir(f"attention/bench_{device}_{args.dtype}")
    report = {"device": device, "dtype": args.dtype, "heads": HEADS, "kv_heads": KV_HEADS, "head_dim": DIM, "environment": environment(), "cells": []}
    for length in args.lengths:
        for case, q_len in (("prefill", length), ("decode", 1)):
            q, k, v = make_inputs(q_len, length, dtype, batch=1, heads=HEADS, kv_heads=KV_HEADS, dim=DIM, seed=length)
            q, k, v = q.to(device), k.to(device), v.to(device)
            causal = case == "prefill"
            want = reference_attention(q.double(), k.double(), v.double(), causal=causal)
            cell = {"case": case, "length": length, "queries": q_len, "backends": {}}
            for name, fn in backends(causal).items():
                try:
                    result, stats = measure(fn, (q, k, v), device, args.repeats)
                    stats["max_error"] = float((result.double() - want).abs().max())
                except torch.cuda.OutOfMemoryError:
                    stats = {"error": "out of memory"}
                    torch.cuda.empty_cache()
                cell["backends"][name] = stats
            print(case, length, {n: (round(s["median_ms"], 2), s.get("peak_extra_mb")) if "median_ms" in s else s["error"] for n, s in cell["backends"].items()}, flush=True)
            report["cells"].append(cell)
            write_json(out_dir / "report.json", report)


if __name__ == "__main__":
    main()
