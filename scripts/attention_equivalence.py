#!/usr/bin/env python3
"""End-to-end check: greedy tokens with tiled attention equal greedy tokens with sdpa on the real target model.

Both runs use the hand-written cached decoder and the same explicit masks, so only the attention function differs.
Cells: single prompts (64 and 256 tokens at two offsets, 64 new tokens), a left-padded batch, and validation rows with real
stop tokens, for each tile size. Float32 is the equality result. Fp16 is reported next to it, because fp16 greedy paths
can flip near ties (docs/STATUS.md, Phase 3). A mismatch is recorded, not raised. Results go to results/attention/equivalence.
"""
import argparse
import json

import torch

from trialmatch.attention.hf import use_tiled_attention
from trialmatch.decoding.experiment import STARTS, left_pad, load_model, load_tokenizer, token_pool, validation_prompts
from trialmatch.decoding.greedy import compare, decode_cached
from trialmatch.runtime import TARGET_REVISION, check_versions, environment, results_dir, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float16"])
    parser.add_argument("--tiles", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--rows", type=int, default=16)
    args = parser.parse_args()

    check_versions()
    out = results_dir("attention/equivalence")
    write_json(out / "environment.json", environment())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    pad_id = tokenizer.pad_token_id
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    prompts = validation_prompts()
    pool = token_pool(tokenizer, prompts)
    records = []

    def record(**fields):
        records.append(fields)
        print(json.dumps(fields), flush=True)
        (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    for dtype in args.dtypes:
        model = load_model(dtype, device)
        inputs = []
        for start in STARTS:
            for length in args.prompt_lengths:
                inputs.append(("single", {"start": start, "prompt": length}, left_pad([pool[start:start + length]], pad_id, device), args.new_tokens, ()))
            sequences = [pool[start:start + length] for length in args.prompt_lengths]
            inputs.append(("padded_batch", {"start": start, "prompts": args.prompt_lengths}, left_pad(sequences, pad_id, device), args.new_tokens, ()))
        batch = [torch.tensor(tokenizer(p).input_ids) for p in prompts[:args.rows]]
        inputs.append(("validation_rows", {"rows": len(batch)}, left_pad(batch, pad_id, device), 96, stop_ids))

        baseline = {i: decode_cached(model, *case[2], case[3], case[4], pad_id, keep_logits=True) for i, case in enumerate(inputs)}
        for tile in args.tiles:
            previous = use_tiled_attention(model, tile)
            for i, (kind, label, (ids, mask), new_tokens, stops) in enumerate(inputs):
                tiled = decode_cached(model, ids, mask, new_tokens, stops, pad_id, keep_logits=True)
                record(kind=kind, dtype=dtype, tile=tile, **label, **compare(baseline[i], tiled))
            model.set_attn_implementation(previous)
        del model, baseline
        torch.cuda.empty_cache()

    summary = {
        "revision": TARGET_REVISION, "device": device, "n_records": len(records),
        "tokens_equal": {d: all(r["tokens_equal"] for r in records if r["dtype"] == d) for d in args.dtypes},
        "max_logit_diff": {d: max(r["max_logit_diff"] or 0 for r in records if r["dtype"] == d) for d in args.dtypes},
        "min_top2_margin": {d: min(r["min_top2_margin"] for r in records if r["dtype"] == d and r["min_top2_margin"] is not None) for d in args.dtypes},
        "mismatches": [r for r in records if not r["tokens_equal"]],
    }
    write_json(out / "report.json", summary)
    print({k: v for k, v in summary.items() if k != "mismatches"}, "mismatches", len(summary["mismatches"]), flush=True)


if __name__ == "__main__":
    main()
