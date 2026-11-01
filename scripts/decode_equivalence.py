#!/usr/bin/env python3
"""Compare the cached decoder with the uncached reference on the real target model, in float32 and fp16.

Every grid cell forces continuation past EOS, so the output length is exact. Also runs a left-padded batch against
the same prompts alone, validation rows with real stop tokens, and model.generate as a cross-check only.
Results go to results/decoding/equivalence. A mismatch is recorded, not raised, so it can be investigated.
"""
import argparse
import json
import time

import torch

from trialmatch.decoding.experiment import STARTS, left_pad, load_model, load_tokenizer, token_pool, validation_prompts
from trialmatch.decoding.greedy import compare, decode_cached, decode_uncached
from trialmatch.eval.generate import GREEDY
from trialmatch.runtime import TARGET_REVISION, check_versions, environment, results_dir, write_json


def library_tokens(model, ids, mask, new_tokens, pad_id, stop_ids):
    """model.generate as an outside check. Its stop list comes from stop_ids alone, so an empty list runs past EOS."""
    config = model.generation_config
    saved = config.eos_token_id
    config.eos_token_id = list(stop_ids) or None
    with torch.inference_mode():
        generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=new_tokens, pad_token_id=pad_id, **GREEDY)
    config.eos_token_id = saved
    return generated[:, ids.shape[1]:]


def check_pair(model, ids, mask, new_tokens, pad_id, stop_ids=()):
    reference = decode_uncached(model, ids, mask, new_tokens, stop_ids, pad_id, keep_logits=True)
    cached = decode_cached(model, ids, mask, new_tokens, stop_ids, pad_id, keep_logits=True)
    report = compare(reference, cached)
    library = library_tokens(model, ids, mask, new_tokens, pad_id, stop_ids)
    report["matches_generate"] = library.shape == cached.tokens.shape and bool((library == cached.tokens).all())
    report["cache_growth_ok"] = cached.cache_lengths == list(range(ids.shape[1], ids.shape[1] + len(cached.cache_lengths)))
    return report, cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float16"])
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--row-tokens", type=int, default=96)
    args = parser.parse_args()

    check_versions()
    out = results_dir("decoding/equivalence")
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
        alone = {}
        for start in STARTS:
            for length in args.prompt_lengths:
                for new_tokens in args.output_lengths:
                    began = time.perf_counter()
                    ids, mask = left_pad([pool[start:start + length]], pad_id, device)
                    report, cached = check_pair(model, ids, mask, new_tokens, pad_id)
                    alone[start, length, new_tokens] = cached.tokens[0]
                    record(kind="grid", dtype=dtype, start=start, prompt=length, output=new_tokens, seconds=round(time.perf_counter() - began, 1), **report)

        for start in STARTS:
            for new_tokens in args.output_lengths:
                sequences = [pool[start:start + length] for length in args.prompt_lengths]
                ids, mask = left_pad(sequences, pad_id, device)
                report, cached = check_pair(model, ids, mask, new_tokens, pad_id)
                same = [bool((cached.tokens[row] == alone[start, length, new_tokens]).all()) for row, length in enumerate(args.prompt_lengths)]
                record(kind="padded_batch", dtype=dtype, start=start, prompts=args.prompt_lengths, output=new_tokens, equals_alone=same, **report)

        for begin in range(0, min(args.rows, len(prompts)), args.batch_size):
            batch = [torch.tensor(tokenizer(p).input_ids) for p in prompts[begin:begin + args.batch_size]]
            ids, mask = left_pad(batch, pad_id, device)
            report, _ = check_pair(model, ids, mask, args.row_tokens, pad_id, stop_ids)
            record(kind="validation_rows", dtype=dtype, first_row=begin, rows=len(batch), width=ids.shape[1], **report)
        del model
        torch.cuda.empty_cache()

    summary = {
        "revision": TARGET_REVISION, "device": device, "n_records": len(records),
        "all_tokens_equal": all(r["tokens_equal"] for r in records),
        "all_match_generate": all(r["matches_generate"] for r in records),
        "all_cache_growth_ok": all(r["cache_growth_ok"] for r in records),
        "all_padded_equal_alone": all(all(r["equals_alone"]) for r in records if r["kind"] == "padded_batch"),
        "max_logit_diff": {d: max(r["max_logit_diff"] for r in records if r["dtype"] == d) for d in args.dtypes},
        "min_top2_margin": {d: min(r["min_top2_margin"] for r in records if r["dtype"] == d) for d in args.dtypes},
        "mismatches": [r for r in records if not r["tokens_equal"]],
    }
    write_json(out / "report.json", summary)
    print({k: v for k, v in summary.items() if k != "mismatches"}, "mismatches", len(summary["mismatches"]), flush=True)


if __name__ == "__main__":
    main()
