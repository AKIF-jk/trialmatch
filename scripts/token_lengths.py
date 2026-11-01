#!/usr/bin/env python3
"""Measure tokenized example lengths on train and val with the pinned Qwen tokenizer.

Needs only `tokenizers` and tokenizer.json, so it runs on a laptop. The test split is not read.
Fails if any example exceeds MAX_LEN, because truncation would remove prompt or target text.
"""
import json
import statistics

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from trialmatch.data.encode import MAX_LEN, encode_example
from trialmatch.data.load import load_rows
from trialmatch.data.prompt import ORDERS
from trialmatch.data.split import load_manifest, select_rows
from trialmatch.runtime import TARGET_MODEL, TARGET_REVISION, results_dir, write_json

if __name__ == "__main__":
    tokenizer = Tokenizer.from_file(hf_hub_download(TARGET_MODEL, "tokenizer.json", revision=TARGET_REVISION))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    end_id = tokenizer.token_to_id("<|im_end|>")
    rows, manifest = load_rows(), load_manifest()
    report = {"max_len": MAX_LEN, "im_end_id": end_id}
    for split in ("train", "val"):
        for order in ORDERS:
            examples = [encode_example(r, encode, order) for r in select_rows(rows, manifest, split)]
            assert all(e["input_ids"][-1] == end_id for e in examples)
            lengths = sorted(len(e["input_ids"]) for e in examples)
            targets = sorted(len(e["input_ids"]) - e["prompt_length"] for e in examples)
            report[f"{split}/{order}"] = {
                "examples": len(lengths), "total_max": lengths[-1], "total_median": statistics.median(lengths),
                "prompt_max": max(e["prompt_length"] for e in examples), "target_max": targets[-1], "target_median": statistics.median(targets),
                "truncated": 0,
            }
    write_json(results_dir("phase1") / "token_lengths.json", report)
    print(json.dumps(report, indent=1))
