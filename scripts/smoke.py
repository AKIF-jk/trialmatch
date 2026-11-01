#!/usr/bin/env python3
"""Check that a Kaggle kernel can run the project, and record what it provides.

Covers the Phase 0 unknowns: internet access, pinned model downloads, available
hardware, library versions, explicit-cache support and a short greedy decode.
"""
import json
import time
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from trialmatch.runtime import DATASET, DRAFT_MODEL, DRAFT_REVISION, TARGET_MODEL, TARGET_REVISION, environment, results_dir, write_json


def check_model(name, revision, device, dtype):
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype).to(device).eval()
    load_seconds = time.perf_counter() - start

    messages = [{"role": "user", "content": "Reply with one short sentence about clinical trials."}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=16, do_sample=False, repetition_penalty=1.0,
            temperature=None, top_p=None, top_k=None,
        )
    generate_seconds = time.perf_counter() - start

    # Confirm the cache API the hand-written decoders depend on.
    cache = DynamicCache()
    with torch.inference_mode():
        out = model(**inputs, past_key_values=cache, use_cache=True)
    seq_before = cache.get_seq_length()
    cache.crop(seq_before - 1)

    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "dtype": str(next(model.parameters()).dtype),
        "attn_implementation": model.config._attn_implementation,
        "load_seconds": round(load_seconds, 1),
        "generate_seconds": round(generate_seconds, 2),
        "text": tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True),
        "cache_seq_length": seq_before,
        "cache_seq_length_after_crop": cache.get_seq_length(),
        "logits_shape": list(out.logits.shape),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None,
    }


def main():
    out = results_dir("smoke")
    report = {"environment": environment(), "stages": {}}
    print(json.dumps(report["environment"], indent=2))

    url = f"https://datasets-server.huggingface.co/size?dataset={DATASET}"
    with urllib.request.urlopen(url, timeout=60) as response:
        report["stages"]["hf_dataset_reachable"] = response.status == 200

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    for label, name, revision in (("target", TARGET_MODEL, TARGET_REVISION), ("draft", DRAFT_MODEL, DRAFT_REVISION)):
        report["stages"][label] = check_model(name, revision, device, dtype)
        print(label, json.dumps(report["stages"][label], indent=2))

    write_json(out / "report.json", report)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
