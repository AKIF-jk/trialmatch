#!/usr/bin/env python3
"""Validation-only baselines: the untuned base models on the Phase 1 prompt with greedy decoding.

    python scripts/baseline.py qwen phi3 [--order evidence_first] [--limit N] [--batch-size 8]

Each model writes results/baseline/<name>/{outputs.jsonl,report.json}. The prompt text is checked against
the tokenizer's own chat template for every row before any generation. The test split cannot be loaded here.
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trialmatch.data.load import load_rows
from trialmatch.data.prompt import SYSTEM, build_user, chat_prompt, phi3_prompt
from trialmatch.data.split import load_manifest, select_rows
from trialmatch.eval.generate import GREEDY, generate
from trialmatch.eval.metrics import evaluate
from trialmatch.runtime import PHI_MODEL, PHI_REVISION, TARGET_MODEL, TARGET_REVISION, check_versions, environment, results_dir, write_json

MODELS = {
    "qwen": (TARGET_MODEL, TARGET_REVISION, chat_prompt),
    "phi3": (PHI_MODEL, PHI_REVISION, phi3_prompt),
}


def render_prompts(tokenizer, build, rows, order):
    """Hand-written prompt text per row, asserted equal to tokenizer.apply_chat_template."""
    prompts = []
    for row in rows:
        user = build_user(row.note, row.criterion_type, row.criterion_text, order)
        text = build(user)
        official = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
        if text != official:
            raise AssertionError(f"annotation {row.annotation_id}: prompt differs from the chat template")
        prompts.append(text)
    return prompts


def run_model(name, rows, args, device):
    model_id, revision, build = MODELS[name]
    out = results_dir(f"baseline/{name}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
    prompts = render_prompts(tokenizer, build, rows, args.order)
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|end|>", "<|endoftext|>") if t in tokenizer.get_vocab()]
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float16).to(device).eval()
    load_seconds = time.perf_counter() - start

    start = time.perf_counter()
    outputs = generate(model, tokenizer, prompts, args.batch_size, args.max_new_tokens, stop_ids)
    generate_seconds = time.perf_counter() - start

    with open(out / "outputs.jsonl", "w") as f:
        for row, o in zip(rows, outputs):
            f.write(json.dumps({"annotation_id": row.annotation_id, **o}) + "\n")
    texts = [o["text"] for o in outputs]
    report = {
        "model": model_id, "revision": revision, "split": "val", "order": args.order, "dtype": "float16",
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "stop_token_ids": stop_ids, "generation": GREEDY,
        "n_rows": len(rows), "prompt_matches_chat_template": True,
        "load_seconds": round(load_seconds, 1), "generate_seconds": round(generate_seconds, 1),
        "hit_max_new_tokens": sum(o["new_tokens"] >= args.max_new_tokens for o in outputs),
        "strict": evaluate(rows, texts), "repair": evaluate(rows, texts, repair=True),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None,
    }
    write_json(out / "report.json", report)
    s = report["strict"]
    print(name, "acc", round(s["labels"]["overall"]["accuracy"], 3), "macroF1", round(s["labels"]["overall"]["macro_f1"], 3),
          "malformed", s["malformed"], "evidenceF1", round(s["evidence"]["overall"]["macro"]["f1"], 3), flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=MODELS)
    parser.add_argument("--order", default="evidence_first", choices=("evidence_first", "label_first"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    check_versions()
    rows = select_rows(load_rows(), load_manifest(), "val")[:args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    write_json(results_dir("baseline") / "environment.json", environment())
    for name in args.models:
        run_model(name, rows, args, device)


if __name__ == "__main__":
    main()
