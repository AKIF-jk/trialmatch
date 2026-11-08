#!/usr/bin/env python3
"""Final evaluation: the untuned base model and the three fine-tuned seed adapters on the same rows, with the same decoding.

    python scripts/final_eval.py --expect-commit SHA --fetch seed42/adapter_model.safetensors=URL ... --fetch seed42/adapter_config.json=URL ...
    python scripts/final_eval.py --split val ...        # dry run of the identical code path; needs no freeze

This is the only script that may open the test split (`allow_test=True` appears nowhere else). Before it does, it checks:
configs/freeze.json exists and every frozen file still has its recorded hash; the running commit is the one the freeze tag
points at (--expect-commit); no earlier test run is on record in results/final/test/run.json, unless --rerun-reason says why
a rerun is needed, in which case the reason is written into the results. Run it once. Do not tune after it.

Each model writes results/final/<split>/<name>/{outputs.jsonl,report.json}; the majority baseline and the seed summary go
to results/final/<split>/{majority,summary.json}. Raw outputs are saved before any scoring, so a scoring change never needs
a new generation run. Decoding is greedy (do_sample=False, repetition_penalty=1.0) for every model.
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from trialmatch.data.load import load_rows
from trialmatch.data.prompt import SYSTEM, build_user, row_prompt
from trialmatch.data.split import load_manifest, select_rows
from trialmatch.eval.baselines import majority_outputs
from trialmatch.eval.freeze import FreezeError, open_test_gate
from trialmatch.eval.summary import headline, score_texts, seed_summary
from trialmatch.runtime import TARGET_MODEL, TARGET_REVISION, check_versions, environment, results_dir, write_json

ROOT = Path(os.path.abspath(__file__)).parent.parent
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
ORDER = "evidence_first"
SEEDS = ("seed42", "seed43", "seed44")
GENERATION = dict(do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--models", nargs="+", default=["base", *SEEDS], choices=["base", *SEEDS])
    p.add_argument("--expect-commit", help="commit SHA the freeze tag points at; required for the test split")
    p.add_argument("--rerun-reason", help="why the test split is being run again; written into the results")
    p.add_argument("--fetch", action="append", default=[], help="<folder>/<file>=<url>, repeatable")
    p.add_argument("--adapter-root", default="adapters")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--limit", type=int, help="first N rows; dry runs on val only")
    return p.parse_args(argv)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_adapters(config, models, root):
    """Folder per requested seed, with the weights checked against the checksum recorded in configs/final.json."""
    folders = {}
    for name in models:
        if name == "base":
            continue
        folder = config["adapters"]["folders"][name.removeprefix("seed")]
        path = Path(root) / folder
        digest = sha256_of(path / "adapter_model.safetensors")
        expected = config["adapters"]["sha256"][folder]
        if digest != expected:
            raise RuntimeError(f"adapter {folder}: checksum {digest[:12]} differs from the recorded {expected[:12]}")
        folders[name] = path
    return folders


def check_prompts(tokenizer, rows):
    """The hand-written prompt must equal the tokenizer's own chat template for every row, as in the baselines."""
    for row in rows:
        user = build_user(row.note, row.criterion_type, row.criterion_text, ORDER)
        official = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
        if row_prompt(row, ORDER) != official:
            raise AssertionError(f"annotation {row.annotation_id}: prompt differs from the chat template")


def load_model(adapter):
    """Pinned fp16 base on the GPU; an adapter is attached unmerged, the same way the validation scores were produced."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, revision=TARGET_REVISION, dtype=torch.float16).to("cuda")
    if adapter:
        # Kaggle ships torchao 0.10.0 and peft 0.20.0 refuses it. We use no torchao, so hide it, as scripts/train.py does.
        sys.modules.setdefault("torchao", None)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    return model.eval()


def run_model(name, adapter, tokenizer, rows, prompts, stop_ids, args, out):
    import torch

    from trialmatch.eval.generate import generate

    start = time.perf_counter()
    model = load_model(adapter)
    load_seconds = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    outputs = generate(model, tokenizer, prompts, args.batch_size, args.max_new_tokens, stop_ids)
    generate_seconds = time.perf_counter() - start
    meta = {
        "load_seconds": round(load_seconds, 1),
        "generate_seconds": round(generate_seconds, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "hit_max_new_tokens": sum(o["new_tokens"] >= args.max_new_tokens for o in outputs),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    with open(out / "outputs.jsonl", "w") as f:
        for row, o in zip(rows, outputs):
            f.write(json.dumps({"annotation_id": row.annotation_id, **o}) + "\n")
    return [o["text"] for o in outputs], meta


def recorded_validation(config, name):
    """Strict validation headline of the run this seed's adapter came from, for the dry-run reproduction check."""
    run = config["runs"][name.removeprefix("seed")]
    report = json.loads((ROOT / "results/train" / run / "report.json").read_text())
    return headline(report["best"]["strict"])


def support(rows):
    counts = {}
    for r in rows:
        counts.setdefault(r.criterion_type, {}).setdefault(r.label, 0)
        counts[r.criterion_type][r.label] += 1
    return counts


def main():
    args = parse_args()
    config = json.loads((ROOT / "configs/final.json").read_text())
    commit = os.environ.get("TRIALMATCH_COMMIT", "")
    freeze, earlier_runs = None, []
    if args.split == "test":
        if args.limit:
            raise SystemExit("--limit is for dry runs on val; a partial test run would not count as the one run")
        try:
            freeze, earlier_runs = open_test_gate(ROOT, args.expect_commit, commit, args.rerun_reason)
        except FreezeError as exc:
            raise SystemExit(f"test split stays locked: {exc}")
    check_versions()

    from trialmatch.decoding.experiment import fetch_files

    fetch_files(args.fetch, args.adapter_root)
    adapters = check_adapters(config, args.models, args.adapter_root)

    all_rows, manifest = load_rows(), load_manifest()
    train_rows = select_rows(all_rows, manifest, "train")
    out = results_dir(f"final/{args.split}")
    if args.split == "test":
        rows = select_rows(all_rows, manifest, "test", allow_test=True)
        started = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": commit, "reason": args.rerun_reason, "models": args.models}
        # Written before any generation: a run that crashes after opening the split still counts as a run on record.
        write_json(out / "run.json", {"runs": [*earlier_runs, started]})
    else:
        rows = select_rows(all_rows, manifest, args.split)[:args.limit]
    write_json(out / "environment.json", environment())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, revision=TARGET_REVISION, padding_side="left")
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS]
    check_prompts(tokenizer, rows)
    prompts = [row_prompt(r, ORDER) for r in rows]
    print(f"{args.split}: {len(rows)} rows, {len({r.patient_id for r in rows})} patients, support {json.dumps(support(rows))}", flush=True)

    reports, texts_by_model = {}, {}
    for name in args.models:
        model_out = out / name
        model_out.mkdir(exist_ok=True)
        texts, meta = run_model(name, adapters.get(name), tokenizer, rows, prompts, stop_ids, args, model_out)
        scores = score_texts(rows, texts, boot_n=args.bootstrap)
        report = {"name": name, "split": args.split, "n_rows": len(rows), "adapter": None if name == "base" else config["adapters"]["folders"][name.removeprefix("seed")], **meta, **scores}
        if args.split == "val" and not args.limit and name != "base":
            expected = recorded_validation(config, name)
            found = headline(scores["strict"])
            report["recorded_validation"] = {k: {"recorded": expected[k], "found": found[k]} for k in ("accuracy", "macro_f1", "evidence_macro_f1")}
        write_json(model_out / "report.json", report)
        reports[name], texts_by_model[name] = report, texts
        h = headline(scores["strict"])
        print(name, "acc", round(h["accuracy"], 3), "macroF1", round(h["macro_f1"], 3), "evidenceF1", round(h["evidence_macro_f1"], 3), "malformed", h["malformed"], flush=True)

    (out / "majority").mkdir(exist_ok=True)
    majority = score_texts(rows, majority_outputs(train_rows, rows), boot_n=args.bootstrap)
    write_json(out / "majority" / "report.json", {"name": "majority", "split": args.split, "n_rows": len(rows), **majority})

    fine_tuned = {n: r["strict"] for n, r in reports.items() if n != "base"}
    summary = {
        "split": args.split,
        "n_rows": len(rows),
        "n_patients": len({r.patient_id for r in rows}),
        "support": support(rows),
        "commit": commit,
        "freeze": None if freeze is None else {"tag": freeze["tag"], "code_commit": freeze["code_commit"]},
        "rerun_reason": args.rerun_reason,
        "earlier_test_runs": len(earlier_runs),
        "model": {"name": TARGET_MODEL, "revision": TARGET_REVISION, "dtype": "float16"},
        "adapters": config["adapters"],
        "decoding": {"batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "stop_tokens": list(STOP_TOKENS), "order": ORDER, **GENERATION},
        "models": {n: {"strict": headline(r["strict"]), "repair": headline(r["repair"]), "bootstrap_strict": r["bootstrap_strict"]} for n, r in reports.items()},
        "majority": {"strict": headline(majority["strict"]), "bootstrap_strict": majority["bootstrap_strict"]},
        "fine_tuned_seeds": seed_summary(fine_tuned) if len(fine_tuned) > 1 else None,
    }
    write_json(out / "summary.json", summary)
    if args.split == "test":
        log = json.loads((out / "run.json").read_text())
        log["runs"][-1]["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json(out / "run.json", log)
    print("done", flush=True)


if __name__ == "__main__":
    main()
