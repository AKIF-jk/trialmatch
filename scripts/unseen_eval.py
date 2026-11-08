#!/usr/bin/env python3
"""Run the fine-tuned seeds on the hand-written cases in configs/unseen_cases.json and score them with the Phase 2 evaluator.

    python scripts/unseen_eval.py --fetch seed42/adapter_model.safetensors=URL ... [--models seed42 seed43 seed44]

The cases are not from the dataset, so this never reads a dataset split. Their expected labels were written by the assistant and
are not clinician-reviewed: read the result as a sanity check on notes the model has never seen, not as a benchmark. Writes
results/final/unseen/{cases.jsonl,<model>/outputs.jsonl,report.json}. Decoding matches scripts/final_eval.py. Not part of the frozen set.
"""
import argparse
import json
import os
from pathlib import Path

from final_eval import GENERATION, ORDER, STOP_TOKENS, check_adapters, load_model  # scripts/ is on sys.path when run as a script

from trialmatch.data.load import LABELS, Row
from trialmatch.data.note import format_note
from trialmatch.data.prompt import row_prompt
from trialmatch.eval.metrics import evaluate
from trialmatch.eval.parse import parse_output
from trialmatch.runtime import TARGET_MODEL, TARGET_REVISION, check_versions, environment, results_dir, write_json

ROOT = Path(os.path.abspath(__file__)).parent.parent


def build_rows(spec):
    """Rows for every criterion of every hand-written patient, with a numbered note in the dataset's format."""
    rows = []
    for patient in spec["patients"]:
        note = format_note(patient["sentences"])
        for criterion in patient["criteria"]:
            if criterion["label"] not in LABELS[criterion["type"]]:
                raise ValueError(f"{patient['id']}: label {criterion['label']!r} is not valid for {criterion['type']}")
            if any(not 0 <= i < len(patient["sentences"]) for i in criterion["evidence"]):
                raise ValueError(f"{patient['id']}: evidence id outside the note for {criterion['text']!r}")
            rows.append(Row(len(rows), patient["id"], "unseen", note, criterion["type"], criterion["text"], tuple(criterion["evidence"]), criterion["label"]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["seed42", "seed43", "seed44"], choices=["seed42", "seed43", "seed44"])
    parser.add_argument("--fetch", action="append", default=[], help="<folder>/<file>=<url>, repeatable")
    parser.add_argument("--adapter-root", default="adapters")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    check_versions()
    from transformers import AutoTokenizer

    from trialmatch.decoding.experiment import fetch_files
    from trialmatch.eval.generate import generate

    config = json.loads((ROOT / "configs/final.json").read_text())
    spec = json.loads((ROOT / "configs/unseen_cases.json").read_text())
    rows = build_rows(spec)
    fetch_files(args.fetch, args.adapter_root)
    adapters = check_adapters(config, args.models, args.adapter_root)
    out = results_dir("final/unseen")
    write_json(out / "environment.json", environment())
    with open(out / "cases.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r.annotation_id, "patient": r.patient_id, "type": r.criterion_type, "criterion": r.criterion_text, "expected_label": r.label, "expected_evidence": list(r.evidence_ids)}) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, revision=TARGET_REVISION, padding_side="left")
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in STOP_TOKENS]
    prompts = [row_prompt(r, ORDER) for r in rows]
    report = {"n_cases": len(rows), "authorship": spec["authorship"], "decoding": {"batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, **GENERATION}, "models": {}}
    for name in args.models:
        model = load_model(adapters[name])
        outputs = generate(model, tokenizer, prompts, args.batch_size, args.max_new_tokens, stop_ids)
        del model
        (out / name).mkdir(exist_ok=True)
        with open(out / name / "outputs.jsonl", "w") as f:
            for r, o in zip(rows, outputs):
                p = parse_output(o["text"])
                f.write(json.dumps({"id": r.annotation_id, "text": o["text"], "label": p.label, "evidence": list(p.evidence_ids)}) + "\n")
        scores = evaluate(rows, [o["text"] for o in outputs])
        o = scores["labels"]["overall"]
        report["models"][name] = {"accuracy": o["accuracy"], "macro_f1": o["macro_f1"], "evidence_macro_f1": scores["evidence"]["overall"]["macro"]["f1"], "malformed": scores["malformed"], "strict": scores}
        print(name, "label accuracy", round(o["accuracy"], 3), "evidence macro-F1", round(scores["evidence"]["overall"]["macro"]["f1"], 3), "malformed", scores["malformed"], flush=True)
    write_json(out / "report.json", report)
    print("done", flush=True)


if __name__ == "__main__":
    main()
