#!/usr/bin/env python3
"""Tag every test prediction of the fine-tuned seeds with its kinds of mistake. CPU only, reads saved outputs.

    PYTHONPATH=. python scripts/error_analysis.py [--models seed42 seed43 seed44] [--out results/final/analysis]

Writes records.jsonl (one line per model and record), summary.json, and shows nothing the model did not already output.
The rows come from the dataset by annotation id, taken from the saved outputs, so the test split is read only because
the evaluation already ran. Not part of the frozen set.
"""
import argparse
import json
from pathlib import Path

from trialmatch.analysis.errors import analyse, summarise
from trialmatch.data.load import load_rows

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["seed42", "seed43", "seed44"])
    parser.add_argument("--results", default=str(ROOT / "results/final/test"))
    parser.add_argument("--out", default=str(ROOT / "results/final/analysis"))
    args = parser.parse_args()

    outputs = {m: [json.loads(line) for line in open(Path(args.results) / m / "outputs.jsonl")] for m in args.models}
    ids = [o["annotation_id"] for o in outputs[args.models[0]]]
    if any([o["annotation_id"] for o in outputs[m]] != ids for m in args.models):
        raise SystemExit("models were not run on the same rows in the same order")
    by_id = {r.annotation_id: r for r in load_rows()}
    rows = [by_id[i] for i in ids]
    tagged = analyse(rows, {m: [o["text"] for o in outputs[m]] for m in args.models})
    summary = summarise(tagged)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "records.jsonl", "w") as f:
        for m, records in tagged.items():
            for rec in records:
                f.write(json.dumps({"model": m, **rec}) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for m, s in summary["models"].items():
        print(m, json.dumps(s["tag_counts"]))
        print("  wrong pairs:", json.dumps(s["wrong_label_pairs"]))
    print("wrong in all models:", len(summary["wrong_in_all_models"]))


if __name__ == "__main__":
    main()
