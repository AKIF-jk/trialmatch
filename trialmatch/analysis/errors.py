"""Error analysis on saved model outputs: tag every record with the kinds of mistake it contains.

Runs after the test evaluation, on the raw outputs it saved. It scores nothing new and tunes nothing.
Tags (a record can carry several):
  label_wrong            predicted label differs from the expert label (malformed output counts as wrong)
  evidence_unsupported   cites at least one sentence the expert did not cite
  evidence_missed        leaves out at least one sentence the expert cited
  evidence_none_given    cites nothing although the expert cited something
  cites_but_gold_empty   cites something although the expert cited nothing
  decision_without_evidence  says included / not included / excluded without citing any sentence
  info_gap_with_evidence     says "not enough information" while citing sentences
  gold_included_no_evidence  expert label is `included` but the expert cited nothing (annotation oddity, 7 in the dataset)
  rare_class_gold        expert label is `not included` or `excluded`
"""
from trialmatch.data.load import LABELS
from trialmatch.data.note import parse_note
from trialmatch.eval.parse import parse_output

DECISIONS = ("included", "not included", "excluded")
RARE = ("not included", "excluded")
TAGS = (
    "label_wrong", "evidence_unsupported", "evidence_missed", "evidence_none_given", "cites_but_gold_empty",
    "decision_without_evidence", "info_gap_with_evidence", "gold_included_no_evidence", "rare_class_gold",
)


def tag_record(row, text):
    """Return a dict describing one prediction against its expert annotation."""
    parsed = parse_output(text)
    gold, cited = set(row.evidence_ids), set(parsed.evidence_ids)
    pred = parsed.label if parsed.ok and parsed.label in LABELS[row.criterion_type] else None
    tags = []
    if pred != row.label:
        tags.append("label_wrong")
    if parsed.ok and cited - gold:
        tags.append("evidence_unsupported")
    if parsed.ok and gold - cited:
        tags.append("evidence_missed")
    if parsed.ok and gold and not cited:
        tags.append("evidence_none_given")
    if parsed.ok and cited and not gold:
        tags.append("cites_but_gold_empty")
    if pred in DECISIONS and not cited:
        tags.append("decision_without_evidence")
    if pred == "not enough information" and cited:
        tags.append("info_gap_with_evidence")
    if row.label == "included" and not gold:
        tags.append("gold_included_no_evidence")
    if row.label in RARE:
        tags.append("rare_class_gold")
    return {
        "annotation_id": row.annotation_id, "patient_id": row.patient_id, "criterion_type": row.criterion_type,
        "gold_label": row.label, "pred_label": pred, "malformed": not parsed.ok,
        "gold_evidence": sorted(gold), "pred_evidence": sorted(cited), "tags": tags,
    }


def analyse(rows, texts_by_model):
    """rows: usable rows in output order; texts_by_model: {model: [raw text per row]}. Returns per-record tags per model."""
    out = {}
    for model, texts in texts_by_model.items():
        if len(texts) != len(rows):
            raise ValueError(f"{model}: {len(texts)} outputs for {len(rows)} rows")
        out[model] = [tag_record(r, t) for r, t in zip(rows, texts)]
    return out


def summarise(tagged):
    """Counts per tag and per (gold -> predicted) pair for each model, plus the rows every model gets wrong."""
    summary = {"models": {}, "wrong_in_all_models": []}
    for model, records in tagged.items():
        pairs = {}
        for rec in records:
            if "label_wrong" in rec["tags"]:
                key = f"{rec['gold_label']} -> {rec['pred_label']}"
                pairs[key] = pairs.get(key, 0) + 1
        summary["models"][model] = {
            "n": len(records),
            "tag_counts": {t: sum(t in r["tags"] for r in records) for t in TAGS},
            "wrong_label_pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
        }
    models = list(tagged)
    for i, rec in enumerate(tagged[models[0]]):
        if all("label_wrong" in tagged[m][i]["tags"] for m in models):
            summary["wrong_in_all_models"].append(rec["annotation_id"])
    return summary


def sentences_for(row, ids):
    sentences = parse_note(row.note)
    return {i: sentences[i] for i in ids if 0 <= i < len(sentences)}
