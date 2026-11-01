"""Shared evaluator for every phase: label metrics and evidence metrics, overall and split by criterion type.

Rules, fixed before any test result exists:
- Label list per scope is fixed (inclusion 4, exclusion 4, overall 6), so classes with no rows still appear.
- Precision, recall and F1 are 0 when their denominator is 0 (zero_division=0). `macro_f1` averages the whole
  fixed list. `macro_f1_supported` averages only classes with gold support, as a cross-check. Support is
  reported for every class.
- Malformed output, or a label not valid for the criterion type, is a wrong prediction. It adds to no class's
  predicted count and lands in the `<invalid>` column of the confusion matrix.
- Evidence per record: both sets empty scores 1 (P, R, F1). Any other empty case scores 0 for the empty-denominator
  term, so predicted-empty with gold non-empty has R=0 and P=0, and predicted non-empty with gold empty has P=0 and R=0.
  Malformed output scores 0 on all three. Ids outside the note count as false positives.
- Evidence macro is the mean over records. It is also reported over records with non-empty gold only.
"""
from trialmatch.data.load import LABELS
from trialmatch.data.note import parse_note
from trialmatch.eval.parse import parse_output

INVALID = "<invalid>"
SCOPES = {
    "overall": ("included", "not included", "excluded", "not excluded", "not enough information", "not applicable"),
    "inclusion": LABELS["inclusion"],
    "exclusion": LABELS["exclusion"],
}


def _ratio(num, den):
    return num / den if den else 0.0


def _prf(tp, fp, fn):
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return p, r, _ratio(2 * p * r, p + r)


def label_report(gold, pred, labels):
    """gold and pred are label lists; pred entries outside `labels` (or None) count as invalid."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * (len(labels) + 1) for _ in labels]
    for g, p in zip(gold, pred):
        matrix[index[g]][index.get(p, len(labels))] += 1
    per_class, f1s, supported = {}, [], []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(matrix[j][i] for j in range(len(labels)))
        p, r, f = _prf(tp, predicted - tp, support - tp)
        per_class[label] = {"precision": p, "recall": r, "f1": f, "support": support, "predicted": predicted}
        f1s.append(f)
        if support:
            supported.append(f)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    return {
        "n": len(gold),
        "accuracy": _ratio(correct, len(gold)),
        "macro_f1": _ratio(sum(f1s), len(f1s)),
        "macro_f1_supported": _ratio(sum(supported), len(supported)),
        "per_class": per_class,
        "confusion": {"labels": [*labels, INVALID], "matrix": matrix},
    }


def record_evidence(gold, pred, sentences, ok=True):
    """Score one record. Returns a dict with p, r, f1, tp, fp, fn, invalid (out-of-range ids) and gold_nonempty."""
    gold, pred = set(gold), set(pred)
    base = {"gold_nonempty": bool(gold), "invalid": 0}
    if not ok:
        return {**base, "p": 0.0, "r": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(gold)}
    base["invalid"] = sum(not 0 <= i < sentences for i in pred)
    if not gold and not pred:
        return {**base, "p": 1.0, "r": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    tp, fp, fn = len(gold & pred), len(pred - gold), len(gold - pred)
    p, r, f1 = _prf(tp, fp, fn)
    return {**base, "p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evidence_report(records):
    def macro(items):
        return {name: _ratio(sum(x[key] for x in items), len(items)) for name, key in (("precision", "p"), ("recall", "r"), ("f1", "f1"))}

    tp, fp, fn = (sum(x[k] for x in records) for k in ("tp", "fp", "fn"))
    p, r, f1 = _prf(tp, fp, fn)
    nonempty = [x for x in records if x["gold_nonempty"]]
    return {
        "n": len(records),
        "n_gold_nonempty": len(nonempty),
        "micro": {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn},
        "macro": macro(records),
        "macro_gold_nonempty": macro(nonempty),
        "invalid_ids": sum(x["invalid"] for x in records),
    }


def evaluate(rows, texts, repair=False):
    """Score raw model outputs against rows (usable rows of one split). Returns a JSON-serializable dict."""
    if len(rows) != len(texts):
        raise ValueError(f"{len(rows)} rows but {len(texts)} outputs")
    if any(not r.usable for r in rows):
        raise ValueError("unusable rows must be dropped before evaluation")
    parsed = [parse_output(t, repair) for t in texts]
    preds = [p.label if p.ok and p.label in LABELS[r.criterion_type] else None for r, p in zip(rows, parsed)]
    golds = [r.label for r in rows]
    evidence = [record_evidence(r.evidence_ids, p.evidence_ids, len(parse_note(r.note)), p.ok) for r, p in zip(rows, parsed)]
    report = {
        "n": len(rows),
        "malformed": sum(not p.ok for p in parsed),
        "invalid_label": sum(p.ok and label is None for p, label in zip(parsed, preds)),
        "repaired": sum(p.repaired for p in parsed),
        "errors": sorted({p.error for p in parsed if p.error}),
        "labels": {},
        "evidence": {},
    }
    report["malformed_rate"] = _ratio(report["malformed"], len(rows))
    for scope, labels in SCOPES.items():
        pick = [i for i, r in enumerate(rows) if scope == "overall" or r.criterion_type == scope]
        report["labels"][scope] = label_report([golds[i] for i in pick], [preds[i] for i in pick], labels)
        report["evidence"][scope] = evidence_report([evidence[i] for i in pick])
    return report
