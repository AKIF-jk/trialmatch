"""Summaries on top of the Phase 2 evaluator: patient-level bootstrap intervals and mean/spread over seeds.

Nothing here changes how a prediction is scored. The bootstrap reuses the evaluator's own record scoring, and a
test checks that its point estimates equal `evaluate`.
"""
import random
import statistics

from trialmatch.data.load import LABELS
from trialmatch.data.note import parse_note
from trialmatch.eval.metrics import SCOPES, evaluate, evidence_report, label_report, record_evidence
from trialmatch.eval.parse import parse_output

BOOTSTRAP_METRICS = ("accuracy", "macro_f1", "evidence_macro_f1")


def score_records(rows, texts, repair=False):
    """Per-record gold label, predicted label (None if malformed or invalid for the type) and evidence score."""
    parsed = [parse_output(t, repair) for t in texts]
    gold = [r.label for r in rows]
    pred = [p.label if p.ok and p.label in LABELS[r.criterion_type] else None for r, p in zip(rows, parsed)]
    evidence = [record_evidence(r.evidence_ids, p.evidence_ids, len(parse_note(r.note)), p.ok) for r, p in zip(rows, parsed)]
    return gold, pred, evidence


def _metrics(gold, pred, evidence):
    labels = label_report(gold, pred, SCOPES["overall"])
    return {
        "accuracy": labels["accuracy"],
        "macro_f1": labels["macro_f1"],
        "evidence_macro_f1": evidence_report(evidence)["macro"]["f1"],
    }


def _quantile(values, q):
    values = sorted(values)
    pos = q * (len(values) - 1)
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def patient_bootstrap(rows, texts, repair=False, n=1000, seed=0, level=0.95):
    """Percentile interval from resampling whole patients with replacement, since rows of one patient are not independent."""
    gold, pred, evidence = score_records(rows, texts, repair)
    by_patient = {}
    for i, row in enumerate(rows):
        by_patient.setdefault(row.patient_id, []).append(i)
    patients = sorted(by_patient)
    rng = random.Random(seed)
    draws = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(n):
        picked = [i for p in rng.choices(patients, k=len(patients)) for i in by_patient[p]]
        scores = _metrics([gold[i] for i in picked], [pred[i] for i in picked], [evidence[i] for i in picked])
        for name in BOOTSTRAP_METRICS:
            draws[name].append(scores[name])
    point = _metrics(gold, pred, evidence)
    tail = (1 - level) / 2
    result = {"n_resamples": n, "n_patients": len(patients), "seed": seed, "level": level}
    for name in BOOTSTRAP_METRICS:
        result[name] = {"point": point[name], "lo": _quantile(draws[name], tail), "hi": _quantile(draws[name], 1 - tail)}
    return result


def score_texts(rows, texts, boot_n=1000, seed=0):
    """Strict and repaired evaluator reports plus the strict bootstrap interval for one set of raw outputs."""
    return {
        "strict": evaluate(rows, texts),
        "repair": evaluate(rows, texts, repair=True),
        "bootstrap_strict": patient_bootstrap(rows, texts, n=boot_n, seed=seed),
    }


def headline(report):
    """Flat dict of the numbers the paper tables use, from one `evaluate` report."""
    overall = report["labels"]["overall"]
    flat = {
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "macro_f1_supported": overall["macro_f1_supported"],
        "evidence_micro_f1": report["evidence"]["overall"]["micro"]["f1"],
        "evidence_macro_f1": report["evidence"]["overall"]["macro"]["f1"],
        "malformed": report["malformed"],
    }
    for label, stats in overall["per_class"].items():
        flat[f"f1[{label}]"] = stats["f1"]
    return flat


def mean_sd(values):
    """Mean and sample standard deviation (n - 1), the convention used for the validation seed table."""
    values = list(values)
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def seed_summary(reports):
    """reports maps a seed to its `evaluate` report. Returns per-seed values with mean and spread for each headline number."""
    flat = {seed: headline(report) for seed, report in reports.items()}
    names = next(iter(flat.values()))
    return {name: {"per_seed": {seed: flat[seed][name] for seed in flat}, **mean_sd(flat[seed][name] for seed in flat)} for name in names}
