"""Majority-class baseline. It emits raw output text, so it goes through the same parser and evaluator as a model."""
from collections import Counter

from trialmatch.data.prompt import build_target


def majority_outputs(train_rows, rows, by_type=True):
    """Predict the most frequent training label, with empty evidence. Ties break on label order for determinism."""
    def top(items):
        counts = Counter(r.label for r in items)
        return min(counts, key=lambda label: (-counts[label], label))

    overall = top(train_rows)
    per_type = {t: top([r for r in train_rows if r.criterion_type == t]) for t in {r.criterion_type for r in train_rows}}
    return [build_target([], per_type[r.criterion_type] if by_type else overall) for r in rows]
