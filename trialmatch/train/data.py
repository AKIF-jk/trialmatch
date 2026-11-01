"""Training data helpers that need no torch: encoding, rare-class oversampling, batching, LR schedule."""
import math

import numpy as np

from trialmatch.data.encode import IGNORE, encode_example

RARE = ("excluded", "not included")


def encode_rows(rows, encode, order="evidence_first"):
    return [encode_example(row, encode, order) for row in rows]


def oversample(rows, factor, labels=RARE):
    """Repeat rows whose label is in `labels` so each appears `factor` times. factor 1 changes nothing."""
    if factor < 1:
        raise ValueError("factor must be at least 1")
    return [row for row in rows for _ in range(factor if row.label in labels else 1)]


def epoch_order(n, seed, epoch):
    return np.random.default_rng([seed, epoch]).permutation(n).tolist()


def batches(indices, size):
    return [indices[i:i + size] for i in range(0, len(indices), size)]


def collate(examples, pad_id):
    """Right-pad to the longest example. Padding gets mask 0 and label IGNORE."""
    longest = max(len(e["input_ids"]) for e in examples)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for e in examples:
        pad = longest - len(e["input_ids"])
        out["input_ids"].append(e["input_ids"] + [pad_id] * pad)
        out["attention_mask"].append([1] * len(e["input_ids"]) + [0] * pad)
        out["labels"].append(e["labels"] + [IGNORE] * pad)
    return out


def cosine_factor(step, total, warmup):
    """Linear warmup to 1, then cosine decay to 0. step counts optimizer steps taken so far."""
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
