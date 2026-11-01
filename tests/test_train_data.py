import math
from collections import Counter

import pytest

from trialmatch.data.encode import IGNORE
from trialmatch.data.split import load_manifest, select_rows
from trialmatch.train.data import RARE, batches, collate, cosine_factor, encode_rows, epoch_order, oversample
from tests.test_data import toy_encode


def test_oversample_repeats_only_rare_labels(rows):
    train = select_rows(rows, load_manifest(), "train")
    base, more = Counter(r.label for r in train), Counter(r.label for r in oversample(train, 3))
    for label, n in base.items():
        assert more[label] == n * (3 if label in RARE else 1)
    assert oversample(train, 1) == train
    with pytest.raises(ValueError):
        oversample(train, 0)


def test_epoch_order_is_a_reproducible_permutation():
    a, b = epoch_order(50, 42, 0), epoch_order(50, 42, 0)
    assert a == b and sorted(a) == list(range(50))
    assert epoch_order(50, 42, 1) != a and epoch_order(50, 43, 0) != a


def test_batches_cover_everything_once():
    out = batches(list(range(10)), 4)
    assert [len(x) for x in out] == [4, 4, 2] and sum(out, []) == list(range(10))


def test_collate_pads_and_masks(rows):
    train = [r for r in select_rows(rows, load_manifest(), "val")][:3]
    examples = encode_rows(train, toy_encode)
    out = collate(examples, pad_id=7)
    longest = max(len(e["input_ids"]) for e in examples)
    for e, ids, mask, labels in zip(examples, out["input_ids"], out["attention_mask"], out["labels"]):
        n = len(e["input_ids"])
        assert len(ids) == len(mask) == len(labels) == longest
        assert ids[:n] == e["input_ids"] and ids[n:] == [7] * (longest - n)
        assert mask == [1] * n + [0] * (longest - n)
        assert labels[:n] == e["labels"] and labels[n:] == [IGNORE] * (longest - n)


def test_cosine_schedule_shape():
    total, warmup = 100, 5
    factors = [cosine_factor(s, total, warmup) for s in range(total)]
    assert factors[0] == pytest.approx(0.2) and factors[warmup - 1] == pytest.approx(1.0)
    assert factors[warmup] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(factors[warmup:], factors[warmup + 1:]))
    assert factors[-1] < 0.01 and math.isclose(cosine_factor(total, total, warmup), 0.0, abs_tol=1e-12)
    assert cosine_factor(0, 10, 0) == pytest.approx(1.0)
