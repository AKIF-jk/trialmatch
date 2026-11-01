import json
import math

import pytest

from trialmatch.data.load import Row
from trialmatch.data.note import format_note
from trialmatch.data.prompt import build_target, row_target
from trialmatch.data.split import load_manifest, select_rows
from trialmatch.eval.baselines import majority_outputs
from trialmatch.eval.metrics import INVALID, SCOPES, evaluate, label_report, record_evidence
from trialmatch.eval.parse import parse_output

NOTE = format_note(["a", "b", "c", "d"])


def row(i, kind, label, evidence=(), text="crit"):
    return Row(i, "p", "t", NOTE, kind, text, tuple(evidence), label)


def out(label, ids):
    return json.dumps({"evidence_sentence_ids": ids, "label": label})


def test_parse_valid_and_dedup():
    p = parse_output('{"evidence_sentence_ids":[2,0,2],"label":"included"}')
    assert p.ok and p.label == "included" and p.evidence_ids == (0, 2) and not p.repaired


@pytest.mark.parametrize("text", [
    "", "not json", "[]", '{"label":"included"}', '{"label":1,"evidence_sentence_ids":[]}',
    '{"label":"included","evidence_sentence_ids":"0"}', '{"label":"included","evidence_sentence_ids":[true]}',
    '{"label":"included","evidence_sentence_ids":[1.5]}', '{"label":"included","evidence_sentence_ids":[],"x":1}',
])
def test_parse_rejects_bad_shapes(text):
    p = parse_output(text)
    assert not p.ok and p.error


def test_repair_is_opt_in_and_flagged():
    wrapped = 'Sure:\n```json\n' + out("included", [1]) + '\n```'
    assert not parse_output(wrapped).ok
    p = parse_output(wrapped, repair=True)
    assert p.ok and p.repaired and p.evidence_ids == (1,)
    assert not parse_output("no braces", repair=True).ok


def test_label_report_hand_computed():
    labels = ("a", "b", "c")
    gold = ["a", "a", "b", "b", "b"]
    pred = ["a", "b", "b", "b", None]
    r = label_report(gold, pred, labels)
    assert r["accuracy"] == pytest.approx(3 / 5)
    a, b, c = (r["per_class"][x] for x in labels)
    assert (a["precision"], a["recall"]) == (1.0, 0.5)
    assert (b["precision"], b["recall"]) == (pytest.approx(2 / 3), pytest.approx(2 / 3))
    assert c == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0, "predicted": 0}
    assert r["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert r["macro_f1_supported"] == pytest.approx(2 / 3)
    assert r["confusion"]["labels"] == ["a", "b", "c", INVALID]
    assert r["confusion"]["matrix"] == [[1, 1, 0, 0], [0, 2, 0, 1], [0, 0, 0, 0]]


def test_evidence_empty_set_rules():
    assert record_evidence([], [], 4)["f1"] == 1.0
    miss = record_evidence([1], [], 4)
    assert (miss["p"], miss["r"], miss["f1"], miss["fn"]) == (0.0, 0.0, 0.0, 1)
    extra = record_evidence([], [1], 4)
    assert (extra["p"], extra["r"], extra["f1"], extra["fp"]) == (0.0, 0.0, 0.0, 1)
    bad = record_evidence([], [], 4, ok=False)
    assert (bad["p"], bad["r"], bad["f1"]) == (0.0, 0.0, 0.0)


def test_evidence_invalid_ids_are_false_positives():
    r = record_evidence([0], [0, 9, -1], 4)
    assert (r["tp"], r["fp"], r["fn"], r["invalid"]) == (1, 2, 0, 2)
    assert r["p"] == pytest.approx(1 / 3) and r["r"] == 1.0


def test_evaluate_end_to_end():
    rows = [
        row(1, "inclusion", "included", [0, 1]),
        row(2, "inclusion", "not enough information"),
        row(3, "exclusion", "not excluded"),
        row(4, "exclusion", "excluded", [2]),
        row(5, "exclusion", "not excluded", [3]),
    ]
    texts = [
        out("included", [0]),                 # right label, half the evidence
        out("not enough information", []),    # both empty
        out("included", []),                  # label invalid for exclusion type
        "garbage",                            # malformed
        out("not excluded", [3, 9]),          # right, one invalid id
    ]
    rep = evaluate(rows, texts)
    assert rep["n"] == 5 and rep["malformed"] == 1 and rep["invalid_label"] == 1
    assert rep["malformed_rate"] == pytest.approx(0.2)
    assert rep["labels"]["overall"]["accuracy"] == pytest.approx(3 / 5)
    assert rep["labels"]["inclusion"]["accuracy"] == 1.0
    assert rep["labels"]["exclusion"]["accuracy"] == pytest.approx(1 / 3)
    assert sum(c["support"] for c in rep["labels"]["overall"]["per_class"].values()) == 5
    assert rep["labels"]["exclusion"]["confusion"]["matrix"][0][-1] == 1  # "not excluded" row -> invalid
    ev = rep["evidence"]["overall"]
    assert ev["n"] == 5 and ev["n_gold_nonempty"] == 3 and ev["invalid_ids"] == 1
    assert ev["micro"]["tp"] == 2 and ev["micro"]["fp"] == 1 and ev["micro"]["fn"] == 2
    # f1 per record: 2/3, 1 (both empty), 1 (both empty; label invalid but JSON ok), 0 (malformed), 2/3
    assert ev["macro"]["f1"] == pytest.approx((2 / 3 + 1 + 1 + 0 + 2 / 3) / 5)
    assert ev["macro_gold_nonempty"]["f1"] == pytest.approx((2 / 3 + 0 + 2 / 3) / 3)
    json.dumps(rep)


def test_evaluate_input_checks():
    with pytest.raises(ValueError):
        evaluate([row(1, "inclusion", "included")], [])
    with pytest.raises(ValueError):
        evaluate([row(1, "inclusion", "included", text=None)], [out("included", [])])


def test_perfect_output_scores_one_on_every_scope():
    rows = [row(1, "inclusion", "included", [0]), row(2, "exclusion", "excluded", [1, 2])]
    rep = evaluate(rows, [row_target(r) for r in rows])
    for scope in SCOPES:
        assert rep["labels"][scope]["accuracy"] == 1.0
        assert rep["evidence"][scope]["macro"]["f1"] == 1.0
        assert rep["evidence"][scope]["micro"]["f1"] == 1.0


def test_majority_baseline_per_type_and_deterministic():
    train = [row(1, "inclusion", "included"), row(2, "inclusion", "included"), row(3, "inclusion", "not included"),
             row(4, "exclusion", "not excluded"), row(5, "exclusion", "excluded"),
             row(6, "exclusion", "not enough information"), row(7, "exclusion", "not excluded")]
    test = [row(8, "inclusion", "included"), row(9, "exclusion", "excluded")]
    outs = majority_outputs(train, test)
    assert outs == [build_target([], "included"), build_target([], "not excluded")]
    assert majority_outputs(train, test) == outs
    tie = [row(1, "inclusion", "not included"), row(2, "inclusion", "included")]
    assert majority_outputs(tie, tie[:1]) == [build_target([], "included")]
    rep = evaluate(test, outs)
    assert rep["labels"]["overall"]["accuracy"] == 0.5


def test_real_validation_split_oracle_and_baseline(rows):
    manifest = load_manifest()
    train, val = select_rows(rows, manifest, "train"), select_rows(rows, manifest, "val")
    oracle = evaluate(val, [row_target(r) for r in val])
    assert oracle["malformed"] == 0 and oracle["labels"]["overall"]["accuracy"] == 1.0
    assert oracle["evidence"]["overall"]["micro"]["f1"] == 1.0
    base = evaluate(val, majority_outputs(train, val))
    assert base["malformed"] == 0 and 0 < base["labels"]["overall"]["accuracy"] < 1
    assert math.isclose(base["evidence"]["overall"]["macro"]["f1"], sum(not r.evidence_ids for r in val) / len(val))
    assert sum(c["support"] for c in base["labels"]["overall"]["per_class"].values()) == len(val)
