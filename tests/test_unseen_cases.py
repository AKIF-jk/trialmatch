import json
from pathlib import Path

from trialmatch.data.load import LABELS
from trialmatch.data.note import parse_note
from trialmatch.eval.metrics import evaluate

REPO = Path(__file__).resolve().parent.parent
SPEC = json.loads((REPO / "configs/unseen_cases.json").read_text())


def test_cases_are_well_formed_and_cover_every_label():
    seen = set()
    for patient in SPEC["patients"]:
        assert 4 <= len(patient["sentences"]) <= 8
        for c in patient["criteria"]:
            assert c["label"] in LABELS[c["type"]], (patient["id"], c["text"])
            assert all(0 <= i < len(patient["sentences"]) for i in c["evidence"]), (patient["id"], c["text"])
            assert c["evidence"] == sorted(set(c["evidence"]))
            seen.add(c["label"])
    assert seen == {"included", "not included", "excluded", "not excluded", "not enough information", "not applicable"}


def test_no_hand_written_sentence_comes_from_the_dataset(rows):
    dataset = {s.strip().lower() for r in rows for s in parse_note(r.note)}
    own = [s for patient in SPEC["patients"] for s in patient["sentences"]]
    assert not [s for s in own if s.strip().lower() in dataset]
    criteria = {(r.criterion_text or "").strip().lower() for r in rows}
    assert not [c["text"] for p in SPEC["patients"] for c in p["criteria"] if c["text"].strip().lower() in criteria]


def test_expected_answers_score_perfectly_with_the_evaluator():
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    from trialmatch.data.load import Row
    from trialmatch.data.note import format_note

    built = []
    for patient in SPEC["patients"]:
        note = format_note(patient["sentences"])
        for c in patient["criteria"]:
            built.append(Row(len(built), patient["id"], "unseen", note, c["type"], c["text"], tuple(c["evidence"]), c["label"]))
    texts = [json.dumps({"evidence_sentence_ids": list(r.evidence_ids), "label": r.label}) for r in built]
    report = evaluate(built, texts)
    assert report["labels"]["overall"]["accuracy"] == 1.0
    assert report["evidence"]["overall"]["macro"]["f1"] == 1.0
