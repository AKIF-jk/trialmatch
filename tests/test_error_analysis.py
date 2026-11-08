import json

from trialmatch.analysis.errors import analyse, summarise, tag_record
from trialmatch.data.load import Row
from trialmatch.data.note import format_note

NOTE = format_note(["a", "b", "c", "d"])


def row(i, kind, label, evidence=()):
    return Row(i, "p", "t", NOTE, kind, "criterion", tuple(evidence), label)


def out(label, ids):
    return json.dumps({"evidence_sentence_ids": ids, "label": label})


def tags(r, text):
    return tag_record(r, text)["tags"]


def test_exact_prediction_has_no_tags():
    assert tags(row(1, "inclusion", "included", [0, 2]), out("included", [0, 2])) == []


def test_wrong_label_with_extra_and_missing_evidence():
    found = tags(row(1, "inclusion", "included", [0, 1]), out("not included", [1, 3]))
    assert found == ["label_wrong", "evidence_unsupported", "evidence_missed"]


def test_nothing_cited_where_expert_cited():
    found = tags(row(1, "exclusion", "not excluded", [2]), out("not excluded", []))
    assert found == ["evidence_missed", "evidence_none_given"]


def test_cites_where_expert_cited_nothing_and_information_gap_with_evidence():
    found = tags(row(1, "inclusion", "not enough information"), out("not enough information", [1]))
    assert found == ["evidence_unsupported", "cites_but_gold_empty", "info_gap_with_evidence"]


def test_decision_without_evidence_and_rare_gold():
    found = tags(row(1, "inclusion", "not included"), out("included", []))
    assert found == ["label_wrong", "decision_without_evidence", "rare_class_gold"]


def test_gold_included_without_expert_evidence_is_flagged():
    assert "gold_included_no_evidence" in tags(row(1, "inclusion", "included"), out("included", [0]))


def test_malformed_output_is_a_wrong_label_and_nothing_else():
    rec = tag_record(row(1, "inclusion", "included", [0]), "not json")
    assert rec["tags"] == ["label_wrong"] and rec["malformed"] and rec["pred_label"] is None


def test_label_invalid_for_the_criterion_type_counts_as_wrong():
    rec = tag_record(row(1, "inclusion", "included", [0]), out("not excluded", [0]))
    assert "label_wrong" in rec["tags"] and rec["pred_label"] is None


def test_summary_counts_pairs_and_rows_wrong_in_every_model():
    rows = [row(1, "exclusion", "not enough information"), row(2, "inclusion", "included", [0]), row(3, "inclusion", "included", [0])]
    texts = {
        "a": [out("not excluded", []), out("included", [0]), out("not included", [0])],
        "b": [out("not excluded", []), out("included", [0]), out("included", [0])],
    }
    summary = summarise(analyse(rows, texts))
    assert summary["wrong_in_all_models"] == [1]
    assert summary["models"]["a"]["wrong_label_pairs"] == {"not enough information -> not excluded": 1, "included -> not included": 1}
    assert summary["models"]["b"]["tag_counts"]["label_wrong"] == 1


def test_mismatched_lengths_are_rejected():
    import pytest

    with pytest.raises(ValueError):
        analyse([row(1, "inclusion", "included")], {"a": []})
