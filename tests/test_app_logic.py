"""App logic that needs no model: segmentation, criteria parsing, limits, aggregation, explanations and reports."""
import json

import pytest

from trialmatch.app import screen, text
from trialmatch.app.screen import ELIGIBLE, INELIGIBLE, REVIEW, interpret
from trialmatch.data.note import parse_note

NOTE = [
    "A 45 y.o. man presents with chest pain.", "BP was 120/80 and temp 98.6 F.", "He has no history of diabetes.",
    "Dr. Smith started aspirin.", "Plan: admit.",
]


def test_segment_basic_and_abbreviations():
    got = text.segment_note("A 45 y.o. man presents with chest pain. BP was 120/80 and temp 98.6 F. He has no history of diabetes. Dr. Smith started aspirin.")
    assert got == NOTE[:4]


def test_segment_lines_and_blank_lines():
    assert text.segment_note("First line\n\n  second   line  \r\nThird. Fourth!") == ["First line", "second line", "Third.", "Fourth!"]


def test_segment_empty():
    assert text.segment_note("  \n \n") == []


def test_segment_question_mark_and_bracket():
    assert text.segment_note("Is he diabetic? (No.) Fine.") == ["Is he diabetic?", "(No.)", "Fine."]


def test_numbered_round_trip():
    assert text.parse_numbered_note(text.numbered(NOTE)) == NOTE
    assert parse_note(text.numbered(NOTE)) == NOTE


def test_numbering_broken_gives_fix():
    with pytest.raises(text.InputError, match="Renumber"):
        text.parse_numbered_note("0. a\n2. b")
    with pytest.raises(text.InputError, match="Renumber"):
        text.parse_numbered_note("just words")


def test_renumber_rebuilds():
    assert text.renumber("0. a\n5. b\n\n  c  \n9) d") == ["a", "b", "c", "d"]
    assert text.parse_numbered_note(text.numbered(text.renumber("0. a\n5. b"))) == ["a", "b"]


def test_note_limits():
    with pytest.raises(text.InputError, match="empty"):
        text.parse_numbered_note("   ")
    with pytest.raises(text.InputError, match="limit"):
        text.validate_note(["x"] * (text.MAX_SENTENCES + 1))
    with pytest.raises(text.InputError, match="characters"):
        text.validate_note(["x" * 5000, "y" * 5000])
    with pytest.raises(text.InputError, match="empty"):
        text.validate_note(["ok", " "])


def test_split_items_bullets_with_continuations():
    block = "* Type 2 diabetes\n  confirmed by test\n* Age over 18\nSome trailing words"
    assert text.split_items(block) == ["Type 2 diabetes confirmed by test", "Age over 18 Some trailing words"]


def test_split_items_numbered_and_plain():
    assert text.split_items("1. aaa\n2) bbb\n(a) ccc") == ["aaa", "bbb", "ccc"]
    assert text.split_items("aaa\n\nbbb\n") == ["aaa", "bbb"]
    assert text.split_items("Patients must have:\n- aaa\n- bbb") == ["Patients must have:", "aaa", "bbb"]


def test_split_eligibility_headings():
    eligibility = "Inclusion Criteria:\n\n* adult\n* diabetes\n\nExclusion Criteria:\n\n* pregnancy\n* dialysis"
    inc, exc, warnings = text.split_eligibility(eligibility)
    assert (inc, exc, warnings) == (["adult", "diabetes"], ["pregnancy", "dialysis"], [])


def test_split_eligibility_exclusion_first_and_preamble():
    inc, exc, warnings = text.split_eligibility("Intro text\nExclusion Criteria:\n- a1\nInclusion Criteria:\n- b2")
    assert (inc, exc) == (["b2"], ["a1"])
    assert warnings and "ignored" in warnings[0]


def test_split_eligibility_without_headings_warns():
    inc, exc, warnings = text.split_eligibility("- a\n- b")
    assert (inc, exc) == (["a", "b"], []) and "heading" in warnings[0]


def test_validate_criteria():
    assert text.validate_criteria(["abc"], ["def"]) == [("inclusion", "abc"), ("exclusion", "def")]
    with pytest.raises(text.InputError, match="No criteria"):
        text.validate_criteria([], [])
    with pytest.raises(text.InputError, match="limit"):
        text.validate_criteria(["abc"] * (text.MAX_CRITERIA + 1), [])
    with pytest.raises(text.InputError, match="Malformed"):
        text.validate_criteria(["12 34"], [])
    with pytest.raises(text.InputError, match="Malformed"):
        text.validate_criteria([], ["ab"])
    with pytest.raises(text.InputError, match="characters"):
        text.validate_criteria(["x" * (text.MAX_CRITERION_CHARS + 1)], [])


def test_fetch_trial_rejects_bad_id_without_network():
    for bad in ("", "nct123", "NCT1234567", "NCT123456789", "../etc"):
        with pytest.raises(text.InputError, match="NCT"):
            text.fetch_trial(bad)


def test_fetch_trial_parses(monkeypatch):
    payload = {"protocolSection": {"identificationModule": {"briefTitle": "T"}, "eligibilityModule": {"eligibilityCriteria": "Inclusion Criteria:\n* a"}}}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *args):
            return json.dumps(payload).encode()

    seen = []
    monkeypatch.setattr(text.urllib.request, "urlopen", lambda url, timeout: (seen.append(url), Response())[1])
    got = text.fetch_trial(" nct00000620 ")
    assert got == {"nct_id": "NCT00000620", "title": "T", "eligibility": "Inclusion Criteria:\n* a"}
    assert seen[0].startswith("https://clinicaltrials.gov/api/v2/studies/NCT00000620?") and len(seen) == 1


def test_fetch_trial_errors(monkeypatch):
    def not_found(url, timeout):
        raise text.urllib.error.HTTPError(url, 404, "nf", {}, None)

    monkeypatch.setattr(text.urllib.request, "urlopen", not_found)
    with pytest.raises(text.InputError, match="no study"):
        text.fetch_trial("NCT00000620")

    def offline(url, timeout):
        raise text.urllib.error.URLError("down")

    monkeypatch.setattr(text.urllib.request, "urlopen", offline)
    with pytest.raises(text.InputError, match="could not be reached"):
        text.fetch_trial("NCT00000620")


# ---- interpretation ------------------------------------------------------------------------------------------------

def make(ctype, raw, n=5):
    return interpret(0, ctype, "criterion text", raw, n)


def test_interpret_ok():
    r = make("inclusion", '{"evidence_sentence_ids":[2,0],"label":"included"}')
    assert (r.status, r.label, r.evidence_ids, r.dropped_ids) == ("ok", "included", [0, 2], [])


def test_interpret_repairs_code_fence():
    r = make("exclusion", '```json\n{"evidence_sentence_ids":[1],"label":"not excluded"}\n```')
    assert (r.status, r.label, r.evidence_ids) == ("repaired", "not excluded", [1])
    assert any("first JSON object" in m for m in screen.missing_information(r))


def test_interpret_unparseable_and_wrong_shape():
    for raw in ("", "no json here", "{", '{"label":"included"}', "[1,2]", '{"evidence_sentence_ids":"0","label":"included"}'):
        r = make("inclusion", raw)
        assert (r.status, r.label) == ("unparsed", None), raw


def test_interpret_label_from_other_type_is_invalid():
    r = make("inclusion", '{"evidence_sentence_ids":[],"label":"not excluded"}')
    assert (r.status, r.label) == ("invalid_label", None)
    assert make("exclusion", '{"evidence_sentence_ids":[],"label":"included"}').label is None


def test_interpret_drops_out_of_range_ids():
    r = make("inclusion", '{"evidence_sentence_ids":[0,5,99],"label":"included"}', n=5)
    assert (r.evidence_ids, r.dropped_ids) == ([0], [5, 99])
    assert any("[5, 99]" in m for m in screen.missing_information(r))


# ---- aggregation ---------------------------------------------------------------------------------------------------

def results(*pairs):
    return [screen.CriterionResult(i, t, "c", "ok" if label else "unparsed", label) for i, (t, label) in enumerate(pairs)]


@pytest.mark.parametrize("pairs,outcome", [
    ([("inclusion", "included"), ("exclusion", "not excluded")], ELIGIBLE),
    ([("inclusion", "not included"), ("exclusion", "not excluded")], INELIGIBLE),
    ([("inclusion", "included"), ("exclusion", "excluded")], INELIGIBLE),
    ([("inclusion", "not enough information"), ("exclusion", "not excluded")], REVIEW),
    ([("inclusion", "not included"), ("exclusion", "not enough information")], INELIGIBLE),
    ([("inclusion", "included"), ("exclusion", None)], REVIEW),
    ([("inclusion", "not applicable"), ("exclusion", "not applicable")], ELIGIBLE),
    ([("inclusion", "not applicable"), ("exclusion", "not enough information")], REVIEW),
    ([("inclusion", "excluded")], ELIGIBLE),  # Not a valid inclusion label, so only the exact rule strings block.
])
def test_aggregate_rules(pairs, outcome):
    assert screen.aggregate(results(*pairs))[0] == outcome


def test_aggregate_reasons_name_criteria():
    outcome, reasons = screen.aggregate(results(("inclusion", "included"), ("exclusion", "excluded"), ("inclusion", "not included")))
    assert outcome == INELIGIBLE and len(reasons) == 2 and "Criterion 2" in reasons[0] and "Criterion 3" in reasons[1]


def test_all_not_applicable_says_nothing_checked():
    assert "nothing was checked" in screen.aggregate(results(("inclusion", "not applicable")))[1][0]


# ---- explanation, missing information, display ---------------------------------------------------------------------

def test_template_uses_only_cited_sentences():
    r = make("inclusion", '{"evidence_sentence_ids":[2],"label":"not included"}')
    explanation = screen.template_explanation(r, NOTE)
    assert "[2]" in explanation and NOTE[2] in explanation and NOTE[0] not in explanation and "not included" in explanation


def test_empty_evidence_does_not_change_label_but_flags():
    r = make("exclusion", '{"evidence_sentence_ids":[],"label":"not excluded"}')
    assert r.label == "not excluded"
    assert "No sentence was cited" in screen.template_explanation(r, NOTE)
    assert any("no cited sentence" in m for m in screen.missing_information(r))


def test_not_enough_information_missing_text():
    r = make("inclusion", '{"evidence_sentence_ids":[],"label":"not enough information"}')
    assert screen.missing_information(r) == ["The note does not say enough to decide: criterion text"]
    assert make("inclusion", '{"evidence_sentence_ids":[],"label":"not applicable"}').label == "not applicable"
    assert screen.missing_information(make("inclusion", '{"evidence_sentence_ids":[],"label":"not applicable"}')) == []


def test_unreadable_output_text():
    r = make("inclusion", "garbage")
    assert "could not be read" in screen.template_explanation(r, NOTE)
    assert r.shown_label == "unreadable output"


def test_highlight_escapes_markup():
    out = screen.highlight_html(["<script>x</script>", "ok"], [0])
    assert "<script>" not in out and "&lt;script&gt;" in out and out.count("<mark>") == 1


# ---- report --------------------------------------------------------------------------------------------------------

def test_report_contents():
    res = [make("inclusion", '{"evidence_sentence_ids":[0],"label":"included"}'), make("exclusion", "garbage")]
    res[1].index = 1
    res[0].rationale = "Because chest pain."
    prov = {"commit": "abc", "decode": {"do_sample": False}}
    report = screen.report_dict(NOTE, res, prov, extra={"timing": {"total_seconds": 1.0}})
    assert report["notice"] == screen.NOTICE and report["provenance"] == prov and report["timing"]["total_seconds"] == 1.0
    assert report["overall_outcome"]["outcome"] == REVIEW
    assert report["criteria"][0]["cited_sentences"] == [{"id": 0, "text": NOTE[0]}]
    json.dumps(report)
    md = screen.report_markdown(report)
    assert screen.NOTICE in md and md.count(screen.NOTICE) == 2
    assert "Needs review" in md and "unreadable output" in md and "Model rationale" in md and '"commit": "abc"' in md
    assert NOTE[0] in md


def lab_rows(speed=1.0):
    row = lambda mode, total, **extra: {"mode": mode, "backend": "sdpa", "tokens": 20, "tokens_equal": True, "total_seconds": total, "prefill_seconds": 0.1,
                                         "decode_seconds": total - 0.1, "tokens_per_second": 20 / total, **extra}
    return [
        row("no cache", 2.0), row("KV cache", 1.0),
        row("speculative", 1.2, acceptance=0.9, tokens_per_round=2.5, target_forwards=9, draft_seconds=0.5, verify_seconds=0.6),
        row("speculative (prompt lookup)", 0.6, acceptance=0.5, tokens_per_round=1.8, target_forwards=11, draft_seconds=0.0, verify_seconds=0.5),
    ]


def test_lab_commentary_compares_each_speculative_mode():
    text = screen.lab_commentary(lab_rows())
    assert "same tokens" in text and "2.00x the speed of no cache" in text
    assert "Speculative is 0.83x the speed of KV cache (slower)" in text
    assert "Speculative (prompt lookup) is 1.67x the speed of KV cache (faster)" in text
    assert text.count("Cost per round") == 2


def test_lab_commentary_flags_mismatch_and_handles_missing_modes():
    rows = lab_rows()
    rows[2]["tokens_equal"] = False
    assert "Token mismatch in: speculative." in screen.lab_commentary(rows)
    assert "x the speed of KV cache" not in screen.lab_commentary(rows[:1])
    table = screen.lab_table(lab_rows())
    assert [r[3] for r in table] == ["yes"] * 4 and table[3][9] == "0.50" and table[0][9] == "n/a"
