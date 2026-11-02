"""Gradio wiring with stub model functions: input validation, error recovery, reports. Needs gradio, not torch."""
import json
from pathlib import Path

import pytest

pytest.importorskip("gradio")
from gradio_client import Client

from trialmatch.app import screen
from trialmatch.app.ui import EXAMPLES, build_ui

PROV = {"device": "stub", "dtype": "float32", "app_commit": "test"}


def stub_screen(sentences, criteria, rationale):
    outputs = {
        "inclusion": '{"evidence_sentence_ids":[0],"label":"included"}',
        "exclusion": '{"evidence_sentence_ids":[],"label":"not enough information"}',
    }
    results = [screen.interpret(i, kind, t, outputs[kind], len(sentences)) for i, (kind, t) in enumerate(criteria)]
    if rationale:
        results[0].rationale = "stub rationale"
    return results, {"criteria": len(criteria)}, PROV


class Calls:
    lab = []
    fail = None


def stub_lab(sentences, ctype, criterion, modes, backend, tile, k, draft, new_tokens, force, repeats):
    Calls.lab.append((modes, backend, tile, k, draft, new_tokens, force, repeats))
    if Calls.fail:
        raise Calls.fail
    rows = [
        {"mode": m, "backend": backend, "tokens": 14, "tokens_equal": True, "prefill_seconds": 0.1, "decode_seconds": 0.4,
         "total_seconds": 0.5, "tokens_per_second": 28.0, "peak_extra_mb": None, "acceptance": 0.9 if m == "speculative" else None,
         "tokens_per_round": 2.5 if m == "speculative" else None, "target_forwards": 6 if m == "speculative" else None,
         "draft_seconds": 0.2, "verify_seconds": 0.2}
        for m in modes
    ]
    return rows, '{"evidence_sentence_ids":[0],"label":"included"}', PROV


@pytest.fixture(scope="module")
def client():
    demo = build_ui(stub_screen, stub_lab, lookup_fn=lambda nct: (_ for _ in ()).throw(AssertionError("no network in tests")))
    _, url, _ = demo.launch(prevent_thread_lock=True, server_name="127.0.0.1", quiet=True)
    try:
        yield Client(url, verbose=False)
    finally:
        demo.close()


NUMBERED = "0. Man with diabetes.\n1. No pregnancy."


def test_segment_and_renumber(client):
    numbered, status = client.predict("A man has diabetes. He takes metformin.", api_name="/segment")
    assert numbered == "0. A man has diabetes.\n1. He takes metformin." and status == ""
    assert client.predict("0. a\n7. b", api_name="/renumber") == "0. a\n1. b"


def test_segment_empty_keeps_numbered_and_explains(client):
    _, status = client.predict("   ", api_name="/segment")
    assert "empty" in status


def test_screen_end_to_end(client):
    status, outcome, table, detail, md, js = client.predict(NUMBERED, "Diabetes\nAge 18+", "Pregnancy", True, api_name="/screen")
    assert "Done" in status and "Educational screening prototype" in status
    assert "Needs review" in outcome
    assert [row[3] for row in table["data"]] == ["included", "included", "not enough information"]
    assert "<mark>" in detail and "stub rationale" in detail and "Man with diabetes." in detail
    text = Path(md).read_text()
    assert "Educational screening prototype, not a clinical enrollment decision." in text and "Needs review" in text
    report = json.loads(Path(js).read_text())
    assert report["provenance"]["app_commit"] == "test" and len(report["criteria"]) == 3 and report["timing"]["criteria"] == 3


@pytest.mark.parametrize("numbered,inc,exc,expect", [
    ("", "Diabetes", "", "empty"),
    ("0. a\n3. b", "Diabetes", "", "Renumber"),
    (NUMBERED, "", "", "No criteria"),
    (NUMBERED, "\n".join(f"criterion number {i}" for i in range(13)), "", "limit"),
    (NUMBERED, "12 34", "", "Malformed"),
    (NUMBERED, "x" * 700, "", "characters"),
])
def test_invalid_inputs_give_message_and_no_outputs(client, numbered, inc, exc, expect):
    status, outcome, table, detail, md, js = client.predict(numbered, inc, exc, False, api_name="/screen")
    assert "Cannot run" in status and expect in status
    assert outcome == "" and table["data"] == [[""] * 6] or table["data"] == []
    assert md is None and js is None


def test_recovery_after_error(client):
    assert "Cannot run" in client.predict("", "x y z", "", False, api_name="/screen")[0]
    assert "Done" in client.predict(NUMBERED, "Diabetes", "", False, api_name="/screen")[0]


def test_lab_end_to_end_and_arguments(client):
    Calls.lab.clear()
    Calls.fail = None
    status, table, commentary, generated, js = client.predict(
        NUMBERED, "inclusion", "Diabetes", ["no cache", "KV cache", "speculative", "speculative (prompt lookup)"], "tiled online-softmax", 16, 4, "base", 24, True, 2, api_name="/lab"
    )
    assert Calls.lab[-1] == (["no cache", "KV cache", "speculative", "speculative (prompt lookup)"], "tiled", 16, 4, "base", 24, True, 2)
    assert "Live run" in status and len(table["data"]) == 4 and table["data"][3][3] == "yes"
    assert "same tokens" in commentary and "label" in generated
    assert json.loads(Path(js).read_text())["provenance"]["device"] == "stub"


def test_lab_validation_and_gpu_error(client):
    assert "at least one" in client.predict(NUMBERED, "inclusion", "Diabetes", [], "sdpa", 64, 2, "LoRA", 32, False, 1, api_name="/lab")[0]
    assert "Cannot run" in client.predict(NUMBERED, "inclusion", "", ["KV cache"], "sdpa", 64, 2, "LoRA", 32, False, 1, api_name="/lab")[0]
    Calls.fail = RuntimeError("ZeroGPU quota exceeded for this visitor")
    try:
        status = client.predict(NUMBERED, "inclusion", "Diabetes", ["KV cache"], "sdpa", 64, 2, "LoRA", 32, False, 1, api_name="/lab")[0]
    finally:
        Calls.fail = None
    assert "quota" in status and "try again" in status


def test_examples_are_valid():
    from trialmatch.app import text

    for example in EXAMPLES.values():
        sentences = text.validate_note(text.segment_note(example["note"]))
        text.validate_criteria(text.split_items(example["inclusion"]), text.split_items(example["exclusion"]))
        assert len(sentences) >= 4
