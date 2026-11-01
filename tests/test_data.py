import copy
import dataclasses
import json
from collections import Counter

import pyarrow.parquet as pq
import pytest

from trialmatch.data.encode import IGNORE, TruncationError, encode_example
from trialmatch.data.load import LABELS, Row, dataset_path
from trialmatch.data.note import format_note, parse_note
from trialmatch.data.prompt import build_target, row_prompt, row_target
from trialmatch.data.split import SIZES, LockedSplitError, assign_patients, build_manifest, load_manifest, select_rows, verify_manifest


def test_split_is_reproducible(rows):
    assert build_manifest(rows) == build_manifest(rows)
    assert assign_patients(r.patient_id for r in reversed(rows)) == assign_patients(r.patient_id for r in rows)


def test_committed_manifest_matches_rebuild(rows):
    assert load_manifest() == build_manifest(rows)


def test_split_sizes_and_no_patient_overlap(rows):
    manifest = load_manifest()
    patients = manifest["patients"]
    assert {k: len(v) for k, v in patients.items()} == SIZES
    assert sum(len(v) for v in patients.values()) == len(set().union(*map(set, patients.values())))
    owner = {pid: name for name, pids in patients.items() for pid in pids}
    for split, ids in manifest["annotations"].items():
        assert {r.patient_id for r in rows if r.annotation_id in set(ids)} <= set(patients[split])
    all_ids = [a for ids in manifest["annotations"].values() for a in ids]
    assert sorted(all_ids) == [r.annotation_id for r in rows]
    assert {owner[r.patient_id] for r in rows} == set(SIZES)


def test_manifest_hash_detects_changes(rows):
    manifest = build_manifest(rows)
    tampered = copy.deepcopy(manifest)
    tampered["patients"]["test"][0] = tampered["patients"]["train"][0]
    with pytest.raises(ValueError):
        verify_manifest(tampered)


def test_manifest_records_revision_and_class_counts(rows):
    manifest = load_manifest()
    assert manifest["dataset_revision"] == "1cfcafde94a1560a33b4addc1638664fe28fc059"
    assert manifest["unusable_annotation_ids"] == [883]
    counted = Counter()
    for split in manifest["class_counts"].values():
        for kind in split.values():
            counted.update(kind)
    assert sum(counted.values()) == len(rows)
    for split, kinds in manifest["class_counts"].items():
        for kind, counts in kinds.items():
            assert set(counts) <= set(LABELS[kind])


def test_test_split_is_locked(rows):
    manifest = load_manifest()
    with pytest.raises(LockedSplitError):
        select_rows(rows, manifest, "test")
    assert select_rows(rows, manifest, "test", allow_test=True)
    assert {r.annotation_id for r in select_rows(rows, manifest, "train")}.isdisjoint(manifest["annotations"]["test"])


def test_null_criterion_is_flagged_and_excluded(rows):
    manifest = load_manifest()
    row = next(r for r in rows if r.annotation_id == 883)
    assert not row.usable
    for split in SIZES:
        assert 883 not in {r.annotation_id for r in select_rows(rows, manifest, split, allow_test=True)}
    with pytest.raises(ValueError):
        row_prompt(row)


def test_row_has_no_gpt4_or_excluded_fields():
    names = {f.name for f in dataclasses.fields(Row)}
    assert not {n for n in names if n.startswith("gpt4") or n in {"explanation_correctness", "training", "trial_title"}}


def test_prompts_and_targets_hold_no_gpt4_text(rows):
    raw = {r["annotation_id"]: r for r in pq.read_table(dataset_path()).to_pylist()}
    for row in (r for r in rows if r.usable):
        # An explanation may repeat a note sentence word for word, so remove the allowed fields first.
        text = (row_prompt(row) + row_target(row)).replace(row.note, "").replace(row.criterion_text, "")
        record = raw[row.annotation_id]
        assert record["gpt4_explanation"] not in text
        assert record["trial_title"] not in text
        for label in ("gpt4_eligibility", "explanation_correctness"):
            assert f'"{label}"' not in text


def test_note_round_trip(rows):
    for note in {r.note for r in rows}:
        sentences = parse_note(note)
        assert format_note(sentences) == note
        assert parse_note(format_note(sentences)) == sentences


def test_note_numbering_must_start_at_zero():
    with pytest.raises(ValueError):
        parse_note("1. a\n2. b")
    with pytest.raises(ValueError):
        parse_note("0. a\n2. b")


def test_evidence_ids_are_in_range_and_round_trip(rows):
    for row in (r for r in rows if r.usable):
        target = json.loads(row_target(row))
        assert tuple(target["evidence_sentence_ids"]) == row.evidence_ids
        assert target["label"] == row.label
        assert all(0 <= i < len(parse_note(row.note)) for i in target["evidence_sentence_ids"])


def test_target_order_and_format():
    assert build_target([2, 0, 2], "included") == '{"evidence_sentence_ids":[0,2],"label":"included"}'
    assert build_target([], "not excluded", "label_first") == '{"label":"not excluded","evidence_sentence_ids":[]}'
    with pytest.raises(ValueError):
        build_target([], "included", "other")


def toy_encode(text):
    # One id per character, except <|im_end|> which is a single id, like the real special token.
    ids, i = [], 0
    while i < len(text):
        if text.startswith("<|im_end|>", i):
            ids.append(999999)
            i += len("<|im_end|>")
        else:
            ids.append(ord(text[i]))
            i += 1
    return ids


def test_loss_mask_covers_only_assistant_tokens(rows):
    row = next(r for r in rows if r.usable and r.evidence_ids)
    example = encode_example(row, toy_encode, max_len=10**6)
    ids, labels, n = example["input_ids"], example["labels"], example["prompt_length"]
    assert len(ids) == len(labels)
    assert all(label == IGNORE for label in labels[:n])
    assert labels[n:] == ids[n:]
    assert ids[-1] == 999999
    decoded = "".join(chr(i) for i in labels[n:-1])
    assert decoded == row_target(row)


def test_encoding_fails_instead_of_truncating(rows):
    row = next(r for r in rows if r.usable)
    with pytest.raises(TruncationError):
        encode_example(row, toy_encode, max_len=100)
