"""Prompt and target construction. Inputs are the numbered note, criterion type, criterion text and its label set."""
import json

from trialmatch.data.load import LABELS, Row
from trialmatch.data.note import parse_note

SYSTEM = (
    "You are a clinical trial screening assistant. Decide how one patient note relates to one trial "
    "criterion, and cite the note sentences that support the decision. Use only the note."
)

DEFINITIONS = {
    "inclusion": {
        "included": "the note shows the patient meets the criterion",
        "not included": "the note shows the patient does not meet the criterion",
        "not enough information": "the note has no information to decide",
        "not applicable": "the criterion does not apply to this patient",
    },
    "exclusion": {
        "excluded": "the note shows the patient meets the exclusion criterion",
        "not excluded": "the note shows the patient does not meet the exclusion criterion",
        "not enough information": "the note has no information to decide",
        "not applicable": "the criterion does not apply to this patient",
    },
}

ORDERS = ("evidence_first", "label_first")


def schema(order):
    parts = {"evidence_first": '{"evidence_sentence_ids":[<ids>],"label":"<label>"}', "label_first": '{"label":"<label>","evidence_sentence_ids":[<ids>]}'}
    return parts[order]


def build_user(note, criterion_type, criterion_text, order="evidence_first"):
    labels = "\n".join(f"- {label}: {meaning}" for label, meaning in DEFINITIONS[criterion_type].items())
    return (
        f"Patient note (numbered sentences):\n{note}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {criterion_text}\n\n"
        f"Labels:\n{labels}\n\n"
        f"Answer with one JSON object and nothing else: {schema(order)}\n"
        "Use an empty list when no sentence supports the label."
    )


def build_target(evidence_ids, label, order="evidence_first"):
    ids = sorted(set(int(i) for i in evidence_ids))
    pairs = [("evidence_sentence_ids", ids), ("label", label)]
    if order == "label_first":
        pairs.reverse()
    elif order != "evidence_first":
        raise ValueError(f"unknown order {order!r}")
    return json.dumps(dict(pairs), separators=(",", ":"))


def chat_prompt(user, system=SYSTEM):
    """Qwen2.5 ChatML text ending at the assistant turn. Phase 2 checks it matches tokenizer.apply_chat_template."""
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def row_prompt(row: Row, order="evidence_first"):
    if not row.usable:
        raise ValueError(f"annotation {row.annotation_id} has no criterion text")
    return chat_prompt(build_user(row.note, row.criterion_type, row.criterion_text, order))


def row_target(row: Row, order="evidence_first"):
    sentences = len(parse_note(row.note))
    if any(not 0 <= i < sentences for i in row.evidence_ids):
        raise ValueError(f"annotation {row.annotation_id}: evidence id outside 0..{sentences - 1}")
    return build_target(row.evidence_ids, row.label, order)
