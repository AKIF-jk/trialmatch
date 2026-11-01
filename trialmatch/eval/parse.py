"""Parse model output into a label and evidence ids. Strict by default; the one repair step is opt-in and flagged."""
import json
from dataclasses import dataclass

KEYS = {"evidence_sentence_ids", "label"}


@dataclass(frozen=True)
class Parsed:
    ok: bool
    label: str | None = None
    evidence_ids: tuple[int, ...] = ()
    error: str | None = None
    repaired: bool = False


def _load(text, repair):
    try:
        return json.loads(text.strip()), False
    except json.JSONDecodeError as exc:
        if not repair or "{" not in text:
            raise exc
    # Repair: decode the first JSON object and drop text around it (code fences, chatter).
    value, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
    return value, True


def parse_output(text, repair=False):
    """Return Parsed. ok is False when the text is not one JSON object with exactly the two expected keys.

    Label validity depends on the criterion type, so the evaluator checks it. Evidence ids are kept as given
    (deduplicated and sorted); the evaluator counts out-of-range ids as false positives.
    """
    try:
        value, repaired = _load(text, repair)
    except json.JSONDecodeError as exc:
        return Parsed(False, error=f"json: {exc.msg}")
    if not isinstance(value, dict) or set(value) != KEYS:
        return Parsed(False, error="shape: expected an object with keys evidence_sentence_ids and label", repaired=repaired)
    ids, label = value["evidence_sentence_ids"], value["label"]
    if not isinstance(label, str):
        return Parsed(False, error="shape: label is not a string", repaired=repaired)
    if not isinstance(ids, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
        return Parsed(False, error="shape: evidence_sentence_ids is not a list of integers", repaired=repaired)
    return Parsed(True, label=label, evidence_ids=tuple(sorted(set(ids))), repaired=repaired)
