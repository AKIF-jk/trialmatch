"""Screening logic that needs no model: reading outputs, the aggregation rule, explanations, evidence display and reports."""
import html
import json
from dataclasses import asdict, dataclass, field

from trialmatch.data.load import LABELS
from trialmatch.eval.parse import parse_output

NOTICE = "Educational screening prototype, not a clinical enrollment decision."
AGGREGATION_NOTE = (
    "Rule-based aggregation of the criterion labels, not a ground-truth verdict. Any not included or excluded gives "
    "Likely ineligible. Otherwise any not enough information (or unreadable output) gives Needs review. Otherwise "
    "Preliminarily eligible. not applicable criteria do not count."
)
LAB_MODES = ("no cache", "KV cache", "speculative", "speculative (prompt lookup)")
TILES = (16, 64, 256)
INELIGIBLE, REVIEW, ELIGIBLE = "Likely ineligible", "Needs review", "Preliminarily eligible"
BLOCKING = {"inclusion": "not included", "exclusion": "excluded"}
DECISIVE = {"included", "not included", "excluded", "not excluded"}


@dataclass
class CriterionResult:
    index: int
    criterion_type: str
    text: str
    status: str  # ok, repaired (strict parse failed, first JSON object used), unparsed, invalid_label
    label: str | None
    evidence_ids: list = field(default_factory=list)  # Valid ids only.
    dropped_ids: list = field(default_factory=list)  # Ids outside the note, ignored.
    raw_output: str = ""
    rationale: str | None = None  # Optional model text. Never used for the label.

    @property
    def shown_label(self):
        return self.label if self.label else "unreadable output"


def interpret(index, criterion_type, text, raw_output, n_sentences):
    """Turn raw model text into a CriterionResult. Strict JSON first, then the logged repair step. Never raises."""
    parsed = parse_output(raw_output)
    if not parsed.ok:
        parsed = parse_output(raw_output, repair=True) if "{" in raw_output else parsed
    base = dict(index=index, criterion_type=criterion_type, text=text, raw_output=raw_output)
    if not parsed.ok:
        return CriterionResult(status="unparsed", label=None, **base)
    if parsed.label not in LABELS[criterion_type]:
        return CriterionResult(status="invalid_label", label=None, **base)
    valid = [i for i in parsed.evidence_ids if 0 <= i < n_sentences]
    dropped = [i for i in parsed.evidence_ids if not 0 <= i < n_sentences]
    return CriterionResult(status="repaired" if parsed.repaired else "ok", label=parsed.label, evidence_ids=valid, dropped_ids=dropped, **base)


def aggregate(results):
    """(outcome, reasons). reasons is a list of sentences naming the criteria that decided the outcome."""
    blocking = [r for r in results if r.label == BLOCKING[r.criterion_type]]
    if blocking:
        return INELIGIBLE, [f"Criterion {r.index + 1} ({r.criterion_type}) is {r.label}." for r in blocking]
    unsure = [r for r in results if r.label in (None, "not enough information")]
    if unsure:
        return REVIEW, [f"Criterion {r.index + 1} ({r.criterion_type}) is {r.shown_label}." for r in unsure]
    if all(r.label == "not applicable" for r in results):
        return ELIGIBLE, ["Every criterion was labelled not applicable, so nothing was checked."]
    return ELIGIBLE, ["No criterion is blocking and none needs more information."]


def cited(result, sentences):
    return [(i, sentences[i]) for i in result.evidence_ids]


def template_explanation(result, sentences):
    """Explanation built only from the label and the cited sentences. It adds no claim of its own."""
    if result.label is None:
        return "The model output could not be read, so there is no label to explain."
    head = f"Labelled {result.label} for this {result.criterion_type} criterion."
    quotes = cited(result, sentences)
    if not quotes:
        return head + " No sentence was cited."
    return head + " Cited: " + " ".join(f"[{i}] \"{s}\"" for i, s in quotes)


def missing_information(result):
    """What the user should add or check. Empty evidence never changes the label, it only raises a flag."""
    notes = []
    if result.label is None:
        notes.append("Model output could not be read. Check this criterion by hand.")
    elif result.label == "not enough information":
        notes.append(f"The note does not say enough to decide: {result.text}")
    elif result.label in DECISIVE and not result.evidence_ids:
        notes.append("The label has no cited sentence, so nothing in the note was pointed to as support. Check it by hand.")
    if result.dropped_ids:
        notes.append(f"The model cited sentence ids {result.dropped_ids} that are not in the note. They were ignored.")
    if result.status == "repaired":
        notes.append("The output was not strict JSON. The first JSON object in it was used.")
    return notes


def highlight_html(sentences, evidence_ids):
    """The note with cited sentences marked. Text is escaped, so pasted markup is shown, never run."""
    marked = set(evidence_ids)
    rows = []
    for i, sentence in enumerate(sentences):
        text = html.escape(sentence)
        rows.append(f"<div><b>{i}.</b> <mark>{text}</mark></div>" if i in marked else f"<div><b>{i}.</b> {text}</div>")
    return "\n".join(rows)


def outcome_html(outcome, reasons):
    colour = {INELIGIBLE: "#b3261e", REVIEW: "#8a6100", ELIGIBLE: "#1b6e3c"}[outcome]
    items = "".join(f"<li>{html.escape(r)}</li>" for r in reasons)
    return (
        f"<div style='border-left:6px solid {colour};padding:8px 14px'>"
        f"<div style='font-size:1.3em;font-weight:600;color:{colour}'>{html.escape(outcome)}</div>"
        f"<ul style='margin:6px 0'>{items}</ul>"
        f"<div style='font-size:0.85em;opacity:0.8'>{html.escape(AGGREGATION_NOTE)}</div></div>"
    )


def summary_rows(results):
    return [
        [r.index + 1, r.criterion_type, r.text, r.shown_label, ", ".join(map(str, r.evidence_ids)) or "none", r.status]
        for r in results
    ]


def report_dict(sentences, results, provenance, extra=None):
    outcome, reasons = aggregate(results)
    return {
        "notice": NOTICE,
        "overall_outcome": {"outcome": outcome, "reasons": reasons, "rule": AGGREGATION_NOTE},
        "provenance": provenance,
        "note_sentences": list(sentences),
        "criteria": [
            {
                **asdict(r), "cited_sentences": [{"id": i, "text": s} for i, s in cited(r, sentences)],
                "explanation": template_explanation(r, sentences), "missing_information": missing_information(r),
            }
            for r in results
        ],
        **(extra or {}),
    }


def report_markdown(report):
    out = ["# Eligibility screening report", "", f"> {report['notice']}", ""]
    out += [f"**Overall (rule-based): {report['overall_outcome']['outcome']}**", ""]
    out += [f"- {r}" for r in report["overall_outcome"]["reasons"]] + ["", f"_{report['overall_outcome']['rule']}_", ""]
    out += ["## Criteria", ""]
    for c in report["criteria"]:
        out += [f"### {c['index'] + 1}. {c['criterion_type'].title()}: {c['text']}", "", f"- Label: **{c['label'] or 'unreadable output'}** (output status: {c['status']})"]
        ids = ", ".join(map(str, c["evidence_ids"])) or "none"
        out += [f"- Cited sentence ids: {ids}"]
        out += [f"  - [{s['id']}] {s['text']}" for s in c["cited_sentences"]]
        out += [f"- Explanation (template from cited sentences): {c['explanation']}"]
        out += [f"- Missing information: {m}" for m in c["missing_information"]]
        if c["rationale"]:
            out += [f"- Model rationale (generated from the cited sentences only, may be wrong): {c['rationale']}"]
        out += [""]
    out += ["## Patient note", ""] + [f"{i}. {s}" for i, s in enumerate(report["note_sentences"])] + ["", "## Provenance", ""]
    out += ["```json", json.dumps(report["provenance"], indent=2, default=str), "```", "", f"> {report['notice']}", ""]
    return "\n".join(out)


# ---- inference lab ---------------------------------------------------------------------------------------------------

LAB_HEADERS = [
    "Mode", "Attention", "Tokens", "Tokens equal to KV-cache sdpa run", "Prefill s", "Decode s", "Total s", "Tokens/s (total)",
    "Peak extra MB", "Acceptance", "Tokens per round", "Target passes",
]


def _fmt(value, digits=3):
    return "n/a" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else value)


def lab_table(rows):
    return [
        [
            r["mode"], r["backend"], r["tokens"], "yes" if r["tokens_equal"] else "NO", _fmt(r["prefill_seconds"]), _fmt(r["decode_seconds"]),
            _fmt(r["total_seconds"]), _fmt(r["tokens_per_second"], 1), _fmt(r.get("peak_extra_mb"), 1), _fmt(r.get("acceptance"), 2),
            _fmt(r.get("tokens_per_round"), 2), _fmt(r.get("target_forwards")),
        ]
        for r in rows
    ]


def lab_commentary(rows):
    """Plain-language reading of one lab run, built only from its measured rows."""
    by = {r["mode"]: r for r in rows}
    lines = []
    bad = [r["mode"] for r in rows if not r["tokens_equal"]]
    lines.append(
        "All modes produced the same tokens as the KV-cache sdpa run." if not bad else
        f"**Token mismatch in: {', '.join(bad)}.** The comparison is against a cached sdpa run. In fp16 a near tie between two logits can flip "
        "when the tensor shapes change (Phases 3 to 5 of the report); tiled attention is not bit-equal to sdpa."
    )
    if "KV cache" in by and "no cache" in by:
        lines.append(f"KV cache is {by['no cache']['total_seconds'] / by['KV cache']['total_seconds']:.2f}x the speed of no cache on this prompt (total time, prefill included).")
    for name in ("speculative", "speculative (prompt lookup)"):
        if "KV cache" not in by or name not in by:
            continue
        s, c = by[name], by["KV cache"]
        ratio = c["total_seconds"] / s["total_seconds"]
        verdict = "faster" if ratio > 1.02 else "slower" if ratio < 0.98 else "about the same"
        lines.append(
            f"{name.capitalize()} is {ratio:.2f}x the speed of KV cache ({verdict}). Acceptance {_fmt(s.get('acceptance'), 2)}, "
            f"{_fmt(s.get('tokens_per_round'), 2)} tokens per round, {s.get('target_forwards')} target passes against {c['tokens']} tokens."
        )
        if s.get("draft_seconds") is not None and s["target_forwards"] > 1:
            rounds = s["target_forwards"] - 1
            lines.append(
                f"Cost per round: draft {1000 * s['draft_seconds'] / rounds:.1f} ms plus target pass {1000 * s['verify_seconds'] / rounds:.1f} ms. "
                "It wins only when the tokens a round emits are cheaper than the same number of single-token cached steps."
            )
    return "\n\n".join(lines)
