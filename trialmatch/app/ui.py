"""Gradio interface. It takes the model calls as functions, so the same UI runs with the real engine (app.py), on ZeroGPU,
or with stubs in tests. Nothing here imports torch."""
import html
import json
import tempfile
import time
from pathlib import Path

import gradio as gr

from trialmatch.app import screen, text
from trialmatch.app.screen import LAB_MODES, NOTICE, TILES

EXAMPLES = {
    "Type 2 diabetes, hand-written": {
        "note": (
            "58-year-old woman seen for a follow-up visit. She has had type 2 diabetes for 9 years and takes metformin 1000 mg twice daily. "
            "Her last HbA1c was 8.1%. Blood pressure today was 138/84. She does not smoke. She is not pregnant and had a hysterectomy in 2015. "
            "Creatinine is 0.9 mg/dL. She reports no chest pain or shortness of breath."
        ),
        "inclusion": "Adults aged 18 to 75 years\nDiagnosis of type 2 diabetes mellitus\nHbA1c between 7.5% and 11%\nHistory of cardiovascular disease",
        "exclusion": "Pregnancy or breastfeeding\nSerum creatinine above 1.5 mg/dL\nCurrent insulin therapy",
    },
    "Asthma, hand-written": {
        "note": (
            "A 24-year-old man with asthma since childhood presents with wheeze after exercise. He uses albuterol twice a week. "
            "He was admitted to hospital for an exacerbation 8 months ago. He has never smoked. Spirometry shows FEV1 of 72% predicted. "
            "He takes no oral steroids."
        ),
        "inclusion": "Physician-diagnosed asthma\nFEV1 between 50% and 80% of predicted\nAge 12 to 65 years",
        "exclusion": "Current smoker\nOral corticosteroid use in the last 4 weeks\nHospital admission for asthma in the last 3 months",
    },
}

LAB_NOTE = (
    "0. 67-year-old man with heart failure and an ejection fraction of 30%.\n1. He takes carvedilol and lisinopril.\n"
    "2. He has had atrial fibrillation for 3 years.\n3. He has no history of stroke.\n4. Creatinine is 1.1 mg/dL."
)
LAB_CRITERION = "Diagnosis of heart failure with reduced ejection fraction"

CSS = ".notice{border:1px solid #b3261e;padding:6px 12px;border-radius:6px;font-weight:600} mark{background:#ffe08a;color:#000;padding:1px 3px}"


def _file(name, content):
    path = Path(tempfile.mkdtemp()) / name
    path.write_text(content)
    return str(path)


def _error(message):
    return f"<div style='border-left:6px solid #b3261e;padding:6px 12px'><b>Cannot run.</b> {html.escape(message)}</div>"


def friendly_error(exc):
    """Message for an exception raised while running on the GPU. Quota and timeout wording varies, so match on keywords."""
    message = str(exc)
    low = message.lower()
    if any(w in low for w in ("quota", "zerogpu", "gpu task", "abort", "timeout", "timed out")):
        return (
            "The GPU quota or the time limit was reached. Wait a few minutes (anonymous visitors get about 2 minutes of GPU per day, "
            "signed-in free accounts about 5), use fewer criteria or fewer new tokens, then try again. " + message[:200]
        )
    if "out of memory" in low:
        return "The GPU ran out of memory. Shorten the note or use fewer criteria and try again."
    return f"Unexpected error ({type(exc).__name__}): {message[:300]}. Try again; if it repeats, shorten the input."


def detail_html(sentences, results):
    cards = []
    for r in results:
        label = html.escape(r.shown_label)
        ids = ", ".join(map(str, r.evidence_ids)) or "none"
        missing = "".join(f"<li>{html.escape(m)}</li>" for m in screen.missing_information(r))
        parts = [
            f"<div><b>Explanation (template, built from the cited sentences only):</b> {html.escape(screen.template_explanation(r, sentences))}</div>",
            f"<div style='margin-top:6px'><b>Evidence in the note</b> (cited ids: {ids})</div><div style='margin:4px 0 8px'>{screen.highlight_html(sentences, r.evidence_ids)}</div>",
            f"<div><b>Missing information / flags:</b><ul style='margin:2px 0'>{missing}</ul></div>" if missing else "",
            f"<div><b>Model rationale</b> (optional, generated from the cited sentences only, may be wrong): {html.escape(r.rationale)}</div>" if r.rationale else "",
        ]
        cards.append(
            f"<details {'open' if r.index == 0 else ''} style='border:1px solid #8884;border-radius:6px;padding:6px 10px;margin:6px 0'>"
            f"<summary><b>{r.index + 1}. {html.escape(r.criterion_type.title())}:</b> {html.escape(r.text)} &mdash; <b>{label}</b></summary>"
            + "".join(parts) + "</details>"
        )
    return "".join(cards)


def build_ui(screen_fn, lab_fn, lookup_fn=text.fetch_trial):
    """screen_fn(sentences, criteria, rationale) -> (results, timing, provenance).
    lab_fn(sentences, type, criterion, modes, backend, tile, k, draft, new_tokens, force, repeats) -> (rows, generated_text, provenance)."""

    def segment(raw):
        try:
            return text.numbered(text.validate_note(text.segment_note(raw or ""))), ""
        except text.InputError as exc:
            return gr.update(), _error(str(exc))

    def do_renumber(numbered):
        return text.numbered(text.renumber(numbered or ""))

    def load_example(name):
        example = EXAMPLES[name]
        sentences = text.segment_note(example["note"])
        return example["note"], text.numbered(sentences), example["inclusion"], example["exclusion"]

    def fetch(nct):
        try:
            trial = lookup_fn(nct)
        except text.InputError as exc:
            return gr.update(), gr.update(), _error(str(exc))
        inclusion, exclusion, warnings = text.split_eligibility(trial["eligibility"])
        info = f"**{trial['nct_id']}**: {trial['title']}  \nPublic ClinicalTrials.gov metadata. " + " ".join(warnings)
        return "\n".join(inclusion), "\n".join(exclusion), info

    def estimate(inclusion, exclusion, rationale):
        n = len(text.split_items(inclusion or "")) + len(text.split_items(exclusion or ""))
        seconds = 10 + 3 * n + (15 if rationale else 0)
        return f"{n} criteria entered (limit {text.MAX_CRITERIA}). Estimated GPU time about {seconds} s of your daily quota."

    def run_screen(numbered, inclusion, exclusion, rationale):
        empty = ("", [], "", None, None)
        try:
            sentences = text.parse_numbered_note(numbered)
            criteria = text.validate_criteria(text.split_items(inclusion or ""), text.split_items(exclusion or ""))
        except text.InputError as exc:
            return (_error(str(exc)), *empty)
        started = time.perf_counter()
        try:
            results, timing, provenance = screen_fn(sentences, criteria, bool(rationale))
        except Exception as exc:  # noqa: BLE001 - every failure on the GPU path becomes a message, never a stack trace.
            return (_error(friendly_error(exc)), *empty)
        outcome, reasons = screen.aggregate(results)
        timing["wall_seconds"] = time.perf_counter() - started
        report = screen.report_dict(sentences, results, provenance, extra={"timing": timing})
        md = _file("screening_report.md", screen.report_markdown(report))
        js = _file("screening_report.json", json.dumps(report, indent=2, default=str))
        status = f"Done in {timing['wall_seconds']:.1f} s on {provenance['device']}. {NOTICE}"
        return status, screen.outcome_html(outcome, reasons), screen.summary_rows(results), detail_html(sentences, results), md, js

    def run_lab(numbered, ctype, criterion, modes, backend, tile, k, draft, new_tokens, force, repeats):
        empty = ([], "", "", None)
        try:
            sentences = text.parse_numbered_note(numbered)
            text.validate_criteria([criterion] if ctype == "inclusion" else [], [criterion] if ctype == "exclusion" else [])
            if not modes:
                raise text.InputError("Choose at least one decoding mode.")
        except text.InputError as exc:
            return (_error(str(exc)), *empty)
        try:
            rows, generated, provenance = lab_fn(
                sentences, ctype, criterion.strip(), list(modes), "sdpa" if backend == "sdpa" else "tiled", int(tile), int(k), draft, int(new_tokens), bool(force), int(repeats)
            )
        except Exception as exc:  # noqa: BLE001
            return (_error(friendly_error(exc)), *empty)
        payload = {"notice": NOTICE, "provenance": provenance, "rows": rows, "generated_text": generated}
        status = f"Live run on {provenance['device']} ({provenance['dtype']}). Numbers are from this run, not from a stored chart."
        return status, screen.lab_table(rows), screen.lab_commentary(rows), generated, _file("inference_lab.json", json.dumps(payload, indent=2, default=str))

    with gr.Blocks(title="TrialMatch", css=CSS) as demo:
        gr.Markdown("# TrialMatch: clinical trial criterion screening")
        gr.HTML(f"<div class='notice'>{html.escape(NOTICE)}</div>")
        with gr.Tabs():
            with gr.Tab("Screening"):
                gr.Markdown(
                    "1. Paste a patient note and press **Segment note**; fix the numbered sentences if needed. 2. Enter criteria, one per line "
                    "(or fetch them by NCT id). 3. Press **Screen**. Use your own text; do not paste real patient identifiers."
                )
                with gr.Row():
                    example = gr.Dropdown(list(EXAMPLES), label="Hand-written example", value=None)
                    load = gr.Button("Load example")
                note_raw = gr.Textbox(lines=6, label="Patient note (free text)")
                with gr.Row():
                    segment_btn = gr.Button("Segment note")
                    renumber_btn = gr.Button("Renumber after editing")
                numbered = gr.Textbox(lines=10, label="Numbered sentences (editable, format 0. first, 1. second, ...)")
                with gr.Row():
                    nct = gr.Textbox(label="Optional NCT id (public ClinicalTrials.gov metadata only)", placeholder="NCT00000620", scale=3)
                    fetch_btn = gr.Button("Fetch criteria", scale=1)
                trial_info = gr.Markdown()
                with gr.Row():
                    inclusion = gr.Textbox(lines=6, label="Inclusion criteria (one per line)")
                    exclusion = gr.Textbox(lines=6, label="Exclusion criteria (one per line)")
                rationale = gr.Checkbox(label="Add a model rationale per criterion (optional, uses the untuned model on the cited sentences only; adds GPU time)")
                cost = gr.Markdown()
                run_btn = gr.Button("Screen", variant="primary")
                status = gr.HTML()
                outcome = gr.HTML()
                table = gr.Dataframe(headers=["#", "Type", "Criterion", "Label", "Cited sentence ids", "Output status"], interactive=False, wrap=True, label="Criterion results")
                detail = gr.HTML()
                with gr.Row():
                    md_file = gr.File(label="Markdown report")
                    json_file = gr.File(label="JSON report")

                outputs = [status, outcome, table, detail, md_file, json_file]
                # The status HTML carries errors and the done line, so the first output is always a string.
                segment_btn.click(segment, note_raw, [numbered, status], api_name="segment")
                renumber_btn.click(do_renumber, numbered, numbered, api_name="renumber")
                load.click(load_example, example, [note_raw, numbered, inclusion, exclusion])
                fetch_btn.click(fetch, nct, [inclusion, exclusion, trial_info], api_name="fetch")
                for box in (inclusion, exclusion, rationale):
                    box.change(estimate, [inclusion, exclusion, rationale], cost)
                run_btn.click(run_screen, [numbered, inclusion, exclusion, rationale], outputs, concurrency_limit=1, api_name="screen")

            with gr.Tab("Inference Lab"):
                gr.Markdown(
                    "Run one criterion prompt live through the decoding mechanisms written for this project and compare them. Tokens are checked "
                    "against a KV-cache run with sdpa attention. In the report's T4 and CPU benchmarks speculative decoding with a 0.5B draft model was "
                    "**slower** than cached greedy: one draft step costs 0.8x a target step, so no K breaks even. Prompt lookup has no draft model "
                    "(it copies the tokens that followed an earlier match in the text) and was faster on the T4. Run both and see what this device does."
                )
                lab_note = gr.Textbox(lines=6, value=LAB_NOTE, label="Numbered note")
                with gr.Row():
                    lab_type = gr.Radio(["inclusion", "exclusion"], value="inclusion", label="Criterion type")
                    lab_criterion = gr.Textbox(value=LAB_CRITERION, label="Criterion", scale=3)
                copy_btn = gr.Button("Copy note and first criterion from Screening")
                modes = gr.CheckboxGroup(list(LAB_MODES), value=list(LAB_MODES), label="Decoding modes")
                with gr.Row():
                    backend = gr.Radio(["sdpa", "tiled online-softmax"], value="sdpa", label="Attention backend")
                    tile = gr.Dropdown(list(TILES), value=64, label="Tile size (tiled only)")
                    k = gr.Slider(1, 8, value=2, step=1, label="K, draft tokens per round")
                    draft = gr.Radio(["LoRA", "base"], value="LoRA", label="Draft model (0.5B)")
                with gr.Row():
                    new_tokens = gr.Slider(8, 64, value=32, step=8, label="Max new tokens")
                    force = gr.Checkbox(label="Keep generating past the stop token (longer decode)")
                    repeats = gr.Slider(1, 3, value=1, step=1, label="Repeats (median)")
                lab_btn = gr.Button("Run live comparison", variant="primary")
                lab_status = gr.HTML()
                lab_table = gr.Dataframe(headers=screen.LAB_HEADERS, interactive=False, wrap=True, label="Measured results")
                lab_text = gr.Markdown()
                lab_out = gr.Textbox(label="Generated text of the reference run", interactive=False)
                lab_file = gr.File(label="Lab run (JSON)")

                def copy_from_screening(numbered_text, inc, exc):
                    first = (text.split_items(inc or "") or text.split_items(exc or "") or [LAB_CRITERION])[0]
                    return numbered_text or LAB_NOTE, "inclusion" if text.split_items(inc or "") else "exclusion", first

                copy_btn.click(copy_from_screening, [numbered, inclusion, exclusion], [lab_note, lab_type, lab_criterion])
                lab_btn.click(
                    run_lab, [lab_note, lab_type, lab_criterion, modes, backend, tile, k, draft, new_tokens, force, repeats],
                    [lab_status, lab_table, lab_text, lab_out, lab_file], concurrency_limit=1, api_name="lab",
                )
        gr.Markdown(f"_{NOTICE}_ Model: Qwen2.5-1.5B-Instruct with a LoRA adapter trained on TrialGPT criterion annotations. Source: see the repository README.")
    return demo
