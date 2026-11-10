# AI Use

This project was built with Claude Code (Anthropic) as a coding and writing assistant, under the author's direction. The author is responsible for the submission and has reviewed the work.

## What the assistant did

- **Planning and research support.** Drafted the phase plan (`docs/PLAN.md`), compared model candidates, and read and summarised the seven papers in `docs/sources.md`. Every citation in the report was checked against Hugging Face paper metadata (title, authors, arXiv ID). Parts of papers that were not read (appendices, the TrialGPT Methods section) are listed in `docs/sources.md`.
- **Code.** Wrote the source under `trialmatch/`, `scripts/`, `tools/` and `app.py`: data preparation and split, prompt and encoding, metrics and parsing, LoRA training, the no-cache and KV-cached decoders, reference and tiled attention, speculative and prompt-lookup decoding, and the Gradio application.
- **Experiment management.** Built and submitted the Kaggle batch jobs, read their logs, and saved results under `results/`.
- **Tests.** Wrote the pytest suite (`tests/`) and ran it on a CPU Kaggle kernel; 487 passed and 5 skipped at the last full run.
- **Analysis and writing.** Wrote the first drafts of `docs/error_analysis.md`, `docs/unseen_cases.md`, `scripts/make_report_assets.py` and the report text (`report/main.tex`), and the status log (`docs/STATUS.md`).
- **Specific items to note.** (1) The expected labels in `configs/unseen_cases.json` (3 patients, 26 criteria) were written by the assistant and were not reviewed by a clinician; they support a sanity check only. (2) The ten error cases and their readings in `docs/error_analysis.md` are the assistant's interpretation of the saved outputs and were not tested as causes.

## Libraries and models reused

PyTorch, Hugging Face Transformers 5.16.1, PEFT, Accelerate, Gradio, NumPy, scikit-learn, pytest. Models: Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct (Apache-2.0), Phi-3-mini-4k-instruct (MIT, baseline only). Dataset: `ncbi/TrialGPT-Criterion-Annotations`. The library code is used through its public interfaces and was exercised by the tests above; the decoders and attention are written against the model's forward pass and do not call `generate` (except HF assisted generation, used only as a float32 cross-check).

## What was not AI-generated

- The decision to use the model, split and evaluation rules as pinned in the plan, and the decision to use Kaggle for the demo when hosting was refused.
- All experiment runs on Kaggle hardware: every number in the report comes from files under `results/`, not from model recall.

## Mistakes the process caught (kept for honesty)

A local CPU PyTorch install against the project rules (removed); a wrong tarball tree hash for an empty `__init__.py`; source files missing from restricted job trees; Kaggle library version drift. Each is recorded in `docs/STATUS.md`.
