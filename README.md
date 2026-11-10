# TrialMatch

Criterion-level clinical trial eligibility screening. Input: a patient note and one trial criterion. Output: a label from the dataset's own categories (`included`, `not included`, `excluded`, `not excluded`, `not enough information`, `not applicable`) and the IDs of the note sentences that support it.

Model: Qwen2.5-1.5B-Instruct with a LoRA adapter, trained on the TrialGPT Criterion Annotations. The repository also holds hand-written decoders built on the model's forward pass: a KV-cached greedy decoder, a tiled online-softmax attention backend, and a speculative decoder (0.5B draft model or prompt lookup), plus a Gradio application that compares them live.

**Educational screening prototype, not a clinical enrollment decision.**

Report: `report/main.pdf` (IEEE LaTeX source in `report/main.tex`). Project status log: `docs/STATUS.md`. AI use: `AI_USE.md`.

## Links

| Item | Where |
|---|---|
| Code | https://github.com/AKIF-jk/trialmatch |
| Adapters (LoRA, seeds 42/43/44 and the 0.5B draft) | https://huggingface.co/AkifJK/trialmatch-qwen2.5-1.5b-lora, revision `6fd2471b3894cdbe84d46f9d1278233a3279e77e` |
| Split manifest | `splits/manifest.json` (sha256 `357486d0...`) |
| Demo video | see the submission message (link also in the report appendix) |
| Application URL | **No permanent URL.** See "Running the application". |

Download an adapter (about 17 MB for the target, adapters only; base models come from the Hub):

```bash
pip install huggingface_hub
hf download AkifJK/trialmatch-qwen2.5-1.5b-lora --revision 6fd2471b3894cdbe84d46f9d1278233a3279e77e --include "seed44/*" --local-dir adapters
```

The adapter repo is made public at submission time. Before that it is private and needs a token with read access.

## Pinned inputs

| Input | Value |
|---|---|
| Dataset | `ncbi/TrialGPT-Criterion-Annotations` at `1cfcafde94a1560a33b4addc1638664fe28fc059` (licence field: `other`) |
| Target model | `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Draft model | `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604a` |
| Libraries | `requirements-kaggle.txt` (torch 2.11.0, transformers 5.16.1, peft 0.20.0) |

## Split

Patient IDs are sorted, shuffled with `numpy.random.default_rng(42)`, then assigned 37/8/8 (train/val/test) patients: 603, 206 and 206 annotations. `splits/manifest.json` lists patient IDs, annotation IDs and per-class counts. Annotation 883 has no criterion text and is flagged and excluded from training and metrics. `gpt4_*` columns, `explanation_correctness`, the dataset's `training` column and `trial_title` are never used as inputs. Test rows load only through `scripts/final_eval.py` (explicit `allow_test=True`; a test enforces that no other file passes it).

## Layout

```
trialmatch/data/       split, note parsing, prompt, tokenization
trialmatch/train/      LoRA training data helpers
trialmatch/eval/       parsing, metrics, baselines, generation, freeze record
trialmatch/decoding/   greedy.py (no cache and KV cache), speculative.py, prompt_lookup.py
trialmatch/attention/  reference.py, tiled.py, hf.py (backend for the model)
trialmatch/app/        text handling, screening logic, engine, Gradio UI
app.py                 application entry point
scripts/               train, evaluation, benchmarks, report assets (one script per experiment)
configs/               final.json (frozen config), freeze.json, unseen_cases.json
results/               every log and report the paper quotes
report/                paper source, generated tables and figures
tests/                 pytest suite (CPU)
tools/                 Kaggle kernel builder, Space staging
```

## Setup

Python 3.12 and a CUDA GPU for anything that runs the models. Do not install PyTorch on a small machine to run the tests: the data, metric, split and application-logic tests need no torch.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
PYTHONPATH=. .venv/bin/python -m pytest -q tests      # torch-free tests; torch tests skip
```

The torch tests (cache equivalence, attention, speculative decoding on a tiny random Qwen2 in float64) run on a CPU Kaggle kernel; the last full run was 487 passed and 5 skipped (`results/tests/`). On a machine with torch, install `requirements-kaggle.txt` and run the same command.

## Running the application

The application is `app.py` (Gradio). It loads four models at start-up (target with the seed-44 adapter, the 0.5B draft base and LoRA, and the untuned 1.5B for optional rationales).

**On any CUDA GPU (about 8 GB of GPU memory):**

```bash
pip install -r requirements-kaggle.txt gradio==5.50.0 huggingface_hub pyarrow
export HF_TOKEN=...        # read access to the adapter repo while it is private
python app.py              # http://localhost:7860
```

On CPU the models load in float32 (about 12 GB RAM) and one criterion takes tens of seconds.

**Kaggle demo (what was used for the video):** the notebook `akifjawad/trialmatch-demo` downloads the code, loads the adapters and prints a temporary `*.gradio.live` link. It needs the Kaggle secret `HF_TOKEN`; run it with a T4, read the link from the log, and stop the session afterwards.

**Known gap:** there is no permanent hosted URL. Hugging Face refused to host a ZeroGPU or CPU Gradio Space for this account (HTTP 402 on 2026-11-02), so the Space `AkifJK/trialmatch` stays paused. The brief says an evaluator should not need to run a notebook; the `python app.py` route above is the alternative. `docs/DEPLOY.md` has the details and the Space deployment steps for an account that can host one.

Screening tab: paste a note, press *Segment note* (edit the numbered sentences if needed), enter criteria one per line or load them from an NCT ID, press *Screen*. Each result shows label, cited IDs, highlighted evidence, missing-information flags and a template explanation built only from the cited sentences. The overall outcome is a rule on the labels (any `not included` or `excluded`: "Likely ineligible"; else any `not enough information`: "Needs review"; else "Preliminarily eligible"). Reports download as Markdown and JSON. Inference Lab tab: one prompt through no cache, KV cache, model-draft speculative and prompt-lookup speculative decoding, with K and attention backend (sdpa or tiled) selectable; all numbers come from the live run.

## Reproducing the results

All heavy jobs ran on Kaggle (T4) as batch kernels built by `tools/build_kernel.py` from a commit, and wrote to `results/<job>/`. The scripts below can be run on any CUDA machine with the pinned libraries.

| Step | Command | Output |
|---|---|---|
| Split | `python scripts/make_split.py` | `splits/manifest.json` |
| Untuned baselines (validation) | `python scripts/baseline.py <model names from MODELS in the script>` | `results/baseline/` |
| Train (frozen config, one seed) | `python scripts/train.py --name over4_s44 --oversample 4 --micro-batch 4 --seed 44` (other values are the defaults; see `configs/final.json`) | `results/train/` |
| Decoding equivalence and benchmark | `python scripts/decode_equivalence.py`, `python scripts/decode_benchmark.py` | `results/decoding/`, `results/final_bench/decoding/` |
| Attention check, equivalence, benchmark | `python scripts/attention_check.py`, `attention_equivalence.py`, `attention_bench.py` | `results/attention/`, `results/final_bench/attention/` |
| Speculative equivalence and benchmark | `python scripts/speculative_equivalence.py`, `python scripts/speculative_benchmark.py` | `results/speculative/`, `results/final_bench/speculative/` |
| Freeze record | `python scripts/freeze.py` | `configs/freeze.json` (tag `final-freeze`) |
| Test evaluation (one run) | `python scripts/final_eval.py --split test --expect-commit <sha of final-freeze> --adapter-root <adapters>` | `results/final/test/` |
| Error analysis | `python scripts/error_analysis.py` | `results/final/analysis/`, `docs/error_analysis.md` |
| Hand-written unseen cases | `python scripts/unseen_eval.py` | `results/final/unseen/` |
| All paper tables and figures | `python3 scripts/make_report_assets.py` (needs matplotlib) | `report/tables/`, `report/figures/` |
| Paper | `cd report && pdflatex main.tex && pdflatex main.tex` | `report/main.pdf` |

Check each script's `--help` for arguments; the test run refuses to start unless the files match `configs/freeze.json` and no earlier run is recorded in `configs/test_runs.json`. The test split was opened once (2026-11-08), and the code was not changed afterwards.

## Results in one table

Test split (206 rows, 8 patients), greedy decoding, fp16. Support: included 27, not included 7, excluded 0, not excluded 91, not enough information 66, not applicable 15. The test split has no `excluded` row, so the six-label macro-F1 is capped at 0.833.

| Model | Accuracy | Macro-F1 | Evidence macro-F1 |
|---|---|---|---|
| Majority class (per criterion type) | 0.626 | 0.226 | 0.704 |
| Qwen2.5-1.5B untuned (repaired parsing) | 0.146 | 0.121 | 0.320 |
| Fine-tuned, mean of 3 seeds | 0.696 | 0.431 | 0.804 |

Intervals over 8 patients are wide (accuracy 0.53 to 0.83 per seed), the rare classes are mostly missed, and the model rarely answers `not enough information` for exclusion criteria. See the report for all tables, caveats and error analysis.

## Benchmark logs

Raw reports and per-record logs are committed under `results/` (`report.json`, `records*.jsonl`, `environment.json`). Each records its commit, library versions and hardware.
