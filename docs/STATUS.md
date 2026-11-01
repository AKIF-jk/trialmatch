# Project Status

Read this first, then `docs/PLAN.md`. Update this file at the end of every phase or work session, and commit and push it.

**Last updated:** 2026-11-01
**Current phase:** Phase 2 started. The shared evaluator and the untuned baselines are done (see Phase 2 Results). Phase 1 code is done. The Phase 1 exit-gate tests pass locally and on a CPU Kaggle kernel (15 tests, `results/tests/`). The source review is done for the main text of all 7 papers. `docs/sources.md` lists what was not read (appendices, the TrialGPT Methods section, parts of the Qwen2.5 evaluation).
**Deadline:** November 10, 2026, plus a 2-hour grace period. Target submission is November 9.

## Next Steps

1. Phase 2, next: write the LoRA training script (`trialmatch/train` plus a `scripts/` entry point) with the starting config in `docs/PLAN.md`. Run a short smoke job first (a few steps, check loss falls and the adapter saves), then the sweep.
2. Then the sweep (`r`, output order, oversampling, last-2-layers), seeds 42/43/44, and the draft LoRA.

## Accounts And Locations

| Item | Value |
|---|---|
| GitHub repo | `AKIF-jk/trialmatch` (private until submission) |
| Kaggle user | `akifjawad` |
| Source kernel | `akifjawad/trialmatch-src` (version 5 at commit `aee7889`; version 3 was a mistaken partial upload and failed) |
| Test kernel | `akifjawad/trialmatch-tests` (CPU, version 1, commit `f07a7b0`, 15 passed) |
| Hugging Face user | `AkifJK` |
| Dataset revision | `ncbi/TrialGPT-Criterion-Annotations` at `1cfcafde94a1560a33b4addc1638664fe28fc059` |
| GPU quota | 30 h/week, resets November 7, 00:00 UTC. About 1.4 min used on November 1. |

## Phase 1 Results

- **Split:** 37/8/8 patients, giving 603 train, 206 val and 206 test annotations. Seed 42, `splits/manifest.json`, sha256 `357486d0`. The seed was not changed after seeing class counts.
- **Rare classes:** train has 8 `excluded` and 13 `not included`. Val has 2 `excluded` and 3 `not included`. Val macro-F1 on those classes will be very noisy, so report support next to every score.
- **Unusable row:** annotation 883 (null criterion) is in the manifest, flagged in `unusable_annotation_ids`, and dropped by `select_rows`.
- **Lengths:** `results/phase1/token_lengths.json`. Longest example is 623 tokens, `MAX_LEN` is 2048, nothing is truncated, and `encode_example` raises instead of truncating. Target is at most 29 tokens.
- **Prompt:** Qwen ChatML text written by hand in `trialmatch/data/prompt.py`, with label definitions. Phase 2 must check it equals `tokenizer.apply_chat_template` on the Kaggle tokenizer. Output order is a parameter (`evidence_first` or `label_first`).
- **Test guard:** `select_rows(..., "test")` raises `LockedSplitError` unless `allow_test=True`. Only `scripts/final_eval.py` (not written yet) may pass it. Only aggregate class counts for test are in the manifest, as the plan requires.
- **Local env:** `.venv` holds numpy, pyarrow, huggingface_hub, tokenizers and pytest (`requirements-dev.txt`). Run tests with `PYTHONPATH=. .venv/bin/python -m pytest -q tests`. Tests download the 176 KB dataset from the pinned revision.

## Phase 2 Results

- **Evaluator:** `trialmatch/eval/{parse,metrics,baselines}.py`, tests in `tests/test_eval.py` (34 tests pass locally with the Phase 1 tests). Input is raw model text plus rows, so every model, baseline and decoding mode is scored through the same code. The rules (fixed label lists, zero-division 0, empty-evidence handling, invalid and malformed outputs) are in the `metrics.py` docstring. Strict JSON parsing is the default. The one repair step (first JSON object in the text) is opt-in via `repair=True` and counted in `repaired`.
- **Headline metric:** `labels.overall.macro_f1` over the fixed 6-label list. `macro_f1_supported` is the cross-check. Always quote `support` per class.
- **Majority baseline (val, per criterion type, empty evidence):** accuracy 0.621, macro-F1 0.215, evidence macro-F1 0.563. Global-majority variant: accuracy 0.490, macro-F1 0.140 with 66 invalid labels, because `not excluded` is invalid for inclusion rows. Report the per-type one.

- **Untuned baselines (val, 206 rows, greedy, fp16, T4, commit `aee7889`, `results/baseline/`):** the prompt text equals `tokenizer.apply_chat_template` for all 206 rows on both tokenizers. This was also the first T4 run through the source-tarball path, and it worked.

| Model | Parse | Accuracy | Macro-F1 | Malformed | Evidence macro-F1 |
|---|---|---|---|---|---|
| Majority (per type) | n/a | 0.621 | 0.215 | 0 | 0.563 |
| Qwen2.5-1.5B-Instruct | strict | 0.015 | 0.015 | 97 | 0.358 |
| Qwen2.5-1.5B-Instruct | repaired | 0.180 | 0.125 | 0 | 0.451 |
| Phi-3-mini-4k-instruct | strict | 0.184 | 0.141 | 64 | 0.293 |
| Phi-3-mini-4k-instruct | repaired | 0.417 | 0.229 | 0 | 0.565 |

  - **Why strict is so low:** Qwen wraps 97 outputs in a json code fence. Phi-3 appends extra text to 64. The repair step (first JSON object in the text) is logged as `repaired` and recovers them. Report both.
  - **Real weakness after repair:** Qwen predicts `excluded` for 140 rows (support 2) and `not excluded` for none (support 101). Phi-3 predicts `excluded` 68 times, `not excluded` 69 times. Neither beats the majority baseline on accuracy. Phi-3 repaired macro-F1 0.229 is just above 0.215. Fine-tuning has a large gap to close, and the comparison is meaningful.
  - **Cost:** Qwen load 24.5 s, generate 36.2 s, peak 3.18 GB. Phi-3 load 43.6 s, generate 160 s, peak 8.8 GB, 9 outputs hit the 96-token cap. About 6 min of T4 time in total.
  - **Phi-3 pin:** `microsoft/Phi-3-mini-4k-instruct` at `f39ac1d28e925b323eae81227eaba4464caced4e`, MIT licence, not gated.

## Known Constraints

- **Local machine:** 7 GB RAM and about 1.7 GB free disk. Do not install PyTorch or download models locally.
- **Kaggle attachments:** `kernelDataSources` is silently dropped by the MCP server. Pass outputs between kernels through signed output URLs from `list_notebook_session_output`. Fetch a fresh URL for each job, because their lifetime is unknown.
- **Delta source build:** `python3 tools/build_kernel.py src --base-url URL --base-commit SHA` inlines only files that differ from `SHA`. `URL` is the `src.tar.gz` output URL of the previous source kernel version, and `SHA` is the commit that version was built from (version 2 is `3cc217d`). The kernel asserts the base tree hash and the new tree hash. Tested locally for change, delete and wrong-base cases. Run on Kaggle from `3cc217d` to `f07a7b0`: both hash asserts passed and 20 files were packed. The next delta must use the version 5 `src.tar.gz` URL with `--base-commit aee7889`. `tools/build_kernel.py job` needs its positionals before the options (`job scripts/x.py arg1 arg2 --source-url URL`). Without the flags, the build inlines every file.
- **Source kernel size:** every byte of a delta is retyped through `save_notebook`, so keep committed files small. The manifest is written one key per line for this reason. Check that the text sent matches `build/src_kernel.py` in full. The hash asserts catch a truncated or mistyped upload.
- **Kaggle slugs:** Create a kernel with `newTitle`. Update one with `slug: "akifjawad/<name>"`.
- **Blocked actions:** A secret gist upload and `gh repo create` were blocked by the permission classifier. Do not route around a block. Tell the user and let them decide.
- **Hugging Face writes:** The connector and plugin tools are read-only. Use `~/.local/bin/hf`, which is logged in as `AkifJK` and passed a private create, upload and read-back test. The test repo has been deleted.
- **Library pin:** Stay on transformers 5.16.1. Use `cache.crop(-n)`, because positive crop lengths are removed in 5.18.
- **Generation settings:** Always pass `do_sample=False`, `repetition_penalty=1.0`, `temperature=None`, `top_p=None`, `top_k=None`. The model defaults are sampling settings.
- **Test set:** Do not load or inspect it before Phase 7. It runs once, from a tagged commit.

## Open Items For The User

None. Resolved on 2026-11-01: the rubric total is 135 (the table is right and the 100 in the heading is wrong), and the HF email is verified. The two existing Spaces use `cpu-basic`, so both free ZeroGPU slots should be available.

## Kernels In Flight

None. Source kernel `akifjawad/trialmatch-src` is now version 5 (commit `aee7889`, built as a delta from `f07a7b0`, both hash asserts passed, 25 files). Its `src.tar.gz` URL changes per fetch, so get a fresh one from `list_notebook_session_output`. The next delta uses `--base-commit aee78891498f9e0367138b02835bfda9fe394c6d`. Baseline kernel: `akifjawad/trialmatch-baseline` v1, complete.
