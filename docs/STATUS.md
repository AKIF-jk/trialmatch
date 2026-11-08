# Project Status

Read this first, then `docs/PLAN.md`. Update this file at the end of every phase or work session, and commit and push it.

**Last updated:** 2026-11-08
**Current phase:** Phase 6 done with Kaggle as the demo route (exit gate checked in a browser on 2026-11-08, see Phase 6 Results). Phases 7 (freeze and one-shot test evaluation) and 8 (paper, README, video, submission) are next and not started. Phase 5 done (exit gate met: speculative tokens equal greedy in every test, float32 and fp16; with a model draft it is slower than cached greedy on both T4 and CPU, and the reason is measured; prompt lookup, which has no draft model, is 1.4 to 1.74 times faster on the T4). Phase 4 done. Phase 3 done. Phase 2 mostly done. Evaluator, untuned baselines, sweep and 3-seed runs are done and the config is frozen (`configs/final.json`). The three `over4` target adapters and the draft adapter are on HF (private). Phase 1 code is done. The Phase 1 exit-gate tests pass locally and on a CPU Kaggle kernel (15 tests, `results/tests/`). The source review is done for the main text of all 7 papers. `docs/sources.md` lists what was not read (appendices, the TrialGPT Methods section, parts of the Qwen2.5 evaluation).
**Deadline:** November 10, 2026, plus a 2-hour grace period. Target submission is November 9.

## Next Steps

1. Phase 7 and 8 (deadline Nov 10, target submission Nov 9): tag the frozen commit, run `scripts/final_eval.py` once on the test split (not written yet), error analysis, tables and figures from `results/`, IEEE paper, README, `AI_USE.md`, video, submission. Phase 6 is closed; what remains of it is below.
   - **Phase 6 leftovers.** The demo is the Kaggle notebook `akifjawad/trialmatch-demo` (attach the Kaggle secret `HF_TOKEN`, Save & Run All, read the `*.gradio.live` link from the log, stop the session after). The HF Space cannot run (ZeroGPU refused), so there is no permanent URL: say so in the README and the report's Limitations, and note the Task 6 line that an evaluator must not need to run a notebook (G1 risk; ask the instructor). Document `python app.py` on any CUDA GPU as the evaluator route. Replace the `HF_TOKEN` secrets with read-only tokens. `tests/test_app_ui.py` never ran (the browser check covered that wiring).
2. The rare-class gap (no run predicts `excluded` correctly) is still open. Revisit only if time allows, on validation data only.

## Accounts And Locations

| Item | Value |
|---|---|
| GitHub repo | `AKIF-jk/trialmatch` (private until submission) |
| Kaggle user | `akifjawad` |
| Source kernel | `akifjawad/trialmatch-src` (version 19 at local branch `kaggle-v18` commit `45adff9`, a partial tree for the app engine tests; version 17 is the last full one, at commit `894901b`; version 3 was a mistaken partial upload and failed) |
| Hugging Face Space | `AkifJK/trialmatch` (private, SDK gradio, files uploaded, secret `HF_TOKEN` set, state `PAUSED`, no usable hardware). Created as a static Space and switched by uploading a Gradio `README.md`, because creating a Gradio Space directly returns 402. |
| Kaggle demo kernel | `akifjawad/trialmatch-demo` (T4, internet on, needs Kaggle secret `HF_TOKEN`; v1 failed on the symlink bug below, the fix is in the Space repo; the next run was the live link that was checked) |
| Kaggle self-test kernel | `akifjawad/trialmatch-app-selftest` v2 (T4, `results/app/selftest/`, its `src.tar.gz` output is the full app tree, tree hash `1411bee2`) |
| Test kernel | `akifjawad/trialmatch-tests` (CPU, version 10, commit `894901b`, 435 passed and 3 skipped, `results/tests/`) |
| Hugging Face user | `AkifJK` |
| Adapter repo | `AkifJK/trialmatch-qwen2.5-1.5b-lora` (private), revision `6fd2471b3894cdbe84d46f9d1278233a3279e77e`: `seed42/`, `seed43/`, `seed44/` (target adapters, first uploaded at `a4c9e329`), `draft/` (0.5B draft adapter), each with `adapter_model.safetensors` + `adapter_config.json`, and a model card. Read back and checksums matched. Make it public only at submission. |
| Dataset revision | `ncbi/TrialGPT-Criterion-Annotations` at `1cfcafde94a1560a33b4addc1638664fe28fc059` |
| GPU quota | 30 h/week, resets November 7, 00:00 UTC. About 4.9 h used as of November 2 (4.44 h before Phase 5, plus about 1,800 s of Phase 5 kernel time). |

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

- **Training script smoke (commit `74b76d0`, `results/train_smoke/`):** `scripts/train.py` ran 6 steps on 48 train rows and validated on 16 val rows, in both modes. LoRA `r=16` has 4,358,144 trainable parameters, loss 0.49 to 0.23, peak 11.84 GB. `last2` (layers 26 and 27 in float32) has 93,595,648 trainable parameters, peak 12.66 GB. Both save a best checkpoint. These 16-row scores mean nothing, so do not quote them.
  - **Peak memory is close to the T4's 14.6 GB.** Float32 logits for batch 8 at the longest sequence (623 tokens) are several GB. Use `--micro-batch 4` for real runs. The default is 8 and the smoke runs used it.
  - **Bugs found and fixed by the smoke runs:** Kaggle's torchao 0.10.0 makes peft 0.20.0 fail at `get_peft_model` (the script hides torchao from peft); validation-loss batch 16 ran out of memory (now `--val-loss-batch 4` and `expandable_segments`); `last2` generation needs fp16 autocast because of its float32 layers (applied to `last2` only).
  - **First GradScaler step in `last2` shows `grad inf`.** That is the scaler lowering its scale and skipping the step. It is expected.


- **Sweep (val 206 rows, seed 42, 3 epochs, micro-batch 4, commit `74b76d0`, `results/train/<run>/report.json`):** 602 usable train rows (603 minus annotation 883). No outputs were malformed in any run, so strict and repaired scores are equal.

| Run | Change | Val macro-F1 by epoch | Best epoch | Accuracy | Evidence macro-F1 | `not applicable` F1 | Peak GB |
|---|---|---|---|---|---|---|---|
| `base_r16` | base: r=16, alpha 32, evidence-first, no oversampling | 0.360/0.362/0.346 | 2 | 0.738 | 0.723 | 0.00 | 7.41 |
| `r8` | r=8, alpha 16 | 0.355/0.371/0.354 | 2 | 0.748 | 0.745 | 0.00 | 7.37 |
| `r32` | r=32, alpha 64 | 0.359/0.420/0.428 | 3 | 0.728 | 0.741 | 0.45 | 7.47 |
| `label_first` | label-first output order | 0.332/0.429/0.440 | 3 | 0.752 | 0.742 | 0.47 | 7.41 |
| `over4` | rare labels x4 in train | 0.401/0.365/0.426 | 3 | 0.738 | 0.739 | 0.40 | 7.41 |
| `last2` | last 2 layers, lr 2e-5 | 0.254/0.298/0.295 | 2 | 0.675 | 0.621 | 0.00 | 8.56 |

  - **Rank by the pre-set rule (best strict val macro-F1):** `label_first` 0.440, `r32` 0.428, `over4` 0.426, `r8` 0.371, `base_r16` 0.362, `last2` 0.298. For reference the majority baseline is 0.215 and the untuned models are 0.125 to 0.229, so every LoRA run is a real gain.
  - **The gaps among the top three and the base are not reliable.** The macro-F1 differences come almost entirely from `not applicable` (support 12), which the base run never predicts. `not included` (support 3) and `excluded` (support 2) score F1 0.00 in every run. `over4` predicted `not included` 3 times and all were wrong. Seed variance has not been measured, so read the table as one draw.
  - **`last2` is clearly worse** (accuracy 0.675, evidence micro-F1 0.305), but only one learning rate (2e-5) was tried. Report that limit.
  - **Cost:** about 11 to 13 min per run on a T4, peak memory 7.4 GB (8.6 GB for `last2`) with micro-batch 4. The two kernels ran in parallel in about 37 min wall time.


- **Seeds 42/43/44 on the top three configs (val 206 rows, `results/train/*/report.json`, `results/train/seeds_summary.json`):**

| Config | Val macro-F1 per seed (best epoch) | Mean | SD | Accuracy mean | Evidence macro-F1 mean | `not applicable` F1 per seed |
|---|---|---|---|---|---|---|
| `over4` (r=16, rare labels x4) | 0.426 / 0.440 / 0.473 | 0.446 | 0.024 | 0.731 | 0.739 | 0.40 / 0.00 / 0.47 |
| `r32` (alpha 64) | 0.428 / 0.446 / 0.429 | 0.434 | 0.010 | 0.739 | 0.737 | 0.45 / 0.47 / 0.45 |
| `label_first` (r=16) | 0.440 / 0.366 / 0.363 | 0.390 | 0.044 | 0.741 | 0.744 | 0.47 / 0.00 / 0.00 |

  - **The seed-42 lead of `label_first` was a lucky draw.** Seeds 43 and 44 score 0.366 and 0.363 because they never predict `not applicable`. Single-seed rankings in the sweep table above should not be quoted.
  - **`over4` has the best mean, but `over4` and `r32` are not distinguishable** (ranges 0.426 to 0.473 and 0.428 to 0.446 overlap, and the gap is half an SD). `r32` is more stable and always finds `not applicable` (F1 0.45 to 0.47). `over4` is the only config that ever gets a rare class right: seed 43 scores `excluded` F1 0.50 (2 predicted, 1 correct, support 2), seed 44 scores `not included` F1 0.22 (6 predicted, 1 correct, support 3). That is one correct row each, so it is a hint and not a result.
  - **Rare classes are still essentially unsolved.** `excluded` has support 2 and `not included` has support 3 on val, so no claim about them is possible from these numbers.
  - **Cost:** 11 to 13 min per run, peak 7.4 to 7.5 GB, no malformed outputs in any of the 9 runs.
  - **Frozen config:** `configs/final.json` (`over4`: r=16, alpha 32, evidence-first, rare labels x4, 3 epochs, lr 2e-4, micro-batch 4), chosen by the pre-set rule (highest mean val macro-F1). It is a close call with `r32`, and the report must say so. Checkpoint per run is its best epoch by val macro-F1. The adapter for the app is `over4_s44` (val 0.473). Phase 7 scores all three seed adapters on test once and reports mean and spread.


- **Draft LoRA (`draft_over4`, Qwen2.5-0.5B-Instruct, frozen args, seed 42, source v10 commit `92b8887`, `results/train/draft_over4/report.json`):** 2,162,688 trainable of 496,195,456 parameters. Val macro-F1 by epoch 0.310 / 0.400 / 0.438, best epoch 3. Accuracy 0.738, evidence macro-F1 0.661, no malformed outputs. It predicted `not included` 3 times (none right) and never `excluded`. About 5 min of T4 time, peak 5.3 GB. This is a single seed and is only the draft for speculative decoding, so its scores are context and not a result to quote on their own. The adapter is on HF under `draft/`.

## Phase 3 Results

- **Code (commit `97ae4db`, source v13):** `trialmatch/decoding/greedy.py` has `decode_uncached` and `decode_cached`, which call `model.forward` directly. They build their own 4-D causal and padding mask and their positions from `cumsum(mask)-1`. Both keep logits for the last column only (`logits_to_keep=1`), so the reference does not pay for unused `lm_head` work. `scripts/decode_equivalence.py` and `scripts/decode_benchmark.py` run the experiments. `tests/test_decoding.py` runs on a small random Qwen2 and skips locally because there is no torch.
- **Plan change:** Qwen2 in transformers 5.16.1 takes no `cache_position`. The write slot follows the cache length, so the decoder does not pass it and instead asserts the cache length after every call.
- **Tests:** 65 passed on a CPU kernel (`akifjawad/trialmatch-tests` v5, `results/tests/`).
- **Bugs found and fixed:** kept logits were cast to float32, which rounded the float64 comparison. `tests/test_train_data.py` imported `tests.test_data`, which a different `tests` package shadows on Kaggle.
- **Equivalence (T4, `results/decoding/equivalence/`):** 3 prompt lengths x 3 output lengths x 2 prompts, plus left-padded batches and 32 validation rows with real stop tokens. Float32 is `records_float32_run1.jsonl` (commit `677c04e`, 58 min). That kernel hit its 3600 s timeout during fp16, so the file also holds 8 partial fp16 records. Fp16 is `records.jsonl` and `report.json` (commit `97ae4db`, 16 min). The 8 repeated fp16 records are identical across the two runs, so these results are deterministic from run to run.

| Dtype | Records | Tokens equal | Matches `generate` | Max logit difference | Min top-2 margin |
|---|---|---|---|---|---|
| float32 | 28 | 28 | 28 | 8.5e-4 | 0.0022 |
| fp16 | 28 | 25 | 27 | 0.402 | 0.0 |

- **Cause of the 3 fp16 mismatches: logit rounding at near ties, not a cache bug.** At the first differing step all earlier tokens are the same, so the inputs match. The top two logits there are within 2 fp16 ulps. In this range (16 to 32) one ulp is 1/64 = 0.015625. Each gap is no larger than the numerical difference between the paths at that step.

| Case | First differing step | Uncached logits (uncached token, cached token) | Cached logits (same order) | Max logit difference at step | Cached matches `generate` |
|---|---|---|---|---|---|
| Single, start 0, 64 x 256 | 180 | 26.515625, 26.5 | 26.515625, 26.515625 (tie) | 0.035 | yes |
| Single, start 1500, 1024 x 256 | 223 | 20.171875, 20.15625 | 20.15625, 20.171875 | 0.027 | no |
| Padded batch, start 1500, row 2 (prompt 1024), output 256 | 195 | 22.0, 22.0 (tie) | 21.9375, 21.96875 | 0.078 | yes |

  - **Supporting evidence:** the same inputs in float32 show no mismatch. All 3 cases are at output 256, at step 180 or later. No validation row differed, and those outputs stop at or before 96 tokens.
  - **Limits:** the padded-batch row also differs from the same prompt run alone (`equals_alone` false for that row), and that second comparison has no logits recorded. The first-divergence logits explain only the cached-vs-uncached flip. The cause of the numerical difference is not isolated beyond that it comes from different tensor shapes on the same inputs. `generate` differs from the hand-written cached path in one case because it builds its own mask, which is also a guess.
  - **Consequence:** greedy token equality is not guaranteed across code paths in fp16 at long outputs. Report float32 as the equality result and fp16 flips as a numerical finding.
- **T4 benchmark (fp16, sdpa, warm-up, 5 repeats, synchronized, median, `results/decoding/benchmark_cuda_float16/report.json`, about 20 min):**

| Prompt x output | Batch | Uncached total s | Cached total s | Speedup | Uncached tok/s | Cached tok/s | Peak GB (uncached / cached) | Tokens equal |
|---|---|---|---|---|---|---|---|---|
| 64 x 16 | 1 | 0.50 | 0.50 | 1.0 | 31.9 | 32.3 | 2.89 / 2.89 | yes |
| 64 x 64 | 1 | 2.02 | 2.02 | 1.0 | 31.7 | 31.6 | 2.89 / 2.89 | yes |
| 64 x 256 | 1 | 11.65 | 8.20 | 1.4 | 22.0 | 31.2 | 2.91 / 2.90 | **no** |
| 256 x 16 | 1 | 1.01 | 0.50 | 2.0 | 15.8 | 32.1 | 2.90 / 2.91 | yes |
| 256 x 64 | 1 | 4.28 | 2.01 | 2.1 | 15.0 | 31.9 | 2.91 / 2.91 | yes |
| 256 x 256 | 1 | 19.40 | 8.04 | 2.4 | 13.2 | 31.9 | 2.92 / 2.91 | yes |
| 1024 x 16 | 1 | 3.22 | 0.64 | 5.0 | 5.0 | 24.9 | 2.95 / 2.98 | yes |
| 1024 x 64 | 1 | 13.14 | 2.14 | 6.2 | 4.9 | 30.0 | 2.96 / 2.98 | yes |
| 1024 x 256 | 1 | 56.48 | 7.96 | 7.1 | 4.5 | 32.1 | 2.97 / 2.98 | yes |
| 349 x 64 (padded rows) | 8 | 33.69 | 2.62 | 12.9 | 15.2 | 195.6 | 3.09 / 3.16 | yes |

  - **The one `no` is the 64 x 256 flip explained above.** The benchmark's flag compares every run in the cell with the first, so it does not name a path.
  - **Reading:** cached time is about 31 ms per token at every prompt length. At prompt 64 the uncached pass costs about the same per step, so the cache saves nothing until the sequence grows. Why short sequences cost the same has not been measured; per-step overhead outside the attention work is a guess.
  - **Prefill** is equal on both paths (0.03 to 0.17 s at batch 1).
  - **Cache size** is about 28 KB per token per row in fp16, which matches the plan's estimate. The cached path's peak is at most 0.06 GB above the uncached path's in every cell, and slightly below it in some short-prompt cells.
- **CPU benchmark (float32, 2 threads on 4 CPUs, warm-up, 5 repeats, median, `results/decoding/benchmark_cpu_float32/report.json`, 89 min):** tokens equal in every cell. Reduced grid, because a 1024-token or 256-output uncached CPU run would take far too long.

| Prompt x output | Uncached total s | Cached total s | Speedup | Uncached tok/s | Cached tok/s | Peak RSS GB (both) |
|---|---|---|---|---|---|---|
| 64 x 16 | 36.75 | 6.71 | 5.5 | 0.44 | 2.38 | 6.67 |
| 64 x 64 | 181.40 | 22.02 | 8.2 | 0.35 | 2.91 | 6.68 |
| 256 x 16 | 109.92 | 11.59 | 9.5 | 0.15 | 1.38 | 6.71 |
| 256 x 64 | 478.63 | 27.26 | 17.6 | 0.13 | 2.35 | 6.71 |

  - **Reading:** the cache helps much more on CPU than on the T4 at short prompts (5.5x against 1.0x at 64 x 16). CPU cost grows with the tokens processed, and the T4 cost at these lengths does not. Prefill matches on both paths (about 2.1 s at prompt 64, 6.7 s at prompt 256). Peak RSS is dominated by the float32 model, so the cache does not show in it.

## Phase 4 Results

- **Code (commit `e0a9aa6`, source v15):** `trialmatch/attention/reference.py` (full score matrix, stable softmax, bottom-right causal rule), `tiled.py` (online softmax over key tiles, running max `m`, normalizer `l`, accumulator `acc`; statistics in float32, or float64 for float64 input), `hf.py` (registers a named backend `tiled_<tile>` with `AttentionInterface` and `AttentionMaskInterface`, switched with `use_tiled_attention(model, tile)`), `cases.py` (shared inputs and masks). Grouped KV heads are not expanded: `q` is viewed as `[B, Hkv, rep, Tq, D]` and matmul broadcasts over `rep`. Rows with no allowed key output zeros, in both functions. The tile loop does not skip fully masked tiles. It handles them through the `-inf` shift rule, and the tests cover them.
- **Tests:** 371 passed on the CPU kernel (full suite, `results/tests/`). New: `tests/test_attention.py` (294 cases) and `tests/test_attention_hf.py` (12). They cover T 1, 7, 64, 256, 1024, tiles 1, 16, 64 and larger than T, six mask kinds (none, causal, padding, causal plus padding, rectangular, sparse), float32 and float64, fully masked rows and tiles, the running maximum in the first or last tile, extreme scores, and negative controls (naive softmax overflows, a wrong scale changes the output). The reference is checked against `F.scaled_dot_product_attention` in float64.
- **Bug found by the tests:** the first HF run failed 10 of 12 tests because the backend compared `attention_mask.dtype != bool` instead of `torch.bool`. Fixed in `e0a9aa6`.
- **Error table (`results/attention/check/report.json`, 240 cases):** against the float64 reference, worst max error is 1.08e-6 in float32 (tolerance 1e-5) and 2.1e-15 in float64 (tolerance 1e-12). Worst mean error 4.3e-8 and 8.1e-17. No NaN.
- **End-to-end tokens (T4, `results/attention/equivalence/`, Qwen2.5-1.5B, the hand-written cached decoder with sdpa against tiled, tiles 16/64/256, 7 inputs per tile: 4 single prompts, 2 left-padded batches, 16 validation rows with real stop tokens):**

| Dtype | Records | Tokens equal | Max logit difference | Min top-2 margin |
|---|---|---|---|---|
| float32 | 21 | 21 | 8.0e-4 | 0.0021 |
| fp16 | 21 | 19 | 0.33 | 0.0 |

  - **The 2 fp16 flips** are both validation rows, row 2, step 11 (tile 16 and tile 64). The sdpa logits for the two tokens tie exactly at 28.53125. The tiled path gives token 220 a logit higher by 1 fp16 ulp (28.546875, tile 16) or 2 ulps (28.5625, tile 64). One ulp in this range is 0.015625. Same kind of near tie as Phase 3. It is not a defect in the tiled function, but the tiled path is not bit-equal to sdpa, so ties can break the other way. Tile 256 flipped nothing in the same rows, but its min top-2 margin is 0.0, so a tie exists there too. Not isolated beyond that.
  - **Consequence:** equality is claimed in float32. Fp16 token equality is not guaranteed between attention implementations.
- **Attention benchmark (T4, fp16, 12 query heads, 2 KV heads, head dim 128, batch 1, 5 repeats, median ms and peak extra MB, `results/attention/bench_cuda_float16/report.json`):**

| Case | T | Reference | Tiled 64 | Tiled 256 | Tiled 1024 | sdpa |
|---|---|---|---|---|---|---|
| prefill | 1024 | 6.38 ms, 151 MB | 10.47 ms, 36 MB | 6.48 ms, 60 MB | 5.70 ms, 157 MB | 5.16 ms, 138 MB |
| prefill | 4096 | 70.9 ms, 2345 MB | 110.5 ms, 145 MB | 69.8 ms, 242 MB | 61.6 ms, 821 MB | 53.4 ms, 1896 MB |
| decode | 1024 | 0.96 ms, 12 MB | 6.97 ms, 0.5 MB | 2.34 ms, 1.8 MB | 1.35 ms, 7.1 MB | 1.05 ms, 20 MB |
| decode | 4096 | 1.15 ms, 48 MB | 25.7 ms, 0.5 MB | 6.98 ms, 1.8 MB | 2.48 ms, 7.2 MB | 1.23 ms, 80 MB |

  - Also measured at T 256 and 2048 (in the report). The max error against float64 is 9e-4 for prefill and 3e-5 to 1e-4 for decode, equal across all five backends, so it is fp16 output rounding and not an algorithm difference.
  - **Reading:** tiling cuts the temporary memory a lot. At T 4096 prefill, tile 64 uses 145 MB where the reference uses 2345 MB (16 times less) and tile 256 uses 242 MB (10 times less). The Python loop is slower. Tile 64 is 1.56 times the reference time at prefill T 4096, and 22 times at decode T 4096, because one decode step runs 64 loop iterations (4096/64) of several small kernels in every layer. Tile 1024 at prefill beats the reference (61.6 against 70.9 ms) but keeps 821 MB. `sdpa` is fastest at prefill (53.4 ms at T 4096), and its 1896 MB peak shows that it did not save memory in this setup. Which sdpa backend it used was not checked. The tiled function is a correctness and memory demonstration, not a speed path. A fused kernel avoids both the launch overhead and the score-matrix traffic, which a Python loop cannot.
  - **Not measured:** end-to-end decode speed with the tiled backend inside the model. Phase 6 shows it live as an app mode.
- **Local PyTorch install mistake:** this session installed CPU PyTorch into a scratch venv outside the repo, against the constraint below, and then deleted it with its pip cache after the user pointed it out. Nothing from it was used. All results above come from Kaggle.

## Phase 5 Results

- **Code (commit `894901b`, source v17):** `trialmatch/decoding/speculative.py`. Batch 1, no padding. Target cache holds `seq[:-1]`. Each round the draft proposes k tokens, the target reads `[seq[-1], d1..dk]` in one pass (`logits_to_keep=k+1`), the longest matching prefix m is accepted, and the target's token after it is emitted (correction if m < k, bonus if m == k). The target cache is cropped to `seq + d1..dm` with `cache.crop(-n)`. The draft (`ModelDraft`, own cache) is truncated to the same prefix, and at the start of the next round it feeds the tokens it has not read (the correction or bonus, and `d_k` after a full accept). k is limited to `remaining - 1`. A stop token ends the output at its first occurrence, from the draft or the target. `decode_speculative` takes any draft with `start`, `propose`, `truncate`, so tests use stubs.
- **Tests (`tests/test_speculative.py`, CPU kernel v10, 435 passed and 3 skipped in the full suite):** tiny random Qwen2 in float64 against `decode_cached`. Stub drafts that always agree (rounds = `ceil((N-1)/(K+1))`, acceptance 1.0), always disagree (1 token per round, target forwards = N) and a scripted mix (accepted count per round equals the script). A random draft model for K 0, 1, 2, 4, 8 and 5 prompt/output sizes (tokens equal, max logit difference under 1e-9). The target as its own draft (acceptance 1.0). A stop token at positions 0, 1, 2, 5, 11. Target pass sizes and cache length follow the accepted text (forward hook). Draft forward count. The tiled attention backend and float32. Input rejection. Negative control: with `crop` turned off the run fails. The first CPU run failed that control with a `RuntimeError` (sdpa saw a mask shorter than the keys) before the decoder's own cache assertion; the test now accepts either error. The 3 skips are cases where the stop token already appears earlier in a random output.
- **Equality on the real models (T4, `results/speculative/equivalence/`).** Target Qwen2.5-1.5B with the seed-44 adapter merged in (`over4_s44`), draft Qwen2.5-0.5B base or with the draft adapter merged in, K 1/2/4/8, validation rows with real stop tokens plus token-pool prompts with 64 forced new tokens, all against `decode_cached` with the same masks.

| Dtype | Cases x drafts x K | Tokens equal | Max logit difference | Min top-2 margin | HF assisted generation equal |
|---|---|---|---|---|---|
| float32 | 12 x 2 x 4 = 96 | 96 | 1.2e-4 | 0.027 | 96 of 96 |
| fp16 | 20 x 2 x 4 = 160 | 160 | 0.067 | 0.031 | not run |

  - **No fp16 flip appeared here, unlike Phases 3 and 4.** Those flips were at output 256, and this run uses at most 96 new tokens (validation rows) and 64 (pool prompts). Treat it as "no flip seen", not a guarantee. The largest logit difference between the paths is 0.067 in fp16 (about 4 ulps if the logits are in the 16 to 32 range, which was not checked), and the smallest top-2 margin is 0.031, so a near tie could still break the other way.
  - **Draft quality on the validation JSON (acceptance = accepted / proposed, fp16, 16 rows):** base draft 0.93 / 0.92 / 0.69 / 0.57 and LoRA draft 0.97 / 0.93 / 0.72 / 0.61 for K = 1 / 2 / 4 / 8. The LoRA draft is better at every K. On pool prompts (not JSON) the two are within 0.03 of each other, as expected.
- **T4 benchmark (fp16, sdpa, 12 validation rows, one prompt at a time, warm-up, 3 repeats, median of the sum over prompts, `results/speculative/bench_cuda_float16/report.json`, about 18 min).** Speedup is greedy total time over speculative total time (prefill included). The JSON workload uses the fine-tuned target (164 greedy tokens over 12 prompts, so about 14 tokens each). The text workload uses the untuned target and draft with a free-text prompt (1131 tokens, cap 128).

| Workload | Draft | K | Acceptance | Tokens per round | Target passes | Speedup |
|---|---|---|---|---|---|---|
| JSON | base | 1 / 2 / 4 / 8 | 0.94 / 0.91 / 0.66 / 0.56 | 1.85 / 2.81 / 3.38 / 5.07 | 94 / 66 / 57 / 42 | 0.90 / 0.94 / 0.75 / 0.64 |
| JSON | LoRA | 1 / 2 / 4 / 8 | 0.97 / 0.94 / 0.70 / 0.63 | 1.95 / 2.81 / 3.53 / 5.63 | 90 / 66 / 55 / 39 | 0.92 / 0.93 / 0.76 / 0.71 |
| Text | base | 1 / 2 / 4 / 8 | 0.81 / 0.72 / 0.62 / 0.43 | 1.80 / 2.43 / 3.44 / 4.34 | 645 / 481 / 358 / 270 | 0.93 / 0.87 / 0.77 / 0.59 |

  - Greedy cached: JSON total 5.56 s (IQR 0.10), 29.5 tokens/s. Text total 37.5 s (IQR 1.3), 30.2 tokens/s. Decode is 32.8 ms per token in both. Speculative IQRs are 0.04 to 0.22 s (JSON) and 0.3 to 1.2 s (text), so every gap above is larger than the spread.
  - **Every setting is slower than greedy. Best is 0.94, worst 0.59.** The target passes drop (JSON LoRA K=8: 39 against 164 tokens) but each round costs more than the passes it replaces.
  - **Why (measured):** one draft step takes 26 to 29 ms and one target pass 33.5 to 35.6 ms, whatever K. A target pass over 2 to 9 tokens takes 33.5 to 35.6 ms against 32.8 ms for one token, so on the T4 the cost is per call, not per token, at these lengths (measured here; Phase 3 showed the same flatness across prompt lengths). So a round costs about `K x 28 + 35` ms, and the check matches the data: JSON base K=2 is 2 x 28.1 + 35.0 = 91.2 ms against 91.2 ms measured, K=8 is 8 x 27.7 + 35.6 = 257 ms against 257.5 ms. That round emits 2.81 tokens at K=2, so 32.5 ms per token against 32.8 for greedy: break even before prefill. Prefill adds the draft's prompt pass on top. The draft is only 0.8 times as costly as the target (24 layers against 28, launch bound), so no K can win. A larger K lowers acceptance faster than it saves passes.
  - **Prefill:** speculative total minus decode shows the draft's prompt pass. On T4 it is small (JSON total speedup 0.94 against decode speedup 1.01 at K=2 for both drafts).
  - **One fp16 token flip in the text workload.** 1 of 12 prompts differs from greedy at K = 1, 2 and 4 (base draft) and none at K=8. The benchmark keeps no logits, so the cause is not isolated. It matches the Phase 3 and 4 pattern (fp16 near ties across different tensor shapes), but this is a guess. The float32 equality results above do not have it. All 12 JSON prompts are equal in every cell. The fine-tuned target's merged output is valid JSON for 12 of 12 prompts.
  - **Memory:** peak extra memory above the loaded models is 34 MB (JSON) and 25 MB (text) for greedy, and the same for every speculative cell. The draft's cache is small (about 12 KB per token in fp16) and the models, not the caches, dominate. Both models stay resident, so speculative decoding adds the 0.5B draft weights (about 1 GB in fp16).
- **CPU benchmark (float32, 2 threads on 4 CPUs, 4 rows, K 2 and 4, 1 repeat, text capped at 64 tokens, `results/speculative/bench_cpu_float32/report.json`).** Reduced grid, single repeat, so no IQR: read gaps under about 10% as noise.

| Workload | Draft | K | Acceptance | Total speedup | Decode-only speedup |
|---|---|---|---|---|---|
| JSON (53 tokens over 4 prompts) | base | 2 / 4 | 0.94 / 0.63 | 0.82 / 0.82 | 0.94 / 0.93 |
| JSON | LoRA | 2 / 4 | 0.97 / 0.77 | 0.87 / 0.87 | 1.06 / 1.08 |
| Text (243 tokens) | base | 2 / 4 | 0.69 / 0.58 | 0.85 / 0.91 | 0.86 / 0.93 |

  - All 24 prompt comparisons (4 prompts x 6 cells) are token-equal in float32.
  - **Slower on CPU too. My expectation that the CPU would show a speedup was wrong.** Greedy is 543 to 553 ms per token. A draft step is 208 to 261 ms (about 0.4 times the target, so the draft really is cheaper here). But a verification pass over 3 to 5 tokens takes 1.04 to 1.18 s, about 1.9 to 2.1 times a one-token step, because CPU cost grows with tokens processed (compute bound). A round at K=2 costs about 2 x 0.22 + 1.1 = 1.5 s for 2.88 tokens, which is 0.53 s per token against 0.55 for greedy: roughly break even in decode (0.94 to 1.08, within the noise of one repeat). Total speedup is lower because the draft's prompt pass adds 11 to 12 s on top of a 37 s greedy prefill (JSON outputs are only about 13 tokens, so prefill dominates the total: 37 s of 64 s).
  - **The LoRA draft decode-only values above 1 (1.06, 1.08) are one repeat and are not a result.**
- **What would make speculative decoding win:** a draft much cheaper than the target per call (here 0.8 times on T4 for a 0.5B model). Prompt lookup has a draft cost of about zero and does win on the T4; see the addendum below. Do not claim a speedup for the model drafts in the report.
- **Not done:** HF assisted generation was only run in float32 (cross-check); batch sizes above 1; draft prefill reuse; sampling (only greedy is verified).
- **Quirks found:** `list_notebook_session_output` needs `kernelSlug` without the owner prefix and `userName` separately (`akifjawad/trialmatch-src` as the slug is denied). Kernel logs appear only after the kernel ends.
- **Job arguments:** adapter files reach the job as `--fetch <folder>/<file>=<signed output URL>` (target and draft adapters came from the `best/` folders of `trialmatch-seeds-y` and `trialmatch-draft`). Jobs here were written straight into `save_notebook` with the arguments inlined (same template as `tools/build_kernel.py job`, commit `894901b`, tree hash `c5cdc271`). The signed URLs expire, so refetch before rerunning.

### Phase 5 addendum: prompt lookup (commit `b28d99a`)

Added after a classmate's table showed prompt lookup beating greedy on a T4 while model drafters lost, which matches the cost model above.

- **Code:** `trialmatch/decoding/prompt_lookup.py` (`PromptLookupDraft`: last n tokens, n from 3 down to 1, most recent earlier match, propose the tokens that followed, up to k; no match means no proposal). `decode_speculative` now accepts a draft that proposes fewer than k tokens or none: it sets k to the proposal length, a round with none is a plain one-token step, and a draft that proposes more than asked raises `ValueError`. Behaviour for fixed-length drafts is unchanged (the Phase 5 tests were rerun). `scripts/speculative_benchmark.py` and `scripts/speculative_equivalence.py` take `--drafts base lora lookup` and `--tag` (results go to `..._lookup` files so the model-draft reports are not overwritten). n-gram sizes were not tuned.
- **Tests:** `tests/test_prompt_lookup.py` (matching rule: longest n-gram first, most recent match, min n-gram, cut at the end of the text; tokens equal `decode_cached` in float64 with logits within 1e-9 for K 1/2/4/8/16, 1/2/9/40 new tokens, periodic and random prompts; stop tokens; rounds with k = 0 emit one token; a search over 40 seeds finds both an empty round and an accepted proposal; an over-long proposal is rejected). CPU kernel `trialmatch-tests-pl` v1: 487 passed, 5 skipped (all skips are stop-token cases where the token already appears earlier). The passing test count includes the 435 from before.
- **Source kernel:** `akifjawad/trialmatch-src-pl` v1 (commit `b28d99a`, delta from the v17 tarball of `trialmatch-src`, both hash asserts passed, 50 files). The shared `trialmatch-src` was not touched, because another session works on Phase 6. The jobs used their own kernels: `trialmatch-tests-pl`, `trialmatch-lookup-t4`, `trialmatch-lookup-cpu`.
- **Equality (T4, float32, target `over4_s44`, K 2/4/8/16, 8 validation rows plus 4 pool prompts, `results/speculative/equivalence/*_float32_lookup*`):** 48 of 48 token-equal to greedy, max logit difference 1.3e-4, min top-2 margin 0.027. HF assisted generation was not run for lookup. The CPU benchmark below is also token-equal in float32 for all 24 prompt comparisons.
- **T4 benchmark (fp16, 12 validation rows, 3 repeats, `results/speculative/bench_cuda_float16_lookup/report.json`, about 11 min).** Speedup is greedy total time over speculative total time. Greedy in this run: JSON 5.36 s (IQR 0.04), 30.6 tokens/s; text 36.2 s (IQR 0.14), 31.2 tokens/s.

| Workload | K | Acceptance | Tokens per round | Rounds with no proposal | Tokens/s | Speedup | Prompts token-equal |
|---|---|---|---|---|---|---|---|
| JSON | 2 | 0.55 | 1.67 | 40% | 46.6 | 1.52 | 12 / 12 |
| JSON | 4 | 0.43 | 1.92 | 46% | 52.3 | 1.71 | 12 / 12 |
| JSON | 8 | 0.22 | 1.92 | 46% | 52.3 | 1.71 | 12 / 12 |
| JSON | 16 | 0.11 | 1.92 | 46% | 51.1 | 1.67 | 12 / 12 |
| Text | 2 | 0.40 | 1.61 | 24% | 48.4 | 1.40 | 10 / 12 |
| Text | 4 | 0.28 | 1.81 | 28% | 54.0 | 1.63 | 11 / 12 |
| Text | 8 | 0.17 | 1.95 | 29% | 57.7 | 1.74 | 11 / 12 |
| Text | 16 | 0.09 | 1.90 | 32% | 55.4 | 1.74 | 11 / 12 |

  - **Why it wins on the T4:** a round costs one target pass (32.1 to 33.2 ms) plus 0.2 ms of lookup, against 28 ms per draft step for the 0.5B models, and a pass over 2 to 17 tokens costs the same as over one. Each round emits about 1.9 tokens, so decode runs at about 32.4 ms per 1.9 tokens against 32.8 per token (decode-only speedup 1.87 on JSON, 1.76 on text at K=8). Acceptance is low (0.1 to 0.55) because the guesses are crude, and 24 to 46% of rounds find no match, but no draft cost is lost on them. Larger K stops helping once the matches are used up (JSON K=4, 8 and 16 give the same rounds). The speedup spread over repeats is small (IQR 0.02 to 0.13 s against totals of 3 to 26 s), so the gaps between K=2 and the others are real; K=4, 8 and 16 differ by less than the spread on JSON.
  - **Compared with the model drafts on the same T4 and rows:** best model draft 0.94 (JSON) and 0.93 (text). Lookup is 1.7 and 1.74 at its best K. Memory: no draft weights (the 0.5B drafts add about 1 GB).
  - **fp16 token flips in the text workload:** 2 of 12 prompts at K=2 and 1 of 12 at K=4, 8 and 16 differ from greedy (the model drafts had 1 of 12 at K=1, 2, 4). JSON is 12 of 12 at every K. The benchmark keeps no logits, so the cause is not isolated. It matches the earlier near-tie pattern across different tensor shapes (a verification pass over several tokens against single-token steps), but this is a guess. The float32 check above has no flips.
- **CPU benchmark (float32, 2 threads, 4 rows, K 2/4/8, 1 repeat, text capped at 64 tokens, `results/speculative/bench_cpu_float32_lookup/report.json`).** One repeat and 4 prompts, so no IQR: read gaps under about 10% as noise. All 24 prompt comparisons are token-equal.

| Workload | K | Acceptance | Rounds with no proposal | Total speedup | Decode-only speedup |
|---|---|---|---|---|---|
| JSON (53 tokens) | 2 / 4 / 8 | 0.66 / 0.52 / 0.26 | 43% / 50% / 50% | 1.03 / 1.10 / 1.03 | 1.08 / 1.31 / 1.13 |
| Text (243 tokens) | 2 / 4 / 8 | 0.30 / 0.21 / 0.13 | 27% / 29% / 31% | 0.83 / 0.90 / 0.89 | 0.79 / 0.87 / 0.86 |

  - **On the CPU lookup is roughly break even, not a win.** A verification pass takes 579 to 715 ms against 364 to 372 ms for a one-token step (these rounds include the ones with no proposal, which cost a plain step), because CPU cost grows with the tokens verified. With about 1.4 to 2.0 tokens per round that cancels most of the gain. On JSON the prefill (33 of 51 s) hides the rest.
  - **This CPU host was faster than the one in the earlier CPU table** (greedy 364 to 372 ms per token here against 543 to 553 before). Compare speedups only within one run, never absolute times across the two tables.
- **Comparison with the classmate's table (T4 screening, fine-tuned 1.5B target):** their prompt lookup gets 1.32 (screening) and 1.58 (explanation) at its best K, and their model drafters get 0.72 to 0.81. Ours gets 1.52 to 1.71 (JSON) and 1.40 to 1.74 (text), and 0.59 to 0.94 for the model drafts. Same ordering, same reason. Their CPU model draft gets 1.37 and ours 0.82 to 0.91; those differ and the cause is unknown (different hardware, threads and batch are likely, but not checked).
- **Not done for lookup:** HF assisted generation cross-check, fp16 equality run with logits kept, n-gram size search, batch sizes above 1.

## Phase 6 Results (in progress)

- **Code (commit `656073b`, plus the prompt-lookup lab mode added afterwards):** `app.py` (loads the models at import, wraps the model calls in `@spaces.GPU` with a dynamic duration), `trialmatch/app/text.py` (note segmentation, numbered-note parsing, criteria parsing, ClinicalTrials.gov lookup by NCT id, input limits: 80 sentences, 8000 characters, 12 criteria, 600 characters per criterion), `screen.py` (reads model output with the Phase 2 parser, strict first then the logged repair step; label checked against the criterion type; ids outside the note dropped and flagged; the aggregation rule; template explanation from cited sentences only; missing-information flags; Markdown and JSON reports), `engine.py` (screening in left-padded batches of 8 through `decode_cached`; optional rationale from the untuned 1.5B on the cited sentences only; the Inference Lab), `ui.py` (Gradio tabs). `tools/deploy_space.py` stages `build/space`. `docs/DEPLOY.md` has the steps.
- **Inference Lab:** one prompt through no cache, KV cache, model-draft speculative (base or LoRA 0.5B draft) and prompt-lookup speculative, with sdpa or tiled attention (tile 16/64/256), K 1 to 8, up to 64 new tokens, 1 to 3 repeats (median). Tokens are compared with a cached sdpa run. Columns: prefill, decode, total, tokens/s, peak extra GPU memory, acceptance, tokens per round, target passes. The text under the table is built from the measured rows only. An option keeps generating past the stop token, because the JSON answer is only about 15 tokens.
- **Decisions:** the labels come from the fine-tuned model only. The rationale is optional, labelled as model text, and never feeds the label. An empty evidence list never changes a label; it only adds a flag. `not applicable` criteria do not count in the aggregation. An unreadable output counts as "needs review". Adapters load from the pinned HF revision and are checked against the sha256 in `configs/final.json`. All four models load at import time because ZeroGPU wants that. ZeroGPU supports torch 2.11.0 and Python 3.12 (the Kaggle versions), the free tier gives 2 GPU minutes a day to anonymous visitors and 5 to free accounts, and the GPU is an RTX Pro 6000 Blackwell (so lab timings differ from the T4 numbers in the report; the app records the device).
- **Tests run:** `tests/test_app_logic.py` (41, torch-free, pass on the laptop; segmentation, criteria parsing, limits, NCT lookup with a faked network, interpretation of good and bad model output, aggregation rules, explanations, reports, lab table and text). `tests/test_app_engine.py` (11 passed on the Kaggle CPU kernel `akifjawad/trialmatch-app-tests` v1, `results/app_tests/`): tiny random Qwen2 in float64 with a stand-in tokenizer; batch equals single, order kept across batches, screening never raises on junk output, scripted JSON is read correctly, rationale prompts contain only cited sentences, all lab modes give the cached tokens with sdpa and with tiled attention, the attention setting is restored, provenance fields.
- **Hosting blocked (2026-11-02):** `hf repos create AkifJK/trialmatch --type space --flavor zero-a10g` returned `402 Payment Required` ("Your account is not eligible to host ZeroGPU Spaces for free"), and the same with `--flavor cpu-basic` ("hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription"). This contradicts the earlier note that the two free ZeroGPU slots were available. The user then found a way to create the Space: make it first (static SDK is allowed, `hf repos create ... --sdk static --private` worked), upload the app with a Gradio `README.md` (SDK becomes `gradio`), set the secret. That part worked. Changing the hardware to ZeroGPU still fails: `hf spaces settings --hardware zero-a10g` returns the same 402, and the web settings page still shows it as paid. The Space stays `PAUSED`. The user chose the Kaggle route for the demo.
- **Secret caution:** the `HF_TOKEN` secret on the Space was set from the local login token (write scope unknown). Replace it with a read-only token, and use a read-only token for the Kaggle secret too.
- **Self-test on a Kaggle T4 (`akifjawad/trialmatch-app-selftest` v2, source tree hash `1411bee2` matched, `scripts/serve_app.py --selftest`, `results/app/selftest/`):** the real app launched on localhost and was driven through `gradio_client`. Engine load 40 s (target with the seed-44 adapter, both drafts and the untuned 1.5B, all on the GPU; adapters from kernel outputs, sha256 checked). Whole self-test 38 s.
  - **Screening (hand-written diabetes example, 7 criteria, rationale on):** 4.1 s on the T4, no malformed outputs, Markdown and JSON reports written with the notice and provenance. Invalid inputs each gave a plain message and no stack trace: empty note, broken numbering (0. then 5.), no criteria. A valid run right after each worked.
  - **Labels on that example are partly wrong, which matches the model's modest validation scores (macro-F1 0.47):** "HbA1c between 7.5% and 11%" got `not enough information` while the note says 8.1% and the model cited that very sentence; "History of cardiovascular disease" got `not included` citing the pregnancy and hysterectomy sentence, when the note says nothing about it; "Serum creatinine above 1.5" got `not enough information` while the note gives 0.9. The rule-based outcome follows the labels, so it said "Likely ineligible". The rationales (untuned model) restate the label and in one case invent support. Keep these as error-analysis material for Phase 7 (the example note is hand-written, not from the dataset).
  - **Inference Lab, 13 to 48 tokens, one repeat, Tesla T4 fp16:** every mode matched the cached sdpa tokens in all three runs (sdpa, sdpa with 48 forced tokens and 2 repeats, tiled attention tile 64 with 32 forced tokens). Total time against KV cache: no cache 0.85x to 0.64x of its speed (KV cache 1.18x to 1.56x faster, 1.02x with tiled attention); model-draft speculative 0.87x, 0.90x, 0.86x (slower; a round costs 74 to 158 ms of draft plus 36 to 100 ms of target pass); prompt-lookup speculative 1.49x, 1.22x, 1.34x (faster; acceptance 0.62, 0.21, 0.41, 1.4 to 1.7 tokens per round). This reproduces the Phase 5 result live: the draft model costs more than it saves, lookup does not. Single runs, so read small gaps as noise. Tiled attention is slower than sdpa (KV cache 2.83 s against 1.60 s for 32 to 48 tokens).
  - **Lab with no mode selected** gave a plain message.
- **Kaggle demo notebook (`akifjawad/trialmatch-demo`).** The script reads the Kaggle secret `HF_TOKEN`, installs `gradio==5.50.0`, downloads the Space repo (`snapshot_download(..., repo_type="space")`), sets `TRIALMATCH_SHARE=1` and runs `app.py`, which loads the adapters from the private model repo and prints a public `*.gradio.live` link. First run failed: `FileNotFoundError: .../hub/configs/final.json`. Cause: `engine.py` used `Path(__file__).resolve()`, which follows the Hub cache symlinks into the `blobs` folder, so the project root was wrong. Fixed with `os.path.abspath` (commit `5f8b903`, Space commit `2e9e553`). `trialmatch/data/split.py` has the same `resolve()` pattern but is not on the app's load path.
- **Live check of the demo link in a browser (2026-11-02, Tesla T4):** the page needed about 40 s to render the first time (fonts through the tunnel). Screening on the diabetes example: 7 criteria in 1.0 s, labels identical to the self-test (greedy, deterministic), report files offered. Inference Lab, defaults (13 tokens, 1 repeat): all four modes token-equal to the KV-cache run; KV cache 1.41x no cache; model-draft speculative 1.04x KV cache (faster this time, within noise of one short run, against 0.86x to 0.90x in the self-test); prompt-lookup 1.64x. The invalid-input cases were not repeated in the browser.
- **UI problems found by that check, fixed:** "Load example" with nothing selected showed a raw red "Error" (`EXAMPLES[None]` raised); it now shows a warning. The cost line said "of your daily quota", which is wrong off ZeroGPU; the wording is now neutral. Both are in commit `eb4d6f4` and the Space repo (`494294e`) and apply on the next notebook start. The running link had the old behaviour.
- **Exit-gate check on a fresh demo link (2026-11-08, `trialmatch-demo` restarted so it ran the fixed code, Tesla T4, new link):**
  - **Fixes live:** "Load example" with nothing selected shows a warning toast ("Pick an example in the dropdown first.") instead of an error; the cost line reads "on ZeroGPU this counts against the daily quota".
  - **Invalid input and recovery:** Screen with an empty note shows "Cannot run. The note is empty. Paste a patient note and press Segment note." with no results; the Asthma example (not run before) then screened correctly: 6 criteria with rationale in 3.1 s (0.8 s labels, 2.3 s rationale), outcome "Needs review". Labels again partly wrong: "FEV1 between 50% and 80%" got `not enough information` while the note gives 72% and the model cited that sentence.
  - **Report download:** Markdown and JSON files offered; JSON read back in the browser: notice present (twice in the Markdown), 6 criteria, adapter folder `seed44` with sha256 starting `6d6cc082e604` (matches `configs/final.json`), greedy settings recorded, device Tesla T4, fp16, timestamp. The app commit field reads `space-AkifJK/trialmatch` (set by the notebook), not a git hash.
  - **Inference Lab, default settings (13 tokens, one repeat):** all four modes token-equal to the KV-cache run. KV cache 1.77x no cache; model-draft speculative 0.99x KV cache; prompt lookup 1.66x. With tiled attention (tile 64): all four token-equal again; KV cache 1.07x no cache; model draft 1.00x; lookup 1.61x; tiled KV cache 1.05 s against 0.44 s with sdpa.
  - **Reliability finding:** on the first Screen click of that session the browser's event stream to the server was aborted (`ERR_ABORTED` on `queue/data`), so the page stayed on "processing" for more than 30 s although the server had finished. Clicking Screen again worked at once. Treat the gradio.live tunnel as flaky: have a reload ready and rehearse before the video.
  - **Wording fix:** a ratio of 1.00 was printed as "(faster)". The commentary now says "about the same" within 2% (`screen.py`, test added, 42 logic tests pass; in the Space repo as `d769f67`, live from the next notebook start).
- **Not run:** `tests/test_app_ui.py` (needs gradio on a machine that has the UI file; the self-test and the browser checks covered the same wiring against the real engine). The ZeroGPU path (`@spaces.GPU`, `app.py` under ZeroGPU) has never run. Quota and timeout error text in `friendly_error` is a guess. `Engine.load` with the Hub adapter path (`fetch_adapter`) ran for the first time in the demo notebook; the self-test used local adapter folders.
- **Kaggle bookkeeping:** `trialmatch-src` v19 is commit `45adff9` on the local branch `kaggle-v18` (894901b plus `engine.py`, `screen.py`, `text.py`, `tests/test_app_engine.py` only, to keep the retyped delta at 43 KB). It does not hold `ui.py`, the prompt-lookup files or the speculative.py change. The prompt-lookup session uses its own `trialmatch-src-pl`. v18 failed on an unescaped quote in my transcription (the syntax error showed in the log), v19 passed both hash asserts. The self-test job rebuilds the full app tree itself (v19 tarball, the prompt-lookup files from `trialmatch-src-pl`, `ui.py`, `engine.py` and `serve_app.py` inline, three edits to `screen.py`) and writes it as `src.tar.gz` in the output of `trialmatch-app-selftest` v2, tree hash `1411bee2`, so later app fixes can be small deltas against that tarball (get a fresh URL; it is not a git commit: it is local branch `kaggle-v18` at `56d88c3` plus the `screen.py` of `main`). Raw-string inlining (`r'''...'''`) was far easier to transcribe than the generated repr.

## Known Constraints

- **Local machine:** 7 GB RAM and about 1.7 GB free disk. Do not install PyTorch or download models locally.
- **Kaggle attachments:** `kernelDataSources` is silently dropped by the MCP server. Pass outputs between kernels through signed output URLs from `list_notebook_session_output`. Fetch a fresh URL for each job, because their lifetime is unknown.
- **Delta source build:** `python3 tools/build_kernel.py src --base-url URL --base-commit SHA` inlines only files that differ from `SHA`. `URL` is the `src.tar.gz` output URL of the previous source kernel version, and `SHA` is the commit that version was built from (version 2 is `3cc217d`). The kernel asserts the base tree hash and the new tree hash. Tested locally for change, delete and wrong-base cases. Run on Kaggle from `3cc217d` to `f07a7b0`: both hash asserts passed and 20 files were packed. The next delta must use a fresh version 17 `src.tar.gz` URL with `--base-commit 894901b639911cea66c717f3ea4e8eeddc9ad399`. The `list_notebook_session_output` tool takes `kernelSlug` without the owner prefix. `tools/build_kernel.py job` needs its positionals before the options (`job scripts/x.py arg1 arg2 --source-url URL`). It cannot take arguments that start with `--`, so build with a dummy arg `x` and replace `['x']` in `build/<name>_kernel.py` with the real list (this also allows the bare `--` run separator). Without the flags, the build inlines every file.
- **Source kernel size:** every byte of a delta is retyped through `save_notebook`, so keep committed files small. The manifest is written one key per line for this reason. Check that the text sent matches `build/src_kernel.py` in full. The hash asserts catch a truncated or mistyped upload.
- **Kaggle slugs:** Create a kernel with `newTitle`. Update one with `slug: "akifjawad/<name>"`.
- **Kaggle sessions:** T4 jobs use `enableGpu: true` and `machineShape: "NvidiaTeslaT4"`. A timed-out kernel keeps the files it wrote and its log, so write results incrementally. Status then reads `CANCEL_ACKNOWLEDGED`. `cancel_notebook_session` needs a session ID, and no tool returns one for a running kernel. It only appears in output URLs (`/kf/<id>/`) after files exist, so stop a running kernel from the Kaggle page.
- **Equivalence cost:** a float32 run with logits kept uses most of a 3600 s timeout. Give each dtype its own run.
- **Blocked actions:** A secret gist upload and `gh repo create` were blocked by the permission classifier. Do not route around a block. Tell the user and let them decide.
- **Hugging Face writes:** The connector and plugin tools are read-only. Use `~/.local/bin/hf`, which is logged in as `AkifJK` and passed a private create, upload and read-back test. The test repo has been deleted. Gradio Spaces cannot be created directly on this account (402); a static Space can, and uploading a Gradio `README.md` converts it. `short_description` in the Space card must be 60 characters or fewer. Passing a secret: `HF_TOKEN=... hf spaces secrets add AkifJK/trialmatch -s HF_TOKEN` reads it from the environment.
- **Library pin:** Stay on transformers 5.16.1. Use `cache.crop(-n)`, because positive crop lengths are removed in 5.18.
- **Generation settings:** Always pass `do_sample=False`, `repetition_penalty=1.0`, `temperature=None`, `top_p=None`, `top_k=None`. The model defaults are sampling settings.
- **Test set:** Do not load or inspect it before Phase 7. It runs once, from a tagged commit.

## Open Items For The User

- **Hosting: decided.** The user chose Kaggle for the demo (no other option available). Still worth asking the instructor whether that satisfies G1.
- **Tokens.** Replace the `HF_TOKEN` secret on the Space and in the Kaggle secret with read-only tokens.
- **Demo.** The exit gate was checked on 2026-11-08. For the video: restart `trialmatch-demo` shortly before, open the new link, rehearse once (the tunnel dropped one event stream), and stop the session afterwards. The newest wording fix (`d769f67`) is only live after a restart.
- Resolved on 2026-11-01: the rubric total is 135 (the table is right and the 100 in the heading is wrong), and the HF email is verified.

## Kernels In Flight

`akifjawad/trialmatch-demo` runs only when the user starts it (serves a live link; stop the session after the demo to save GPU quota). Finished in Phase 6: `trialmatch-app-tests` v1 (CPU, 11 engine tests), `trialmatch-app-selftest` v1 (failed on the old `screen.py`) and v2 (T4, passed, about 2.5 min). Finished in Phase 5 (prompt lookup added later): `trialmatch-src-pl` v1, `trialmatch-tests-pl` v1, `trialmatch-lookup-t4` v1 (about 11 min), `trialmatch-lookup-cpu` v1 (about 15 min). Finished in Phase 5: `trialmatch-tests` v9 and v10 (CPU tests), `trialmatch-spec-equiv-f32` v1 (about 7 min on T4), `trialmatch-spec-equiv-f16` v1 (about 4 min), `trialmatch-spec-bench-t4` v1 (about 18 min), `trialmatch-spec-bench-cpu` v1 (about 20 min). Finished in Phase 4: `trialmatch-tests` v6 to v8 (CPU tests), `trialmatch-attn-check` v1 (CPU error table), `trialmatch-attn-bench-t4` v1 (about 30 s), `trialmatch-attn-equiv-t4` v1 (about 7 min). Finished: `trialmatch-decode-equivalence` v2 (fp16; v1 timed out), `trialmatch-decode-bench-cpu` v1, `trialmatch-decode-bench-t4` v1, `trialmatch-draft` v1 (draft training). Finished: `trialmatch-seeds-x` v1 and `trialmatch-seeds-y` v1 (seed runs, source v9), sweep kernels (`trialmatch-sweep-a` v1, `trialmatch-sweep-b` v1). Their best adapters sit in the kernel outputs and are not yet downloaded or uploaded to HF. Old notes follow.

- (done) `akifjawad/trialmatch-sweep-a` v1: `base_r16`, `r8` (alpha 16), `r32` (alpha 64). T4, 3 epochs, micro-batch 4, seed 42.
- `akifjawad/trialmatch-sweep-b` v1: `label_first`, `over4` (rare labels x4), `last2` (lr 2e-5). Same settings; later runs pass `--order evidence_first` explicitly because runs inherit the first run's flags.
- Both started 2026-11-01 from source v9 (tree hash `9b7b27a6`, commit `74b76d0`; the job's recorded commit is `937dcfa`, which differs only in docs and results). Expect about 30 min each. Results go to `results/train/<run>/`.

Finished earlier: Source kernel `akifjawad/trialmatch-src` is version 9 (commit `74b76d0`, delta from `b31148d`, both hash asserts passed, 29 files). Its `src.tar.gz` URL changes per fetch, so get a fresh one from `list_notebook_session_output`. Finished kernels: `trialmatch-baseline` v1, `trialmatch-train-smoke` v4.
