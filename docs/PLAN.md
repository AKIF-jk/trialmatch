# Assignment 2 Workflow: Clinical Trial Criterion Matching

## Context

Build a Transformer system that labels a patient against each trial criterion and cites supporting sentence IDs. It must include a hand-written KV-cached decoder, tiled online-softmax attention, a speculative decoder, a browser app, an IEEE paper, a GitHub repo, a video and a live defense.

- **Deadline:** November 10, 2026, plus a 2-hour grace period. Late work gets zero. Target submission is November 9.
- **Compute:** Kaggle through the MCP server. GPU quota is 30 h/week, 0 h used so far, and it resets on November 7.

## Stack Check

| Choice | Verdict | Evidence and conditions |
|---|---|---|
| Qwen2.5-1.5B-Instruct (target) | Fine | Apache-2.0, not gated, `Qwen2ForCausalLM`, 28 layers, hidden size 1536, 12 query heads, 2 KV heads, 32,768 context, 1.54B params, about 3.1 GB in fp16 or bf16. Fits one T4. |
| Qwen2.5-0.5B-Instruct (draft) | Fine | Apache-2.0, 24 layers, hidden size 896, 494M params, about 1 GB. `tokenizer.json` SHA-256 is identical to the target's, and vocab size is the same (151,936). |
| LoRA | Fine | Optional in the brief, and suits 37 training patients. The report must compare it with at least one alternative. |
| Gradio | Fine, with hosting change | Kaggle cannot serve a persistent app and the laptop cannot run the model. Host on a free ZeroGPU Space (account confirmed eligible). |
| Kaggle MCP | Fine, with workflow change | It offers batch save-and-run, status, logs and output download, but no interactive cells were found and no dataset-creation tool surfaced. Design every job as a self-contained batch run. |
| GitHub | Ready through `gh` CLI | Private repo `AKIF-jk/trialmatch` exists and matches local `main`. `gh` is logged in with `repo` scope. No GitHub connector is listed in this session, and none is needed. |
| Hugging Face connector | Read-only tools | Account `AkifJK`, free tier, created about 2025-12-01, so past the 30-day ZeroGPU rule. Tools can read repos, docs and papers, and can invoke Spaces. They cannot create repos or upload files, even though the token has `contribute-repos`. The token expires 2026-11-01 16:42 UTC, so reconnect when it lapses. |

Local machine limits: i5-6300U, 7 GB RAM, no usable GPU, **1.6 GB free disk**. Use it only for code, Git, LaTeX and small result files. Do not install PyTorch or download models locally.

## Findings That Shape The Design

- **Dataset:** 1,015 annotations, 53 patients, 103 trials. 356 inclusion and 659 exclusion.
- **Severe imbalance:** `not excluded` has 484 rows, `excluded` has 10, `not included` has 23. The test split may contain few or no rare-class rows.
- **Metric labels:** Fix the label list and zero-division rule before seeing test results, and report support for every class. Also report a majority-class baseline.
- **Null criterion:** Annotation 883 (`sigir-201519`) has a null `criterion_text`. Keep it in the manifest, flag it, exclude it from training and metrics, and disclose this in the report.
- **Evidence:** 610 rows have no expert evidence, including 7 `included` rows. Use those 7 in contradiction analysis. Empty evidence must never imply a label.
- **Excluded columns:** Drop `gpt4_*` and `explanation_correctness` from all inputs. Ignore the `training` column because it conflicts with the required split (44 patients have mixed flags). Do not use `trial_title` as input because Task 1 does not list it.
- **No gold explanations:** The only explanations are GPT-4's, so do not use them as targets. Train on label and sentence IDs only.
- **Generation defaults:** Both models default to `do_sample=true` and `repetition_penalty=1.1`. Every comparison must explicitly set `do_sample=False` and `repetition_penalty=1.0`.
- **Cache shape:** The target uses grouped KV heads, so each KV tensor is `[B, 2, T, 128]`. That is about 28.7 KB per token in fp16, or about 12.3 KB for the draft.
- **Sentence IDs:** Notes use a `0. ... 1. ...` format. The app must segment unseen notes into the same format.
- **Rubric total:** The heading says 100 marks but the table sums to 135. Confirmed by the user on 2026-11-01: the total is 135. Work is prioritized using the table.

## Kaggle Execution Loop

GitHub is the source of truth. The repo stays private until submission so that this individual work is not exposed early.

1. Commit the code.
2. When source changes, run `tools/build_kernel.py src` and send it as a new version of the private `akifjawad/trialmatch-src` kernel. It writes `src.tar.gz` to its output. Use `--base-url URL --base-commit SHA` for a delta build that downloads the previous tarball and inlines only changed files, since every source byte is retyped through the MCP call.
3. Run `tools/build_kernel.py job --source-url URL <script>`. The job kernel downloads the tarball from the signed output URL and verifies a tree hash.
4. Submit with `save_notebook` as a private script kernel with internet on. Use T4 GPU for GPU jobs and CPU for tests, which saves quota.
5. Poll `get_notebook_session_status`, then read logs and file URLs from `list_notebook_session_output` and save outputs to `results/<job>/`.
6. Chain jobs through signed output URLs. `kernelDataSources` is silently dropped by the MCP server, and secret gists were blocked as public uploads.

Rules:

- Do not put tokens in notebook text. Upload adapters to HF from the local machine.
- Pin the model revisions (`989aa7980e` target, `7ae557604a` draft) and the dataset revision (`1cfcafde94`, license listed as `other`, to be reported in A2).
- Use the HF connector for pinned-revision checks, the literature review (`hf://papers/<id>/paper.md`) and docs (`hf://docs/...`), and to test the deployed Space with `dynamic_space`.
- Each job writes its config, commit SHA, library versions, hardware details and JSONL logs.
- Use a session timeout on every job. Keep runs short and save checkpoints to `/kaggle/working`.

## Phases

### Phase 0: Foundations (Nov 1) — DONE

Results are in `results/smoke/`. Open follow-up: the T4 has only run the direct smoke kernel, not the source-tarball path, so the first Phase 2 GPU job checks that.

- Create the repo structure: `trialmatch/{data,train,eval,decoding,attention,app}`, `configs/`, `scripts/`, `tests/`, `tools/`, `results/`, `report/`, `splits/`, `AI_USE.md`.
- Run a CPU and a T4 smoke job to confirm internet, HF download, log access, machine-shape names, and whether a ZeroGPU-compatible PyTorch (2.8 or newer) works on the T4.
- Freeze the dependency lock file.
- Free local disk space, since 98% used could break Git and LaTeX. Add the missing `algorithm.sty` and `algorithmic.sty` to `report/`.
- **Exit gate:** A kernel built from a commit runs and returns its outputs through the MCP server.

### Phase 1: Data And Split (Nov 1-2) — B1, B2

- Sort patient IDs, shuffle with `np.random.default_rng(42)`, then assign 37/8/8.
- Publish `splits/manifest.json` with patient IDs, annotation IDs, per-class counts and the dataset revision. Add a leakage test and a hash check.
- Build the prompt from the numbered note, criterion type, criterion text and the label set for that type.
- Target output: `{"evidence_sentence_ids":[0,2],"label":"included"}`.
- Mask prompt tokens with `-100` so loss covers only the assistant tokens.
- Measure token lengths and document truncation. Current character lengths suggest none will be needed, so fail loudly if any example would lose text.
- Add a test-data guard. Test rows load only with an explicit flag used by `scripts/final_eval.py`.
- Start the source review: TrialGPT, Qwen2.5, LoRA, Attention Is All You Need, FlashAttention, and the two speculative-decoding papers.
- **Exit gate:** Tests pass for split reproducibility, no patient overlap, no GPT-4 fields in prompts, and sentence-ID round trips.

### Phase 2: Baselines And Fine-Tuning (Nov 2-4) — A1-A3, B3, B4, C1, C2

- Write the shared evaluator first. It reports accuracy, macro-F1, per-class precision/recall/F1 and confusion matrices, overall and split by inclusion and exclusion.
- Evidence metrics: micro and macro precision, recall and F1.
- Empty-set rule: both empty scores 1 for that record in macro. Predicted empty with gold non-empty scores recall 0. Predicted non-empty with gold empty scores precision 0. Also report macro on non-empty gold records only.
- Invalid sentence IDs count as false positives. Malformed JSON counts as a wrong prediction and its rate is reported. Any repair step is logged.
- Run validation-only baselines for Qwen2.5-1.5B and Phi-3-mini (the rejected candidate), plus the majority-class baseline.
- Start LoRA from: `r=16`, `alpha=32`, dropout 0.05, targets `q_proj,k_proj,v_proj,o_proj`, learning rate 2e-4, cosine schedule, batch size 8, fp16 base with float32 adapter, GradScaler, gradient checkpointing.
- Select on validation macro-F1. Compare on validation only:
  - `r` of 8, 16 and 32
  - Evidence-first versus label-first output order
  - Rare-class oversampling on and off
  - LoRA versus unfreezing the last 2 layers
- Train the final config with seeds 42, 43 and 44 to measure variance.
- Train a LoRA draft for Phase 5 with the same pipeline, and upload both adapters to HF.
- **Exit gate:** Same-prompt before/after results on validation, training curves, and a frozen config committed by SHA. Test set still untouched.

### Phase 3: KV-Cached Decoding (Nov 3-4) — D1-D3 — DONE

Results are in `results/decoding/` and `docs/STATUS.md`. Qwen2 in transformers 5.16.1 takes no `cache_position`, so the decoder asserts the cache length instead. Three fp16 token mismatches at output 256 have a documented cause (rounding at near ties).

- Reference path: greedy decoding with a full forward pass over all tokens at each step and no cache.
- Cached path: an explicit loop using `DynamicCache`, `attention_mask`, `position_ids` and `cache_position`. Prefill runs once, then each step feeds only the last token and records cache growth.
- Support left-padded batches, with positions computed from `cumsum(mask)-1`.
- Equivalence: at least 3 prompt lengths × 3 output lengths, for example 64, 256, 1024 × 16, 64, 256, with forced continuation past EOS. Compare tokens and maximum logit difference, in float32 and fp16.
- Benchmark on T4 and CPU with warm-up, `torch.cuda.synchronize()`, at least 5 repeats, and median plus IQR. Measure prefill, decode, tokens/s and peak memory.
- **Exit gate:** All token sequences match, or any mismatch has a documented cause.

### Phase 4: Tiled Online-Softmax Attention (Nov 4) — E1-E3 — DONE

Results are in `results/attention/` and `docs/STATUS.md`. Errors are within tolerance in float32 and float64, and float32 tokens match end to end. Two fp16 flips have a documented cause (rounding at near ties). The Python tiled loop is slower than sdpa and saves memory instead.

- Reference: scaled dot-product attention with correct `sqrt(d_k)` scaling, causal masking and a stable softmax.
- Tiled: iterate over KV tiles, keeping a running maximum and normalizer. Rescale the accumulated output when the maximum changes, and handle fully masked tiles and rows without NaNs.
- Test T values of 1, 7, 64, 256 and 1024, with tile sizes of 1, 16, 64 and larger than T. Cover causal, padding and rectangular masks, float32 and float64, and extreme score values. Report maximum and mean error.
- Benchmark reference, tiled and `F.scaled_dot_product_attention` for speed and memory, and explain honestly why a Python tiled loop may lose to a fused kernel.
- Add the tiled function to `Qwen2Attention` with grouped KV-head expansion, and check greedy token equality. This also becomes an app mode.
- **Exit gate:** Errors within tolerance (about 1e-5 for float32, 1e-12 for float64) and tokens match end to end.

### Phase 5: Speculative Decoding (Nov 5) — F1-F3

DONE on Nov 2. Results are in `results/speculative/` and `docs/STATUS.md`. Tokens equal greedy in every test (float64, float32, fp16 on real models). Speculative decoding is slower than cached greedy on both the T4 (0.59 to 0.94) and the CPU (0.82 to 0.91), and the cost model explains why.

- Each round: the draft proposes K tokens, the target verifies them in one forward pass, the longest matching prefix is accepted, and the target's next token is added as the correction or bonus token.
- Use `cache.crop()` on both caches after each round, and feed the correction token to the draft so its state stays aligned.
- Handle EOS inside a draft and limit K near `max_new_tokens`.
- Test with a stub draft that always agrees, one that always disagrees and one with a mix, plus a randomly initialized draft. Tokens must always equal the Phase 3 greedy output. Use HF assisted generation as a cross-check only.
- Experiment with K of 1, 2, 4 and 8, base versus LoRA draft, and short JSON versus longer free text, on T4 and CPU.
- Report acceptance rate, tokens per round, target forward count, latency, tokens/s, peak memory and speedup or slowdown. Explain any slowdown.
- **Exit gate:** 100% token equality across all tests.

### Phase 6: Gradio App On ZeroGPU (Nov 5-7) — C3, C4, G1-G4

**Screening tab**

- Paste a note and see editable numbered sentences.
- Paste or select criteria for inclusion and exclusion. An optional NCT lookup uses public ClinicalTrials.gov metadata only.
- Results show the criterion, label, cited IDs, highlighted evidence text, missing information and an explanation.
- The default explanation is a template built only from cited sentences. An optional, clearly labeled model rationale receives only those sentences.
- Overall outcome uses explicit rules and is labeled as a rule-based aggregation, not a ground-truth verdict:
  - Any `not included` or `excluded` gives "Likely ineligible".
  - Otherwise, any `not enough information` gives "Needs review".
  - Otherwise, "Preliminarily eligible".

**Inference Lab tab**

- Run the same prompt live through no-cache, KV-cache and speculative modes, with selectable K and attention backend.
- Show token equality, prefill and decode time, tokens/s, peak memory and acceptance rate. Results must come from live runs, not static charts.

**Reliability and export**

- Validate empty input, oversized input, too many criteria, malformed criteria and unparseable model output.
- Catch ZeroGPU timeout and quota errors and show a clear retry message.
- Export Markdown and JSON reports with model and adapter revisions, decode settings, timestamp and app commit.
- Show this notice in the app and every report: "Educational screening prototype, not a clinical enrollment decision."

**Deployment**

- Create the model on `cuda` at module level, and wrap inference with `@spaces.GPU(duration=...)` using a dynamic duration. Use Python 3.12 and PyTorch 2.8 or newer.
- Daily quota is 2 min for anonymous visitors and 5 min for free accounts, so cap criteria per run and display the estimated cost.
- Keep the Space private until submission, then make it public.
- Write path: the connector and plugin tools cannot deploy. Adapters and app files go through `~/.local/bin/hf`, which is logged in and passed a private create, upload and read-back test. Space creation uses `hf repos create --type space --sdk gradio --private --flavor zero-a10g`, which is not tested yet.
- Weights: adapters only (tens of MB), with the Space loading base models from HF. Merged weights never touch the laptop.
- Backup: a tested kernel that launches the same app with `share=True`. It depends on a running session and is only for emergencies.
- **Exit gate:** From a clean browser session, the deployed URL completes screening, an invalid-input recovery, a live benchmark and a report download.

### Phase 7: Freeze And Final Evaluation (Nov 7-8) — C1-C4, D3, E3, F3

- Tag the commit with frozen code, prompts, thresholds and checkpoints before opening the test set.
- Run `scripts/final_eval.py` **once**. Evaluate the base and fine-tuned models on the same test rows with the same decode settings.
- Do not tune after this run. If a bug forces a rerun, disclose the bug and the rerun in the report.
- Analyze at least 8 distinct errors, including unsupported evidence, missing information, contradictions and rare classes.
- Run final benchmarks from the tagged commit using the refreshed GPU quota.
- Test with hand-written unseen cases. Never use test patients as demo examples.
- **Exit gate:** All tables and figures regenerate from `results/` with one script.

### Phase 8: Paper, Repo, Video, Submission (Nov 8-9) — A1-A3, H1-H3

- Write the IEEE LaTeX paper with all required sections, architecture diagrams, algorithms, and interpreted tables and graphs. Reference every figure and table in the text. Report failures and rejected options.
- Verify every citation against its source, with at least five credible ones.
- Finish the README with setup, startup, reproduction commands, split, adapter download link and app URL. Add `AI_USE.md` and benchmark logs.
- Check reproducibility by running a fresh kernel from the public repo.
- Record the 5-7 minute demo.
- Defense preparation: explain each component, practice changing K, the tile size and the aggregation rules, and evaluate an unseen case live.
- **Exit gate:** Submit on November 9. November 10 is reserved as a buffer.

## GPU Budget (Estimates)

| Work | T4 hours |
|---|---|
| Smoke tests and baselines | 2 |
| LoRA sweep and final seeds | 8 |
| Draft LoRA | 1 |
| Decoding and attention experiments | 5 |
| Final evaluation and benchmarks | 3 |
| Reruns and buffer | 6 |
| **Total** | **25** |

This fits within the 30 h available before November 7, with another 30 h after the reset. Run unit tests on CPU kernels.

## Actions Only You Can Do

- Make the repo, model and Space public when ready.
- Record the video, submit through Google Classroom, and attend the demo.

## Verification

- `pytest` on a CPU kernel covers the split, leakage, prompt fields, metrics, attention agreement, cache equivalence and speculative equivalence with stub drafts.
- GPU kernels repeat the token-equivalence checks on the real models.
- A rubric checklist maps every code from A1 to H3 to a file, result or app screen, and is reviewed before submission.
- The final check uses a fresh browser and a fresh kernel, relying only on the README.
