# Project Status

Read this first, then `docs/PLAN.md`. Update this file at the end of every phase or work session, and commit and push it.

**Last updated:** 2026-11-01
**Current phase:** Phase 1 code is done. The Phase 1 exit-gate tests pass locally (15 tests). The source review has started: paper identities are checked in `docs/sources.md`, but no paper is read yet.
**Deadline:** November 10, 2026, plus a 2-hour grace period. Target submission is November 9.

## Next Steps

1. Run `pytest tests` once on a CPU Kaggle kernel (delta source build, first use of `--base-url`). This confirms the tests also pass on the Kaggle image.
2. Read the papers in `docs/sources.md` and fill in the Read column.
3. Start Phase 2 as written in `docs/PLAN.md`. Write the shared evaluator first.

## Accounts And Locations

| Item | Value |
|---|---|
| GitHub repo | `AKIF-jk/trialmatch` (private until submission) |
| Kaggle user | `akifjawad` |
| Source kernel | `akifjawad/trialmatch-src` (version 2 at commit `3cc217d`) |
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

## Known Constraints

- **Local machine:** 7 GB RAM and about 1.7 GB free disk. Do not install PyTorch or download models locally.
- **Kaggle attachments:** `kernelDataSources` is silently dropped by the MCP server. Pass outputs between kernels through signed output URLs from `list_notebook_session_output`. Fetch a fresh URL for each job, because their lifetime is unknown.
- **Delta source build:** `python3 tools/build_kernel.py src --base-url URL --base-commit SHA` inlines only files that differ from `SHA`. `URL` is the `src.tar.gz` output URL of the previous source kernel version, and `SHA` is the commit that version was built from (version 2 is `3cc217d`). The kernel asserts the base tree hash and the new tree hash. Tested locally for change, delete and wrong-base cases. Not yet run on Kaggle. Without the flags, the build inlines every file.
- **Kaggle slugs:** Create a kernel with `newTitle`. Update one with `slug: "akifjawad/<name>"`.
- **Blocked actions:** A secret gist upload and `gh repo create` were blocked by the permission classifier. Do not route around a block. Tell the user and let them decide.
- **Hugging Face writes:** The connector and plugin tools are read-only. Use `~/.local/bin/hf`, which is logged in as `AkifJK` and passed a private create, upload and read-back test. The test repo has been deleted.
- **Library pin:** Stay on transformers 5.16.1. Use `cache.crop(-n)`, because positive crop lengths are removed in 5.18.
- **Generation settings:** Always pass `do_sample=False`, `repetition_penalty=1.0`, `temperature=None`, `top_p=None`, `top_k=None`. The model defaults are sampling settings.
- **Test set:** Do not load or inspect it before Phase 7. It runs once, from a tagged commit.

## Open Items For The User

- Ask the instructor whether the rubric is out of 100 or 135. The heading and the table disagree.
- Before Phase 6: confirm the HF email is verified. The two existing Spaces use `cpu-basic`, so both free ZeroGPU slots should be available.

## Kernels In Flight

None.
