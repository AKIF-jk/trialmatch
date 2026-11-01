# Project Status

Read this first, then `docs/PLAN.md`. Update this file at the end of every phase or work session, and commit and push it.

**Last updated:** 2026-11-01
**Current phase:** Phase 0 is done. Phase 1 (data and split) has not started.
**Deadline:** November 10, 2026, plus a 2-hour grace period. Target submission is November 9.

## Next Steps

1. Change `tools/build_kernel.py src` to a delta build, so only changed files are retyped into the source kernel.
2. Start Phase 1 as written in `docs/PLAN.md`.

## Accounts And Locations

| Item | Value |
|---|---|
| GitHub repo | `AKIF-jk/trialmatch` (private until submission) |
| Kaggle user | `akifjawad` |
| Source kernel | `akifjawad/trialmatch-src` (version 2 at commit `3cc217d`) |
| Hugging Face user | `AkifJK` |
| Dataset revision | `ncbi/TrialGPT-Criterion-Annotations` at `1cfcafde94a1560a33b4addc1638664fe28fc059` |
| GPU quota | 30 h/week, resets November 7, 00:00 UTC. About 1.4 min used on November 1. |

## Known Constraints

- **Local machine:** 7 GB RAM and about 1.7 GB free disk. Do not install PyTorch or download models locally.
- **Kaggle attachments:** `kernelDataSources` is silently dropped by the MCP server. Pass outputs between kernels through signed output URLs from `list_notebook_session_output`. Fetch a fresh URL for each job, because their lifetime is unknown.
- **Kaggle slugs:** Create a kernel with `newTitle`. Update one with `slug: "akifjawad/<name>"`.
- **Blocked actions:** A secret gist upload and `gh repo create` were blocked by the permission classifier. Do not route around a block. Tell the user and let them decide.
- **Hugging Face writes:** The connector and plugin tools are read-only. Use `~/.local/bin/hf`, which is logged in as `AkifJK` and passed a create, upload and read-back test on the private repo `AkifJK/trialmatch-write-test`. That repo can be deleted once the user agrees.
- **Library pin:** Stay on transformers 5.16.1. Use `cache.crop(-n)`, because positive crop lengths are removed in 5.18.
- **Generation settings:** Always pass `do_sample=False`, `repetition_penalty=1.0`, `temperature=None`, `top_p=None`, `top_k=None`. The model defaults are sampling settings.
- **Test set:** Do not load or inspect it before Phase 7. It runs once, from a tagged commit.

## Open Items For The User

- Ask the instructor whether the rubric is out of 100 or 135. The heading and the table disagree.
- Before Phase 6: confirm the HF email is verified. The two existing Spaces use `cpu-basic`, so both free ZeroGPU slots should be available.

## Kernels In Flight

None.
