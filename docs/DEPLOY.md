# Deploying the app

The app is `app.py` plus `trialmatch/app/`. It loads Qwen2.5-1.5B-Instruct with the seed-44 LoRA merged in, the 0.5B draft (base and LoRA) and the untuned 1.5B model (for optional rationales) at import time, then serves a Gradio UI with two tabs: Screening and Inference Lab.

## Run it locally (needs a GPU or patience on CPU)

```bash
pip install -r requirements-kaggle.txt gradio==5.50.0 huggingface_hub pyarrow
export HF_TOKEN=...   # read access to the private adapter repo AkifJK/trialmatch-qwen2.5-1.5b-lora
python app.py          # http://localhost:7860
```

On CPU the models load in float32 (about 12 GB of RAM for all four) and one criterion takes tens of seconds.

## Hosting status

Hugging Face refused to create the Space on this account on 2026-11-02: ZeroGPU gave `402 Payment Required` ("not eligible to host ZeroGPU Spaces for free") and a free `cpu-basic` Gradio Space gave `402` too ("hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription"). No Space exists. The steps below work once the account has PRO (or a grant); until then the app runs on Kaggle.

## Run it on a Kaggle T4 (works without PRO)

`scripts/serve_app.py` runs the same app on a Kaggle GPU kernel. Kaggle cannot read the private adapter repo, so the adapters arrive as signed kernel-output URLs through `--fetch <folder>/<file>=<url>` (target adapter in folder `seed44`, draft adapter in `draft`; both are checked against the sha256 in `configs/final.json`).

- `--selftest`: launch on localhost, drive the app through `gradio_client` (segmentation, screening with rationale, empty note, broken numbering, no criteria, recovery, report files, the Inference Lab in every mode with sdpa and tiled attention) and save everything to `results/app/selftest`.
- `--serve-seconds N`: also open a public `*.gradio.live` share link for N seconds. The link is printed as `SHARE URL` in the kernel log, which Kaggle shows only to the owner while the kernel runs; the kernel stops after N seconds (a Kaggle session is at most 12 hours). Start it shortly before a demo or the defense.

## Deploy to a ZeroGPU Space

1. Stage the files (no network, writes `build/space`): `python3 tools/deploy_space.py`
2. Create the Space (private until submission): `hf repos create AkifJK/trialmatch --type space --sdk gradio --private`
3. Set the hardware to ZeroGPU in the Space settings.
4. Add a Space secret `HF_TOKEN` with read access to the adapter repo (the repo is private until submission). Do not put the token in any file.
5. Upload: `hf upload AkifJK/trialmatch build/space . --type space`

Sizes and limits: ZeroGPU supports torch 2.11.0 and Python 3.12. Free visitors get about 2 GPU minutes per day (5 with a free account), so a screening run is capped at 12 criteria and the UI shows an estimate.

## Provenance

`tools/deploy_space.py` writes `trialmatch/app/build_info.json` with the git commit (marked `+dirty` if the tree had changes). Every report carries that commit, the model and adapter revisions, the adapter checksums, decode settings, device, dtype and package versions.
