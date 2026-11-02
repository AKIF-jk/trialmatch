# Deploying the app

The app is `app.py` plus `trialmatch/app/`. It loads Qwen2.5-1.5B-Instruct with the seed-44 LoRA merged in, the 0.5B draft (base and LoRA) and the untuned 1.5B model (for optional rationales) at import time, then serves a Gradio UI with two tabs: Screening and Inference Lab.

## Run it locally (needs a GPU or patience on CPU)

```bash
pip install -r requirements-kaggle.txt gradio==5.50.0 huggingface_hub pyarrow
export HF_TOKEN=...   # read access to the private adapter repo AkifJK/trialmatch-qwen2.5-1.5b-lora
python app.py          # http://localhost:7860
```

On CPU the models load in float32 (about 12 GB of RAM for all four) and one criterion takes tens of seconds.

## Deploy to a ZeroGPU Space

1. Stage the files (no network, writes `build/space`): `python3 tools/deploy_space.py`
2. Create the Space (private until submission): `hf repos create AkifJK/trialmatch --type space --sdk gradio --private`
3. Set the hardware to ZeroGPU in the Space settings.
4. Add a Space secret `HF_TOKEN` with read access to the adapter repo (the repo is private until submission). Do not put the token in any file.
5. Upload: `hf upload AkifJK/trialmatch build/space . --type space`

Sizes and limits: ZeroGPU supports torch 2.11.0 and Python 3.12. Free visitors get about 2 GPU minutes per day (5 with a free account), so a screening run is capped at 12 criteria and the UI shows an estimate.

## Provenance

`tools/deploy_space.py` writes `trialmatch/app/build_info.json` with the git commit (marked `+dirty` if the tree had changes). Every report carries that commit, the model and adapter revisions, the adapter checksums, decode settings, device, dtype and package versions.
