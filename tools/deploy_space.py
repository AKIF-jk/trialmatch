#!/usr/bin/env python3
"""Stage the Space files under build/space (no network). Upload afterwards with the hf CLI, see docs/DEPLOY.md.

The Space holds app.py, the trialmatch package, configs/final.json, requirements.txt and a README with the Space metadata.
It holds no weights: the adapters are fetched at start-up from the pinned revision in configs/final.json.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = """---
title: TrialMatch
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Clinical trial criterion screening with live decoding comparison
---

Educational screening prototype, not a clinical enrollment decision.

TrialMatch labels a patient note against each trial criterion with Qwen2.5-1.5B-Instruct plus a LoRA adapter, cites the
supporting sentence ids, and compares no-cache, KV-cache and speculative decoding live. Code: see the project repository.
"""
REQUIREMENTS = """torch==2.11.0
transformers==5.16.1
peft==0.20.0
accelerate==1.14.0
huggingface_hub
numpy==2.1.3
pyarrow
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "build" / "space"))
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "trialmatch", out / "trialmatch", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "final.json", out / "configs" / "final.json")
    shutil.copy(ROOT / "app.py", out / "app.py")
    (out / "README.md").write_text(README)
    (out / "requirements.txt").write_text(REQUIREMENTS)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "unknown"
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    (out / "trialmatch" / "app" / "build_info.json").write_text(json.dumps({"commit": commit + ("+dirty" if dirty else "")}) + "\n")
    print(f"staged {sum(1 for p in out.rglob('*') if p.is_file())} files in {out} at {commit[:8]}{'+dirty' if dirty else ''}")


if __name__ == "__main__":
    main()
