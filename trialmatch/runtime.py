"""Shared run provenance: pinned revisions, results paths and environment capture."""
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DRAFT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATASET = "ncbi/TrialGPT-Criterion-Annotations"


def results_dir(job):
    path = Path(os.environ.get("TRIALMATCH_RESULTS", "results")) / job
    path.mkdir(parents=True, exist_ok=True)
    return path


def package_versions(names=("torch", "transformers", "peft", "accelerate", "datasets", "gradio", "numpy", "scikit-learn")):
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def check_versions():
    """Fail fast if the Kaggle image no longer matches requirements-kaggle.txt."""
    expected = {
        "torch": "2.11.0", "transformers": "5.16.1", "peft": "0.20.0", "accelerate": "1.14.0",
        "datasets": "4.8.5", "numpy": "2.1.3", "scikit-learn": "1.6.1",
    }
    found = package_versions(expected)
    wrong = {name: found[name] for name, version in expected.items() if (found[name] or "").split("+")[0] != version}
    if wrong:
        raise RuntimeError(f"Package versions differ from the lock: {wrong}")


def environment():
    info = {
        "commit": os.environ.get("TRIALMATCH_COMMIT"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
    }
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpus"] = [
                {"name": torch.cuda.get_device_name(i), "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                for i in range(torch.cuda.device_count())
            ]
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        info["cuda_available"] = None
    try:
        query = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        info["nvidia_smi"] = subprocess.run(query, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info["nvidia_smi"] = None
    return info


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n")
