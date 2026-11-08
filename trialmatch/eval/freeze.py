"""Freeze record and the gate in front of the test split.

configs/freeze.json holds the hash of every file the final evaluation depends on (prompt, parser, metrics, split
manifest, frozen config, the evaluation script itself) plus the adapter checksums and model revisions.
scripts/final_eval.py checks it before the test split opens, so a code change after the freeze stops the run
instead of going unnoticed. A rerun is allowed only with a stated reason, which the results then carry.
"""
import hashlib
import json
from pathlib import Path

FREEZE_FILE = "configs/freeze.json"
# Kaggle jobs see only the committed trialmatch/, scripts/, configs/, tests/ and splits/ trees (tools/build_kernel.py),
# so every frozen file and the run log must live in one of them. A test enforces this.
RUN_LOG = "configs/test_runs.json"
FROZEN = (
    "trialmatch/data/*.py",
    "trialmatch/eval/*.py",
    "trialmatch/runtime.py",
    "scripts/final_eval.py",
    "configs/final.json",
    "splits/manifest.json",
)


class FreezeError(RuntimeError):
    pass


def file_hashes(root, patterns=FROZEN):
    """sha256 per frozen file, keyed by path relative to root.

    Each file is hashed in the form tools/build_kernel.py stores it in the Kaggle tarball (text stripped, one trailing
    newline), so the record made on the laptop equals what the job computes. An empty __init__.py is the case that differs.
    """
    root = Path(root)
    found = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                found[path.relative_to(root).as_posix()] = hashlib.sha256((path.read_text().strip() + "\n").encode()).hexdigest()
    return dict(sorted(found.items()))


def build_freeze(root, code_commit, tag):
    root = Path(root)
    config = json.loads((root / "configs/final.json").read_text())
    manifest = json.loads((root / "splits/manifest.json").read_text())
    return {
        "tag": tag,
        "code_commit": code_commit,
        "manifest_sha256": manifest["sha256"],
        "target": {"model": config["model"], "revision": config["model_revision"]},
        "draft": config["draft"],
        "adapters": config["adapters"],
        "files": file_hashes(root),
    }


def verify_freeze(root):
    """Return the freeze record, or raise FreezeError if it is missing or any frozen file changed since."""
    path = Path(root) / FREEZE_FILE
    if not path.exists():
        raise FreezeError(f"{FREEZE_FILE} is missing: run scripts/freeze.py and tag the commit before opening the test split")
    record = json.loads(path.read_text())
    current = file_hashes(root)
    changed = sorted(name for name in set(record["files"]) | set(current) if record["files"].get(name) != current.get(name))
    if changed:
        raise FreezeError(f"frozen files differ from {FREEZE_FILE}: {changed}")
    return record


def previous_runs(root):
    path = Path(root) / RUN_LOG
    return json.loads(path.read_text())["runs"] if path.exists() else []


def check_rerun(root, reason):
    """Earlier test runs are on record in results/final/test/run.json. Another run needs a reason, disclosed in the report."""
    runs = previous_runs(root)
    if runs and not (reason and reason.strip()):
        raise FreezeError(f"{len(runs)} earlier test run(s) are on record in {RUN_LOG}; a rerun needs --rerun-reason, and the report must disclose it")
    return runs


def open_test_gate(root, expect_commit, commit, rerun_reason=None):
    """Everything that must hold before the test split loads. Returns (freeze record, earlier runs)."""
    record = verify_freeze(root)
    if not expect_commit or len(expect_commit) < 7:
        raise FreezeError("--expect-commit needs the commit SHA of the freeze tag (at least 7 characters)")
    if not commit or not (commit.startswith(expect_commit) or expect_commit.startswith(commit)):
        raise FreezeError(f"running commit {commit!r} is not the expected {expect_commit!r}")
    return record, check_rerun(root, rerun_reason)
