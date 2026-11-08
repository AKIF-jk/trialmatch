#!/usr/bin/env python3
"""Write configs/freeze.json for the current commit. Run it once, after the last code change and before the test split opens.

    python scripts/freeze.py --tag final-freeze

It refuses to run on a dirty tree. Then commit configs/freeze.json and create the git tag on that commit; the commit the tag
points at is the one to pass to scripts/final_eval.py as --expect-commit. It does not read the test split.
"""
import argparse
import json
import subprocess
from pathlib import Path

from trialmatch.eval.freeze import FREEZE_FILE, build_freeze

ROOT = Path(__file__).resolve().parent.parent


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if git("status", "--porcelain"):
        raise SystemExit("working tree is not clean: commit everything first, so the recorded code commit is the code that runs")
    record = build_freeze(ROOT, git("rev-parse", "HEAD"), args.tag)
    path = ROOT / FREEZE_FILE
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}: {len(record['files'])} frozen files, code commit {record['code_commit'][:10]}")
    print(f"next: git add {FREEZE_FILE} && git commit, then git tag {args.tag}")


if __name__ == "__main__":
    main()
