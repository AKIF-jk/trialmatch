#!/usr/bin/env python3
"""Package the committed source and build a Kaggle notebook that runs one job.

The source tarball goes to a secret gist, because notebook text passes through
an MCP call and must stay small. The notebook verifies the tarball's SHA-256
before extracting it, so each run is tied to one exact commit.

Usage:
    python3 tools/build_kernel.py scripts/smoke.py [--requirements requirements/kaggle.txt] [-- job args]

The notebook is written to build/<job>.ipynb.
"""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
GIST_ID_FILE = ROOT / ".gist_id"
GIST_FILE = "src.tar.gz.b64"
SOURCE_PATHS = ["trialmatch", "scripts", "configs", "tests", "splits", "requirements"]

RUNNER = """\
import base64, hashlib, io, os, subprocess, sys, tarfile, urllib.request

JOB, COMMIT, URL, SHA256 = {job!r}, {commit!r}, {url!r}, {sha256!r}
REQUIREMENTS, ARGS = {requirements!r}, {args!r}
SRC, OUT = "/tmp/src", "/kaggle/working"

blob = urllib.request.urlopen(URL, timeout=120).read()
assert hashlib.sha256(blob).hexdigest() == SHA256, "source tarball hash mismatch"
os.makedirs(SRC, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(blob)), mode="r:gz").extractall(SRC, filter="data")

os.makedirs(f"{{OUT}}/logs", exist_ok=True)
log_path = f"{{OUT}}/logs/{{os.path.basename(JOB)}}.log"
env = dict(os.environ, PYTHONPATH=SRC, PYTHONUNBUFFERED="1", TRIALMATCH_COMMIT=COMMIT, TRIALMATCH_RESULTS=f"{{OUT}}/results")

with open(log_path, "w") as log:
    def run(cmd):
        proc = subprocess.Popen(cmd, cwd=SRC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return proc.wait()

    code = 0
    if REQUIREMENTS:
        code = run([sys.executable, "-m", "pip", "install", "-q", "-r", REQUIREMENTS])
    if code == 0:
        code = run([sys.executable, JOB, *ARGS])

assert code == 0, f"job failed with exit code {{code}}; see {{log_path}}"
"""


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def publish(payload_path):
    """Upload the tarball to the secret gist and return its pinned raw URL."""
    if GIST_ID_FILE.exists():
        gist_id = GIST_ID_FILE.read_text().strip()
        subprocess.run(["gh", "gist", "edit", gist_id, "--filename", GIST_FILE, str(payload_path)], cwd=ROOT, check=True)
    else:
        url = subprocess.run(
            ["gh", "gist", "create", "--desc", "trialmatch kernel source", str(payload_path)],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
        gist_id = url.rsplit("/", 1)[-1]
        GIST_ID_FILE.write_text(gist_id)
    meta = json.loads(subprocess.run(["gh", "api", f"gists/{gist_id}"], check=True, capture_output=True, text=True).stdout)
    return f"https://gist.githubusercontent.com/{meta['owner']['login']}/{gist_id}/raw/{meta['history'][0]['version']}/{GIST_FILE}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("--requirements", default="")
    parser.add_argument("args", nargs="*")
    opts = parser.parse_args()

    if git("status", "--porcelain"):
        raise SystemExit("Working tree is dirty. Commit first, so the kernel maps to one commit.")
    commit = git("rev-parse", "HEAD")

    paths = [p for p in SOURCE_PATHS if (ROOT / p).exists() and git("ls-files", p)]
    tarball = subprocess.run(["git", "archive", "--format=tar.gz", "HEAD", "--", *paths], cwd=ROOT, check=True, capture_output=True).stdout
    payload = base64.b64encode(tarball)

    BUILD.mkdir(exist_ok=True)
    payload_path = BUILD / GIST_FILE
    payload_path.write_bytes(payload)
    url = publish(payload_path)

    source = RUNNER.format(
        job=opts.job, commit=commit, url=url, sha256=hashlib.sha256(payload).hexdigest(),
        requirements=opts.requirements, args=opts.args,
    )
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = BUILD / f"{Path(opts.job).stem}.ipynb"
    out.write_text(json.dumps(notebook))
    print(f"commit {commit}\nnotebook {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
