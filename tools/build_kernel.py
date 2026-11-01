#!/usr/bin/env python3
"""Build Kaggle kernel scripts from the committed source.

Source reaches Kaggle through a private "source kernel", so nothing is uploaded
to a public host. The source kernel writes the package to its output, and job
kernels attach it through kernelDataSources and import from there.

    python3 tools/build_kernel.py src
        Write build/src_kernel.py, and build/src_manifest.json holding the
        SHA-256 of every file. Compare the manifest with the kernel's output
        to confirm the text sent to Kaggle matches the commit.

    python3 tools/build_kernel.py job scripts/smoke.py [job args]
        Write build/<job>_kernel.py, which runs one script from the source kernel.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
SOURCE_PATHS = ["trialmatch", "scripts", "configs", "tests", "splits"]
SOURCE_KERNEL = "akifjawad/trialmatch-src"

SRC_TEMPLATE = """\
import hashlib, json, os

COMMIT = {commit!r}
FILES = {files}

manifest = {{}}
for path, text in FILES.items():
    target = os.path.join("/kaggle/working/src", path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", newline="") as handle:
        handle.write(text)
    manifest[path] = hashlib.sha256(text.encode()).hexdigest()
with open("/kaggle/working/src_manifest.json", "w") as handle:
    json.dump({{"commit": COMMIT, "files": manifest}}, handle, indent=2, sort_keys=True)
print("wrote", len(manifest), "files for", COMMIT)
"""

JOB_TEMPLATE = """\
import glob, json, os, shutil, subprocess, sys

JOB, ARGS, EXPECTED_COMMIT = {job!r}, {args!r}, {commit!r}

manifests = glob.glob("/kaggle/input/**/src_manifest.json", recursive=True)
assert len(manifests) == 1, f"expected one attached source kernel, found {{manifests}}"
source = os.path.join(os.path.dirname(manifests[0]), "src")
commit = json.load(open(manifests[0]))["commit"]
assert commit == EXPECTED_COMMIT, f"source kernel is at {{commit}}, job expects {{EXPECTED_COMMIT}}"
shutil.copytree(source, "/tmp/src", dirs_exist_ok=True)

env = dict(os.environ, PYTHONPATH="/tmp/src", PYTHONUNBUFFERED="1", TRIALMATCH_COMMIT=commit, TRIALMATCH_RESULTS="/kaggle/working/results", CUDA_VISIBLE_DEVICES="0")
sys.exit(subprocess.run([sys.executable, JOB, *ARGS], cwd="/tmp/src", env=env).returncode)
"""


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def committed_files():
    """Return tracked source files from HEAD, keyed by repo path."""
    if git("status", "--porcelain", "--", *SOURCE_PATHS):
        raise SystemExit("Source has uncommitted changes. Commit first, so the kernel maps to one commit.")
    paths = [p for p in SOURCE_PATHS if (ROOT / p).exists()]
    return {path: git("show", f"HEAD:{path}") + "\n" for path in git("ls-files", *paths).splitlines()}


def render_files(files):
    """Render files as a readable dict literal, so the kernel text can be reviewed."""
    return "{\n" + "".join(f"{path!r}: {text!r},\n" for path, text in files.items()) + "}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["src", "job"])
    parser.add_argument("job", nargs="?")
    parser.add_argument("args", nargs="*")
    opts = parser.parse_args()

    BUILD.mkdir(exist_ok=True)
    commit = git("rev-parse", "HEAD")
    if opts.kind == "src":
        files = committed_files()
        (BUILD / "src_kernel.py").write_text(SRC_TEMPLATE.format(commit=commit, files=render_files(files)))
        manifest = {path: hashlib.sha256(text.encode()).hexdigest() for path, text in files.items()}
        (BUILD / "src_manifest.json").write_text(json.dumps({"commit": commit, "files": manifest}, indent=2, sort_keys=True))
        out = BUILD / "src_kernel.py"
    else:
        if not opts.job:
            parser.error("job needs a script path")
        out = BUILD / f"{Path(opts.job).stem}_kernel.py"
        out.write_text(JOB_TEMPLATE.format(job=opts.job, args=opts.args, commit=commit))
    print(f"commit {commit}\nwrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
