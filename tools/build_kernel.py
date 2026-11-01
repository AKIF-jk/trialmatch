#!/usr/bin/env python3
"""Build Kaggle kernel scripts from the committed source.

Source reaches Kaggle through a private "source kernel", so nothing is uploaded
to a public host. The source kernel writes a tarball to its output. Job kernels
download it from the signed output URL and verify a hash of the file tree.
kernelDataSources is not used because the MCP server drops it without an error.

    python3 tools/build_kernel.py src
        Write build/src_kernel.py to send with save_notebook.

    python3 tools/build_kernel.py job --source-url URL scripts/smoke.py [job args]
        Write build/<job>_kernel.py, where URL is the src.tar.gz output URL from
        list_notebook_session_output on the source kernel.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
SOURCE_PATHS = ["trialmatch", "scripts", "configs", "tests", "splits"]

SRC_TEMPLATE = """\
import io, tarfile

COMMIT = {commit!r}
FILES = {files}

with tarfile.open("/kaggle/working/src.tar.gz", "w:gz") as archive:
    for path, text in FILES.items():
        data = text.encode()
        info = tarfile.TarInfo(path)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
print("packed", len(FILES), "files for", COMMIT)
"""

JOB_TEMPLATE = """\
import hashlib, io, json, os, subprocess, sys, tarfile, urllib.request

JOB, ARGS, COMMIT, TREE_HASH = {job!r}, {args!r}, {commit!r}, {tree_hash!r}
URL = {url!r}

archive = tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(URL, timeout=120).read()), mode="r:gz")
archive.extractall("/tmp/src", filter="data")
tree = {{m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers()}}
found = hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()
assert found == TREE_HASH, f"source tree hash {{found}} does not match {{TREE_HASH}}"

env = dict(os.environ, PYTHONPATH="/tmp/src", PYTHONUNBUFFERED="1", TRIALMATCH_COMMIT=COMMIT, TRIALMATCH_RESULTS="/kaggle/working/results", CUDA_VISIBLE_DEVICES="0")
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


def tree_hash(files):
    tree = {path: hashlib.sha256(text.encode()).hexdigest() for path, text in files.items()}
    return hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()


def render_files(files):
    """Render files as a readable dict literal, so the kernel text can be reviewed."""
    return "{\n" + "".join(f"{path!r}: {text!r},\n" for path, text in files.items()) + "}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["src", "job"])
    parser.add_argument("job", nargs="?")
    parser.add_argument("args", nargs="*")
    parser.add_argument("--source-url")
    opts = parser.parse_args()

    BUILD.mkdir(exist_ok=True)
    commit = git("rev-parse", "HEAD")
    files = committed_files()
    if opts.kind == "src":
        out = BUILD / "src_kernel.py"
        out.write_text(SRC_TEMPLATE.format(commit=commit, files=render_files(files)))
    else:
        if not (opts.job and opts.source_url):
            parser.error("job needs a script path and --source-url")
        out = BUILD / f"{Path(opts.job).stem}_kernel.py"
        out.write_text(JOB_TEMPLATE.format(job=opts.job, args=opts.args, commit=commit, tree_hash=tree_hash(files), url=opts.source_url))
    print(f"commit {commit}\ntree {tree_hash(files)}\nwrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
