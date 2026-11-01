#!/usr/bin/env python3
"""Build Kaggle kernel scripts from the committed source.

Source reaches Kaggle through a private "source kernel", so nothing is uploaded
to a public host. The source kernel writes a tarball to its output. Job kernels
download it from the signed output URL and verify a hash of the file tree.
kernelDataSources is not used because the MCP server drops it without an error.

    python3 tools/build_kernel.py src
        Write build/src_kernel.py to send with save_notebook. It inlines every file.

    python3 tools/build_kernel.py src --base-url URL --base-commit SHA
        Delta build. URL is the src.tar.gz output URL of the previous source kernel
        version and SHA is the commit that version was built from. Only files that
        differ from SHA are inlined. The kernel checks the base tree hash before
        applying the delta and the new tree hash after.

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

DELTA_TEMPLATE = """\
import hashlib, io, json, tarfile, urllib.request

COMMIT, BASE_COMMIT = {commit!r}, {base_commit!r}
BASE_TREE_HASH, TREE_HASH = {base_tree_hash!r}, {tree_hash!r}
URL = {url!r}
CHANGED = {files}
DELETED = {deleted!r}


def tree_hash(files):
    tree = {{path: hashlib.sha256(data).hexdigest() for path, data in files.items()}}
    return hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()


with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(URL, timeout=120).read()), mode="r:gz") as base:
    files = {{m.name: base.extractfile(m).read() for m in base.getmembers()}}
found = tree_hash(files)
assert found == BASE_TREE_HASH, f"base tree hash {{found}} does not match {{BASE_TREE_HASH}} ({{BASE_COMMIT}})"
for path in DELETED:
    del files[path]
files.update({{path: text.encode() for path, text in CHANGED.items()}})
found = tree_hash(files)
assert found == TREE_HASH, f"new tree hash {{found}} does not match {{TREE_HASH}}"

with tarfile.open("/kaggle/working/src.tar.gz", "w:gz") as archive:
    for path, data in files.items():
        info = tarfile.TarInfo(path)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
print("packed", len(files), "files for", COMMIT, "from", BASE_COMMIT, "changed", len(CHANGED), "deleted", len(DELETED))
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


def files_at(rev):
    """Return tracked source files at a revision, keyed by repo path."""
    paths = [p for p in SOURCE_PATHS if git("ls-tree", "--name-only", rev, p)]
    return {path: git("show", f"{rev}:{path}") + "\n" for path in git("ls-tree", "-r", "--name-only", rev, *paths).splitlines()}


def committed_files():
    """Return tracked source files from HEAD, keyed by repo path."""
    if git("status", "--porcelain", "--", *SOURCE_PATHS):
        raise SystemExit("Source has uncommitted changes. Commit first, so the kernel maps to one commit.")
    return files_at("HEAD")


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
    parser.add_argument("--base-url")
    parser.add_argument("--base-commit")
    opts = parser.parse_args()

    BUILD.mkdir(exist_ok=True)
    commit = git("rev-parse", "HEAD")
    files = committed_files()
    if opts.kind == "src" and (opts.base_url or opts.base_commit):
        if not (opts.base_url and opts.base_commit):
            parser.error("a delta build needs both --base-url and --base-commit")
        base = files_at(opts.base_commit)
        changed = {path: text for path, text in files.items() if base.get(path) != text}
        deleted = sorted(set(base) - set(files))
        out = BUILD / "src_kernel.py"
        out.write_text(DELTA_TEMPLATE.format(commit=commit, base_commit=git("rev-parse", opts.base_commit), base_tree_hash=tree_hash(base), tree_hash=tree_hash(files), url=opts.base_url, files=render_files(changed), deleted=deleted))
        print(f"delta against {opts.base_commit}: {len(changed)} changed, {len(deleted)} deleted")
    elif opts.kind == "src":
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
