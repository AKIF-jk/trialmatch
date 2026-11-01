#!/usr/bin/env python3
"""Write splits/manifest.json from the pinned dataset revision. Re-running must give the same file."""
import json

from trialmatch.data.load import load_rows
from trialmatch.data.split import MANIFEST_PATH, build_manifest

if __name__ == "__main__":
    manifest = build_manifest(load_rows())
    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    # One line per key keeps the file small, since every committed byte is retyped into the source kernel.
    lines = [f" {json.dumps(key)}: {json.dumps(manifest[key], sort_keys=True)}" for key in sorted(manifest)]
    MANIFEST_PATH.write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print("wrote", MANIFEST_PATH, manifest["sha256"])
    print(json.dumps({k: {t: sum(c.values()) for t, c in v.items()} for k, v in manifest["class_counts"].items()}))
