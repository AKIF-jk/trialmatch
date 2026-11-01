#!/usr/bin/env python3
"""Write splits/manifest.json from the pinned dataset revision. Re-running must give the same file."""
import json

from trialmatch.data.load import load_rows
from trialmatch.data.split import MANIFEST_PATH, build_manifest

if __name__ == "__main__":
    manifest = build_manifest(load_rows())
    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print("wrote", MANIFEST_PATH, manifest["sha256"])
    print(json.dumps({k: {t: sum(c.values()) for t, c in v.items()} for k, v in manifest["class_counts"].items()}))
