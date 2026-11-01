"""Patient-level split and its manifest.

Patient IDs are sorted, shuffled with np.random.default_rng(SEED), then assigned 37/8/8.
The committed manifest is the source of truth. Tests rebuild it and compare, so a changed
dataset revision or numpy stream shows up as a failure instead of a silent new split.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from trialmatch.runtime import DATASET, DATASET_REVISION

SEED = 42
SIZES = {"train": 37, "val": 8, "test": 8}
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "splits" / "manifest.json"


class LockedSplitError(RuntimeError):
    pass


def assign_patients(patient_ids):
    ids = sorted(set(patient_ids))
    if len(ids) != sum(SIZES.values()):
        raise ValueError(f"expected {sum(SIZES.values())} patients, found {len(ids)}")
    order = np.random.default_rng(SEED).permutation(len(ids))
    shuffled = [ids[i] for i in order]
    train, val = SIZES["train"], SIZES["val"]
    return {"train": sorted(shuffled[:train]), "val": sorted(shuffled[train:train + val]), "test": sorted(shuffled[train + val:])}


def _digest(manifest):
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def build_manifest(rows):
    patients = assign_patients(r.patient_id for r in rows)
    owner = {pid: name for name, pids in patients.items() for pid in pids}
    annotations = {name: [] for name in SIZES}
    for row in rows:
        annotations[owner[row.patient_id]].append(row.annotation_id)
    by_id = {r.annotation_id: r for r in rows}
    manifest = {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "seed": SEED,
        "method": "sorted patient ids, np.random.default_rng(seed).permutation, then 37/8/8",
        "patients": patients,
        "annotations": annotations,
        "unusable_annotation_ids": sorted(r.annotation_id for r in rows if not r.usable),
        "class_counts": {
            name: {
                kind: dict(sorted(Counter(by_id[a].label for a in ids if by_id[a].criterion_type == kind).items()))
                for kind in ("inclusion", "exclusion")
            }
            for name, ids in annotations.items()
        },
    }
    manifest["sha256"] = _digest(manifest)
    return manifest


def verify_manifest(manifest):
    if manifest["sha256"] != _digest(manifest):
        raise ValueError("manifest hash does not match its content")
    return manifest


def load_manifest(path=MANIFEST_PATH):
    return verify_manifest(json.loads(Path(path).read_text()))


def select_rows(rows, manifest, split, allow_test=False):
    """Return usable rows of one split. The test split needs allow_test=True, set only by scripts/final_eval.py."""
    if split not in SIZES:
        raise ValueError(f"unknown split {split!r}")
    if split == "test" and not allow_test:
        raise LockedSplitError("The test split loads only from scripts/final_eval.py, with allow_test=True.")
    wanted = set(manifest["annotations"][split])
    return [r for r in rows if r.annotation_id in wanted and r.usable]
