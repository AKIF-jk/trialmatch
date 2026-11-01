"""Load the annotation table at the pinned revision, keeping only the fields the task may use.

The GPT-4 columns, `explanation_correctness`, `training` and `trial_title` are dropped here,
so nothing downstream can put them in a prompt or a target.
"""
import json
from dataclasses import dataclass

from trialmatch.runtime import DATASET, DATASET_FILE, DATASET_REVISION

LABELS = {
    "inclusion": ("included", "not included", "not enough information", "not applicable"),
    "exclusion": ("excluded", "not excluded", "not enough information", "not applicable"),
}


@dataclass(frozen=True)
class Row:
    annotation_id: int
    patient_id: str
    trial_id: str
    note: str
    criterion_type: str
    criterion_text: str | None
    evidence_ids: tuple[int, ...]
    label: str

    @property
    def usable(self):
        """False for rows with no criterion text. They stay in the manifest but not in training or metrics."""
        return bool(self.criterion_text and self.criterion_text.strip())


def parse_row(record):
    row = Row(
        annotation_id=int(record["annotation_id"]),
        patient_id=record["patient_id"],
        trial_id=record["trial_id"],
        note=record["note"],
        criterion_type=record["criterion_type"],
        criterion_text=record["criterion_text"],
        evidence_ids=tuple(json.loads(record["expert_sentences"])),
        label=record["expert_eligibility"],
    )
    if row.label not in LABELS[row.criterion_type]:
        raise ValueError(f"annotation {row.annotation_id}: label {row.label!r} is not valid for {row.criterion_type}")
    return row


def dataset_path():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset", revision=DATASET_REVISION)


def load_rows(path=None):
    """Return every annotation as a Row, sorted by annotation_id. Split filtering happens in split.py."""
    import pyarrow.parquet as pq

    records = pq.read_table(path or dataset_path()).to_pylist()
    return sorted((parse_row(r) for r in records), key=lambda r: r.annotation_id)
