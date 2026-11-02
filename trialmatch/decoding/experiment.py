"""Inputs shared by the decoding experiments: pinned model loading, exact-length prompts and left padding."""
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trialmatch.data.load import load_rows
from trialmatch.data.prompt import build_user, chat_prompt
from trialmatch.data.split import load_manifest, select_rows
from trialmatch.runtime import DRAFT_MODEL, DRAFT_REVISION, TARGET_MODEL, TARGET_REVISION

DTYPES = {"float32": torch.float32, "float16": torch.float16}
STARTS = (0, 1500)  # Offsets into the token pool. The first prompt begins with the chat header, the second inside a note.


def load_tokenizer():
    return AutoTokenizer.from_pretrained(TARGET_MODEL, revision=TARGET_REVISION, padding_side="left")


def load_model(dtype, device):
    return AutoModelForCausalLM.from_pretrained(TARGET_MODEL, revision=TARGET_REVISION, dtype=DTYPES[dtype]).to(device).eval()


def validation_prompts():
    return [chat_prompt(build_user(r.note, r.criterion_type, r.criterion_text)) for r in select_rows(load_rows(), load_manifest(), "val")]


def token_pool(tokenizer, prompts, size=4096):
    """Validation prompt tokens joined end to end, so prompts longer than any single row (623 tokens) are real text."""
    pool = []
    for prompt in prompts:
        pool += tokenizer(prompt).input_ids
        if len(pool) >= size:
            return torch.tensor(pool[:size])
    raise ValueError(f"only {len(pool)} tokens available")


def left_pad(sequences, pad_id, device):
    """Left-padded input_ids and attention_mask from 1-D token tensors of any length."""
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, s in enumerate(sequences):
        ids[row, width - len(s):] = s
        mask[row, width - len(s):] = 1
    return ids.to(device), mask.to(device)


def load_model_for(role, dtype, device, adapter=None):
    """Pinned base model for "target" or "draft". With an adapter directory, the LoRA is merged into the weights."""
    name, revision = (TARGET_MODEL, TARGET_REVISION) if role == "target" else (DRAFT_MODEL, DRAFT_REVISION)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=DTYPES[dtype])
    if adapter:
        # Kaggle ships torchao 0.10.0 and peft 0.20.0 refuses it. We use no torchao, so hide it, as scripts/train.py does.
        sys.modules.setdefault("torchao", None)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    return model.to(device).eval()


def fetch_files(specs, root):
    """Download "<folder>/<file>=<url>" specs under root and return the folders. Signed output URLs carry no tokens we store."""
    folders = set()
    for spec in specs:
        path, url = spec.split("=", 1)
        target = Path(root) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(urllib.request.urlopen(url, timeout=300).read())
        folders.add(target.parent)
    return folders


def validation_rows():
    return select_rows(load_rows(), load_manifest(), "val")


class PeakRSS:
    """Peak resident memory of this process during a with block."""

    def __enter__(self):
        self.peak, self.running = 0, True
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        page = os.sysconf("SC_PAGE_SIZE")
        while self.running:
            with open("/proc/self/statm") as f:
                self.peak = max(self.peak, int(f.read().split()[1]) * page)
            time.sleep(0.01)

    def __exit__(self, *exc):
        self.running = False
        self.thread.join()
