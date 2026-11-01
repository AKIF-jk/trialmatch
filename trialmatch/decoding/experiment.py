"""Inputs shared by the decoding experiments: pinned model loading, exact-length prompts and left padding."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trialmatch.data.load import load_rows
from trialmatch.data.prompt import build_user, chat_prompt
from trialmatch.data.split import load_manifest, select_rows
from trialmatch.runtime import TARGET_MODEL, TARGET_REVISION

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
