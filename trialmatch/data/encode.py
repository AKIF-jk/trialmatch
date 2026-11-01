"""Tokenize prompt and target together, masking prompt tokens so loss covers only the assistant tokens."""
from trialmatch.data.prompt import row_prompt, row_target

IGNORE = -100
END_OF_TURN = "<|im_end|>"
MAX_LEN = 2048


class TruncationError(ValueError):
    pass


def encode_example(row, encode, order="evidence_first", max_len=MAX_LEN):
    """encode: callable str -> list[int]. Returns input_ids and labels of equal length.

    The target ends with <|im_end|> so the model learns to stop. Raises instead of truncating.
    """
    prompt_ids = encode(row_prompt(row, order))
    target_ids = encode(row_target(row, order) + END_OF_TURN)
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_len:
        raise TruncationError(f"annotation {row.annotation_id}: {len(input_ids)} tokens exceeds max_len {max_len}")
    return {"input_ids": input_ids, "labels": [IGNORE] * len(prompt_ids) + target_ids, "prompt_length": len(prompt_ids)}
