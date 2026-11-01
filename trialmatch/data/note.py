"""Numbered-sentence notes: "0. first\n1. second". The app must segment unseen notes into this format."""
import re

_BREAK = re.compile(r"\n(?=\d+\. )")


def parse_note(note):
    """Split a numbered note into sentences without their numbers. Fails if numbering is not 0..n-1."""
    parts = _BREAK.split(note)
    sentences = []
    for index, part in enumerate(parts):
        prefix = f"{index}. "
        if not part.startswith(prefix):
            raise ValueError(f"sentence {index} does not start with {prefix!r}: {part[:30]!r}")
        sentences.append(part[len(prefix):])
    return sentences


def format_note(sentences):
    return "\n".join(f"{index}. {sentence}" for index, sentence in enumerate(sentences))
