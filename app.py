"""Space entry point: load the models once at import time, run every model call inside a ZeroGPU window, serve the Gradio app."""
import os

from trialmatch.app.engine import Engine
from trialmatch.app.ui import build_ui

try:
    import spaces
except ImportError:  # Running locally or on a plain GPU host: no GPU window to request.
    spaces = None

ENGINE = Engine.load()


def _gpu(duration):
    return spaces.GPU(duration=duration) if spaces else (lambda fn: fn)


def screen_seconds(sentences, criteria, rationale):
    return min(120, 20 + 4 * len(criteria) + (20 if rationale else 0))


@_gpu(screen_seconds)
def screen_fn(sentences, criteria, rationale):
    results, timing = ENGINE.screen(sentences, criteria, rationale)
    return results, timing, ENGINE.provenance()


def lab_seconds(sentences, ctype, criterion, modes, backend, tile, k, draft, new_tokens, force, repeats):
    slow = 4 if backend == "tiled" else 1
    return min(120, 25 + int(repeats) * len(modes) * (4 + slow * int(new_tokens) // 8))


@_gpu(lab_seconds)
def lab_fn(sentences, ctype, criterion, modes, backend, tile, k, draft, new_tokens, force, repeats):
    rows, text = ENGINE.lab(sentences, ctype, criterion, modes, backend, tile, k, draft, new_tokens, force, repeats)
    return rows, text, ENGINE.provenance()


demo = build_ui(screen_fn, lab_fn)

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=bool(os.environ.get("TRIALMATCH_SHARE")))
