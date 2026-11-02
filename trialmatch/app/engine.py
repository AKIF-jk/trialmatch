"""Model side of the app: criterion screening with the cached decoder, and the live inference comparison.

The Screening tab runs the fine-tuned target (base model with the seed-44 LoRA merged in) through decode_cached, one
batch of left-padded prompts per call. The Inference Lab runs one prompt through the uncached, cached and speculative
decoders with a chosen attention backend and compares tokens against the cached sdpa run. Everything is measured live.
"""
import datetime
import json
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from trialmatch.app import screen
from trialmatch.app.screen import LAB_MODES as MODES, TILES  # noqa: F401
from trialmatch.data.note import format_note
from trialmatch.data.prompt import build_user, chat_prompt
from trialmatch.decoding.greedy import decode_cached, decode_uncached
from trialmatch.decoding.prompt_lookup import PromptLookupDraft
from trialmatch.decoding.speculative import ModelDraft, decode_speculative
from trialmatch.eval.generate import GREEDY
from trialmatch.eval.parse import parse_output
from trialmatch.runtime import DRAFT_MODEL, DRAFT_REVISION, TARGET_MODEL, TARGET_REVISION, package_versions

ROOT = Path(__file__).resolve().parents[2]
MAX_NEW_TOKENS = 64
LAB_MAX_NEW_TOKENS = 64
RATIONALE_TOKENS = 80
BATCH_SIZE = 8


def final_config():
    return json.loads((ROOT / "configs" / "final.json").read_text())


def app_commit():
    """Commit of the deployed code: env var, else build_info.json written by tools/deploy_space.py, else unknown."""
    if os.environ.get("TRIALMATCH_COMMIT"):
        return os.environ["TRIALMATCH_COMMIT"]
    info = Path(__file__).with_name("build_info.json")
    return json.loads(info.read_text()).get("commit", "unknown") if info.exists() else "unknown"


@dataclass
class Adapters:
    """Where the adapters came from, recorded in every report."""

    repo: str
    revision: str
    target_folder: str
    draft_folder: str
    sha256: dict


def fetch_adapter(repo, revision, folder, expected_sha256=None):
    """Download one adapter folder at a pinned revision and check the weights against the recorded checksum."""
    import hashlib

    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo, revision=revision, allow_patterns=[f"{folder}/*"]))
    path = root / folder
    if expected_sha256:
        digest = hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise RuntimeError(f"adapter {folder} checksum {digest[:12]} differs from the recorded {expected_sha256[:12]}")
    return path


def local_adapter(root, folder, expected_sha256):
    import hashlib

    path = Path(root) / folder
    digest = hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"adapter {folder} checksum {digest[:12]} differs from the recorded {expected_sha256[:12]}")
    return path


@contextmanager
def attention_backend(models, backend, tile):
    """Run with sdpa or the tiled online-softmax backend on every model, then restore each model's own setting."""
    from trialmatch.attention.hf import use_tiled_attention

    if backend == "sdpa":
        yield
        return
    previous = [use_tiled_attention(m, tile) for m in models]
    try:
        yield
    finally:
        for model, name in zip(models, previous):
            model.set_attn_implementation(name)


def _median(values):
    return statistics.median(values) if values else None


class Engine:
    """Holds the tokenizer and models. Build the real one with Engine.load(). Tests pass small models directly."""

    def __init__(self, tokenizer, target, drafts, device, dtype_name, adapters=None, base_loader=None):
        self.tokenizer, self.target, self.drafts = tokenizer, target, drafts
        self.device, self.dtype_name, self.adapters = torch.device(device), dtype_name, adapters
        self.pad_id = tokenizer.pad_token_id
        self.stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        self._base_loader, self._base = base_loader, None
        for model in (target, *drafts.values()):
            model.generation_config.eos_token_id = None

    @classmethod
    def load(cls, adapter_root=None):
        """Build the real engine. With adapter_root, the adapters come from <root>/<folder> (checked against the recorded
        sha256) instead of the private Hub repo, for hosts that cannot read it, such as Kaggle."""
        from trialmatch.decoding.experiment import load_model_for, load_tokenizer

        config = final_config()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = "float16" if device == "cuda" else "float32"
        hub = config["adapters"]
        folders, sha = hub["folders"], hub["sha256"]
        deployed = config["deployed_adapter"].split()[0]
        target_folder = next(folders[seed] for seed, run in config["runs"].items() if run == deployed)
        if adapter_root:
            target_path, draft_path = local_adapter(adapter_root, target_folder, sha[target_folder]), local_adapter(adapter_root, folders["draft"], sha["draft"])
        else:
            target_path = fetch_adapter(hub["hf_repo"], hub["hf_revision"], target_folder, sha[target_folder])
            draft_path = fetch_adapter(hub["hf_repo"], hub["hf_revision"], folders["draft"], sha["draft"])
        target = load_model_for("target", dtype, device, target_path)
        drafts = {"base": load_model_for("draft", dtype, device), "LoRA": load_model_for("draft", dtype, device, draft_path)}
        adapters = Adapters(hub["hf_repo"], hub["hf_revision"], target_folder, folders["draft"], {target_folder: sha[target_folder], "draft": sha["draft"]})
        engine = cls(load_tokenizer(), target, drafts, device, dtype, adapters, base_loader=lambda: load_model_for("target", dtype, device))
        if device == "cuda":
            # ZeroGPU wants every model placed on the device at import time, so the untuned model for rationales loads now too.
            engine.base_model()
        return engine  # On CPU it loads on first use: all four models in float32 would not fit in 16 GB.

    # ---- shared helpers --------------------------------------------------------------------------------------------

    def prompt(self, sentences, criterion_type, criterion_text):
        return chat_prompt(build_user(format_note(sentences), criterion_type, criterion_text))

    def _encode(self, prompts):
        batch = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        return batch.input_ids, batch.attention_mask

    def _texts(self, tokens):
        out = []
        for row in tokens:
            kept = row[row != self.pad_id]
            out.append(self.tokenizer.decode(kept, skip_special_tokens=True))
        return out

    def generate_texts(self, model, prompts, max_new_tokens, batch_size=BATCH_SIZE):
        """Greedy text for each prompt with the cached decoder, in input order. Short prompts are batched together."""
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        texts, tokens = [None] * len(prompts), [None] * len(prompts)
        for start in range(0, len(order), batch_size):
            ids_ = order[start:start + batch_size]
            ids, mask = self._encode([prompts[i] for i in ids_])
            result = decode_cached(model, ids, mask, max_new_tokens, self.stop_ids, self.pad_id)
            for i, text, row in zip(ids_, self._texts(result.tokens), result.tokens):
                texts[i], tokens[i] = text, int((row != self.pad_id).sum())
        return texts, tokens

    # ---- screening -------------------------------------------------------------------------------------------------

    def screen(self, sentences, criteria, rationale=False):
        """Label every (type, text) criterion against the note. Returns (results, timing). Never raises on bad model output."""
        started = time.perf_counter()
        prompts = [self.prompt(sentences, kind, text) for kind, text in criteria]
        texts, new_tokens = self.generate_texts(self.target, prompts, MAX_NEW_TOKENS)
        results = [screen.interpret(i, kind, text, raw, len(sentences)) for i, ((kind, text), raw) in enumerate(zip(criteria, texts))]
        label_seconds = time.perf_counter() - started
        rationale_seconds = 0.0
        if rationale:
            t0 = time.perf_counter()
            self.add_rationales(sentences, results)
            rationale_seconds = time.perf_counter() - t0
        timing = {
            "criteria": len(criteria), "labelling_seconds": label_seconds, "rationale_seconds": rationale_seconds,
            "generated_tokens": int(sum(new_tokens)), "prompt_tokens": [len(self.tokenizer(p).input_ids) for p in prompts],
        }
        return results, timing

    def base_model(self):
        """The untuned target used for rationales."""
        if self._base is None:
            if self._base_loader is None:
                raise RuntimeError("no base model available for rationales")
            self._base = self._base_loader()
            self._base.generation_config.eos_token_id = None
        return self._base

    def add_rationales(self, sentences, results):
        """Short free-text reasons from the untuned model. Each prompt holds only the criterion, the label and the cited sentences."""
        todo = [r for r in results if r.label is not None and r.evidence_ids]
        if not todo:
            return
        prompts = []
        for r in todo:
            quoted = "\n".join(f"[{i}] {sentences[i]}" for i in r.evidence_ids)
            user = (
                f"Criterion ({r.criterion_type}): {r.text}\nLabel: {r.label}\n\nSentences:\n{quoted}\n\n"
                "In one or two sentences, explain why this label follows from these sentences. Use only these sentences."
            )
            prompts.append(chat_prompt(user, system="You explain a clinical trial screening label briefly and only from the sentences given."))
        texts, _ = self.generate_texts(self.base_model(), prompts, RATIONALE_TOKENS)
        for r, text in zip(todo, texts):
            r.rationale = text.strip() or None

    # ---- inference lab ---------------------------------------------------------------------------------------------

    def _run(self, mode, ids, mask, new_tokens, force, k, draft_name):
        stops = () if force else self.stop_ids
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            before = torch.cuda.memory_allocated(self.device)
        if mode == "no cache":
            r = decode_uncached(self.target, ids, mask, new_tokens, stops, self.pad_id)
            extra = {}
        elif mode == "KV cache":
            r = decode_cached(self.target, ids, mask, new_tokens, stops, self.pad_id)
            extra = {"cache_mb": r.cache_bytes / 2**20}
        else:
            draft = PromptLookupDraft() if mode == "speculative (prompt lookup)" else ModelDraft(self.drafts[draft_name], self.target.config)
            r = decode_speculative(self.target, draft, ids, mask, new_tokens, k, stops)
            extra = {
                "acceptance": r.acceptance_rate, "tokens_per_round": r.tokens_per_round, "target_forwards": r.target_forwards,
                "draft_forwards": r.draft_forwards, "draft_seconds": r.draft_seconds, "verify_seconds": r.verify_seconds,
                "cache_mb": r.cache_bytes / 2**20,
            }
        peak = (torch.cuda.max_memory_allocated(self.device) - before) / 2**20 if self.device.type == "cuda" else None
        return r, {**extra, "peak_extra_mb": peak}

    def lab(self, sentences, criterion_type, criterion_text, modes, backend="sdpa", tile=64, k=2, draft_name="LoRA",
            new_tokens=32, force=False, repeats=1):
        """Run the chosen modes on one prompt. Returns rows (one per mode) and the generated text of the reference run.

        Tokens are compared with a cached sdpa run of the same prompt, which is not timed unless KV cache is selected
        with sdpa. Each mode runs once untimed first when the backend is not sdpa, to warm up. Times are the median of repeats.
        """
        if not modes:
            raise ValueError("choose at least one mode")
        new_tokens = max(1, min(int(new_tokens), LAB_MAX_NEW_TOKENS))
        ids, mask = self._encode([self.prompt(sentences, criterion_type, criterion_text)])
        stops = () if force else self.stop_ids
        reference = decode_cached(self.target, ids, mask, new_tokens, stops, self.pad_id)
        models = [self.target] + ([self.drafts[draft_name]] if "speculative" in modes else [])
        rows = []
        with attention_backend(models, backend, tile):
            decode_cached(self.target, ids, mask, 2, stops, self.pad_id)  # Warm-up under the chosen backend.
            for mode in modes:
                runs = [self._run(mode, ids, mask, new_tokens, force, k, draft_name) for _ in range(repeats)]
                result, extra = runs[0]
                totals = [r.prefill_seconds + r.decode_seconds for r, _ in runs]
                n = int(result.tokens.shape[1])
                equal = bool(result.tokens.shape == reference.tokens.shape and (result.tokens.cpu() == reference.tokens.cpu()).all())
                rows.append({
                    "mode": mode, "backend": backend if backend == "sdpa" else f"tiled {tile}", "tokens": n, "tokens_equal": equal,
                    "prefill_seconds": _median([r.prefill_seconds for r, _ in runs]), "decode_seconds": _median([r.decode_seconds for r, _ in runs]),
                    "total_seconds": _median(totals), "tokens_per_second": n / _median(totals), "repeats": repeats, **extra,
                })
        text = self._texts(reference.tokens)[0]
        return rows, text

    # ---- provenance ------------------------------------------------------------------------------------------------

    def provenance(self):
        config = final_config()
        return {
            "app_commit": app_commit(),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "target_model": {"name": TARGET_MODEL, "revision": TARGET_REVISION},
            "target_adapter": None if self.adapters is None else {
                "repo": self.adapters.repo, "revision": self.adapters.revision, "folder": self.adapters.target_folder,
                "sha256": self.adapters.sha256[self.adapters.target_folder], "note": config["deployed_adapter"],
            },
            "draft_model": {"name": DRAFT_MODEL, "revision": DRAFT_REVISION},
            "draft_adapter": None if self.adapters is None else {"folder": self.adapters.draft_folder, "sha256": self.adapters.sha256["draft"]},
            "decoding": {**GREEDY, "max_new_tokens": MAX_NEW_TOKENS, "decoder": "decode_cached", "attention": "sdpa"},
            "device": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
            "dtype": self.dtype_name,
            "packages": package_versions(),
        }


def lab_json_valid(text):
    return parse_output(text).ok or parse_output(text, repair=True).ok
