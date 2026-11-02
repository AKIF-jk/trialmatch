"""Hand-written greedy decoders: an uncached reference and a KV-cached path.

Both paths call model.forward directly and build the same inputs, so any difference comes from cache reuse:

- Positions: rotary phases must follow real tokens, not tensor columns. Left padding shifts the columns, so
  positions come from cumsum(mask)-1. The cached path computes them once at prefill, then adds 1 per step.
- Masks: a query at column i may attend to key columns j <= i that are not padding. In the cached path the query
  is only the newest column, but it attends to every cached column, so the mask is [B, 1, 1, T+step].
  The padding mask grows by one column per step, including for rows that have stopped.
- Cache: DynamicCache concatenates each layer's new K and V along the time axis. In transformers 5.16.1 the write
  slot follows the cache length and Qwen2 takes no cache_position, so the decoder asserts the length after every call.

Both paths keep logits for the last column only (logits_to_keep=1), so the reference is not charged for computing
lm_head on positions it never reads. Clocks read after torch.cuda.synchronize at phase boundaries only.
"""
import time
from dataclasses import dataclass

import torch
from transformers import DynamicCache


@dataclass
class DecodeResult:
    tokens: torch.Tensor  # [B, N]. A stopped row keeps its stop token, then holds pad_id.
    prefill_seconds: float  # The first forward pass and its token pick.
    decode_seconds: float  # All later steps.
    forward_tokens: list  # Tokens per row fed to each forward pass.
    cache_lengths: list  # Cache length after each forward pass. Empty for the uncached path.
    cache_bytes: int
    logits: torch.Tensor | None  # [B, N, V] on the CPU when keep_logits is set. Slows the run, so never set in benchmarks.


def positions_from_mask(mask):
    """Position of each real token, counting from the first non-padding column. Padding columns get 0."""
    return (mask.long().cumsum(-1) - 1).clamp(min=0)


def causal_mask(mask, query_length):
    """Boolean mask [B, 1, query, keys], True where attention is allowed. The queries are the newest columns of mask."""
    keys = mask.shape[1]
    causal = torch.ones(query_length, keys, dtype=torch.bool, device=mask.device).tril(keys - query_length)
    return causal[None, None] & mask[:, None, None, :]


def _attention(mask, query_length):
    # A dict skips the library's own mask construction in Qwen2Model.forward, so this mask is the one used.
    return {"full_attention": causal_mask(mask, query_length)}


def _extend(mask):
    return torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)


class _Run:
    """Greedy picks, stop handling and phase timing shared by both paths."""

    def __init__(self, input_ids, max_new_tokens, stop_ids, pad_id, keep_logits):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        self.device = input_ids.device
        self.tokens = input_ids.new_full((input_ids.shape[0], max_new_tokens), pad_id)
        self.stop = torch.tensor(list(stop_ids), dtype=torch.long, device=self.device)
        self.done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=self.device)
        self.pad_id, self.keep_logits = pad_id, keep_logits
        self.logits, self.forward_tokens, self.cache_lengths = [], [], []
        self.count, self.cache_bytes = 0, 0
        self.start = self._now()
        self.prefill_end = None

    def _now(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def pick(self, logits):
        """Greedy token per row from last-column logits [B, V]. Rows that already stopped get pad_id."""
        if self.keep_logits:
            self.logits.append(logits.cpu())  # Native dtype, so float64 distances are not rounded to float32.
        token = logits.argmax(-1).masked_fill(self.done, self.pad_id)
        self.tokens[:, self.count] = token
        self.count += 1
        if len(self.stop):
            self.done |= torch.isin(token, self.stop)
        if self.prefill_end is None:
            self.prefill_end = self._now()
        return token

    @property
    def stopped(self):
        return len(self.stop) > 0 and bool(self.done.all())

    def result(self):
        end = self._now()
        return DecodeResult(
            tokens=self.tokens[:, :self.count], prefill_seconds=self.prefill_end - self.start, decode_seconds=end - self.prefill_end,
            forward_tokens=self.forward_tokens, cache_lengths=self.cache_lengths, cache_bytes=self.cache_bytes,
            logits=torch.stack(self.logits, dim=1) if self.keep_logits else None,
        )


@torch.inference_mode()
def decode_uncached(model, input_ids, attention_mask, max_new_tokens, stop_ids=(), pad_id=0, keep_logits=False):
    """Reference path: every step reruns the model over all tokens, with no state kept between steps."""
    run = _Run(input_ids, max_new_tokens, stop_ids, pad_id, keep_logits)
    ids, mask = input_ids, attention_mask.bool()
    for _ in range(max_new_tokens):
        run.forward_tokens.append(ids.shape[1])
        out = model(
            input_ids=ids, attention_mask=_attention(mask, ids.shape[1]), position_ids=positions_from_mask(mask),
            use_cache=False, logits_to_keep=1,
        )
        token = run.pick(out.logits[:, -1])
        if run.stopped:
            break
        ids, mask = torch.cat([ids, token[:, None]], dim=1), _extend(mask)
    return run.result()


@torch.inference_mode()
def decode_cached(model, input_ids, attention_mask, max_new_tokens, stop_ids=(), pad_id=0, keep_logits=False):
    """Cached path: prefill runs once over the prompt, then each step feeds only the last token."""
    run = _Run(input_ids, max_new_tokens, stop_ids, pad_id, keep_logits)
    cache = DynamicCache(config=model.config)
    feed, mask = input_ids, attention_mask.bool()
    positions = positions_from_mask(mask)
    for _ in range(max_new_tokens):
        run.forward_tokens.append(feed.shape[1])
        out = model(
            input_ids=feed, attention_mask=_attention(mask, feed.shape[1]), position_ids=positions,
            past_key_values=cache, use_cache=True, logits_to_keep=1,
        )
        length = cache.get_seq_length()
        if length != mask.shape[1]:
            raise AssertionError(f"cache holds {length} columns but the mask has {mask.shape[1]}")
        run.cache_lengths.append(length)
        token = run.pick(out.logits[:, -1])
        if run.stopped:
            break
        feed, mask, positions = token[:, None], _extend(mask), positions[:, -1:] + 1
    run.cache_bytes = sum(layer.keys.nbytes + layer.values.nbytes for layer in cache.layers)
    return run.result()


def compare(reference, cached):
    """Token agreement and logit distance. Logits are compared only before the first divergence, since later they see different inputs."""
    ref, new = reference.tokens, cached.tokens
    report = {"shape_equal": ref.shape == new.shape}
    width = min(ref.shape[1], new.shape[1])
    differs = (ref[:, :width] != new[:, :width]).any(0).nonzero()
    report["first_divergence"] = int(differs[0]) if len(differs) else None
    report["tokens_equal"] = report["shape_equal"] and not len(differs)
    if reference.logits is not None and cached.logits is not None:
        keep = report["first_divergence"] if len(differs) else width
        diff = (reference.logits[:, :keep] - cached.logits[:, :keep]).abs()
        top = reference.logits[:, :keep].topk(2, dim=-1).values
        report["max_logit_diff"] = float(diff.max()) if keep else None
        report["mean_logit_diff"] = float(diff.double().mean()) if keep else None
        # A token can only flip when the numerical difference reaches the gap between the top two logits.
        report["min_top2_margin"] = float((top[..., 0] - top[..., 1]).min()) if keep else None
        if len(differs):
            report["divergence"] = _divergence(reference, cached, report["first_divergence"])
    return report


def _divergence(reference, cached, step):
    """Logits behind the first differing token. Inputs up to this step were identical, so any gap here is numerical."""
    row = int((reference.tokens[:, step] != cached.tokens[:, step]).nonzero()[0])
    a, b = int(reference.tokens[row, step]), int(cached.tokens[row, step])
    ref, new = reference.logits[row, step], cached.logits[row, step]
    return {
        "row": row, "reference_token": a, "cached_token": b, "max_logit_diff": float((ref - new).abs().max()),
        "reference_path_logits": [float(ref[a]), float(ref[b])], "cached_path_logits": [float(new[a]), float(new[b])],
    }
