"""Greedy speculative decoding with a draft that proposes tokens and a target that verifies them in one pass.

State kept between rounds, with seq = prompt + every token emitted so far (batch 1, no padding):

- Target cache holds seq[:-1]. The newest token is fed at the start of the next verification pass.
- A round: the draft proposes k tokens d1..dk after seq. The target reads [seq[-1], d1..dk] in one forward pass and
  its logits give the greedy token t0..tk at each of those k+1 positions. t_i is what plain greedy decoding would
  emit after seq + d1..di, as long as d1..di match t0..t(i-1). So with m the length of the longest matching prefix,
  d1..dm are exactly the next greedy tokens and t_m is the next one after them: the correction when m < k, the
  bonus when m == k. The round emits m + 1 tokens, so the output equals plain greedy decoding by construction.
- After the round, the target cache is cropped to the verified prefix (seq + d1..dm), which leaves the new
  seq[:-1]. The draft is told the same prefix length and drops what it cannot keep. At the start of the next
  round it feeds every token of seq it has not seen (the correction or bonus, and d_k when all k were accepted),
  so its state follows the target's accepted text.
- k is limited to remaining - 1, so a round never emits more tokens than are left. A draft may propose fewer than k tokens
  (prompt lookup does); the round then verifies what it got, and with none it is a plain one-token step. A stop token ends the output at
  its first occurrence, whether the draft or the target produced it, as in decode_cached.

Equality with decode_cached holds when both paths give identical logits at every step. The verification pass has a
different tensor shape (k + 1 query columns) from a single-token step, so in fp16 near ties can break differently
(docs/STATUS.md, Phase 3). Float32 and float64 are the equality results.
"""
import time
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from trialmatch.decoding.greedy import _attention


@dataclass
class SpeculativeResult:
    tokens: torch.Tensor  # [1, N], stops at and includes the first stop token.
    prefill_seconds: float  # Target prefill, first token pick and draft prefill.
    decode_seconds: float  # All rounds.
    draft_seconds: float  # Time inside the draft's proposals, part of decode_seconds.
    verify_seconds: float  # Time in the target's verification passes, part of decode_seconds.
    rounds: list = field(default_factory=list)  # One dict per round: k, accepted, emitted.
    target_forwards: int = 0  # Prefill plus one per round.
    draft_forwards: int = 0
    cache_bytes: int = 0  # Target cache at the end.
    logits: torch.Tensor | None = None  # [1, N, V] on the CPU: the target logits behind each emitted token.

    @property
    def proposed(self):
        return sum(r["k"] for r in self.rounds)

    @property
    def accepted(self):
        return sum(r["accepted"] for r in self.rounds)

    @property
    def acceptance_rate(self):
        return self.accepted / self.proposed if self.proposed else None

    @property
    def tokens_per_round(self):
        return sum(r["emitted"] for r in self.rounds) / len(self.rounds) if self.rounds else None

    def summary(self):
        return {
            "tokens": int(self.tokens.numel()), "rounds": len(self.rounds), "proposed": self.proposed, "accepted": self.accepted,
            "acceptance_rate": self.acceptance_rate, "tokens_per_round": self.tokens_per_round,
            "target_forwards": self.target_forwards, "draft_forwards": self.draft_forwards,
            "prefill_seconds": self.prefill_seconds, "decode_seconds": self.decode_seconds,
            "draft_seconds": self.draft_seconds, "verify_seconds": self.verify_seconds, "cache_bytes": self.cache_bytes,
        }


def _forward(model, cache, tokens, past, keep):
    """Feed 1-D tokens after `past` cached columns. Returns logits [keep, V] for the last `keep` columns."""
    n = tokens.numel()
    mask = torch.ones(1, past + n, dtype=torch.bool, device=tokens.device)
    positions = torch.arange(past, past + n, device=tokens.device)[None]
    out = model(
        input_ids=tokens[None], attention_mask=_attention(mask, n), position_ids=positions,
        past_key_values=cache, use_cache=True, logits_to_keep=keep,
    )
    length = cache.get_seq_length()
    if length != past + n:
        raise AssertionError(f"cache holds {length} columns, expected {past + n}")
    return out.logits[0]


class ModelDraft:
    """A causal LM used as the draft. It keeps its own cache and the number of tokens it has read."""

    def __init__(self, model, target_config=None):
        if target_config is not None and model.config.vocab_size != target_config.vocab_size:
            raise ValueError(f"draft vocab {model.config.vocab_size} differs from target vocab {target_config.vocab_size}")
        self.model, self.cache, self.length, self.forwards = model, None, 0, 0

    def _feed(self, tokens):
        logits = _forward(self.model, self.cache, tokens, self.length, 1)
        self.length += tokens.numel()
        self.forwards += 1
        return logits[-1]

    @torch.inference_mode()
    def start(self, prompt_ids):
        """Read the prompt. prompt_ids is [1, P]."""
        self.cache = DynamicCache(config=self.model.config)
        self.length = 0
        self._feed(prompt_ids[0])

    @torch.inference_mode()
    def propose(self, seq, k):
        """Greedy continuation of seq (1-D) for k tokens. Feeds the tokens of seq the draft has not read yet first."""
        logits = self._feed(seq[self.length:])
        proposals = []
        for i in range(k):
            token = logits.argmax(-1)
            proposals.append(token)
            if i < k - 1:
                logits = self._feed(token[None])
        return torch.stack(proposals)

    def truncate(self, valid):
        """Drop cache entries past the first `valid` columns."""
        extra = self.length - valid
        if extra > 0:
            self.cache.crop(-extra)
            self.length = valid


def _now(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.inference_mode()
def decode_speculative(target, draft, input_ids, attention_mask, max_new_tokens, k, stop_ids=(), keep_logits=False):
    """Greedy speculative decoding for one prompt. `draft` needs start(prompt_ids), propose(seq, k) -> [k] and truncate(valid)."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if k < 0:
        raise ValueError("k must not be negative")
    if input_ids.shape[0] != 1 or not bool(attention_mask.bool().all()):
        raise ValueError("speculative decoding takes one prompt without padding")
    device, stops = input_ids.device, set(int(s) for s in stop_ids)
    cache = DynamicCache(config=target.config)
    start = _now(device)

    logits = _forward(target, cache, input_ids[0], 0, 1)
    token = logits[-1].argmax(-1)
    logs = [logits[-1:].cpu()] if keep_logits else []
    seq = torch.cat([input_ids[0], token[None]])
    count = 1
    done = int(token) in stops
    if not done and count < max_new_tokens:
        draft.start(input_ids)
    prefill_end = _now(device)

    rounds, target_forwards, draft_seconds, verify_seconds = [], 1, 0.0, 0.0
    while not done and count < max_new_tokens:
        kk = min(k, max_new_tokens - count - 1)
        length = seq.numel()
        t0 = _now(device)
        drafts = draft.propose(seq, kk) if kk else seq[:0]
        if drafts.numel() > kk:
            raise ValueError(f"draft proposed {drafts.numel()} tokens, at most {kk} were asked for")
        kk = drafts.numel()  # A draft may propose fewer than asked, or none.
        t1 = _now(device)
        logits = _forward(target, cache, torch.cat([seq[-1:], drafts]), length - 1, kk + 1)
        picks = logits.argmax(-1)
        matched = int((drafts == picks[:kk]).long().cumprod(0).sum())
        new = torch.cat([drafts[:matched], picks[matched:matched + 1]])
        emitted = new.tolist()
        for i, value in enumerate(emitted):
            if value in stops:
                emitted, done = emitted[:i + 1], True
                break
        t2 = _now(device)
        draft_seconds, verify_seconds = draft_seconds + t1 - t0, verify_seconds + t2 - t1
        target_forwards += 1
        rounds.append({"k": kk, "accepted": min(matched, len(emitted)), "emitted": len(emitted)})
        if keep_logits:
            logs.append(logits[:len(emitted)].cpu())
        seq = torch.cat([seq, new[:len(emitted)]])
        count += len(emitted)
        if not done:
            if kk > matched:
                cache.crop(-(kk - matched))
            if kk:
                draft.truncate(length + matched)
    end = _now(device)

    return SpeculativeResult(
        tokens=seq[input_ids.shape[1]:][None], prefill_seconds=prefill_end - start, decode_seconds=end - prefill_end,
        draft_seconds=draft_seconds, verify_seconds=verify_seconds, rounds=rounds, target_forwards=target_forwards,
        draft_forwards=getattr(draft, "forwards", 0), cache_bytes=sum(layer.keys.nbytes + layer.values.nbytes for layer in cache.layers),
        logits=torch.cat(logs)[None] if keep_logits else None,
    )
