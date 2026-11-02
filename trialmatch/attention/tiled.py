"""Tiled attention with an online softmax (the FlashAttention recurrence, written as a Python loop over key tiles).

For each query row the loop keeps three running values over the key tiles seen so far:
  m    the largest score,
  l    sum of exp(score - m),
  acc  sum of exp(score - m) * v.
A new tile with scores s and tile maximum t gives m' = max(m, t), then
  alpha = exp(m - m'),  l' = l * alpha + sum(exp(s - m')),  acc' = acc * alpha + exp(s - m') @ v_tile.
The output is acc / l after the last tile, equal to softmax(scores) @ v without ever holding the full score matrix:
the largest temporary is [Tq, tile] per head, not [Tq, Tk].

Masked scores are -inf. While a row has seen no allowed key m is -inf, and exp(-inf - -inf) would be NaN, so the
shift uses 0 instead of an infinite m'. Then every term exp(s - 0) is exp(-inf) = 0, alpha is exp(-inf) = 0, and the row
stays at l = 0, acc = 0 until a real key arrives. A row that never sees one outputs zeros, like the reference.

Grouped KV heads are not expanded: q is viewed as [B, Hkv, rep, Tq, D] and each key tile is [B, Hkv, 1, tile, D],
so matmul broadcasts over rep and no repeated copy of K or V is made.
Statistics and the accumulator are kept in float32 (float64 for float64 inputs) whatever the input dtype.
"""
import math

import torch

from trialmatch.attention.reference import causal_rule, check_shapes, compute_dtype


def tiled_attention(q, k, v, mask=None, causal=False, scale=None, tile=64):
    check_shapes(q, k, v)
    if tile < 1:
        raise ValueError("tile must be at least 1")
    batch, heads, query_length, dim = q.shape
    kv_heads, key_length = k.shape[1], k.shape[2]
    n_rep = heads // kv_heads
    work = compute_dtype(q.dtype)
    scale = 1.0 / math.sqrt(dim) if scale is None else scale

    q5 = q.to(work).view(batch, kv_heads, n_rep, query_length, dim)
    m = torch.full((batch, kv_heads, n_rep, query_length, 1), float("-inf"), dtype=work, device=q.device)
    l = torch.zeros_like(m)
    acc = torch.zeros(batch, kv_heads, n_rep, query_length, v.shape[-1], dtype=work, device=q.device)
    if mask is not None:
        mask = _mask5(mask, batch, kv_heads, n_rep, query_length, key_length)

    for start in range(0, key_length, tile):
        stop = min(start + tile, key_length)
        scores = torch.matmul(q5, k[:, :, None, start:stop].to(work).transpose(-1, -2)) * scale
        allowed = None
        if causal:
            allowed = causal_rule(query_length, key_length, start, stop, q.device)
        if mask is not None:
            allowed = mask[..., start:stop] if allowed is None else allowed & mask[..., start:stop]
        if allowed is not None:
            scores = scores.masked_fill(~allowed, float("-inf"))

        new_m = torch.maximum(m, scores.amax(-1, keepdim=True))
        shift = torch.where(torch.isinf(new_m), torch.zeros_like(new_m), new_m)
        alpha = torch.exp(m - shift)
        weights = torch.exp(scores - shift)
        l = l * alpha + weights.sum(-1, keepdim=True)
        acc = acc * alpha + torch.matmul(weights, v[:, :, None, start:stop].to(work))
        m = new_m

    out = acc / torch.where(l > 0, l, torch.ones_like(l))
    return out.reshape(batch, heads, query_length, v.shape[-1]).to(q.dtype)


def _mask5(mask, batch, kv_heads, n_rep, query_length, key_length):
    """Boolean mask broadcastable to [B, Hq, Tq, Tk] as [B or 1, Hkv or 1, rep or 1, Tq or 1, Tk or 1]."""
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be boolean, got {mask.dtype}")
    if mask.dim() != 4 or mask.shape[-1] != key_length:
        raise ValueError(f"mask must be [B, 1 or H, Tq, {key_length}], got {tuple(mask.shape)}")
    if mask.shape[1] == 1:
        return mask[:, :, None]
    return mask.reshape(mask.shape[0], kv_heads, n_rep, *mask.shape[2:])
