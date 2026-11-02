"""Reference scaled dot-product attention: the full score matrix, a stable softmax and nothing else.

Shapes: q [B, Hq, Tq, D], k and v [B, Hkv, Tk, D] with Hq a multiple of Hkv (grouped KV heads).
Mask: boolean, True where attention is allowed, broadcastable to [B, Hq, Tq, Tk]. causal=True adds the
bottom-right aligned causal rule, so the Tq queries are the last Tq positions of the Tk keys. This is the layout
of a KV-cached step (Tq=1, every cached key visible) and of a full prefill (Tq=Tk).
A query with no allowed key gets an all-zero output, not NaN.
"""
import math

import torch


def compute_dtype(dtype):
    """Math dtype: float64 stays float64, everything else (float16, bfloat16, float32) is computed in float32."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def check_shapes(q, k, v):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k and v must be [B, H, T, D]")
    if k.shape[:3] != v.shape[:3]:
        raise ValueError(f"k {tuple(k.shape)} and v {tuple(v.shape)} disagree")
    if q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3]:
        raise ValueError(f"q {tuple(q.shape)} and k {tuple(k.shape)} disagree on batch or head dimension")
    if q.shape[1] % k.shape[1]:
        raise ValueError(f"{q.shape[1]} query heads is not a multiple of {k.shape[1]} KV heads")


def causal_rule(query_length, key_length, key_start, key_stop, device):
    """Bool [Tq, key_stop-key_start]: key j is visible to query i when j <= i + (Tk - Tq)."""
    keys = torch.arange(key_start, key_stop, device=device)
    queries = torch.arange(query_length, device=device) + (key_length - query_length)
    return keys[None, :] <= queries[:, None]


def reference_attention(q, k, v, mask=None, causal=False, scale=None):
    check_shapes(q, k, v)
    work = compute_dtype(q.dtype)
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    n_rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)

    scores = torch.matmul(q.to(work), k.to(work).transpose(-1, -2)) * scale
    allowed = None
    if causal:
        allowed = causal_rule(q.shape[2], k.shape[2], 0, k.shape[2], q.device)
    if mask is not None:
        allowed = mask if allowed is None else allowed & mask
    if allowed is not None:
        scores = scores.masked_fill(~allowed, float("-inf"))

    peak = scores.amax(-1, keepdim=True)
    peak = torch.where(torch.isinf(peak), torch.zeros_like(peak), peak)  # An all-masked row would give -inf - -inf.
    weights = torch.exp(scores - peak)
    total = weights.sum(-1, keepdim=True)
    weights = weights / torch.where(total > 0, total, torch.ones_like(total))
    return torch.matmul(weights, v.to(work)).to(q.dtype)
