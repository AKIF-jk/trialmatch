"""Shared inputs and masks for the attention tests and the attention check script."""
import torch

MASK_KINDS = ("none", "causal", "padding", "causal_padding", "rectangular", "sparse")


def make_inputs(q_len, k_len, dtype, batch=2, heads=4, kv_heads=2, dim=16, seed=0, q_scale=1.0):
    """Random q [B,H,Tq,D] and k, v [B,Hkv,Tk,D]. Drawn in float64, then cast, so every dtype sees the same values."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, q_len, dim, generator=g, dtype=torch.float64) * q_scale
    k = torch.randn(batch, kv_heads, k_len, dim, generator=g, dtype=torch.float64)
    v = torch.randn(batch, kv_heads, k_len, dim, generator=g, dtype=torch.float64)
    return q.to(dtype), k.to(dtype), v.to(dtype)


def query_length(kind, length):
    return max(1, length // 4) if kind == "rectangular" else length


def make_mask(kind, batch, q_len, k_len, seed=0):
    """(mask, causal) for one kind. mask is a boolean tensor, True where attention is allowed, or None.

    padding: the first few columns of row 1 are padding, as in a left-padded batch. [B,1,1,Tk].
    rectangular: fewer queries than keys, causal from the bottom right, plus a random [B,1,Tq,Tk] mask.
    sparse: a random [B,1,Tq,Tk] mask with 95% of entries blocked, so many rows and tiles have no allowed key.
    """
    g = torch.Generator().manual_seed(seed + 1000)
    if kind == "none":
        return None, False
    if kind == "causal":
        return None, True
    if kind in ("padding", "causal_padding"):
        pad = torch.zeros(batch, dtype=torch.long)
        if batch > 1:
            pad[1] = k_len // 2
        return (torch.arange(k_len)[None, :] >= pad[:, None])[:, None, None, :], kind == "causal_padding"
    if kind == "rectangular":
        return torch.rand(batch, 1, q_len, k_len, generator=g) > 0.2, True
    if kind == "sparse":
        return torch.rand(batch, 1, q_len, k_len, generator=g) > 0.95, False
    raise ValueError(f"unknown mask kind {kind}")
