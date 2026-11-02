"""Plug tiled attention into a Hugging Face Qwen2 model as a named attention backend.

Qwen2Attention looks its function up by config._attn_implementation and calls it with
(module, query [B,Hq,Tq,D], key [B,Hkv,Tk,D], value, attention_mask, dropout, scaling, sliding_window, **kwargs),
expecting (output [B,Tq,Hq,D], weights or None). The function below has that signature. It takes the boolean masks
that sdpa_mask builds (and that the decoders in trialmatch.decoding.greedy pass as a dict), expands nothing for
grouped KV heads, and does not support dropout, sliding windows or returning weights.
"""
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

from trialmatch.attention.tiled import tiled_attention


def backend_name(tile):
    return f"tiled_{tile}"


def register_tiled(tile):
    """Register the backend for one tile size and return its name. Safe to call again."""
    name = backend_name(tile)

    def tiled_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None, is_causal=None, **kwargs):
        if dropout:
            raise NotImplementedError("tiled attention has no dropout")
        if sliding_window is not None:
            raise NotImplementedError("tiled attention has no sliding window")
        if kwargs.get("output_attentions", False):
            raise NotImplementedError("tiled attention does not return attention weights")
        if attention_mask is not None and attention_mask.dtype != bool:
            raise TypeError(f"tiled attention needs a boolean mask, got {attention_mask.dtype}")
        # With no mask, sdpa applies the causal rule for multi-token queries. Do the same.
        is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        causal = bool(attention_mask is None and query.shape[2] > 1 and is_causal)
        out = tiled_attention(query, key, value, mask=attention_mask, causal=causal, scale=scaling, tile=tile)
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface.register(name, tiled_forward)
    AttentionMaskInterface.register(name, sdpa_mask)
    return name


def use_tiled_attention(model, tile):
    """Switch every attention layer of model to the tiled backend. Returns the previous implementation name."""
    previous = model.config._attn_implementation
    model.set_attn_implementation(register_tiled(tile))
    return previous
