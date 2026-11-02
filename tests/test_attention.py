"""Reference and tiled attention tests. They need torch, so they run on a CPU kernel and skip where torch is missing."""
import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F

from trialmatch.attention.cases import MASK_KINDS, make_inputs, make_mask, query_length
from trialmatch.attention.reference import reference_attention
from trialmatch.attention.tiled import tiled_attention

LENGTHS = [1, 7, 64, 256, 1024]
TILES = [1, 16, 64, "larger"]
TOLERANCE = {torch.float32: 1e-5, torch.float64: 1e-12}


def tile_size(tile, length):
    return 2 * length + 1 if tile == "larger" else tile


def oracle(q, k, v, mask, causal):
    """Independent float64 answer: torch's own SDPA with an explicit boolean mask. Rows with no allowed key become 0."""
    q_len, k_len = q.shape[2], k.shape[2]
    allowed = torch.ones(q.shape[0], 1, q_len, k_len, dtype=torch.bool)
    if causal:
        allowed = allowed & torch.ones(q_len, k_len, dtype=torch.bool).tril(k_len - q_len)
    if mask is not None:
        allowed = allowed & mask
    n_rep = q.shape[1] // k.shape[1]
    out = F.scaled_dot_product_attention(q, k.repeat_interleave(n_rep, 1), v.repeat_interleave(n_rep, 1), attn_mask=allowed)
    return torch.nan_to_num(out, nan=0.0)


@pytest.mark.parametrize("kind", MASK_KINDS)
@pytest.mark.parametrize("length", [1, 7, 64, 256])
def test_reference_matches_torch_sdpa(kind, length):
    q_len = query_length(kind, length)
    q, k, v = make_inputs(q_len, length, torch.float64)
    mask, causal = make_mask(kind, 2, q_len, length)
    got = reference_attention(q, k, v, mask=mask, causal=causal)
    assert (got - oracle(q, k, v, mask, causal)).abs().max() < 1e-12


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("kind", MASK_KINDS)
def test_tiled_matches_float64_reference(kind, length, tile, dtype):
    q_len = query_length(kind, length)
    q, k, v = make_inputs(q_len, length, dtype)
    mask, causal = make_mask(kind, 2, q_len, length)
    want = reference_attention(q.double(), k.double(), v.double(), mask=mask, causal=causal)
    got = tiled_attention(q, k, v, mask=mask, causal=causal, tile=tile_size(tile, length))
    assert got.dtype == dtype and not got.isnan().any()
    error = (got.double() - want).abs()
    assert error.max() < TOLERANCE[dtype] and error.mean() < TOLERANCE[dtype]


@pytest.mark.parametrize("tile", [1, 5, 16, 100])
def test_fully_masked_rows_give_zeros_not_nan(tile):
    q, k, v = make_inputs(20, 20, torch.float32)
    mask = torch.ones(2, 1, 20, 20, dtype=torch.bool)
    mask[:, :, 3] = False  # One query row sees nothing, in every tile.
    mask[:, :, 11, :9] = False  # Another sees nothing in the first tiles and something later.
    mask[:, :, 11, 12:] = False
    got = tiled_attention(q, k, v, mask=mask, tile=tile)
    assert not got.isnan().any() and not got.isinf().any()
    assert (got[:, :, 3] == 0).all()
    want = reference_attention(q.double(), k.double(), v.double(), mask=mask)
    assert (got.double() - want).abs().max() < 1e-5


@pytest.mark.parametrize("tile", [1, 3, 16])
def test_causal_padding_rows_are_zero(tile):
    """A left-padded query row has no allowed key under causal plus padding, and must come out as zeros."""
    q, k, v = make_inputs(12, 12, torch.float32)
    mask, causal = make_mask("causal_padding", 2, 12, 12)
    got = tiled_attention(q, k, v, mask=mask, causal=causal, tile=tile)
    assert (got[1, :, :6] == 0).all()
    assert got[1, :, 6:].abs().max() > 0


@pytest.mark.parametrize("where", ["first", "last"])
@pytest.mark.parametrize("tile", [1, 4, 16])
def test_running_maximum_rescale(where, tile):
    """The largest score sits in the first or the last tile, so the accumulated sum is rescaled (last) or kept (first)."""
    q, k, v = make_inputs(3, 16, torch.float64, heads=2, kv_heads=2, seed=5)
    k = k.clone()
    spot = 0 if where == "first" else 15
    k[:, :, spot] = 25 * q[:, :, 0]  # Score with the first query is about 25 * |q|^2 / 4, far above every other key.
    want = reference_attention(q, k, v)
    got = tiled_attention(q, k, v, tile=tile)
    assert (got - want).abs().max() < 1e-12
    # The dominant key carries nearly all the weight for the first query, so the check is not vacuous.
    assert (want[:, :, 0] - v[:, :, spot]).abs().max() < 1e-6


@pytest.mark.parametrize("tile", [1, 16, 64, 300])
@pytest.mark.parametrize("scale", [1e-8, 30.0, 120.0])
def test_extreme_scores_stay_finite_and_accurate(tile, scale):
    """Scores near 1e-8 and near +-1e3 to 1e4 (float32 exp overflows past 88)."""
    q, k, v = make_inputs(64, 256, torch.float32, q_scale=scale)
    want = reference_attention(q.double(), k.double(), v.double(), causal=True)
    got = tiled_attention(q, k, v, causal=True, tile=tile)
    assert not got.isnan().any() and not got.isinf().any()
    assert (got.double() - want).abs().max() < 2e-3
    exact = tiled_attention(q.double(), k.double(), v.double(), causal=True, tile=tile)
    assert (exact - want).abs().max() < 1e-12


def test_unstable_softmax_fails_where_tiled_works():
    """Negative control: a softmax without max subtraction overflows on these scores, so the finite results above mean something."""
    q, k, v = make_inputs(8, 8, torch.float32, q_scale=120.0)
    scores = torch.matmul(q, k.repeat_interleave(2, 1).transpose(-1, -2)) * 0.25
    naive = torch.exp(scores) / torch.exp(scores).sum(-1, keepdim=True)
    assert not torch.isfinite(naive).all()
    assert torch.isfinite(tiled_attention(q, k, v, tile=4)).all()


def test_scaling_uses_sqrt_of_head_dim():
    """Negative control for the scale: attention with scale 1 differs from the default 1/sqrt(16) = 0.25."""
    q, k, v = make_inputs(4, 4, torch.float64)
    default = reference_attention(q, k, v)
    assert (reference_attention(q, k, v, scale=0.25) - default).abs().max() == 0
    assert (reference_attention(q, k, v, scale=1.0) - default).abs().max() > 1e-3
    assert (tiled_attention(q, k, v, scale=1.0, tile=2) - reference_attention(q, k, v, scale=1.0)).abs().max() < 1e-12


def test_grouped_heads_use_group_order():
    """Query head h reads KV head h // rep, as in repeat_kv. Swapping the two KV heads must change the output."""
    q, k, v = make_inputs(5, 9, torch.float64)
    base = tiled_attention(q, k, v, tile=4)
    swapped = tiled_attention(q, k.flip(1), v.flip(1), tile=4)
    assert (base - swapped).abs().max() > 1e-3
    expanded = reference_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1))
    assert (base - expanded).abs().max() < 1e-12


def test_half_precision_input_is_accumulated_in_float32():
    q, k, v = make_inputs(32, 32, torch.float16)
    want = reference_attention(q.double(), k.double(), v.double(), causal=True)
    got = tiled_attention(q, k, v, causal=True, tile=8)
    assert got.dtype == torch.float16
    assert (got.double() - want).abs().max() < 2e-3


def test_rejects_bad_arguments():
    q, k, v = make_inputs(4, 4, torch.float32)
    with pytest.raises(ValueError):
        tiled_attention(q, k, v, tile=0)
    with pytest.raises(TypeError):
        tiled_attention(q, k, v, mask=torch.zeros(2, 1, 4, 4))
    with pytest.raises(ValueError):
        tiled_attention(q, k[:, :, :3], v, tile=2)
    with pytest.raises(ValueError):
        tiled_attention(q[:, :3], k, v, tile=2)
