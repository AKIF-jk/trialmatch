"""Tiled attention inside a small random Qwen2: same greedy tokens as the sdpa backend. Needs torch and transformers."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from trialmatch.attention import hf
from trialmatch.decoding.experiment import left_pad
from trialmatch.decoding.greedy import compare, decode_cached, decode_uncached

PAD = 0


def tiny_model(dtype):
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=97, hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=512, use_sliding_window=False, attn_implementation="sdpa",
    )
    return Qwen2ForCausalLM(config).to(dtype).eval()


def prompt(length, seed):
    return torch.randint(1, 97, (length,), generator=torch.Generator().manual_seed(seed))


def batch():
    return left_pad([prompt(5, 1), prompt(23, 2), prompt(40, 3)], PAD, "cpu")


@pytest.fixture
def spy(monkeypatch):
    calls = []
    real = hf.tiled_attention

    def counted(*args, **kwargs):
        calls.append(kwargs["tile"])
        return real(*args, **kwargs)

    monkeypatch.setattr(hf, "tiled_attention", counted)
    return calls


@pytest.mark.parametrize("tile", [1, 7, 16, 500])
@pytest.mark.parametrize("decode", [decode_uncached, decode_cached])
def test_tokens_equal_sdpa_float64(decode, tile, spy):
    model = tiny_model(torch.float64)
    ids, mask = batch()
    expected = decode(model, ids, mask, 16, keep_logits=True)
    assert hf.use_tiled_attention(model, tile) == "sdpa"
    assert model.config._attn_implementation == hf.backend_name(tile)
    actual = decode(model, ids, mask, 16, keep_logits=True)
    assert spy and set(spy) == {tile}
    report = compare(expected, actual)
    assert report["tokens_equal"] and report["max_logit_diff"] < 1e-9


def test_tokens_equal_sdpa_float32(spy):
    model = tiny_model(torch.float32)
    ids, mask = batch()
    expected = decode_cached(model, ids, mask, 16, keep_logits=True)
    hf.use_tiled_attention(model, 16)
    actual = decode_cached(model, ids, mask, 16, keep_logits=True)
    assert spy
    report = compare(expected, actual)
    assert report["tokens_equal"] and report["max_logit_diff"] < 1e-3


def test_library_masks_and_generate_work_with_tiled_backend(spy):
    """No explicit mask dict: the library builds its own mask for the registered backend, and generate runs the cache."""
    model = tiny_model(torch.float64)
    ids, mask = batch()
    kwargs = dict(input_ids=ids, attention_mask=mask, max_new_tokens=10, pad_token_id=PAD, do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
    model.generation_config.eos_token_id = None
    with torch.inference_mode():
        expected = model.generate(**kwargs)
        hf.use_tiled_attention(model, 8)
        actual = model.generate(**kwargs)
    assert spy
    assert actual.tolist() == expected.tolist()


def test_switch_back_to_sdpa(spy):
    model = tiny_model(torch.float64)
    ids, mask = batch()
    hf.use_tiled_attention(model, 4)
    model.set_attn_implementation("sdpa")
    decode_cached(model, ids, mask, 4)
    assert not spy


def test_unsupported_options_fail_loudly():
    model = tiny_model(torch.float64)
    hf.use_tiled_attention(model, 4)
    layer = model.model.layers[0].self_attn
    function = ALL_ATTENTION_FUNCTIONS.get_interface(model.config._attn_implementation, None)
    q = torch.zeros(1, 4, 2, 12, dtype=torch.float64)
    k = torch.zeros(1, 2, 2, 12, dtype=torch.float64)
    with pytest.raises(NotImplementedError):
        function(layer, q, k, k, None, dropout=0.1)
    with pytest.raises(NotImplementedError):
        function(layer, q, k, k, None, sliding_window=8)
    with pytest.raises(TypeError):
        function(layer, q, k, k, torch.zeros(1, 1, 2, 2))
