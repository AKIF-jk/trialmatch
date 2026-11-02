"""Cached decoder tests on a small random Qwen2. They need torch, so they run on a CPU kernel and skip locally."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from trialmatch.decoding.experiment import left_pad
from trialmatch.decoding.greedy import DecodeResult, causal_mask, compare, decode_cached, decode_uncached, positions_from_mask

PAD = 0


def tiny_model(dtype=torch.float64):
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=97, hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=512, use_sliding_window=False, attn_implementation="sdpa",
    )
    return Qwen2ForCausalLM(config).to(dtype).eval()


@pytest.fixture(scope="module")
def model():
    return tiny_model()


def prompt(length, seed=1):
    return torch.randint(1, 97, (length,), generator=torch.Generator().manual_seed(seed))


def single(length, seed=1):
    return left_pad([prompt(length, seed)], PAD, "cpu")


def test_positions_skip_left_padding():
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert positions_from_mask(mask).tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]


def test_causal_mask_full_and_single_query():
    mask = torch.tensor([[False, True, True]])
    assert causal_mask(mask, 3)[0, 0].tolist() == [[False, False, False], [False, True, False], [False, True, True]]
    assert causal_mask(mask, 1)[0, 0].tolist() == [[False, True, True]]


@pytest.mark.parametrize("length", [4, 16, 64])
@pytest.mark.parametrize("new_tokens", [4, 16, 32])
def test_cached_equals_reference_float64(model, length, new_tokens):
    ids, mask = single(length)
    report = compare(decode_uncached(model, ids, mask, new_tokens, keep_logits=True), decode_cached(model, ids, mask, new_tokens, keep_logits=True))
    assert report["tokens_equal"]
    assert report["max_logit_diff"] < 1e-9


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 1e-3), (torch.float16, 0.1)])
def test_cached_equals_reference_lower_precision(dtype, tolerance):
    model = tiny_model(dtype)
    ids, mask = single(32)
    report = compare(decode_uncached(model, ids, mask, 16, keep_logits=True), decode_cached(model, ids, mask, 16, keep_logits=True))
    assert report["tokens_equal"]
    assert report["max_logit_diff"] < tolerance


def test_cached_feeds_one_token_and_cache_grows(model):
    ids, mask = single(10)
    seen = []
    handle = model.register_forward_pre_hook(lambda _, args, kwargs: seen.append(kwargs["input_ids"].shape[1]), with_kwargs=True)
    try:
        cached = decode_cached(model, ids, mask, 5)
        assert seen == [10, 1, 1, 1, 1] == cached.forward_tokens
        seen.clear()
        reference = decode_uncached(model, ids, mask, 5)
        assert seen == [10, 11, 12, 13, 14] == reference.forward_tokens
    finally:
        handle.remove()
    assert cached.cache_lengths == [10, 11, 12, 13, 14]
    assert reference.cache_lengths == []
    # 3 layers, K and V, batch 1, 2 KV heads, 14 columns, head dim 12, 8-byte values.
    assert cached.cache_bytes == 3 * 2 * 2 * 14 * 12 * 8


def test_explicit_mask_matches_library_mask(model):
    ids, mask = left_pad([prompt(9), prompt(5, seed=2)], PAD, "cpu")
    cached = decode_cached(model, ids, mask, 2, keep_logits=True)
    cache = DynamicCache(config=model.config)
    with torch.inference_mode():
        first = model(input_ids=ids, attention_mask=mask, position_ids=positions_from_mask(mask), past_key_values=cache, use_cache=True).logits[:, -1]
        grown = torch.cat([mask, torch.ones(2, 1, dtype=mask.dtype)], dim=1)
        second = model(input_ids=first.argmax(-1, keepdim=True), attention_mask=grown, position_ids=positions_from_mask(grown)[:, -1:], past_key_values=cache, use_cache=True).logits[:, -1]
    assert (cached.logits[:, 0] - first).abs().max() < 1e-9
    assert (cached.logits[:, 1] - second).abs().max() < 1e-9


@pytest.mark.parametrize("decode", [decode_uncached, decode_cached])
def test_left_padded_batch_matches_rows_alone(model, decode):
    lengths = [3, 17, 40]
    ids, mask = left_pad([prompt(n, seed=n) for n in lengths], PAD, "cpu")
    batch = decode(model, ids, mask, 12, keep_logits=True)
    assert not batch.logits.isnan().any()
    for row, n in enumerate(lengths):
        alone = decode(model, *single(n, seed=n), 12, keep_logits=True)
        assert batch.tokens[row].tolist() == alone.tokens[0].tolist()
        assert (batch.logits[row] - alone.logits[0]).abs().max() < 1e-9


def test_wrong_positions_are_detected(model):
    """Negative control: without position handling the logits change, so the equality tests can fail."""
    ids, mask = single(12)
    cache = DynamicCache(config=model.config)
    with torch.inference_mode():
        first = model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True).logits[:, -1]
        token = first.argmax(-1, keepdim=True)
        grown = torch.cat([mask, torch.ones(1, 1, dtype=mask.dtype)], dim=1)
        right = model(input_ids=token, attention_mask=grown, position_ids=torch.tensor([[12]]), past_key_values=cache, use_cache=True).logits[:, -1]
        cache.crop(-1)
        wrong = model(input_ids=token, attention_mask=grown, position_ids=torch.tensor([[0]]), past_key_values=cache, use_cache=True).logits[:, -1]
    assert (right - wrong).abs().max() > 1e-6


@pytest.mark.parametrize("decode", [decode_uncached, decode_cached])
def test_stop_token_ends_row_and_pads(model, decode):
    ids, mask = left_pad([prompt(8, seed=3), prompt(8, seed=4)], PAD, "cpu")
    free = decode(model, ids, mask, 12).tokens
    stop = int(free[0, 2])
    if stop in free[0, :2].tolist():
        pytest.skip("stop token repeats early")
    stopped = decode(model, ids, mask, 12, stop_ids=[stop], pad_id=PAD).tokens
    assert stopped[0, :3].tolist() == free[0, :3].tolist()
    assert set(stopped[0, 3:].tolist()) <= {PAD}
    hit = [i for i, t in enumerate(free[1].tolist()) if t == stop]
    keep = hit[0] + 1 if hit else stopped.shape[1]
    assert stopped[1, :keep].tolist() == free[1, :keep].tolist()


def test_all_rows_stopping_ends_early(model):
    ids, mask = single(8, seed=3)
    free = decode_cached(model, ids, mask, 12).tokens
    stop = int(free[0, 0])
    result = decode_cached(model, ids, mask, 12, stop_ids=[stop], pad_id=PAD)
    assert result.tokens.shape == (1, 1)
    assert result.forward_tokens == [8]


def test_matches_library_generate(model):
    ids, mask = left_pad([prompt(6, seed=5), prompt(11, seed=6)], PAD, "cpu")
    cached = decode_cached(model, ids, mask, 10)
    model.generation_config.eos_token_id = None
    with torch.inference_mode():
        library = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=10, pad_token_id=PAD, do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
    assert library[:, ids.shape[1]:].tolist() == cached.tokens.tolist()


def test_compare_reports_first_divergence():
    def result(tokens, logits=None):
        return DecodeResult(torch.tensor(tokens), 0.0, 0.0, [], [], 0, logits)

    assert compare(result([[1, 2, 3]]), result([[1, 2, 3]]))["tokens_equal"]
    report = compare(result([[1, 2, 3]]), result([[1, 9, 3]]))
    assert not report["tokens_equal"] and report["first_divergence"] == 1
    assert not compare(result([[1, 2, 3]]), result([[1, 2]]))["tokens_equal"]


def test_compare_reports_logits_at_divergence():
    def result(token, logits):
        return DecodeResult(torch.tensor([[0, token]]), 0.0, 0.0, [], [], 0, torch.tensor([[[5.0, 1.0, 0.0], logits]]))

    report = compare(result(1, [0.0, 2.0, 1.75]), result(2, [0.0, 1.875, 2.0]))
    assert report["first_divergence"] == 1 and report["max_logit_diff"] == 0.0
    assert report["divergence"] == {
        "row": 0, "reference_token": 1, "cached_token": 2, "max_logit_diff": 0.25,
        "reference_path_logits": [2.0, 1.75], "cached_path_logits": [1.875, 2.0],
    }


def test_rejects_no_output(model):
    ids, mask = single(4)
    with pytest.raises(ValueError):
        decode_cached(model, ids, mask, 0)
