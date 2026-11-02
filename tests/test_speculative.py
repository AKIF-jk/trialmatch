"""Speculative decoder tests on small random Qwen2 models. They need torch, so they run on a CPU kernel and skip locally.

Every test compares with decode_cached from the same model, so the expected tokens come from the Phase 3 greedy path.
Stub drafts script how many proposals agree with that greedy output, which covers always-agree, always-disagree and mixed.
"""
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from trialmatch.attention.hf import use_tiled_attention
from trialmatch.decoding.experiment import left_pad
from trialmatch.decoding.greedy import compare, decode_cached
from trialmatch.decoding.speculative import ModelDraft, decode_speculative

PAD, VOCAB = 0, 97


def tiny_model(seed, hidden=48, layers=3, dtype=torch.float64):
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=512, use_sliding_window=False, attn_implementation="sdpa",
    )
    return Qwen2ForCausalLM(config).to(dtype).eval()


@pytest.fixture(scope="module")
def target():
    return tiny_model(7)


@pytest.fixture(scope="module")
def small():
    return tiny_model(11, hidden=32, layers=2)


def single(length, seed=1):
    prompt = torch.randint(1, VOCAB, (length,), generator=torch.Generator().manual_seed(seed))
    return left_pad([prompt], PAD, "cpu")


class Scripted:
    """Stub draft: agrees with the reference greedy output for the first agree(round, k) proposals of each round, then is wrong."""

    def __init__(self, reference, agree):
        self.reference, self.agree, self.round, self.prompt_length = reference, agree, 0, None

    def start(self, prompt_ids):
        self.prompt_length, self.round = prompt_ids.shape[1], 0

    def propose(self, seq, k):
        first = seq.numel() - self.prompt_length  # index in the reference of the token that follows seq
        agree = self.agree(self.round, k)
        self.round += 1
        out = []
        for i in range(k):
            right = int(self.reference[first + i])
            out.append(right if i < agree else (right + 1) % VOCAB)
        return torch.tensor(out)

    def truncate(self, valid):
        pass


def greedy(model, ids, mask, n, stops=(), logits=False):
    return decode_cached(model, ids, mask, n, stop_ids=stops, pad_id=PAD, keep_logits=logits)


def expected_rounds(new_tokens, k):
    return math.ceil((new_tokens - 1) / (k + 1))


@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("new_tokens", [2, 9, 40])
def test_agreeing_draft_accepts_everything(target, k, new_tokens):
    ids, mask = single(12)
    reference = greedy(target, ids, mask, new_tokens)
    result = decode_speculative(target, Scripted(reference.tokens[0], lambda r, kk: kk), ids, mask, new_tokens, k)
    assert result.tokens.tolist() == reference.tokens.tolist()
    assert len(result.rounds) == expected_rounds(new_tokens, k) == result.target_forwards - 1
    assert result.accepted == result.proposed and result.acceptance_rate in (1.0, None)
    assert all(r["emitted"] == r["k"] + 1 for r in result.rounds)


@pytest.mark.parametrize("k", [1, 4, 8])
def test_disagreeing_draft_accepts_nothing(target, k):
    ids, mask = single(12)
    reference = greedy(target, ids, mask, 20)
    result = decode_speculative(target, Scripted(reference.tokens[0], lambda r, kk: 0), ids, mask, 20, k)
    assert result.tokens.tolist() == reference.tokens.tolist()
    assert result.accepted == 0 and result.acceptance_rate == 0.0
    assert [r["emitted"] for r in result.rounds] == [1] * 19
    assert result.target_forwards == 20  # no fewer than plain greedy, one per token


@pytest.mark.parametrize("k", [2, 4, 8])
def test_mixed_draft_accepts_the_scripted_prefix(target, k):
    ids, mask = single(10, seed=3)
    reference = greedy(target, ids, mask, 37)
    agree = lambda r, kk: (3 * r + 1) % (kk + 1)
    result = decode_speculative(target, Scripted(reference.tokens[0], agree), ids, mask, 37, k)
    assert result.tokens.tolist() == reference.tokens.tolist()
    proposing = [r for r in result.rounds if r["k"]]
    assert [r["accepted"] for r in proposing] == [min(agree(i, r["k"]), r["k"]) for i, r in enumerate(proposing)]
    assert [r["emitted"] for r in result.rounds] == [r["accepted"] + 1 for r in result.rounds]
    assert 0 < result.accepted < result.proposed


@pytest.mark.parametrize("k", [0, 1, 2, 4, 8])
@pytest.mark.parametrize("length,new_tokens", [(4, 1), (4, 2), (16, 3), (16, 24), (64, 33)])
def test_random_draft_model_equals_greedy_float64(target, small, k, length, new_tokens):
    ids, mask = single(length)
    reference = greedy(target, ids, mask, new_tokens, logits=True)
    result = decode_speculative(target, ModelDraft(small, target.config), ids, mask, new_tokens, k, keep_logits=True)
    report = compare(reference, result)
    assert report["tokens_equal"], report
    assert report["max_logit_diff"] < 1e-9


def test_target_as_its_own_draft_is_always_accepted(target):
    ids, mask = single(20)
    result = decode_speculative(target, ModelDraft(target), ids, mask, 30, 4)
    assert result.tokens.tolist() == greedy(target, ids, mask, 30).tokens.tolist()
    assert result.acceptance_rate == 1.0
    assert result.target_forwards == 1 + expected_rounds(30, 4)


@pytest.mark.parametrize("k", [1, 3, 8])
@pytest.mark.parametrize("position", [0, 1, 2, 5, 11])
def test_stop_token_ends_output_wherever_it_appears(target, small, k, position):
    ids, mask = single(14, seed=5)
    free = greedy(target, ids, mask, 40).tokens[0]
    stop = int(free[position])
    if stop in free[:position].tolist():
        pytest.skip("stop token appears earlier")
    expected = greedy(target, ids, mask, 40, stops=[stop])
    assert expected.tokens.shape[1] == position + 1
    for draft in (Scripted(free, lambda r, kk: kk), ModelDraft(small), Scripted(free, lambda r, kk: 0)):
        result = decode_speculative(target, draft, ids, mask, 40, k, stop_ids=[stop])
        assert result.tokens.tolist() == expected.tokens.tolist()


def test_target_pass_sizes_and_cache_follow_the_accepted_text(target, small):
    ids, mask = single(10)
    seen = []

    def record(module, args, kwargs):
        seen.append((kwargs["past_key_values"].get_seq_length(), kwargs["input_ids"].shape[1]))

    handle = target.register_forward_pre_hook(record, with_kwargs=True)
    try:
        result = decode_speculative(target, ModelDraft(small), ids, mask, 25, 4)
    finally:
        handle.remove()
    assert seen[0] == (0, 10)
    total = 10 + 1  # prompt plus the first token
    for (cached, fed), record_ in zip(seen[1:], result.rounds):
        assert (cached, fed) == (total - 1, record_["k"] + 1)
        total += record_["emitted"]
    assert len(seen) == result.target_forwards == 1 + len(result.rounds)
    assert total - 10 == result.tokens.shape[1]


def test_draft_forwards_are_counted(target, small):
    ids, mask = single(10)
    draft = ModelDraft(small)
    result = decode_speculative(target, draft, ids, mask, 25, 4)
    # One prefill, then per round: the unread tokens in one pass and k - 1 single-token steps.
    assert result.draft_forwards == draft.forwards == 1 + sum(r["k"] for r in result.rounds)
    assert draft.length <= 10 + result.tokens.shape[1]


def test_no_crop_is_detected(target, small, monkeypatch):
    """Negative control: with cropping off the cache check must fail, so the cache tests can fail."""
    ids, mask = single(10)
    monkeypatch.setattr(DynamicCache, "crop", lambda self, n: None)
    # The oversized cache fails either at the decoder's length check or earlier, when sdpa sees a mask shorter than the keys.
    with pytest.raises((AssertionError, RuntimeError)):
        decode_speculative(target, ModelDraft(small), ids, mask, 25, 4)


def test_tiled_attention_target_equals_greedy():
    target, small = tiny_model(7), tiny_model(11, hidden=32, layers=2)
    ids, mask = single(40)
    reference = greedy(target, ids, mask, 20, logits=True)
    use_tiled_attention(target, 16)
    result = decode_speculative(target, ModelDraft(small), ids, mask, 20, 4, keep_logits=True)
    report = compare(reference, result)
    assert report["tokens_equal"] and report["max_logit_diff"] < 1e-9


def test_float32_equals_greedy():
    target, small = tiny_model(7, dtype=torch.float32), tiny_model(11, hidden=32, layers=2, dtype=torch.float32)
    ids, mask = single(32)
    reference = greedy(target, ids, mask, 24, logits=True)
    result = decode_speculative(target, ModelDraft(small), ids, mask, 24, 4, keep_logits=True)
    report = compare(reference, result)
    assert report["tokens_equal"] and report["max_logit_diff"] < 1e-3


def test_rejects_batches_padding_and_empty_output(target, small):
    ids, mask = left_pad([torch.arange(1, 6), torch.arange(1, 4)], PAD, "cpu")
    with pytest.raises(ValueError):
        decode_speculative(target, ModelDraft(small), ids, mask, 5, 2)
    with pytest.raises(ValueError):
        decode_speculative(target, ModelDraft(small), ids[:1], mask[1:], 5, 2)
    ids, mask = single(4)
    with pytest.raises(ValueError):
        decode_speculative(target, ModelDraft(small), ids, mask, 0, 2)
    with pytest.raises(ValueError):
        decode_speculative(target, ModelDraft(small), ids, mask, 5, -1)


def test_draft_with_other_vocab_is_rejected(target):
    config = Qwen2Config(vocab_size=VOCAB + 1, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2)
    with pytest.raises(ValueError):
        ModelDraft(Qwen2ForCausalLM(config), target.config)


def test_result_statistics(target):
    ids, mask = single(8)
    reference = greedy(target, ids, mask, 13)
    result = decode_speculative(target, Scripted(reference.tokens[0], lambda r, kk: kk), ids, mask, 13, 3)
    assert result.tokens_per_round == 4.0 and result.summary()["rounds"] == 3
    assert result.cache_bytes > 0 and result.prefill_seconds > 0 and result.decode_seconds > 0
