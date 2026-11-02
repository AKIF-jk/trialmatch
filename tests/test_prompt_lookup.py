"""Prompt-lookup draft tests. They need torch, so they run on a CPU kernel and skip locally.

Unit tests pin down the matching rule. End-to-end tests check that tokens still equal decode_cached when the draft proposes
fewer than k tokens or none, on a small random Qwen2 and on prompts that repeat, so that real matches happen.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Qwen2Config, Qwen2ForCausalLM

from trialmatch.decoding.experiment import left_pad
from trialmatch.decoding.greedy import compare, decode_cached
from trialmatch.decoding.prompt_lookup import PromptLookupDraft
from trialmatch.decoding.speculative import decode_speculative

PAD, VOCAB = 0, 97


def propose(tokens, k, **options):
    return PromptLookupDraft(**options).propose(torch.tensor(tokens), k).tolist()


def test_proposes_what_followed_the_earlier_match():
    assert propose([1, 2, 3, 4, 9, 1, 2, 3], 2) == [4, 9]
    assert propose([1, 2, 3, 4, 9, 1, 2, 3], 8) == [4, 9, 1, 2, 3]


def test_no_match_proposes_nothing():
    assert propose([1, 2, 3, 4], 4) == []
    assert propose([5], 4) == []


def test_longer_ngram_wins_over_a_shorter_one():
    # The trigram 1 2 3 matches at the start and is followed by 4. The last token 3 alone was most recently followed by 8.
    tokens = [1, 2, 3, 4, 9, 3, 8, 0, 1, 2, 3]
    assert propose(tokens, 1, max_ngram=3) == [4]
    assert propose(tokens, 1, max_ngram=1) == [8]


def test_most_recent_match_is_used():
    assert propose([1, 2, 5, 1, 2, 6, 0, 1, 2], 1, max_ngram=2) == [6]


def test_min_ngram_blocks_weak_matches():
    assert propose([1, 2, 3, 9, 3], 1, min_ngram=2) == []
    assert propose([1, 2, 3, 9, 3], 1, min_ngram=1) == [9]


def test_proposal_is_cut_at_the_end_of_the_text():
    assert propose([4, 5, 6, 4, 5], 5) == [6, 4, 5]


def test_dtype_and_device_follow_the_input():
    out = PromptLookupDraft().propose(torch.tensor([1, 2, 1], dtype=torch.long), 2)
    assert out.dtype == torch.long and out.tolist() == [2, 1]


def test_rejects_bad_ngram_range():
    with pytest.raises(ValueError):
        PromptLookupDraft(max_ngram=1, min_ngram=2)


def tiny_model(dtype=torch.float64):
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=VOCAB, hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=512, use_sliding_window=False, attn_implementation="sdpa",
    )
    return Qwen2ForCausalLM(config).to(dtype).eval()


@pytest.fixture(scope="module")
def model():
    return tiny_model()


def prompt(kind, length):
    if kind == "periodic":
        return left_pad([torch.arange(1, 8).repeat(length // 7 + 1)[:length]], PAD, "cpu")
    return left_pad([torch.randint(1, VOCAB, (length,), generator=torch.Generator().manual_seed(3))], PAD, "cpu")


@pytest.mark.parametrize("kind", ["periodic", "random"])
@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("new_tokens", [1, 2, 9, 40])
def test_tokens_equal_greedy(model, kind, k, new_tokens):
    ids, mask = prompt(kind, 28)
    reference = decode_cached(model, ids, mask, new_tokens, pad_id=PAD, keep_logits=True)
    result = decode_speculative(model, PromptLookupDraft(), ids, mask, new_tokens, k, keep_logits=True)
    report = compare(reference, result)
    assert report["tokens_equal"], report
    assert report["max_logit_diff"] < 1e-9
    assert all(0 <= r["k"] <= k and r["accepted"] <= r["k"] and r["emitted"] == r["accepted"] + 1 for r in result.rounds)


def random_prompt(seed, length=28):
    return left_pad([torch.randint(1, VOCAB, (length,), generator=torch.Generator().manual_seed(seed))], PAD, "cpu")


def test_round_records_and_real_matches_across_prompts(model):
    """Whether a given random prompt yields matches depends on the model, so look across seeds for both kinds of round."""
    saw_empty = saw_accept = False
    for seed in range(40):
        ids, mask = random_prompt(seed)
        result = decode_speculative(model, PromptLookupDraft(), ids, mask, 40, 8)
        assert result.target_forwards == 1 + len(result.rounds)
        assert result.proposed == sum(r["k"] for r in result.rounds)
        assert all(r["emitted"] == r["accepted"] + 1 for r in result.rounds)
        assert all(r["emitted"] == 1 for r in result.rounds if r["k"] == 0)
        saw_empty = saw_empty or any(r["k"] == 0 for r in result.rounds)
        saw_accept = saw_accept or result.accepted > 0
        if saw_empty and saw_accept:
            return
    pytest.skip(f"no prompt gave both an empty round and an accepted proposal (empty {saw_empty}, accepted {saw_accept})")


@pytest.mark.parametrize("position", [0, 1, 3, 11])
def test_stop_token_with_lookup(model, position):
    ids, mask = prompt("periodic", 28)
    free = decode_cached(model, ids, mask, 40, pad_id=PAD).tokens[0]
    stop = int(free[position])
    if stop in free[:position].tolist():
        pytest.skip("stop token appears earlier")
    expected = decode_cached(model, ids, mask, 40, stop_ids=[stop], pad_id=PAD).tokens
    result = decode_speculative(model, PromptLookupDraft(), ids, mask, 40, 8, stop_ids=[stop])
    assert result.tokens.tolist() == expected.tolist()


class Overlong:
    def start(self, prompt_ids):
        pass

    def truncate(self, valid):
        pass

    def propose(self, seq, k):
        return torch.zeros(k + 1, dtype=torch.long)


def test_draft_proposing_more_than_asked_is_rejected(model):
    ids, mask = prompt("random", 12)
    with pytest.raises(ValueError):
        decode_speculative(model, Overlong(), ids, mask, 10, 4)
