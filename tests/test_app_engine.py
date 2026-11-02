"""Engine tests on small random Qwen2 models in float64 with a stand-in tokenizer. They need torch, so they run on a CPU kernel."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen2Config, Qwen2ForCausalLM

from trialmatch.app import screen
from trialmatch.app.engine import Engine

VOCAB, PAD, STOP = 97, 0, 1
SENTENCES = ["Man with diabetes.", "No pregnancy.", "Takes metformin."]


class Encoding(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return Encoding({k: v.to(device) for k, v in self.items()})


class FakeTokenizer:
    """Characters become ids 2..91. Id 0 pads (as <|endoftext|>) and id 1 is the stop token (as <|im_end|>)."""

    pad_token_id = PAD

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": STOP, "<|endoftext|>": PAD}[token]

    def _ids(self, text):
        return [2 + ord(c) % 90 for c in text]

    def __call__(self, texts, return_tensors=None, padding=False):
        if isinstance(texts, str):
            return Encoding(input_ids=self._ids(texts))
        rows = [self._ids(t) for t in texts]
        width = max(map(len, rows))
        ids = torch.tensor([[PAD] * (width - len(r)) + r for r in rows])
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])
        return Encoding(input_ids=ids, attention_mask=mask)

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in tokens if not (skip_special_tokens and int(t) in (PAD, STOP)))


def tiny(seed, hidden=48, layers=3):
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=VOCAB, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=2048, use_sliding_window=False, attn_implementation="sdpa",
    )
    return Qwen2ForCausalLM(config).to(torch.float64).eval()


@pytest.fixture(scope="module")
def engine():
    return Engine(FakeTokenizer(), tiny(1), {"base": tiny(2, 32, 2), "LoRA": tiny(3, 32, 2)}, "cpu", "float64", base_loader=lambda: tiny(4))


def test_generate_texts_batch_equals_single(engine):
    prompts = [engine.prompt(SENTENCES[:n], "inclusion", f"criterion {n}") for n in (1, 2, 3)]
    batch, counts = engine.generate_texts(engine.target, prompts, 12, batch_size=3)
    single = [engine.generate_texts(engine.target, [p], 12)[0][0] for p in prompts]
    assert batch == single and all(1 <= c <= 12 for c in counts)


def test_generate_texts_keeps_input_order_across_batches(engine):
    prompts = [engine.prompt(SENTENCES, "inclusion", "x" * n) for n in (30, 5, 20, 10, 25)]
    small, _ = engine.generate_texts(engine.target, prompts, 8, batch_size=2)
    big, _ = engine.generate_texts(engine.target, prompts, 8, batch_size=5)
    assert small == big


def test_screen_with_random_model_never_raises(engine):
    results, timing = engine.screen(SENTENCES, [("inclusion", "Diabetes"), ("exclusion", "Pregnancy")])
    assert [r.index for r in results] == [0, 1] and all(r.status in ("unparsed", "invalid_label", "ok", "repaired") for r in results)
    assert timing["criteria"] == 2 and len(timing["prompt_tokens"]) == 2 and timing["labelling_seconds"] > 0


def test_screen_reads_scripted_json(engine, monkeypatch):
    scripted = ['{"evidence_sentence_ids":[0,2],"label":"included"}', '{"evidence_sentence_ids":[9],"label":"not excluded"}']
    monkeypatch.setattr(engine, "generate_texts", lambda model, prompts, n, batch_size=8: (scripted, [10, 10]))
    results, _ = engine.screen(SENTENCES, [("inclusion", "Diabetes"), ("exclusion", "Pregnancy")])
    assert results[0].evidence_ids == [0, 2] and results[1].dropped_ids == [9] and results[1].evidence_ids == []
    assert screen.aggregate(results)[0] == screen.ELIGIBLE


def test_rationale_prompt_has_only_cited_sentences(engine, monkeypatch):
    seen = []

    def fake(model, prompts, n, batch_size=8):
        seen.extend(prompts)
        return ["because"] * len(prompts), [3] * len(prompts)

    monkeypatch.setattr(engine, "generate_texts", fake)
    results = [
        screen.interpret(0, "inclusion", "Diabetes", '{"evidence_sentence_ids":[0],"label":"included"}', 3),
        screen.interpret(1, "exclusion", "Pregnancy", '{"evidence_sentence_ids":[],"label":"not excluded"}', 3),
        screen.interpret(2, "inclusion", "Bad", "garbage", 3),
    ]
    engine.add_rationales(SENTENCES, results)
    assert len(seen) == 1 and SENTENCES[0] in seen[0] and SENTENCES[1] not in seen[0] and SENTENCES[2] not in seen[0]
    assert [r.rationale for r in results] == ["because", None, None]


MODES = ["no cache", "KV cache", "speculative"]


@pytest.mark.parametrize("backend,tile", [("sdpa", 64), ("tiled", 16)])
def test_lab_modes_match_cached_reference(engine, backend, tile):
    rows, text = engine.lab(SENTENCES, "inclusion", "Diabetes", MODES, backend=backend, tile=tile, k=3, new_tokens=12, force=True)
    assert [r["mode"] for r in rows] == MODES and all(r["tokens_equal"] and r["tokens"] == 12 for r in rows)
    spec = rows[2]
    assert 0 <= spec["acceptance"] <= 1 and spec["tokens_per_round"] >= 1 and spec["target_forwards"] >= 2
    assert all(r["total_seconds"] > 0 and r["tokens_per_second"] > 0 for r in rows)
    assert rows[0]["backend"] == ("sdpa" if backend == "sdpa" else f"tiled {tile}")
    assert isinstance(text, str)
    # The attention setting is restored after the run.
    assert engine.target.config._attn_implementation == "sdpa" and engine.drafts["LoRA"].config._attn_implementation == "sdpa"


def test_lab_repeats_and_caps(engine):
    rows, _ = engine.lab(SENTENCES, "exclusion", "Pregnancy", ["KV cache"], new_tokens=500, force=True, repeats=2)
    assert rows[0]["repeats"] == 2 and rows[0]["tokens"] == 64


def test_lab_rejects_no_modes(engine):
    with pytest.raises(ValueError):
        engine.lab(SENTENCES, "inclusion", "Diabetes", [])


def test_lab_row_table_and_commentary(engine):
    rows, _ = engine.lab(SENTENCES, "inclusion", "Diabetes", MODES, new_tokens=10, force=True)
    table = screen.lab_table(rows)
    assert len(table) == 3 and len(table[0]) == len(screen.LAB_HEADERS) and table[0][3] == "yes"
    text = screen.lab_commentary(rows)
    assert "same tokens" in text and "x the speed of KV cache" in text and "Cost per round" in text


def test_provenance(engine):
    prov = engine.provenance()
    assert prov["target_model"]["revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306" and prov["decoding"]["do_sample"] is False
    assert prov["dtype"] == "float64" and prov["device"] == "cpu" and "timestamp_utc" in prov and prov["packages"]["torch"]
