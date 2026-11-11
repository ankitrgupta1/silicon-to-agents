import numpy as np
import pytest

from s2a.prompting import (
    END,
    allowed_next,
    constrained_decode,
    greedy_decode,
    prefill_tokens,
    remaining_context,
    render_chatml,
    toy_scorer,
)

VOCAB = list("abcdefghijklmnopqrstuvwxyz ") + [END]
LABELS = ["positive", "negative", "neutral"]


def test_chatml_layout():
    s = render_chatml([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}])
    assert s == ("<|im_start|>system\nBe brief.<|im_end|>\n"
                 "<|im_start|>user\nHi<|im_end|>\n"
                 "<|im_start|>assistant\n")
    assert not render_chatml([{"role": "user", "content": "Hi"}], add_generation_prompt=False).endswith("assistant\n")
    with pytest.raises(ValueError):
        render_chatml([{"role": "narrator", "content": "x"}])


def test_allowed_next():
    ids = allowed_next("ne", LABELS, VOCAB)
    assert sorted(VOCAB[i] for i in ids) == ["g", "u"]
    assert VOCAB[allowed_next("neutral", LABELS, VOCAB)[0]] == END


@pytest.mark.parametrize("seed", range(20))
def test_constrained_output_is_always_valid(seed):
    assert constrained_decode(toy_scorer(VOCAB, seed), VOCAB, LABELS) in LABELS


def test_unconstrained_output_is_usually_invalid():
    outs = [greedy_decode(toy_scorer(VOCAB, s), VOCAB, max_len=12) for s in range(20)]
    assert sum(o in LABELS for o in outs) == 0


def test_constraint_follows_the_model_preference():
    score = toy_scorer(VOCAB, 0, bias={"n": 5.0, "g": 5.0})   # leans toward "n", then "g": only "negative" fits
    assert constrained_decode(score, VOCAB, LABELS) == "negative"


def test_multi_character_tokens():
    vocab = ['{"s": "', "pos", "neg", "itive", "ative", '"}', END]
    options = ['{"s": "positive"}', '{"s": "negative"}']
    assert constrained_decode(toy_scorer(vocab, 3), vocab, options) in options


def test_prefill_tokens():
    assert prefill_tokens(6000, 500, 100, cached=False) == 650_000
    assert prefill_tokens(6000, 500, 100, cached=True) == 56_000
    assert prefill_tokens(6000, 500, 0, cached=True) == 0


def test_remaining_context():
    assert remaining_context(128_000, system=2_000, tools=6_000, history=30_000, reserved_output=4_000) == 86_000
    with pytest.raises(ValueError):
        remaining_context(8_000, system=5_000, tools=4_000)


def test_decoder_requires_end_token():
    with pytest.raises(ValueError):
        constrained_decode(lambda p: np.zeros(3), ["a", "b", "c"], ["a"])
