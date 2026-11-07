import math

import numpy as np
import pytest

from s2a.lm import (
    BigramModel,
    bits_per_byte,
    cross_entropy,
    nats_to_bits,
    perplexity,
    softmax,
)


def test_softmax_is_stable_and_sums_to_one():
    p = softmax(np.array([[1000.0, 1000.0, 999.0]]))
    assert np.isclose(p.sum(), 1) and np.all(np.isfinite(p))


def test_uniform_prediction_loss_is_log_vocab():
    v = 50_257
    loss = cross_entropy(np.zeros((3, v)), np.array([0, 1, 2]))
    assert loss == pytest.approx(math.log(v))
    assert perplexity(loss) == pytest.approx(v)


def test_confident_and_correct_is_near_zero_loss():
    logits = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    assert cross_entropy(logits, np.array([0, 1])) < 1e-6


def test_confident_and_wrong_is_expensive():
    logits = np.array([[20.0, 0.0, 0.0]])
    assert cross_entropy(logits, np.array([1])) > 19


def test_unit_conversions():
    assert nats_to_bits(math.log(2)) == pytest.approx(1)
    assert bits_per_byte(math.log(2), n_tokens=100, n_bytes=400) == pytest.approx(0.25)


def test_bigram_beats_uniform_and_learns_patterns():
    text = "abababababababab"
    ids = [ord(c) - ord("a") for c in text]
    m = BigramModel(vocab_size=3, smoothing=0.01).fit(ids)
    assert m.loss(ids) < math.log(3) / 10
    assert m.sample(0, 6, rng=np.random.default_rng(0)) == [0, 1, 0, 1, 0, 1, 0]


def test_smoothing_avoids_infinite_loss():
    m = BigramModel(vocab_size=3).fit([0, 1, 0, 1])
    assert math.isfinite(m.loss([0, 2]))


def test_shape_errors():
    with pytest.raises(ValueError):
        cross_entropy(np.zeros(5), np.array([0]))
    with pytest.raises(ValueError):
        BigramModel(vocab_size=0)
