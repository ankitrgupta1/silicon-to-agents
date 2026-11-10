import math

import numpy as np
import pytest

from s2a import lm
from s2a.info import cross_entropy, entropy, kl_divergence, sample_counts, surprise, to_bits


def test_entropy_of_fair_coin_is_one_bit():
    assert to_bits(entropy([0.5, 0.5])) == pytest.approx(1.0)


def test_entropy_extremes():
    assert entropy([1.0, 0.0, 0.0]) == 0.0
    assert entropy(np.full(8, 1 / 8)) == pytest.approx(math.log(8))   # uniform is the maximum


def test_cross_entropy_is_at_least_entropy():
    p, q = [0.7, 0.2, 0.1], [0.4, 0.4, 0.2]
    assert cross_entropy(p, q) > entropy(p)
    assert cross_entropy(p, p) == pytest.approx(entropy(p))


def test_kl_properties():
    p, q = np.array([0.7, 0.2, 0.1]), np.array([0.4, 0.4, 0.2])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert kl_divergence(p, q) > 0
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))   # not symmetric


def test_zero_probability_on_real_outcome_is_infinite():
    assert cross_entropy([0.5, 0.5], [1.0, 0.0]) == math.inf


def test_matches_logit_based_cross_entropy():
    logits = np.array([2.0, 0.5, -1.0])
    q = lm.softmax(logits)
    one_hot = np.array([0.0, 1.0, 0.0])
    assert cross_entropy(one_hot, q) == pytest.approx(lm.cross_entropy(logits[None], np.array([1])))


def test_surprise_and_sampling():
    assert surprise(1.0) == 0.0
    assert surprise(0.5) == pytest.approx(math.log(2))
    counts = sample_counts([0.2, 0.8], 10_000, rng=np.random.default_rng(0))
    assert counts.sum() == 10_000 and abs(counts[1] / 10_000 - 0.8) < 0.02


def test_rejects_non_distributions():
    with pytest.raises(ValueError):
        entropy([0.5, 0.6])
    with pytest.raises(ValueError):
        surprise(0.0)
