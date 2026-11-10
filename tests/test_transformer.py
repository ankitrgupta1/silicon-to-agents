import math

import numpy as np
import pytest

from s2a.lm import cross_entropy
from s2a.transformer import (
    LLAMA3_8B,
    TransformerConfig,
    count_params,
    forward,
    init_params,
    rms_norm,
)

TINY = TransformerConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64)


def test_llama3_8b_param_count():
    assert LLAMA3_8B.param_count() == 8_030_261_248
    b = LLAMA3_8B.param_breakdown()
    assert b["mlp"] / LLAMA3_8B.param_count() == pytest.approx(0.702, abs=0.001)


def test_formula_matches_actual_arrays():
    assert count_params(init_params(TINY)) == TINY.param_count()
    tied = TransformerConfig(50, 32, 2, 4, 2, 64, tie_embeddings=True)
    assert count_params(init_params(tied)) == tied.param_count()


def test_forward_shape():
    logits = forward(init_params(TINY), TINY, [1, 2, 3, 4, 5])
    assert logits.shape == (5, 50)
    assert np.all(np.isfinite(logits))


def test_causal_future_tokens_do_not_change_the_past():
    p = init_params(TINY)
    a = forward(p, TINY, [1, 2, 3, 4, 5])
    b = forward(p, TINY, [1, 2, 3, 9, 9])
    assert np.allclose(a[:3], b[:3])
    assert not np.allclose(a[3:], b[3:])


def test_order_matters():
    p = init_params(TINY)
    a = forward(p, TINY, [1, 2, 3])[-1]
    b = forward(p, TINY, [2, 1, 3])[-1]
    assert not np.allclose(a, b)


def test_untrained_loss_is_near_log_vocab():
    ids = list(np.random.default_rng(1).integers(0, 50, size=40))
    loss = cross_entropy(forward(init_params(TINY), TINY, ids[:-1]), np.array(ids[1:]))
    assert loss == pytest.approx(math.log(50), abs=0.1)


def test_rms_norm_gives_unit_rms():
    x = np.random.default_rng(0).normal(size=(3, 16)) * 50
    y = rms_norm(x, np.ones(16))
    assert np.allclose(np.sqrt((y * y).mean(-1)), 1, atol=1e-4)


def test_config_validation():
    with pytest.raises(ValueError):
        TransformerConfig(50, 30, 1, 4, 2, 64)      # d_model not divisible by heads
    with pytest.raises(ValueError):
        TransformerConfig(50, 32, 1, 4, 3, 64)      # heads not divisible by KV heads
