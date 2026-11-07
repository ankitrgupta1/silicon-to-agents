import numpy as np
import pytest

from s2a.positions import apply_rope, rope_frequencies, sinusoidal_positions

rng = np.random.default_rng(0)


def test_frequencies_start_at_one_and_decrease():
    f = rope_frequencies(8)
    assert f[0] == 1.0
    assert np.all(np.diff(f) < 0)
    assert rope_frequencies(8, base=500_000)[-1] < f[-1]  # larger base -> slower pairs


def test_rope_preserves_length():
    x = rng.normal(size=(6, 16))
    assert np.allclose(np.linalg.norm(apply_rope(x), axis=1), np.linalg.norm(x, axis=1))


def test_position_zero_is_identity():
    x = rng.normal(size=(1, 8))
    assert np.allclose(apply_rope(x, positions=[0]), x)


def test_dot_product_depends_only_on_offset():
    q, k = rng.normal(size=(1, 32)), rng.normal(size=(1, 32))
    def score(m, n):
        return float((apply_rope(q, [m]) @ apply_rope(k, [n]).T).item())
    assert score(5, 2) == pytest.approx(score(105, 102))
    assert score(0, 7) == pytest.approx(score(40, 47))
    assert score(5, 2) != pytest.approx(score(5, 3))


def test_sinusoidal_shape_and_range():
    pe = sinusoidal_positions(50, 16)
    assert pe.shape == (50, 16)
    assert np.all(np.abs(pe) <= 1)
    assert np.allclose(pe[0, 0::2], 0) and np.allclose(pe[0, 1::2], 1)


def test_rejects_odd_dimension():
    with pytest.raises(ValueError):
        rope_frequencies(7)
    with pytest.raises(ValueError):
        apply_rope(np.ones((2, 5)))
