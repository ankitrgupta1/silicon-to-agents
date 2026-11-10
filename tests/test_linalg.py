import numpy as np
import pytest

from s2a.linalg import cosine_similarity, lora_params, low_rank_approx, relative_error

rng = np.random.default_rng(0)


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine_similarity([1, 0], [0, 5]) == pytest.approx(0)
    assert cosine_similarity([1, 2], [-2, -4]) == pytest.approx(-1)
    assert cosine_similarity([1, 2, 3], [10, 20, 30]) == pytest.approx(1)   # length doesn't matter


def test_low_rank_recovers_a_truly_low_rank_matrix():
    M = rng.normal(size=(30, 4)) @ rng.normal(size=(4, 20))                 # rank 4 by construction
    A, B = low_rank_approx(M, 4)
    assert A.shape == (30, 4) and B.shape == (4, 20)
    assert relative_error(M, A @ B) < 1e-10
    assert relative_error(M, np.matmul(*low_rank_approx(M, 2))) > 0.1


def test_error_matches_dropped_singular_values():
    M = rng.normal(size=(12, 9))
    S = np.linalg.svd(M, compute_uv=False)
    A, B = low_rank_approx(M, 3)
    assert np.linalg.norm(M - A @ B) == pytest.approx(np.sqrt((S[3:] ** 2).sum()))


def test_error_decreases_with_rank():
    M = rng.normal(size=(10, 10))
    errs = [relative_error(M, np.matmul(*low_rank_approx(M, r))) for r in range(1, 11)]
    assert all(a > b for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 1e-10


def test_lora_params():
    assert lora_params(4096, 4096, 16) == 131_072
    assert lora_params(4096, 4096, 16) / (4096 * 4096) == pytest.approx(0.0078125)


def test_rank_validation():
    with pytest.raises(ValueError):
        low_rank_approx(np.ones((3, 4)), 4)
