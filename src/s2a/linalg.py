"""Linear-algebra helpers used across the knowledge base.

Small, readable NumPy versions of the ideas on the linear-algebra page:
similarity between vectors, low-rank approximation with the SVD, and the
parameter arithmetic behind LoRA.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors: 1 same direction, 0 unrelated, -1 opposite."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def low_rank_approx(M: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Best rank-`rank` approximation of M, returned as two thin factors (A, B) with M ~ A @ B.

    Uses the SVD: keep the `rank` largest singular values and drop the rest.
    A has shape (rows, rank) and B has shape (rank, cols).
    """
    if rank < 1 or rank > min(M.shape):
        raise ValueError(f"rank must be between 1 and {min(M.shape)}")
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return U[:, :rank] * S[:rank], Vt[:rank]


def relative_error(M: np.ndarray, approx: np.ndarray) -> float:
    """How far off an approximation is, as a fraction of M's size (Frobenius norm)."""
    return float(np.linalg.norm(M - approx) / np.linalg.norm(M))


def lora_params(d_out: int, d_in: int, rank: int) -> int:
    """Trainable numbers in a rank-`rank` LoRA update for a (d_out x d_in) weight matrix."""
    return rank * (d_out + d_in)
