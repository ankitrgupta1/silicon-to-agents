"""Probability and information theory for language models.

Entropy, cross-entropy and KL divergence on explicit probability
distributions. Logits-based helpers (softmax, cross-entropy from logits,
perplexity) live in `s2a.lm`; this module works with probabilities directly
so the definitions are easy to see.

All quantities are in nats (natural log) unless a function says otherwise.
"""

from __future__ import annotations

import math

import numpy as np


def _dist(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or np.any(p < 0) or not math.isclose(p.sum(), 1.0, abs_tol=1e-9):
        raise ValueError("expected a 1-D array of non-negative probabilities summing to 1")
    return p


def entropy(p) -> float:
    """Average surprise of a distribution: -sum p log p. Zero-probability outcomes add nothing."""
    p = _dist(p)
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum()) + 0.0


def cross_entropy(p, q) -> float:
    """Average surprise when outcomes come from p but you predicted q: -sum p log q."""
    p, q = _dist(p), _dist(q)
    nz = p > 0
    if np.any(q[nz] == 0):
        return math.inf
    return float(-(p[nz] * np.log(q[nz])).sum())


def kl_divergence(p, q) -> float:
    """Extra surprise from predicting q when the truth is p: cross_entropy(p, q) - entropy(p)."""
    return cross_entropy(p, q) - entropy(p)


def surprise(prob: float) -> float:
    """Surprise of one outcome, -log p. Probability 1 -> 0; small probabilities -> large."""
    if not 0 < prob <= 1:
        raise ValueError("probability must be in (0, 1]")
    return -math.log(prob) + 0.0          # + 0.0 turns -0.0 into 0.0


def to_bits(nats: float) -> float:
    return nats / math.log(2)


def sample_counts(p, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw n samples from p and count how often each outcome came up."""
    rng = rng or np.random.default_rng()
    return np.bincount(rng.choice(len(p), size=n, p=_dist(p)), minlength=len(p))
