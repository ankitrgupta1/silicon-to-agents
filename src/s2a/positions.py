"""Positional encodings: sinusoidal (absolute) and RoPE (rotary).

Attention on its own has no notion of order: shuffle the tokens and each one
gets the same output. Positional encodings put order back in.

- Sinusoidal (Vaswani et al., 2017) adds a fixed vector to each token's
  embedding, built from sines and cosines of different frequencies.
- RoPE (Su et al., 2021) instead rotates each query and key. Pairs of
  dimensions are rotated by an angle proportional to the token's position, so
  the dot product between a query at position m and a key at position n
  depends only on the offset m - n.
"""

from __future__ import annotations

import numpy as np


def _check_even(d: int) -> None:
    if d <= 0 or d % 2:
        raise ValueError(f"dimension must be a positive even number, got {d}")


def rope_frequencies(d: int, base: float = 10_000.0) -> np.ndarray:
    """Rotation speed (radians per position) of each of the d/2 dimension pairs.

    theta_i = base ** (-2i / d): the first pair spins fast (1 radian per
    position), later pairs progressively slower. A larger `base` makes the
    slowest pairs slower still, which helps with long contexts.
    """
    _check_even(d)
    return base ** (-np.arange(0, d, 2) / d)


def apply_rope(x: np.ndarray, positions: np.ndarray | None = None, base: float = 10_000.0) -> np.ndarray:
    """Rotate vectors x of shape (n, d) by their positions.

    Uses the "interleaved" convention: dimensions (0, 1), (2, 3), ... form the
    rotated pairs. Some libraries pair dimension i with i + d/2 instead; the
    maths and properties are the same.
    """
    x = np.asarray(x, dtype=float)
    n, d = x.shape
    _check_even(d)
    pos = np.arange(n) if positions is None else np.asarray(positions, dtype=float)
    angles = np.outer(pos, rope_frequencies(d, base))          # (n, d/2)
    cos, sin = np.cos(angles), np.sin(angles)
    x_even, x_odd = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x_even * cos - x_odd * sin
    out[:, 1::2] = x_even * sin + x_odd * cos
    return out


def sinusoidal_positions(n: int, d: int, base: float = 10_000.0) -> np.ndarray:
    """The original transformer's absolute encodings, shape (n, d).

    Even dimensions get sin(pos * theta_i), odd dimensions cos(pos * theta_i),
    with the same frequencies as RoPE. These are added to token embeddings.
    """
    angles = np.outer(np.arange(n), rope_frequencies(d, base))
    pe = np.empty((n, d))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe
