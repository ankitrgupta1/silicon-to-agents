"""The language-modeling objective: next-token prediction and its metrics.

A language model outputs a score (logit) for every token in the vocabulary.
Softmax turns the scores into probabilities. Training minimizes cross-entropy:
the average negative log-probability the model gave to the token that actually
came next. Perplexity is exp(cross-entropy).
"""

from __future__ import annotations

import math

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Probabilities over the last axis. Subtracts the max for numerical stability."""
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean negative log-likelihood, in nats.

    logits: (n, vocab) scores, one row per position.
    targets: (n,) the token that actually came next at each position.
    """
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("expected logits of shape (n, vocab) and targets of shape (n,)")
    lp = log_softmax(logits)
    return float(-lp[np.arange(len(targets)), targets].mean())


def perplexity(loss_nats: float) -> float:
    """exp(loss): the effective number of choices the model is unsure between."""
    return math.exp(loss_nats)


def nats_to_bits(loss_nats: float) -> float:
    return loss_nats / math.log(2)


def bits_per_byte(loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    """Tokenizer-independent loss: total bits spent divided by bytes of text."""
    return nats_to_bits(loss_nats) * n_tokens / n_bytes


class BigramModel:
    """The simplest language model: P(next token | current token), by counting.

    `smoothing` adds a pseudo-count to every pair so unseen pairs don't get
    probability zero (which would make the loss infinite).
    """

    def __init__(self, vocab_size: int, smoothing: float = 1.0) -> None:
        if vocab_size <= 0 or smoothing < 0:
            raise ValueError("vocab_size must be positive and smoothing non-negative")
        self.vocab_size = vocab_size
        self.counts = np.full((vocab_size, vocab_size), float(smoothing))

    def fit(self, ids: list[int]) -> "BigramModel":
        for a, b in zip(ids, ids[1:]):
            self.counts[a, b] += 1
        return self

    def logits(self, ids: list[int]) -> np.ndarray:
        """Log-probabilities of the next token after each position, shape (n, vocab)."""
        probs = self.counts / self.counts.sum(axis=1, keepdims=True)
        return np.log(probs[np.asarray(ids)])

    def loss(self, ids: list[int]) -> float:
        """Cross-entropy of predicting ids[1:] from ids[:-1]."""
        return cross_entropy(self.logits(ids[:-1]), np.asarray(ids[1:]))

    def sample(self, start: int, n: int, rng: np.random.Generator | None = None,
               temperature: float = 1.0) -> list[int]:
        rng = rng or np.random.default_rng()
        out = [start]
        for _ in range(n):
            p = softmax(self.logits([out[-1]])[0] / temperature)
            out.append(int(rng.choice(self.vocab_size, p=p)))
        return out
