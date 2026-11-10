"""A modern decoder-only transformer, forward pass only, in NumPy.

"Modern" means the Llama-style recipe most open models share: pre-norm with
RMSNorm, grouped-query attention with RoPE, and a SwiGLU MLP. There is no
training here (that needs autograd); the goal is to make the data flow and
the parameter count easy to read and check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from s2a.lm import softmax
from s2a.positions import apply_rope


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    tie_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head dimension must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def param_breakdown(self) -> dict[str, int]:
        """Where the parameters live (no biases, as in Llama-style models)."""
        d, kv = self.d_model, self.n_kv_heads * self.head_dim
        attention = d * d + 2 * d * kv + d * d          # W_q, W_k, W_v, W_o
        mlp = 3 * d * self.d_ff                          # gate, up, down
        norms = 2 * d
        embed = self.vocab_size * d
        return {
            "embedding": embed,
            "attention": self.n_layers * attention,
            "mlp": self.n_layers * mlp,
            "norms": self.n_layers * norms + d,          # + final norm
            "output_head": 0 if self.tie_embeddings else embed,
        }

    def param_count(self) -> int:
        return sum(self.param_breakdown().values())


# Llama 3 8B, from its published config.
LLAMA3_8B = TransformerConfig(
    vocab_size=128_256, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8, d_ff=14_336
)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Scale each vector to unit root-mean-square, then apply a learned gain."""
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


def swiglu(x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray) -> np.ndarray:
    """The gated MLP: one branch (through SiLU) decides how much of the other passes."""
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down


def attention(x: np.ndarray, layer: dict, cfg: TransformerConfig) -> np.ndarray:
    """Causal grouped-query attention with RoPE. x: (n, d_model)."""
    n, hd = x.shape[0], cfg.head_dim
    q = (x @ layer["w_q"]).reshape(n, cfg.n_heads, hd)
    k = (x @ layer["w_k"]).reshape(n, cfg.n_kv_heads, hd)
    v = (x @ layer["w_v"]).reshape(n, cfg.n_kv_heads, hd)
    group = cfg.n_heads // cfg.n_kv_heads
    future = np.triu(np.ones((n, n), dtype=bool), k=1)
    out = np.empty((n, cfg.n_heads, hd))
    for h in range(cfg.n_heads):
        qh = apply_rope(q[:, h])
        kh = apply_rope(k[:, h // group])               # query heads in a group share K and V
        scores = qh @ kh.T / np.sqrt(hd)
        scores = np.where(future, -np.inf, scores)
        out[:, h] = softmax(scores) @ v[:, h // group]
    return out.reshape(n, cfg.d_model) @ layer["w_o"]


def block(x: np.ndarray, layer: dict, cfg: TransformerConfig) -> np.ndarray:
    """One pre-norm block: each sublayer reads a normalized copy and adds its result back."""
    x = x + attention(rms_norm(x, layer["norm1"]), layer, cfg)
    x = x + swiglu(rms_norm(x, layer["norm2"]), layer["w_gate"], layer["w_up"], layer["w_down"])
    return x


def init_params(cfg: TransformerConfig, rng: np.random.Generator | None = None) -> dict:
    rng = rng or np.random.default_rng(0)
    d, kv = cfg.d_model, cfg.n_kv_heads * cfg.head_dim
    w = lambda *shape: rng.normal(scale=0.02, size=shape)
    params = {
        "embedding": w(cfg.vocab_size, d),
        "layers": [
            {
                "norm1": np.ones(d), "w_q": w(d, d), "w_k": w(d, kv), "w_v": w(d, kv), "w_o": w(d, d),
                "norm2": np.ones(d), "w_gate": w(d, cfg.d_ff), "w_up": w(d, cfg.d_ff), "w_down": w(cfg.d_ff, d),
            }
            for _ in range(cfg.n_layers)
        ],
        "final_norm": np.ones(d),
    }
    params["output_head"] = params["embedding"] if cfg.tie_embeddings else w(cfg.vocab_size, d)
    return params


def count_params(params: dict) -> int:
    """Count the numbers actually stored (a tied output head is counted once)."""
    seen, total = set(), 0
    def visit(obj):
        nonlocal total
        if isinstance(obj, np.ndarray):
            if id(obj) not in seen:
                seen.add(id(obj)); total += obj.size
        elif isinstance(obj, dict):
            for v in obj.values(): visit(v)
        elif isinstance(obj, list):
            for v in obj: visit(v)
    visit(params)
    return total


def forward(params: dict, cfg: TransformerConfig, ids: list[int]) -> np.ndarray:
    """Token ids (n,) -> next-token logits (n, vocab_size)."""
    x = params["embedding"][np.asarray(ids)]             # look up one vector per token
    for layer in params["layers"]:
        x = block(x, layer, cfg)
    return rms_norm(x, params["final_norm"]) @ params["output_head"].T
