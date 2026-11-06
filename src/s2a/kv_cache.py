"""KV-cache sizing.

During generation a transformer stores the key and value vectors of every
previous token, for every layer, so it doesn't recompute them. Per token:

    bytes = 2 (K and V) * layers * kv_heads * head_dim * bytes_per_element
"""

from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_ELEMENT = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
    "fp4": 0.5,
    "int4": 0.5,
}


@dataclass(frozen=True)
class ModelConfig:
    """The attention shape of a model, as far as the KV cache is concerned."""

    name: str
    layers: int
    kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        for field in ("layers", "kv_heads", "head_dim"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")


# Llama 3 8B: 32 layers, 32 query heads, 8 KV heads (GQA), head_dim 128.
LLAMA3_8B = ModelConfig("llama-3-8b", layers=32, kv_heads=8, head_dim=128)


def bytes_per_token(model: ModelConfig, dtype: str = "bf16") -> float:
    """KV-cache bytes needed to store one token across all layers."""
    try:
        size = BYTES_PER_ELEMENT[dtype]
    except KeyError:
        raise ValueError(
            f"unknown dtype {dtype!r}; choose from {sorted(BYTES_PER_ELEMENT)}"
        ) from None
    return 2 * model.layers * model.kv_heads * model.head_dim * size


def kv_cache_bytes(
    model: ModelConfig,
    seq_len: int,
    batch_size: int = 1,
    dtype: str = "bf16",
) -> float:
    """Total KV-cache bytes for `batch_size` sequences of `seq_len` tokens."""
    if seq_len < 0 or batch_size < 0:
        raise ValueError("seq_len and batch_size must be non-negative")
    return bytes_per_token(model, dtype) * seq_len * batch_size


def max_tokens_in_memory(
    model: ModelConfig, memory_bytes: float, dtype: str = "bf16"
) -> int:
    """How many tokens (summed over all sequences) fit in `memory_bytes`."""
    return int(memory_bytes // bytes_per_token(model, dtype))


def human_bytes(n: float) -> str:
    """Format a byte count with binary units, e.g. 1.0 GiB."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024
    raise AssertionError("unreachable")
