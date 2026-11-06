import pytest

from s2a.kv_cache import (
    LLAMA3_8B,
    ModelConfig,
    bytes_per_token,
    human_bytes,
    kv_cache_bytes,
    max_tokens_in_memory,
)

GiB = 1024**3


def test_llama3_8b_per_token_is_128_kib():
    assert bytes_per_token(LLAMA3_8B) == 128 * 1024


def test_llama3_8b_8k_context_is_1_gib():
    assert kv_cache_bytes(LLAMA3_8B, seq_len=8192) == 1 * GiB


def test_scales_linearly_with_batch_and_length():
    one = kv_cache_bytes(LLAMA3_8B, seq_len=1000)
    assert kv_cache_bytes(LLAMA3_8B, seq_len=2000, batch_size=3) == 6 * one


def test_mha_vs_gqa_vs_mqa():
    mha = ModelConfig("mha", layers=32, kv_heads=32, head_dim=128)
    mqa = ModelConfig("mqa", layers=32, kv_heads=1, head_dim=128)
    assert bytes_per_token(mha) == 4 * bytes_per_token(LLAMA3_8B)
    assert bytes_per_token(mqa) == bytes_per_token(LLAMA3_8B) / 8


def test_lower_precision_shrinks_cache():
    assert bytes_per_token(LLAMA3_8B, "fp8") == bytes_per_token(LLAMA3_8B) / 2
    assert bytes_per_token(LLAMA3_8B, "fp4") == bytes_per_token(LLAMA3_8B) / 4


def test_max_tokens_in_memory():
    assert max_tokens_in_memory(LLAMA3_8B, 1 * GiB) == 8192


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        bytes_per_token(LLAMA3_8B, "fp7")
    with pytest.raises(ValueError):
        ModelConfig("bad", layers=0, kv_heads=8, head_dim=128)
    with pytest.raises(ValueError):
        kv_cache_bytes(LLAMA3_8B, seq_len=-1)


@pytest.mark.parametrize(
    "n, expected",
    [(512, "512 B"), (128 * 1024, "128.0 KiB"), (GiB, "1.0 GiB"), (4 * GiB, "4.0 GiB")],
)
def test_human_bytes(n, expected):
    assert human_bytes(n) == expected
