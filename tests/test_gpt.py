import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from s2a.gpt import GPT, estimate_loss, generate, to_numpy_params, train  # noqa: E402
from s2a.transformer import TransformerConfig, forward  # noqa: E402

TINY = TransformerConfig(vocab_size=30, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64)


def test_param_count_matches_formula():
    assert GPT(TINY).num_params() == TINY.param_count()
    tied = TransformerConfig(30, 32, 2, 4, 2, 64, tie_embeddings=True)
    assert GPT(tied).num_params() == tied.param_count()


def test_matches_numpy_forward():
    torch.manual_seed(0)
    model = GPT(TINY).double().eval()
    ids = [3, 1, 4, 1, 5, 9, 2, 6]
    logits, _ = model(torch.tensor([ids]))
    reference = forward(to_numpy_params(model), TINY, ids)
    assert np.allclose(logits[0].detach().numpy(), reference, atol=1e-8)


def test_causal():
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    a, _ = model(torch.tensor([[1, 2, 3, 4, 5]]))
    b, _ = model(torch.tensor([[1, 2, 3, 9, 9]]))
    assert torch.allclose(a[0, :3], b[0, :3], atol=1e-6)
    assert not torch.allclose(a[0, 3:], b[0, 3:])


def test_training_reduces_loss():
    torch.manual_seed(0)
    text = "the cat sat on the mat. " * 60
    chars = sorted(set(text))
    data = torch.tensor([chars.index(c) for c in text])
    cfg = TransformerConfig(len(chars), 32, 2, 4, 2, 64)
    model = GPT(cfg)
    before = estimate_loss(model, data, batches=5, batch_size=16, context=32)
    assert before == pytest.approx(math.log(len(chars)), abs=0.3)
    losses = train(model, data, steps=150, batch_size=16, context=32, lr=3e-3)
    after = estimate_loss(model, data, batches=5, batch_size=16, context=32)
    print(f"\nloss before {before:.3f}, after 150 steps {after:.3f}")
    assert len(losses) == 150
    assert after < 0.5 * before


def test_generate():
    torch.manual_seed(0)
    out = generate(GPT(TINY), [1, 2], n=10, context=8)
    assert len(out) == 12 and out[:2] == [1, 2]
    assert all(0 <= t < TINY.vocab_size for t in out)
