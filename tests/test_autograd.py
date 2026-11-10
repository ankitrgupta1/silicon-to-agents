import math

import pytest

from s2a.autograd import MLP, Value, fit


def numeric_grad(f, args, i, h=1e-6):
    up = list(args); up[i] += h
    down = list(args); down[i] -= h
    return (f(*up) - f(*down)) / (2 * h)


def test_simple_chain_rule():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    L = (a * b + c) * a          # L = a^2 b + a c
    L.backward()
    assert L.data == pytest.approx(8.0)
    assert a.grad == pytest.approx(2 * 2 * -3 + 10)   # 2ab + c = -2
    assert b.grad == pytest.approx(4.0)               # a^2
    assert c.grad == pytest.approx(2.0)               # a


def test_value_used_twice_accumulates():
    a = Value(3.0)
    y = a + a
    y.backward()
    assert a.grad == pytest.approx(2.0)


def test_matches_numerical_gradients():
    def f(x, y, z):
        return ((x * y - z / x) ** 2 + (x + y).tanh() * z.exp() + (y * y + 1).log() - (z * 0.5).relu())

    def f_float(x, y, z):
        return f(Value(x), Value(y), Value(z)).data

    args = [1.3, -0.7, 0.4]
    vals = [Value(v) for v in args]
    f(*vals).backward()
    for i, v in enumerate(vals):
        assert v.grad == pytest.approx(numeric_grad(f_float, args, i), rel=1e-5, abs=1e-7)


def test_backward_twice_does_not_double_count():
    a = Value(2.0)
    y = a * a
    y.backward(); y.backward()
    assert a.grad == pytest.approx(4.0)


def test_deep_chain_does_not_hit_recursion_limit():
    x = Value(1.0)
    y = x
    for _ in range(5000):
        y = y + 1
    y.backward()
    assert x.grad == pytest.approx(1.0)


def test_mlp_learns_xor():
    xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    ys = [0, 1, 1, 0]
    model = MLP(2, [4, 1], seed=0)
    assert len(model.parameters()) == 17
    losses = fit(model, xs, ys, steps=300, lr=0.2)
    assert losses[0] > 0.2
    assert losses[-1] < 0.01
    assert [round(model(x).data) for x in xs] == ys
