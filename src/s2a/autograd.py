"""A tiny automatic-differentiation engine for single numbers.

Every `Value` remembers which values it was computed from and how to pass a
gradient back to them. Calling `backward()` on a result walks that record in
reverse and applies the chain rule, giving the derivative of the result with
respect to every value that fed into it. This is backpropagation; PyTorch does
the same thing on whole tensors.

The design follows the well-known scalar-autograd pattern popularized by
Andrej Karpathy's micrograd.
"""

from __future__ import annotations

import math
import random


class Value:
    def __init__(self, data: float, parents: tuple["Value", ...] = (), op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0                     # d(final result) / d(this value)
        self._parents = parents
        self._op = op
        self._backward = lambda: None       # pushes self.grad to the parents

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- operations: each computes its result and records its local derivative ---

    def __add__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def backward():
            self.grad += out.grad           # d(a+b)/da = 1
            other.grad += out.grad          # d(a+b)/db = 1
        out._backward = backward
        return out

    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def backward():
            self.grad += other.data * out.grad   # d(a*b)/da = b
            other.grad += self.data * out.grad   # d(a*b)/db = a
        out._backward = backward
        return out

    def __pow__(self, k: float) -> "Value":
        out = Value(self.data ** k, (self,), f"**{k}")
        def backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = backward
        return out

    def exp(self) -> "Value":
        out = Value(math.exp(self.data), (self,), "exp")
        def backward():
            self.grad += out.data * out.grad
        out._backward = backward
        return out

    def log(self) -> "Value":
        out = Value(math.log(self.data), (self,), "log")
        def backward():
            self.grad += out.grad / self.data
        out._backward = backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "relu")
        def backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = backward
        return out

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * (other ** -1 if isinstance(other, Value) else 1 / other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return (-self) + other
    def __rtruediv__(self, other): return Value(other) * self ** -1

    # --- backpropagation ---

    def backward(self) -> None:
        """Fill in .grad for every value this one depends on."""
        order: list[Value] = []
        seen: set[int] = set()
        stack = [(self, False)]
        while stack:                         # topological sort, without recursion
            node, done = stack.pop()
            if done:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in node._parents)
        for node in order:
            node.grad = 0.0
        self.grad = 1.0                      # d(self)/d(self)
        for node in reversed(order):         # from the result back to the inputs
            node._backward()


class Neuron:
    def __init__(self, n_in: int, rng: random.Random, activation: bool = True) -> None:
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        total = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return total.tanh() if self.activation else total

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class MLP:
    """A small fully connected network: tanh hidden layers, linear output."""

    def __init__(self, n_in: int, layer_sizes: list[int], seed: int = 0) -> None:
        rng = random.Random(seed)
        sizes = [n_in] + layer_sizes
        self.layers = [
            [Neuron(sizes[i], rng, activation=i < len(layer_sizes) - 1) for _ in range(sizes[i + 1])]
            for i in range(len(layer_sizes))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = [neuron(x) for neuron in layer]
        return x[0] if len(x) == 1 else x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for neuron in layer for p in neuron.parameters()]


def fit(model: MLP, xs: list[list[float]], ys: list[float], steps: int = 200, lr: float = 0.1) -> list[float]:
    """Plain gradient descent on mean squared error. Returns the loss at each step."""
    losses = []
    for _ in range(steps):
        loss = sum(((model(x) - y) ** 2 for x, y in zip(xs, ys)), Value(0.0)) / len(xs)   # forward
        loss.backward()                                                                    # backward
        for p in model.parameters():                                                       # update
            p.data -= lr * p.grad
        losses.append(loss.data)
    return losses
