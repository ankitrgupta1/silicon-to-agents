---
id: prereqs.python-pytorch
title: Python & PyTorch
layer: prereqs
pathways: [fundamentals]
level: intro
prereqs: [prereqs.calculus-backprop]
status: stable
last_reviewed: 2026-11-10
review_every_days: 180
---

# Python & PyTorch

## TL;DR

PyTorch is the library almost all LLM research and training code is written in. You need five ideas to read that code: **tensors** (arrays with a shape), **broadcasting** (how shapes combine), **autograd** (automatic gradients), **modules** (weights plus a forward function), and the **training loop**. Everything else in this repo builds on these.

## Intuition

- A **tensor** is a grid of numbers: a list, a table, or a stack of tables. A batch of text inside a model is a tensor of shape (batch, tokens, width).
- You describe math on **whole tensors at once** instead of looping over numbers. That's shorter to write and far faster, because the work happens in optimized native code, or on a GPU.
- PyTorch quietly **records every operation**, so it can run [backprop](backprop.md) for you. You write the forward pass; the gradients come for free.

## How it works

### Tensors and shapes

Every tensor has a **shape** (its size along each dimension), a **dtype** (what kind of number: `float32` by default, `int64` for token IDs) and a **device** (CPU or a GPU). Most bugs in model code are shape bugs, so read and print `.shape` constantly.

### Matrix multiply and broadcasting

`A @ B` is matrix multiplication: shapes `(n, d) @ (d, k)` give `(n, k)`. With extra leading dimensions, it applies to each matrix in the batch.

For element-wise operations (`+`, `*`, …) between different shapes, PyTorch **broadcasts**: line the shapes up from the right; two dimensions are compatible if they're equal or one of them is 1; a 1 is stretched to match.

```text
(3, 1) + (4,)      ->  (3, 4)
(8, 16, 32) + (32,) -> (8, 16, 32)    # add a bias to every token
(3, 4) + (3,)      ->  error: 4 and 3 don't match
```

Broadcasting is convenient and also the classic source of silent bugs, when two shapes combine in a way you didn't intend.

### Autograd

Create a tensor with `requires_grad=True` (model weights have this by default) and PyTorch tracks everything computed from it. Calling `.backward()` on a single-number result fills in `.grad` on every tracked tensor.

Two details matter in practice:

- **Gradients accumulate.** A second `backward()` *adds* to `.grad`. Training loops clear them each step with `optimizer.zero_grad()`.
- **Turn tracking off when you don't need it.** Wrap evaluation and generation in `torch.no_grad()` to save memory and time.

### Modules

An `nn.Module` bundles parameters with a `forward` function. Modules contain other modules (a model contains blocks, which contain attention and an MLP), and `.parameters()` collects every weight in the tree.

```python
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.out = nn.Linear(4, 1)

    def forward(self, x):
        return self.out(torch.tanh(self.hidden(x)))
```

### The training loop

The same five lines appear in every training script, from this toy to a frontier model:

```python
pred = model(x)                    # forward
loss = loss_fn(pred, y)            # how wrong?
optimizer.zero_grad()              # clear old gradients
loss.backward()                    # backward: fill in .grad
optimizer.step()                   # update the weights
```

### Devices

`model.to("cuda")` and `x.to("cuda")` move the model and data to a GPU. Everything taking part in one operation has to be on the same device.

## Code

The [PyTorch basics lab](../labs/pytorch-basics.ipynb) runs each idea above, with checks built in. It reproduces the worked example from [Calculus & backprop](backprop.md) and confirms PyTorch's gradients match this repo's own [`s2a.autograd`](../src/s2a/autograd.py) engine. For a complete model written this way, read [`s2a.gpt`](../src/s2a/gpt.py).

## Numbers that matter

- **Counting parameters:** `nn.Linear(2, 4)` has 2 × 4 weights + 4 biases = 12; `nn.Linear(4, 1)` has 5. The `TinyNet` above has 17 in total, the same network as in the backprop lab.
- **Bytes per number:** `float32` is 4 bytes; `bfloat16` and `float16` are 2. A model's memory is roughly parameters × bytes per number, so an 8-billion-parameter model is about 32 GB in `float32` and 16 GB in `bfloat16`. The same arithmetic drives the [KV cache](../06-inference/kv-cache.md).
- **Vectorization:** one tensor operation is typically hundreds of times faster than the equivalent Python loop. The lab measures the gap on your machine.

## Beyond the basics

Things you'll meet in real training code, covered where they come up:

- **Mixed precision** (`torch.autocast`): compute in 16-bit floats for speed and memory.
- **`torch.compile`:** traces and optimizes a model for faster execution (see [01 Systems](../01-systems/systems.md)).
- **Data loading, checkpointing and distributed training** (see [02 Distributed](../02-distributed/distributed.md)).
- **Profiling:** finding where time and memory go.

## Current consensus (as of 2026-11)

- PyTorch is the default framework for LLM research and most training and inference code. JAX is the main alternative, used heavily at Google.
- For Apple hardware, MLX offers a similar style of API.

## What changed

- 2026-11: First version of this page.

## Open questions

- None for the basics.

## Resources

- PyTorch, [*Learn the Basics*](https://pytorch.org/tutorials/beginner/basics/intro.html) tutorial series (F)
- PyTorch docs, [*Broadcasting semantics*](https://pytorch.org/docs/stable/notes/broadcasting.html) and [*Autograd mechanics*](https://pytorch.org/docs/stable/notes/autograd.html) (F, P)
- Karpathy, [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html), which introduces PyTorch gradually while building models (F)
- Paszke et al., [*PyTorch: An Imperative Style, High-Performance Deep Learning Library*](https://arxiv.org/abs/1912.01703) (2019) (P)

## Check yourself

Try each one before opening the answer.

**1. What shape is `torch.randn(5, 1, 3) + torch.randn(4, 1)`? And why does `torch.randn(3, 4) + torch.randn(3)` fail?**

<details>
<summary>Answer</summary>

Line the shapes up from the right: `(5, 1, 3)` and `(4, 1)`. Last dimension: 3 and 1 → 3. Middle: 1 and 4 → 4. First: 5 and nothing → 5. Result: **`(5, 4, 3)`**.

For `(3, 4)` and `(3,)`, the last dimensions are 4 and 3: not equal and neither is 1, so it's an error. To add one value per *row*, reshape the second tensor to `(3, 1)`.

</details>

**2. You forget `optimizer.zero_grad()` in your training loop. What goes wrong?**

<details>
<summary>Answer</summary>

Gradients accumulate: each step's `.grad` becomes the sum of all gradients since the start, not just the current one. The updates point in a stale, ever-growing direction, so training becomes unstable or diverges. The lab shows the accumulation directly: two `backward()` calls on `w * w` at `w = 3` give a gradient of 12 instead of 6.

</details>

**3. How many parameters does `nn.Linear(4096, 14336, bias=False)` have, and how much memory is that in `bfloat16`?**

<details>
<summary>Answer</summary>

4,096 × 14,336 = 58,720,256 parameters. At 2 bytes each that's about 117 MB. This is the size of one of the three MLP matrices in each block of Llama 3 8B (see [the transformer block](../04-architectures/transformer.md)).

</details>
