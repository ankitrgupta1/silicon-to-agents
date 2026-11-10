---
id: prereqs.calculus-backprop
title: Calculus & Backprop
layer: prereqs
pathways: [fundamentals]
level: intro
prereqs: []
status: stable
last_reviewed: 2026-11-10
review_every_days: 365
---

# Calculus & Backprop

## TL;DR

A **gradient** says, for every weight in a model, which way and how strongly to change it to reduce the loss. **Backpropagation** computes all of those at once: run the model forward while recording each operation, then walk the record backward applying the chain rule. One backward pass costs about as much as one forward pass, no matter how many weights there are. That efficiency is what makes training neural networks possible.

## Intuition

**A derivative is a sensitivity.** If you nudge an input up a tiny bit, how much does the output move? If the output moves three times as much, the derivative is 3. If it moves the opposite way, the derivative is negative.

**The chain rule multiplies sensitivities.** Suppose a change in the temperature changes ice-cream sales, and ice-cream sales change profit. If sales are 3× as sensitive to temperature and profit is 2× as sensitive to sales, profit is 6× as sensitive to temperature. A neural network is a very long chain of simple steps like this.

**Backprop is bookkeeping for the chain rule.** Every operation in a model (add, multiply, tanh) has a simple local derivative. Backprop starts from the loss and works backward, handing each operation the message "here's how much the loss cares about your output", which it converts into "so here's how much it cares about each of your inputs".

## How it works

Take $L = (a \cdot b + c) \cdot a$ with $a = 2$, $b = -3$, $c = 10$.

**Forward pass:** compute and remember each intermediate value.

```text
d = a * b      = -6
e = d + c      =  4
L = e * a      =  8
```

**Backward pass:** start with $\partial L / \partial L = 1$ and go through the steps in reverse. Each operation has a local rule:

- **Multiply** ($z = x \cdot y$): each input's gradient is the *other* input's value, times the gradient coming in.
- **Add** ($z = x + y$): the incoming gradient passes through unchanged to both inputs.

```text
L = e * a   ->  grad e = a     = 2          grad a += e = 4
e = d + c   ->  grad d = 2                  grad c  = 2
d = a * b   ->  grad a += b * 2 = -6        grad b  = a * 2 = 4
```

Final gradients: $\partial L/\partial a = 4 - 6 = -2$, $\partial L/\partial b = 4$, $\partial L/\partial c = 2$.

Note that $a$ is used twice, so it receives gradient along two paths, and they **add**. This rule (sum over all paths) is why residual connections work so well: the `x + f(x)` in a [transformer block](../04-architectures/transformer.md) gives the gradient a direct path straight through.

**Why backward and not forward?** You could instead nudge each weight separately and rerun the model: one extra run *per weight*. Going backward from the single loss value gives the gradient for *every* weight in one pass. With billions of weights, that's the difference between possible and impossible.

**Using the gradient.** Gradient descent updates each weight a small step against its gradient:

$$
w \leftarrow w - \eta \, \frac{\partial L}{\partial w}
$$

where $\eta$ is the learning rate. Repeat, and the loss goes down. That's [training](../05-training/tiny-gpt.md).

**Vectors and matrices.** Real frameworks do the same thing on whole tensors instead of single numbers. The rules look the same, with matrix shapes to keep track of: for $Y = XW$, the gradient for $W$ is $X^\top$ times the incoming gradient.

## Code

[`s2a.autograd`](../src/s2a/autograd.py) is a complete automatic-differentiation engine for single numbers in about 100 lines, plus a small neural network built on it. Its gradients are tested against numerical nudging. The [backprop lab](../labs/backprop.ipynb) walks through it.

```python
from s2a.autograd import Value

a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = (a * b + c) * a
L.backward()
a.grad, b.grad, c.grad      # (-2.0, 4.0, 2.0)
```

In PyTorch the same idea is `loss.backward()`, applied to tensors.

## Numbers that matter

From the lab, training a 17-weight network on XOR for 300 steps:

| Learning rate | Result |
|---|---|
| 0.01 | Loss 0.237: barely moved |
| 0.05 | Loss 0.120: learning, slowly |
| 0.2 | Loss ≈ 0: solved |
| 0.5 | Diverged after 90 steps |

- **The learning rate has a narrow useful range.** Too small wastes compute; too large blows up. This is why real training uses warmup, schedules and gradient clipping.
- **Cost:** a backward pass takes roughly as much computation as a forward pass, so a training step costs a small multiple of running the model once.
- **Memory:** the backward pass needs the intermediate values from the forward pass, so training must keep them all around. This is the main reason training needs far more memory than inference, and why large-scale training uses tricks like activation checkpointing (see [02 Distributed](../02-distributed/distributed.md)).

## Current consensus (as of 2026-11)

- Reverse-mode automatic differentiation (backprop) is how essentially all neural networks are trained. Frameworks such as PyTorch and JAX do it for you; nobody derives gradients by hand for real models.
- Understanding it still matters in practice: it explains vanishing and exploding gradients, why residual connections and normalization help, and where training memory goes.

## What changed

- 2026-11: First version of this page.

## Open questions

- None for the basics. Research on training without backprop exists, but nothing has come close to replacing it.

## Resources

- Karpathy, [*The spelled-out intro to neural networks and backpropagation: building micrograd*](https://www.youtube.com/watch?v=VMj-3S1tku0) and [micrograd](https://github.com/karpathy/micrograd), which this page's code follows in design (F)
- 3Blue1Brown, [*Neural networks*](https://www.3blue1brown.com/topics/neural-networks) video series, chapters on gradient descent and backpropagation (F)
- Rumelhart, Hinton & Williams, [*Learning representations by back-propagating errors*](https://www.nature.com/articles/323533a0) (1986) (P)
- Baydin et al., [*Automatic Differentiation in Machine Learning: a Survey*](https://arxiv.org/abs/1502.05767) (2018) (P)
- Olah, [*Calculus on Computational Graphs: Backpropagation*](https://colah.github.io/posts/2015-08-Backprop/) (F)

## Check yourself

Try each one before opening the answer.

**1. For $L = (x + y) \cdot y$ with $x = 1$, $y = 3$, find $\partial L/\partial x$ and $\partial L/\partial y$ by walking backward.**

<details>
<summary>Answer</summary>

Forward: $s = x + y = 4$, $L = s \cdot y = 12$.

Backward: from the multiply, grad $s$ = $y$ = 3, and $y$ gets $s$ = 4. From the add, the 3 passes through to both $x$ and $y$.

So $\partial L/\partial x = 3$ and $\partial L/\partial y = 4 + 3 = 7$ (two paths, added). Check: $L = xy + y^2$, so $\partial L/\partial y = x + 2y = 7$.

</details>

**2. Why does the engine use `+=` when storing gradients instead of `=`?**

<details>
<summary>Answer</summary>

A value can feed into several later operations. The loss depends on it through every one of those paths, and the total derivative is the sum over all paths. With `=`, each path would overwrite the previous one and only the last would count. A test in the repo checks this: for `y = a + a`, the gradient of `a` must be 2, not 1.

</details>

**3. A model has 1 billion weights. Roughly how many forward passes would it take to get the full gradient by nudging each weight, versus by backprop?**

<details>
<summary>Answer</summary>

Nudging: about one extra forward pass per weight, so around 1 billion passes for a single training step. Backprop: one forward pass plus one backward pass that costs about the same, so roughly two passes' worth of compute. The gap grows with the number of weights, which is why backprop is the only practical option.

</details>
