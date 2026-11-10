---
id: prereqs.linear-algebra
title: Linear Algebra
layer: prereqs
pathways: [fundamentals]
level: intro
prereqs: []
status: stable
last_reviewed: 2026-11-10
review_every_days: 365
---

# Linear Algebra

## TL;DR

An LLM is mostly matrix multiplication. You need four ideas: a **vector** is a list of numbers that represents something; the **dot product** measures how much two vectors agree; a **matrix** transforms vectors, and multiplying matrices chains transformations; and many big matrices are secretly **low rank**, meaning they can be replaced by two much thinner ones. That last idea is what makes LoRA fine-tuning cheap.

## Intuition

- **Vectors are meanings as coordinates.** A model represents each token as a point in a space with thousands of dimensions. Direction carries meaning: tokens used in similar ways point in similar directions (see [Embeddings & positions](../03-foundations/embeddings-positions.md)).
- **A dot product asks "how aligned are these?"** It's large and positive when two vectors point the same way, near zero when they're unrelated, negative when they're opposed.
- **A matrix is a machine that moves vectors.** It can rotate, stretch, squash or project them. Every layer of a neural network applies one, then a simple non-linear function.
- **Low rank means "less going on than the size suggests."** A table with a million entries might really be built from a handful of patterns.

## How it works

### Vectors and dot products

For vectors $a$ and $b$ with $d$ entries each:

$$
a \cdot b = \sum_{i=1}^{d} a_i b_i = \lVert a \rVert \, \lVert b \rVert \cos\theta
$$

where $\lVert a \rVert$ is the vector's length and $\theta$ the angle between them. Dividing out the lengths gives **cosine similarity**, which only cares about direction. It's the standard way to compare embeddings in search and RAG.

[Attention](../03-foundations/attention.md) is built on this: each score is the dot product of a query and a key.

### Matrices and shapes

A matrix with shape $(n, d)$ has $n$ rows and $d$ columns. The rule that matters most:

$$
(n, d) \;@\; (d, k) \;\rightarrow\; (n, k)
$$

The inner sizes must match and they disappear; the outer sizes remain. Each entry of the result is a dot product of a row from the left with a column from the right. So a matrix product is a whole grid of dot products at once, which is exactly what GPUs are built to compute quickly.

Two properties to remember:

- **Composition:** applying $A$ and then $B$ is the same as applying the single matrix $BA$.
- **Order matters:** in general $AB \neq BA$.

### Why layers need non-linearities

Stacking matrices alone gains nothing: two in a row collapse into one ($B(Ax) = (BA)x$). Putting a non-linear function between them (ReLU, SiLU, tanh) is what lets a deep network represent things a single matrix can't.

### Rank and the SVD

A matrix's **rank** is the number of independent patterns in it. A $(1000, 1000)$ matrix whose rows are all mixtures of 5 patterns has rank 5, and can be written exactly as a $(1000, 5)$ matrix times a $(5, 1000)$ one: 10,000 numbers instead of a million.

The **singular value decomposition (SVD)** breaks any matrix into patterns ordered by importance, each with a strength called a singular value. Keeping only the top $r$ gives the best possible rank-$r$ approximation.

### LoRA

Fine-tuning changes a model's weights a little. LoRA bets that the *change* is low rank. It freezes the original matrix $W$ and learns two thin matrices instead:

$$
W' = W + BA, \qquad B: (d, r), \quad A: (r, k)
$$

Only $A$ and $B$ are trained. $B$ starts at zero, so training begins from exactly the original model. Afterward the update can be merged into $W$, so it adds no cost at inference.

## Code

[`s2a.linalg`](../src/s2a/linalg.py) has cosine similarity, low-rank approximation by SVD and LoRA's parameter arithmetic, tested in CI. The [linear algebra lab](../labs/linear-algebra.ipynb) runs each idea, with pictures.

```python
import numpy as np
from s2a.linalg import low_rank_approx, relative_error, lora_params

M = np.random.default_rng(0).normal(size=(200, 5)) @ np.random.default_rng(1).normal(size=(5, 200))
A, B = low_rank_approx(M, 5)        # two thin matrices: (200, 5) and (5, 200)
relative_error(M, A @ B)            # ~0: nothing lost

lora_params(4096, 4096, 16)         # 131,072
```

## Numbers that matter

**LoRA on one 4,096 × 4,096 weight matrix** (16.8 million numbers):

| Rank | Trainable numbers | Share of the full matrix |
|---|---|---|
| 4 | 32,768 | 0.20% |
| 16 | 131,072 | 0.78% |
| 64 | 524,288 | 3.12% |

**Low rank in the lab** (a 200 × 200 matrix, keeping rank 5, which stores 2,000 numbers instead of 40,000):

| Matrix | Error at rank 5 |
|---|---|
| Built from 5 patterns plus a little noise | 4% |
| Pure random noise | 95% |

Compression only works when there's structure to find. Random noise has none.

**Cost of a matrix product:** multiplying $(n, d)$ by $(d, k)$ takes about $2ndk$ arithmetic operations. This single operation dominates the compute of both training and inference, which is why the [hardware](../00-hardware/hardware.md) and [kernel](../01-systems/systems.md) layers are organized around making it fast.

## Current consensus (as of 2026-11)

- LoRA and its variants (QLoRA, DoRA) remain the standard way to fine-tune large models cheaply; see [05 Training](../05-training/training.md).
- Low-rank ideas show up across the stack: compressing keys and values in attention (MLA), and factorizing or pruning weights for inference.

## What changed

- 2026-11: First version of this page.

## Open questions

- Why are fine-tuning updates so often low rank in practice? There are partial explanations, but no complete theory.

## Resources

- 3Blue1Brown, [*Essence of Linear Algebra*](https://www.3blue1brown.com/topics/linear-algebra) video series: the best source for visual intuition (F)
- Deisenroth, Faisal & Ong, [*Mathematics for Machine Learning*](https://mml-book.github.io/) (free), chapters 2–4 (F)
- Strang, *Introduction to Linear Algebra*, and his MIT 18.06 lectures (F)
- Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685) (2021) (B, P)

## Check yourself

Try each one before opening the answer.

**1. Two embeddings have a large dot product but a cosine similarity of only 0.1. What does that tell you?**

<details>
<summary>Answer</summary>

The vectors are long but point in quite different directions. The dot product is the product of the two lengths and the cosine of the angle, so big lengths can produce a big dot product even when the angle is close to 90°. If you care about direction (meaning) and not magnitude, compare with cosine similarity, or normalize the vectors first.

</details>

**2. `X` has shape (32, 128) and `W` has shape (128, 512). What is the shape of `X @ W`, and roughly how many arithmetic operations does it take?**

<details>
<summary>Answer</summary>

Shape **(32, 512)**: the inner 128s match and disappear. Each of the 32 × 512 = 16,384 outputs is a dot product of 128 pairs (128 multiplies and about as many adds), so roughly 2 × 32 × 128 × 512 ≈ 4.2 million operations.

</details>

**3. A model has width 4,096. You apply rank-8 LoRA to the query and value matrices (each 4,096 × 4,096) in all 32 layers. How many trainable numbers is that?**

<details>
<summary>Answer</summary>

One rank-8 update for a 4,096 × 4,096 matrix has 8 × (4,096 + 4,096) = 65,536 numbers. Two matrices per layer × 32 layers = 64 updates, so 64 × 65,536 = **4,194,304**, about 4.2 million. Compare that with the 8 billion parameters of a model this size: roughly 0.05%.

(In a model with grouped-query attention the value matrix is smaller, so the real count is a little lower.)

</details>
