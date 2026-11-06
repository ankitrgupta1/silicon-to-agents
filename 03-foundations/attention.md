---
id: foundations.attention
title: Attention
layer: 03-foundations
pathways: [fundamentals, builder, practitioner]
level: intro
prereqs: [prereqs.linear-algebra, prereqs.probability-information]
status: stable
last_reviewed: 2026-11-06
review_every_days: 180
---

# Attention

## TL;DR

Attention lets each token build its representation by taking a weighted average of information from other tokens, with weights computed from how well they match. It is the core operation of the transformer. The variants (MHA, MQA, GQA, MLA) mostly trade quality against the memory that inference must keep around: the KV cache.

## Intuition

Think of each token as asking a question (a **query**), advertising what it contains (a **key**), and carrying content (a **value**). A token compares its query against every key, turns the match scores into weights that sum to one, and reads out a blend of the values. "it" in *"The cat sat because it was tired"* learns to put most of its weight on "cat".

In a language model, a token may only look at itself and earlier tokens. This **causal mask** is what lets the model generate text one token at a time.

## How it works

For a sequence of $n$ token vectors stacked into $X \in \mathbb{R}^{n \times d}$, learned matrices project them into queries, keys and values:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

Scaled dot-product attention is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

- $QK^\top$ is an $n \times n$ table of match scores.
- Dividing by $\sqrt{d_k}$ keeps the scores from growing with the head dimension, which would push softmax into near one-hot outputs with tiny gradients.
- $M$ is the causal mask: $0$ where a token may look, $-\infty$ where it may not (future positions).
- Softmax turns each row into weights; multiplying by $V$ blends the values.

**Multi-head attention (MHA)** runs $h$ smaller attention operations in parallel, each with its own projections, so different heads can track different relationships (syntax, coreference, position). Their outputs are concatenated and projected back to size $d$.

### Why the variants exist: the KV cache

When generating, the model reuses the keys and values of all previous tokens instead of recomputing them. That stored data is the **KV cache**, and it often limits how many users or how much context a GPU can serve. Per token, per layer, it holds $2 \times (\text{number of KV heads}) \times d_{\text{head}}$ numbers.

| Variant | Idea | Effect on the KV cache |
|---|---|---|
| MHA | Every query head has its own K and V | Largest |
| MQA | All query heads share one K/V head | Smallest; can cost quality |
| GQA | Groups of query heads share a K/V head | Middle ground; standard in most open models |
| MLA | Compress K and V into a small shared latent vector, expand when needed | Very small while keeping quality |

## Code

A minimal causal attention head in NumPy:

```python
import numpy as np

def causal_attention(X, W_q, W_k, W_v):
    """X: (n, d). Returns (n, d_v)."""
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)                      # (n, n)
    mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)             # hide the future
    scores -= scores.max(axis=-1, keepdims=True)         # numerical stability
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V

rng = np.random.default_rng(0)
n, d, d_k = 5, 16, 8
X = rng.normal(size=(n, d))
out = causal_attention(X, *(rng.normal(size=(d, d_k)) for _ in range(3)))
print(out.shape)  # (5, 8)
```

## Numbers that matter

Using the [KV-cache calculator](../calculators/kv-cache.ipynb) with a Llama-3-8B-style configuration (32 layers, 32 query heads, head dimension 128, BF16):

| Attention type | KV heads | KV cache per token | For 8,192 tokens |
|---|---|---|---|
| MHA | 32 | 512 KiB | 4 GiB |
| GQA (what Llama 3 8B uses) | 8 | 128 KiB | 1 GiB |
| MQA | 1 | 16 KiB | 128 MiB |

Attention compute grows with the square of sequence length, which is why long context is expensive and why FlashAttention (see [01 Systems](../01-systems/systems.md)) avoids writing the $n \times n$ score table to slow memory.

## Current consensus (as of 2026-11)

- GQA is the default in dense open models. DeepSeek's MLA is the main alternative where KV-cache size matters most.
- Many 2025–26 models are **hybrids**: most layers use cheaper linear attention or state-space blocks, with full softmax attention every few layers. See [04 Architectures](../04-architectures/architectures.md).
- RoPE is the standard way to give attention position information (see Foundations: embeddings & positions, planned).

## What changed

- 2026-11: First version of this page.
- 2026-11: Added hidden answers to *Check yourself*.

## Open questions

- How far can hybrids push the ratio of linear to full attention before long-context retrieval quality drops?

## Resources

- Seminal: Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017) (P)
- Shazeer, [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150) (MQA, 2019) (P)
- Ainslie et al., [*GQA: Training Generalized Multi-Query Transformer Models*](https://arxiv.org/abs/2305.13245) (2023) (P)
- DeepSeek-AI, [*DeepSeek-V2*](https://arxiv.org/abs/2405.04434) (introduces MLA, 2024) (P)
- Dao et al., [*FlashAttention*](https://arxiv.org/abs/2205.14135) (2022) (P)
- Karpathy, [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (F, B, P)
- Raschka, *Build a Large Language Model (From Scratch)*, chapter 3 (F, B)

## Check yourself

Try each one before opening the answer.

**1. Why does removing the $\sqrt{d_k}$ scaling hurt training as $d_k$ grows?**

<details>
<summary>Answer</summary>

A dot product of two random $d_k$-dimensional vectors with unit-variance entries has variance $d_k$, so scores spread out as heads get wider. Softmax over large-magnitude scores becomes nearly one-hot: one token gets almost all the weight, and gradients for the others become tiny. Dividing by $\sqrt{d_k}$ brings the variance back to about 1, keeping softmax in the range where it learns well.

</details>

**2. Change the code to multi-head attention with $h = 2$ heads.**

<details>
<summary>Answer</summary>

Project as before, split the columns into $h$ heads, attend within each head, then concatenate:

```python
def multi_head(X, W_q, W_k, W_v, h):
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    d = Q.shape[-1] // h
    heads = []
    for i in range(h):
        q, k, v = (M[:, i*d:(i+1)*d] for M in (Q, K, V))
        scores = q @ k.T / np.sqrt(d)
        scores = np.where(np.triu(np.ones_like(scores, bool), 1), -np.inf, scores)
        w = np.exp(scores - scores.max(-1, keepdims=True))
        heads.append((w / w.sum(-1, keepdims=True)) @ v)
    return np.concatenate(heads, axis=-1)                # real models then apply W_o
```

Real implementations reshape to `(h, n, d_head)` and batch the heads instead of looping.

</details>

**3. A model has 48 layers, 8 KV heads and head dimension 128, stored in FP8 (1 byte). How much KV cache does a 32k-token context need?**

<details>
<summary>Answer</summary>

Per token: $2 \times 48 \times 8 \times 128 \times 1 = 98{,}304$ bytes = 96 KiB. For 32,768 tokens: 96 KiB × 32,768 = **3 GiB**.

</details>
