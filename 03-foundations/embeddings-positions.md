---
id: foundations.embeddings-positions
title: Embeddings & Positions
layer: 03-foundations
pathways: [fundamentals, practitioner]
level: intro
prereqs: [foundations.tokenization, prereqs.linear-algebra]
status: stable
last_reviewed: 2026-11-07
review_every_days: 180
---

# Embeddings & Positions

## TL;DR

An **embedding** turns each token ID into a vector of numbers the model can compute with; similar meanings end up as nearby vectors. Attention alone ignores word order, so models also need **positional information**. Today that's almost always **RoPE**, which rotates queries and keys by an angle based on position, so attention sees how far apart two tokens are.

## Intuition

**Embeddings.** A token ID like `15339` means nothing by itself. The model keeps a big table with one learned vector per token in the vocabulary and looks up the row for each ID. Training nudges these vectors so tokens used in similar ways end up close together. Directions in the space can carry meaning too: the classic word2vec example is *king − man + woman ≈ queen*.

**Positions.** Attention compares every token with every other token, but nothing in that comparison says which came first. Without positional information, *"dog bites man"* and *"man bites dog"* look the same to it. Positional encodings fix this, either by marking each token with its absolute position or, better, by letting attention see the **relative** distance between tokens.

## How it works

### Token embeddings

The embedding table is a matrix $E$ of shape (vocabulary size × model width $d$). Looking up token $t$ just takes row $t$. Mathematically it's a one-hot vector times $E$, but in practice it's an index lookup. At the other end, the model turns its final vector back into scores over the vocabulary with an output matrix of the same shape. Some models reuse $E$ for both (**weight tying**); many large models keep them separate.

### Absolute positions

- **Sinusoidal** (original transformer): add a fixed vector to each token embedding, built from sines and cosines at many frequencies, like the hands of a clock running at different speeds.
- **Learned** (GPT-2, BERT): add a learned vector per position. Simple, but it can't handle positions beyond the longest one seen in training (1,024 for GPT-2).

### Relative positions and RoPE

What attention really needs is "how far back is that token", not "this is token 812". Relative schemes build that in. ALiBi, for example, subtracts a penalty from attention scores that grows with distance.

**RoPE (rotary position embedding)** is the standard today. It groups the dimensions of each query and key into pairs, treats each pair as a 2-D point, and rotates it by an angle proportional to the token's position $m$:

$$
\text{angle}_i = m \cdot \theta_i, \qquad \theta_i = \text{base}^{-2i/d}
$$

Rotating both the query (at position $m$) and the key (at position $n$) means their dot product depends only on the **difference** $m - n$, because rotating two vectors by different angles changes their dot product only by the angle between the rotations. So attention scores carry relative position for free, with no extra parameters and nothing added to the embeddings.

Different pairs rotate at different speeds. The first pair turns 1 radian per position and tracks fine, local order. Later pairs turn very slowly and can distinguish positions far apart. The `base` sets how slow the slowest pairs get.

### Longer contexts

A model trained on 8k-token sequences has never seen rotation angles beyond position 8k. Ways to stretch it:

- **Bigger base:** slows all rotations so long distances stay in a familiar range. Llama 3 uses a base of 500,000 instead of the original 10,000.
- **Position interpolation:** squeeze new positions into the trained range (position 16k is treated like 8k).
- **NTK-aware scaling and YaRN:** stretch the slow pairs more than the fast ones, keeping local detail sharp. Usually followed by a short fine-tune on long sequences.

## Code

[`s2a.positions`](../src/s2a/positions.py) implements RoPE and sinusoidal encodings in NumPy, tested in CI. Explore them in the [positions lab](../labs/positions.ipynb).

```python
import numpy as np
from s2a.positions import apply_rope

rng = np.random.default_rng(0)
q, k = rng.normal(size=(1, 64)), rng.normal(size=(1, 64))

def score(m, n):  # query at position m, key at position n
    return (apply_rope(q, [m]) @ apply_rope(k, [n]).T).item()

score(10, 7), score(1010, 1007)   # equal: only the offset (3) matters
```

## Numbers that matter

- **Embedding tables are big.** For Llama 3 8B, vocabulary 128,256 × width 4,096 ≈ 525 million parameters per table, about 6.5% of the model. A larger vocabulary (see [Tokenization](tokenization.md)) makes them bigger still.
- **RoPE range.** The slowest pair turns once every $2\pi / \theta_{\text{last}}$ positions. With base 10,000 and head dimension 128 that's about 54,000 positions; with base 500,000 it's about 2.6 million, which is part of how Llama 3 handles long contexts.

## Current consensus (as of 2026-11)

- RoPE is the default positional encoding for open LLMs, usually with a large base and a scaling method such as YaRN for long context.
- Some recent architectures mix in layers with **no** positional encoding (NoPE), relying on the causal mask and other layers to carry order. Hybrid models with linear-attention or state-space layers get order from their recurrence (see [04 Architectures](../04-architectures/architectures.md)).
- Separate **sentence embeddings** (one vector per passage, from models trained with contrastive learning) power search and RAG. These are related but distinct from a language model's token embeddings; see [07 Applications](../07-applications/applications.md).

## What changed

- 2026-11: First version of this page.

## Open questions

- How far can RoPE scaling go before quality at long range falls apart, compared with NoPE layers or hybrid models?

## Resources

- Mikolov et al., [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781) (word2vec, 2013) (F, P)
- Press & Wolf, [*Using the Output Embedding to Improve Language Models*](https://arxiv.org/abs/1608.05859) (weight tying, 2017) (P)
- Su et al., [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) (RoPE, 2021) (P)
- Press, Smith & Lewis, [*Train Short, Test Long*](https://arxiv.org/abs/2108.12409) (ALiBi, 2021) (P)
- Chen et al., [*Extending Context Window of LLMs via Positional Interpolation*](https://arxiv.org/abs/2306.15595) (2023) (P)
- Peng et al., [*YaRN: Efficient Context Window Extension of LLMs*](https://arxiv.org/abs/2309.00071) (2023) (P)
- Kazemnejad et al., [*The Impact of Positional Encoding on Length Generalization in Transformers*](https://arxiv.org/abs/2305.19466) (NoPE, 2023) (P)
- Reimers & Gurevych, [*Sentence-BERT*](https://arxiv.org/abs/1908.10084) (2019) (B)

## Check yourself

Try each one before opening the answer.

**1. Why would a transformer with no positional information treat "dog bites man" and "man bites dog" the same way?**

<details>
<summary>Answer</summary>

Attention computes each token's output as a weighted sum over all tokens, with weights from query–key dot products. Shuffling the input tokens just shuffles the outputs the same way; nothing in the computation depends on order. So each word gets the same representation in both sentences, and the model can't tell who bit whom. (The causal mask in decoder models leaks a little order information, which is why NoPE layers can work at all, but it's a weak signal.)

</details>

**2. Why does RoPE make the query–key score depend only on the distance between tokens?**

<details>
<summary>Answer</summary>

For one dimension pair, the query is rotated by $m\theta$ and the key by $n\theta$. A dot product between two rotated vectors equals the dot product of the originals with one of them rotated by the *difference* of the angles, $(m-n)\theta$. Every pair works the same way, so the total score depends on $m - n$, not on $m$ and $n$ separately. The lab checks this numerically.

</details>

**3. A model was trained on 4k-token sequences with RoPE base 10,000. Why might it break at 32k tokens, and what are two ways to fix it?**

<details>
<summary>Answer</summary>

At positions past 4k, the slow-rotating pairs reach angles the model never saw in training, so attention scores at long distances behave unpredictably. Fixes: **position interpolation** (scale positions down so 32k maps into the trained 0–4k range), or **raise the base / use NTK-aware scaling or YaRN** (slow the rotations, stretching slow pairs most). Either way, a short fine-tune on long sequences helps the model adapt.

</details>
