---
id: architectures.transformer
title: The Transformer Block
layer: 04-architectures
pathways: [fundamentals, practitioner]
level: intermediate
prereqs: [foundations.attention, foundations.embeddings-positions, foundations.lm-objective]
status: stable
last_reviewed: 2026-11-09
review_every_days: 180
---

# The Transformer Block

## TL;DR

A language model is a stack of identical **blocks**. Each block does two things: **attention** lets tokens gather information from earlier tokens, and an **MLP** processes each token on its own. Both *add* their result to a running vector per token, the **residual stream**. Today's standard recipe is pre-norm with RMSNorm, grouped-query attention with RoPE, and a SwiGLU MLP. Most of a model's parameters sit in the MLPs, not in attention.

## Intuition

Picture one vector per token flowing up through the model like a shared notepad. This is the residual stream.

- **Attention is communication.** Each token looks back at earlier tokens and copies over what's relevant ("which noun does *it* refer to?"). See [Attention](../03-foundations/attention.md).
- **The MLP is computation.** Each token, alone, transforms what it has gathered ("given these facts, what follows?").
- **Both write by adding.** A block never replaces the notepad. It reads a copy, works out an update, and adds it on. Early blocks tend to handle local, surface patterns; later blocks build on what earlier ones wrote.

Stack 30 to 100 of these blocks, then turn the final vector at each position into scores over the vocabulary. That's the whole model.

## How it works

The complete forward pass, for token IDs in and next-token scores out:

```text
x = embedding[token_ids]                  # one vector per token
for each block:
    x = x + attention(norm(x))            # tokens talk to earlier tokens
    x = x + mlp(norm(x))                  # each token thinks on its own
logits = norm(x) @ output_head            # scores over the vocabulary
```

**Residual connections** (the `x + …`). Adding instead of replacing gives gradients a direct path from the loss back to every layer, which is what makes very deep networks trainable.

**Pre-norm.** The original transformer normalized *after* each addition. Modern models normalize the *input* to each sublayer and leave the residual stream itself untouched. This is much more stable to train at depth.

**RMSNorm.** Rescales each vector so its root-mean-square is 1, then applies a learned per-dimension gain:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}} \odot g
$$

It drops LayerNorm's mean-subtraction and bias, which turned out not to matter, and is cheaper.

**Attention sublayer.** Causal [grouped-query attention](../03-foundations/attention.md), with [RoPE](../03-foundations/embeddings-positions.md) applied to queries and keys. Position information enters here, at every layer, not at the embedding.

**SwiGLU MLP.** A gated feed-forward layer with three weight matrices:

$$
\text{MLP}(x) = W_{\text{down}}\big(\text{SiLU}(W_{\text{gate}}\,x) \odot W_{\text{up}}\,x\big)
$$

One branch (through the smooth SiLU activation) acts as a gate that decides how much of the other branch passes. The hidden width is typically 2.5–4× the model width.

### Original (2017) versus modern

| Piece | Original transformer | Modern LLM (Llama-style) |
|---|---|---|
| Structure | Encoder + decoder | Decoder only |
| Normalization | LayerNorm, after each sublayer | RMSNorm, before each sublayer |
| Positions | Sinusoidal, added to embeddings | RoPE, inside attention |
| Attention | Multi-head | Grouped-query (shared K/V heads) |
| MLP | ReLU, 2 matrices, 4× width | SwiGLU, 3 matrices |
| Biases | Yes | Mostly none |

## Code

[`s2a.transformer`](../src/s2a/transformer.py) is a complete modern transformer forward pass in about 100 lines of NumPy, reusing the RoPE and softmax code from earlier topics. The [transformer block lab](../labs/transformer-block.ipynb) steps through it.

```python
import numpy as np
from s2a.transformer import TransformerConfig, init_params, forward, LLAMA3_8B

cfg = TransformerConfig(vocab_size=100, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2, d_ff=172)
logits = forward(init_params(cfg), cfg, [5, 17, 42, 8])
logits.shape                    # (4, 100): vocabulary scores at each position

LLAMA3_8B.param_count()         # 8,030,261,248
```

It's forward-only: training needs automatic differentiation, which comes with the tiny GPT in [05 Training](../05-training/training.md).

## Numbers that matter

Where Llama 3 8B's 8.03 billion parameters live (computed by `LLAMA3_8B.param_breakdown()` from its published shape: 32 layers, width 4,096, 32 heads, 8 KV heads, MLP width 14,336, vocabulary 128,256):

| Part | Parameters | Share |
|---|---|---|
| MLPs | 5,637 M | 70.2% |
| Attention | 1,342 M | 16.7% |
| Input embedding | 525 M | 6.5% |
| Output head | 525 M | 6.5% |
| Norms | 0.3 M | ~0% |

- **Per block:** 176 M in the MLP versus 42 M in attention. Attention gets the fame, but the MLPs hold most of the weights.
- **GQA saves parameters too:** with 8 KV heads instead of 32, the key and value matrices are a quarter of the size, on top of the 4× smaller [KV cache](../06-inference/kv-cache.md).
- **Why the SwiGLU width looks odd:** three matrices instead of two means that, to match the size of a classic 4× MLP, the hidden width is scaled by ⅔ (to about 2.67×). Llama 3 8B uses 3.5×.
- **Compute:** a forward pass costs roughly 2 FLOPs per parameter per token (each weight takes part in one multiply and one add), so about 16 billion FLOPs per token here, ignoring the attention-score term that grows with context.

## Current consensus (as of 2026-11)

- The recipe above (pre-norm, RMSNorm, RoPE, GQA, SwiGLU, no biases) is the shared baseline of nearly all open dense LLMs.
- Common refinements: normalizing queries and keys before the dot product (QK-norm) for stability, and other small changes to normalization and initialization.
- The big architectural changes are happening *around* the block rather than inside it: replacing the dense MLP with a mixture of experts, and replacing some attention layers with linear-attention or state-space layers. See the planned pages in [04 Architectures](architectures.md).

## What changed

- 2026-11: First version of this page.

## Open questions

- For a fixed parameter budget, when is deeper better than wider? Practice favors fairly deep models, but the trade-off interacts with latency, since depth can't be parallelized.

## Resources

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017) (P)
- He et al., [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) (residual connections, 2015) (P)
- Xiong et al., [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745) (pre-norm, 2020) (P)
- Zhang & Sennrich, [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467) (2019) (P)
- Shazeer, [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202) (SwiGLU, 2020) (P)
- Touvron et al., [*LLaMA: Open and Efficient Foundation Language Models*](https://arxiv.org/abs/2302.13971) (the modern recipe in one place, 2023) (P)
- Elhage et al., [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (the residual-stream view, 2021) (P)
- Karpathy, [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (F, B, P)
- Raschka, *Build a Large Language Model (From Scratch)*, chapter 4 (F, B)

## Check yourself

Try each one before opening the answer.

**1. Why do blocks add to the residual stream instead of replacing it?**

<details>
<summary>Answer</summary>

Two reasons. For training: addition gives the gradient a direct path from the output back to every layer, so even a 100-layer model gets a usable learning signal in its earliest blocks. For function: each block only has to learn a small *update*, and anything written by earlier blocks stays available to later ones unless something deliberately overwrites it.

</details>

**2. A model has width 2,048, 16 heads, 4 KV heads and MLP width 7,168. How many parameters are in one block (ignore norms)?**

<details>
<summary>Answer</summary>

Head dimension is 2,048 / 16 = 128, so the K and V projections map to 4 × 128 = 512.

- Attention: $W_Q$ and $W_O$ are 2,048 × 2,048 each (8.39 M together); $W_K$ and $W_V$ are 2,048 × 512 each (2.10 M together). Total 10.49 M.
- MLP: 3 × 2,048 × 7,168 = 44.04 M.

About **54.5 M** per block, 81% of it in the MLP. Check it with `TransformerConfig(...).param_breakdown()`.

</details>

**3. In the lab, changing the last two tokens leaves the outputs at earlier positions exactly unchanged. Which part of the model guarantees that, and why does it matter?**

<details>
<summary>Answer</summary>

The causal mask in attention: position *t* can only read positions up to *t*. Every other operation (embedding lookup, norms, MLP) works on each position separately, so attention is the only place information moves between positions, and the mask makes it one-way.

It matters twice. In training, one pass yields a valid next-token prediction at every position (see [the language-modeling objective](../03-foundations/lm-objective.md)). In generation, earlier positions' keys and values never change when new tokens arrive, which is exactly why they can be stored in the [KV cache](../06-inference/kv-cache.md).

</details>
