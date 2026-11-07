---
id: foundations.lm-objective
title: The Language-Modeling Objective
layer: 03-foundations
pathways: [fundamentals, practitioner]
level: intro
prereqs: [foundations.tokenization, prereqs.probability-information]
status: stable
last_reviewed: 2026-11-07
review_every_days: 180
---

# The Language-Modeling Objective

## TL;DR

An LLM is trained on one task: **predict the next token**. For every position in its training text, it outputs a probability for each token in the vocabulary. The loss, **cross-entropy**, is how surprised it was by the token that actually came next. **Perplexity** turns that loss into an intuitive number: roughly how many tokens the model is torn between. Everything an LLM knows comes from getting better at this one objective.

## Intuition

Imagine reading a sentence with the next word covered up and betting on what it is. If you put most of your money on the right word, you lose little; if the right word was something you considered unlikely, you lose a lot. Cross-entropy is that loss, averaged over every word in a huge amount of text.

To bet well, a model has to pick up spelling, grammar, facts ("The capital of France is …"), reasoning patterns and style, because all of them make the next token more predictable. That's why such a simple objective produces such general abilities.

## How it works

**From scores to probabilities.** At each position the model produces one score (a **logit**) per vocabulary token. Softmax turns them into probabilities that sum to 1:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

**The loss.** If the true next token is $y$, the loss at that position is $-\ln p_y$. Averaged over all $N$ positions:

$$
\mathcal{L} = -\frac{1}{N}\sum_{t=1}^{N} \ln p_\theta(x_{t+1} \mid x_1, \dots, x_t)
$$

- Probability 1 on the right token: loss 0.
- Probability 0.5: loss 0.69.
- Probability 0.001: loss 6.9. Confident mistakes are expensive.

**Every position at once.** A training sequence of 4,096 tokens gives 4,096 predictions, not one. The causal mask (see [Attention](attention.md)) stops each position from seeing the future, so all of them can be computed in a single parallel pass. Feeding the real previous tokens rather than the model's own guesses is called **teacher forcing**.

**Perplexity** is $e^{\mathcal{L}}$. A model that guesses uniformly over $V$ tokens has loss $\ln V$ and perplexity exactly $V$. A perplexity of 10 means the model is, on average, as unsure as if it were choosing among 10 equally likely tokens.

**Bits per byte.** Loss depends on the tokenizer: a model with bigger tokens makes fewer, harder predictions. To compare models with different tokenizers, convert to bits (divide nats by $\ln 2$) and divide by the number of bytes of text instead of the number of tokens.

**Generation** reuses the same probabilities: pick a next token (greedily, or by sampling), append it, repeat. **Temperature** divides the logits before softmax. Below 1 sharpens the distribution; above 1 flattens it.

## Code

[`s2a.lm`](../src/s2a/lm.py) has softmax, cross-entropy, perplexity, bits-per-byte and a counting-based **bigram model**, all tested in CI. The [language-modeling lab](../labs/lm-objective.ipynb) trains and samples from it.

```python
import numpy as np
from s2a.lm import cross_entropy, perplexity

logits = np.zeros((3, 50_257))                # know nothing: uniform over GPT-2's vocabulary
loss = cross_entropy(logits, np.array([0, 1, 2]))
loss, perplexity(loss)                        # (10.82, 50257.0)
```

## Numbers that matter

**Starting points.** An untrained model is close to uniform: loss $\ln 50{,}257 \approx 10.8$ for GPT-2's vocabulary, about 11.8 for Llama 3's 128k vocabulary. Watching the training loss fall from there is the first health check of any pretraining run.

**From the lab** (characters as tokens, a vocabulary of 31, on a short paragraph), held-out results:

| Model | What it knows | Loss (nats) | Perplexity |
|---|---|---|---|
| Uniform | Nothing | 3.43 | 31.0 |
| Unigram | How common each character is | 3.01 | 20.2 |
| Bigram | The previous character | 2.24 | 9.4 |

Each step uses more context and lowers the loss. Transformers take the same idea much further by attending to thousands of previous tokens.

**Train versus held-out loss.** In the lab the bigram model scores 1.91 on its training text but 2.30 on held-out text with too little smoothing. That gap is overfitting. LLM pretraining usually sees each token about once, so train and validation loss stay close.

## Variations on the objective

- **Multi-token prediction:** extra output heads predict 2, 3 or more tokens ahead. This gives a richer training signal, and the heads can be reused as drafts for [speculative decoding](../06-inference/inference.md).
- **Stability tricks:** z-loss (penalizes logits from drifting very large) and label smoothing (never targets probability exactly 1).
- **Masked language modeling** (BERT) predicts blanked-out tokens using both sides of the context. It's good for embeddings and classification, but it isn't a generative objective.
- **Post-training** keeps the same machinery but changes what counts as a good answer: supervised fine-tuning on curated examples, then preference optimization and reinforcement learning (see [05 Training](../05-training/training.md)).

## Current consensus (as of 2026-11)

- Next-token cross-entropy remains the pretraining objective for essentially all LLMs. Scaling laws are stated in terms of it: loss falls predictably as models, data and compute grow.
- Some frontier models add multi-token prediction during pretraining.
- Benchmarks matter more than perplexity for comparing finished chat models, but loss and bits-per-byte on held-out data remain the main signals during training.

## What changed

- 2026-11: First version of this page.

## Open questions

- How much of a model's downstream ability can be predicted from its pretraining loss alone?

## Resources

- Karpathy, [*The spelled-out intro to language modeling: building makemore*](https://www.youtube.com/watch?v=PaCmpygFfXo) (bigram models, loss, sampling) (F, B, P)
- Shannon, *Prediction and Entropy of Printed English* (1951), the original estimate of how predictable English is (F, P)
- Radford et al., [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019) (P)
- Kaplan et al., [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361) (2020) (P)
- Gloeckle et al., [*Better & Faster Large Language Models via Multi-token Prediction*](https://arxiv.org/abs/2404.19737) (2024) (P)

## Check yourself

Try each one before opening the answer.

**1. A model's validation loss is 2.3 nats per token. What's its perplexity, and what does that mean intuitively?**

<details>
<summary>Answer</summary>

Perplexity is $e^{2.3} \approx 10$. On average the model is as uncertain as if it were choosing uniformly among about 10 tokens at each step, out of a vocabulary of perhaps 100,000. It's a geometric average, so the model is very confident about some tokens and quite unsure about others.

</details>

**2. Model A has loss 2.0 per token with a 32k vocabulary; model B has 2.2 per token with a 128k vocabulary. Is A better?**

<details>
<summary>Answer</summary>

You can't tell from these numbers. B's bigger tokens cover more text each, so it makes fewer predictions for the same passage, and each one is harder. Compare **bits per byte** instead: convert each loss to bits, multiply by tokens, divide by the bytes of the same evaluation text. If B uses about 25% fewer tokens for that text, its total bits can be lower even though its per-token loss is higher.

</details>

**3. Why can one 4,096-token training sequence give 4,096 training examples instead of one?**

<details>
<summary>Answer</summary>

Each position is its own next-token prediction: predict token 2 from token 1, token 3 from tokens 1–2, and so on. The causal mask stops each position from seeing later tokens, so all 4,096 predictions can be computed in one parallel forward pass and their losses averaged. This is a big part of why transformers train efficiently compared with older recurrent models, which had to step through the sequence one token at a time.

</details>
