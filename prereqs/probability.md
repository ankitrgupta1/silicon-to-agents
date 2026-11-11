---
id: prereqs.probability-information
title: Probability & Information
layer: prereqs
pathways: [fundamentals]
level: intro
prereqs: []
status: stable
last_reviewed: 2026-11-10
review_every_days: 365
---

# Probability & Information

## TL;DR

A language model's output is a **probability distribution** over the next token. Four numbers describe how good it is: **surprise** ($-\log p$, the cost of one outcome), **entropy** (how unpredictable the truth is), **cross-entropy** (how surprised the model is on average, which is the training loss), and **KL divergence** (the gap between the two, which is how far the model is from the truth). Training an LLM means pushing cross-entropy down toward the entropy floor.

## Intuition

- **Surprise is the cost of being caught off guard.** If you said something was 50% likely and it happened, you're mildly surprised. If you said 0.1%, you're very surprised. Measured in **bits**, every halving of the probability costs one more bit.
- **Entropy is how much is genuinely unknown.** A fair coin carries 1 bit of uncertainty; a coin that always lands heads carries none. No model can be less surprised, on average, than the entropy of the thing it's predicting.
- **Cross-entropy is how surprised *your model* is.** If your beliefs differ from reality, you pay extra.
- **KL divergence is that extra.** It's zero only when the model matches reality exactly.

## How it works

### Distributions and softmax

A **probability distribution** assigns each possible outcome a non-negative number, and they add up to 1. A language model produces a distribution over its vocabulary at every position, by applying **softmax** to its scores (logits): $p_i = e^{z_i} / \sum_j e^{z_j}$. Exponentiating makes everything positive; dividing makes it sum to 1 (see [the language-modeling objective](../03-foundations/lm-objective.md)).

### Surprise and entropy

The **surprise** of an outcome with probability $p$ is $-\log p$. **Entropy** is the average surprise when outcomes come from the distribution itself:

$$
H(p) = -\sum_i p_i \log p_i
$$

It's largest for a uniform distribution ($\log n$ for $n$ outcomes) and zero when one outcome is certain. With the natural log, the unit is **nats**; with $\log_2$, it's **bits** (1 nat ≈ 1.44 bits).

### Cross-entropy and KL divergence

If the truth is $p$ but you predict $q$:

$$
H(p, q) = -\sum_i p_i \log q_i \qquad\qquad D_{\text{KL}}(p \,\|\, q) = H(p, q) - H(p) \geq 0
$$

In language modeling the "truth" at each position is the single token that actually came next, so $p$ puts all its probability on that token and the cross-entropy reduces to $-\log q(\text{actual next token})$. Averaged over a dataset, that is the training loss.

Because the entropy of real text is fixed, **minimizing cross-entropy is the same as minimizing the KL divergence** from the true distribution of text. The loss can approach the entropy floor but never cross it.

Two properties of KL to remember:

- **It isn't symmetric:** $D_{\text{KL}}(p \| q) \ne D_{\text{KL}}(q \| p)$ in general, so it's not a true distance.
- **It punishes zeros harshly:** giving probability 0 to something that happens makes it infinite. That's why models never output exact zeros, and why counting-based models need smoothing.

### Where these show up later

- **Training loss and perplexity:** cross-entropy, and $e^{\text{loss}}$ ([LM objective](../03-foundations/lm-objective.md)).
- **Temperature:** changes the entropy of the sampling distribution.
- **Distillation:** a small model is trained to minimize KL from a large model's distributions, which carry more information than single correct answers.
- **RLHF and RL for reasoning:** a KL penalty keeps the fine-tuned model close to the original so it doesn't drift into nonsense while chasing reward (see [05 Training](../05-training/training.md)).

## Code

[`s2a.info`](../src/s2a/info.py) has entropy, cross-entropy, KL divergence and surprise, tested in CI (including a check that it agrees with the logit-based loss in [`s2a.lm`](../src/s2a/lm.py)). The [probability lab](../labs/probability.ipynb) runs each idea.

```python
from s2a.info import entropy, cross_entropy, kl_divergence, to_bits

to_bits(entropy([0.5, 0.5]))                  # 1.0 bit: a fair coin
truth, model = [0.7, 0.2, 0.1], [0.4, 0.4, 0.2]
cross_entropy(truth, model)                   # 0.986 nats
entropy(truth) + kl_divergence(truth, model)  # the same: 0.802 + 0.184
```

## Numbers that matter

**Surprise of one outcome:**

| Probability given | Surprise (nats) | Surprise (bits) |
|---|---|---|
| 1 | 0 | 0 |
| 0.5 | 0.69 | 1 |
| 0.1 | 2.30 | 3.3 |
| 0.01 | 4.61 | 6.6 |
| 0.001 | 6.91 | 10.0 |

**From the lab:** a true distribution with entropy 0.802 nats, scored against four models.

| Model | Cross-entropy | KL (the excess) |
|---|---|---|
| Perfect | 0.802 | 0 |
| Close | 0.829 | 0.027 |
| Uniform guess | 1.099 | 0.297 |
| Confident and wrong | 2.707 | 1.905 |

Being confidently wrong is far worse than admitting ignorance.

**Temperature and entropy** (five options, so at most 2.32 bits): temperature 0.25 gives 0.16 bits, nearly deterministic; 1.0 gives 1.74 bits; 5.0 gives 2.29 bits, nearly uniform.

**Reference points:** an untrained model with GPT-2's 50,257-token vocabulary starts near $\ln 50{,}257 \approx 10.8$ nats per token. Shannon estimated English at roughly 1 bit per character.

## Current consensus (as of 2026-11)

- Cross-entropy is the universal pretraining loss, and KL divergence is the standard tool for distillation and for keeping fine-tuned models close to their starting point in RLHF-style training.

## What changed

- 2026-11: First version of this page.

## Open questions

- What is the true entropy of natural language? It sets the floor for every model's loss, and current estimates are rough.

## Resources

- 3Blue1Brown, [*Solving Wordle using information theory*](https://www.youtube.com/watch?v=v68zYyaEmEA), an intuitive introduction to entropy (F)
- Olah, [*Visual Information Theory*](https://colah.github.io/posts/2015-09-Visual-Information/) (F)
- Deisenroth, Faisal & Ong, [*Mathematics for Machine Learning*](https://mml-book.github.io/), chapter 6 (F)
- MacKay, [*Information Theory, Inference, and Learning Algorithms*](https://www.inference.org.uk/itila/book.html) (free) (P)
- Shannon, [*A Mathematical Theory of Communication*](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) (1948) (P)

## Check yourself

Try each one before opening the answer.

**1. A model gives the correct next token probability 0.25. What's its loss on that token, in nats and in bits?**

<details>
<summary>Answer</summary>

$-\ln 0.25 \approx 1.39$ nats, which is $-\log_2 0.25 = 2$ bits. Two halvings from certainty, so two bits: as surprised as if it had been choosing between four equally likely tokens.

</details>

**2. Two models have the same accuracy: each puts the highest probability on the right answer equally often. Why can their cross-entropy differ a lot?**

<details>
<summary>Answer</summary>

Accuracy only checks which option ranks first. Cross-entropy depends on how much probability the right answer got, and on how badly the model is caught out when it's wrong. A model that is confidently wrong now and then (probability 0.001 on what actually happened) pays almost 7 nats each time, while a hedging model pays much less. That's why loss is a more sensitive training signal than accuracy.

</details>

**3. RLHF training adds a penalty for KL divergence between the fine-tuned model and the original. What would go wrong without it?**

<details>
<summary>Answer</summary>

The model optimizes whatever the reward model scores highly, and reward models have blind spots. Without a tether, it can drift into strange outputs that exploit those blind spots (reward hacking) and lose the fluency and knowledge it learned in pretraining. The KL penalty makes every move away from the original distribution cost something, so the model only changes where the reward gain is worth it.

</details>
