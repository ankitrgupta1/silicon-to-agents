---
id: training.tiny-gpt
title: Train a Tiny GPT
layer: 05-training
pathways: [fundamentals, practitioner]
level: intermediate
prereqs: [architectures.transformer, foundations.lm-objective, prereqs.python-pytorch]
status: stable
last_reviewed: 2026-11-09
review_every_days: 180
---

# Train a Tiny GPT

## TL;DR

Training is a loop: show the model a batch of text, measure how badly it predicted each next token, work out how every weight contributed to that error (**backpropagation**), and nudge each weight a little in the direction that reduces it (**gradient descent**). Repeat a few thousand times and a model with random weights starts producing text that looks like its training data. The same loop, scaled up enormously, trains every LLM.

## Intuition

Think of the model as a machine with hundreds of thousands of dials (its weights), all set randomly. You feed it text, and it's bad at guessing what comes next. For each dial, you ask: *if I turned this slightly, would the guesses get better or worse?* That answer, for every dial at once, is the **gradient**. You turn every dial a tiny step in its helpful direction and try again with new text.

No single step does much. But each one makes the next token slightly less surprising, and over thousands of steps the model picks up spelling, then words, then structure.

## How it works

**1. Batches.** Pick random windows of text, say 32 windows of 64 tokens. The targets are the same windows shifted by one: for every position, the answer is the token that came next (see [the language-modeling objective](../03-foundations/lm-objective.md)).

**2. Forward pass.** Run the [transformer](../04-architectures/transformer.md) to get predictions at every position, and compute the cross-entropy loss: one number saying how wrong the batch was.

**3. Backward pass.** Backpropagation applies the chain rule from the loss back through every operation, giving the gradient of the loss with respect to every weight. PyTorch records the operations in the forward pass and does this automatically when you call `loss.backward()`.

**4. Update.** The optimizer changes each weight by a small step against its gradient. The step size is the **learning rate**. Almost all LLMs use **AdamW**, which adapts the step for each weight using running averages of its recent gradients, and gently shrinks weights toward zero (weight decay).

```python
for step in range(steps):
    x, y = get_batch(data, batch_size, context)   # inputs, and inputs shifted by one
    logits, loss = model(x, y)                    # forward
    optimizer.zero_grad()
    loss.backward()                               # backward: gradients for every weight
    optimizer.step()                              # update
```

**Held-out data.** Keep some text the model never trains on, and measure the loss there too. If training loss keeps falling while held-out loss stalls or rises, the model is memorizing rather than learning patterns (overfitting).

**What to watch for**

- Loss starts near $\ln(\text{vocabulary size})$. If it doesn't, something is wrong with the initialization or the loss.
- Loss that explodes or turns into `NaN` usually means the learning rate is too high. Loss that barely moves means it's too low.
- **Gradient clipping** (capping the gradient's overall size) guards against the occasional huge step.

**Generating text** uses the trained model in a loop: predict, sample one token, append, repeat. The simple version here recomputes the whole context every time; real systems avoid that with a [KV cache](../06-inference/kv-cache.md).

## Code

- [`s2a.gpt`](../src/s2a/gpt.py): the model, training loop, loss estimate and sampling in PyTorch, about 150 lines. It's the same architecture as the NumPy [`s2a.transformer`](../src/s2a/transformer.py), and a test checks that the two produce identical outputs from the same weights.
- **[Tiny GPT lab](../labs/tiny-gpt.ipynb):** train a character-level model on Shakespeare. Runs on CPU in several minutes, or faster on a free Colab GPU.

## Numbers that matter

The lab's model, computed from its configuration (4 blocks, width 128, 4 heads, 2 KV heads, MLP width 344, 65 characters):

| | Lab model | Llama 3 8B |
|---|---|---|
| Parameters | 742,784 | 8,030,261,248 |
| Share in MLPs | 71% | 70% |
| Vocabulary | 65 characters | 128,256 tokens |
| Starting loss, $\ln V$ | 4.17 | 11.76 |

- **Same shape, different scale.** The lab model is about 10,800 times smaller, but its parameters are distributed almost identically.
- **Tokens seen:** the default run is 2,000 steps × 32 windows × 64 characters ≈ 4.1 million characters, about four passes over the 1 MB dataset. LLM pretraining sees trillions of tokens, mostly just once.
- **Reference point:** a bigram model, which only looks at the previous character, reaches a loss of about 2.5 on this dataset in Karpathy's walkthrough (linked below). A transformer that uses its context should end up clearly below that.

## What changes at real scale

The loop stays the same. What grows around it:

| Here | At scale | Where |
|---|---|---|
| Characters | BPE tokens | [Tokenization](../03-foundations/tokenization.md) |
| 1 MB of text | Trillions of filtered, deduplicated tokens | [05 Training](training.md): data |
| Constant learning rate | Warmup, then decay | [05 Training](training.md): pretraining |
| One CPU or GPU, 32-bit floats | Thousands of GPUs, mixed precision | [02 Distributed](../02-distributed/distributed.md) |
| Minutes | Weeks to months | [00 Hardware](../00-hardware/hardware.md) |
| Stop here | Then fine-tuning and RL | [05 Training](training.md): post-training |

## Current consensus (as of 2026-11)

- AdamW with learning-rate warmup and decay, gradient clipping and mixed precision is the standard pretraining setup. Newer optimizers (such as Muon) are being explored but haven't replaced it.
- Small from-scratch models remain the standard way to learn and to test ideas cheaply; Karpathy's nanoGPT and nanochat are the common starting points.

## What changed

- 2026-11: First version of this page.

## Open questions

- How well do training tricks found on tiny models transfer to large ones?

## Resources

- Karpathy, [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY), the classic walkthrough this lab follows in spirit (F, B, P)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat) (B, P)
- Karpathy, [*The spelled-out intro to neural networks and backpropagation: building micrograd*](https://www.youtube.com/watch?v=VMj-3S1tku0) (F)
- Loshchilov & Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101) (AdamW, 2019) (P)
- Kingma & Ba, [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980) (2015) (P)
- Raschka, *Build a Large Language Model (From Scratch)*, chapter 5 (F, B)
- Stanford CS336, assignment 1 (build and train a transformer LM from scratch) (P)

## Check yourself

Try each one before opening the answer.

**1. Your model's vocabulary has 65 characters and its loss at step 0 is 6.8. Is that a problem?**

<details>
<summary>Answer</summary>

Yes. A model that knows nothing should guess roughly uniformly, giving a loss near $\ln 65 \approx 4.17$. A starting loss of 6.8 means the untrained model is *confidently wrong*: its initial weights produce large logits. The usual cause is initialization that's too large (the lab uses a standard deviation of 0.02). It will still train, but it wastes early steps just unlearning that confidence.

</details>

**2. Training loss is 0.9 and still falling; held-out loss bottomed out at 1.5 and is now rising. What's happening, and what could you do?**

<details>
<summary>Answer</summary>

The model is overfitting: memorizing the training text instead of learning patterns that carry over. Options: stop training at the point where held-out loss was lowest (early stopping), use more data, use a smaller model, or add regularization such as dropout or more weight decay. With only 1 MB of text and many passes over it, this happens quickly. Large LLMs mostly avoid it by seeing each token about once.

</details>

**3. Why does `generate` in this lab get slower as the text gets longer, and how do real systems avoid it?**

<details>
<summary>Answer</summary>

For every new character it reruns the model on the whole context window, recomputing keys and values for tokens it has already processed. Because of the causal mask, those never change when new tokens arrive, so real systems compute them once and store them in a [KV cache](../06-inference/kv-cache.md). Each new token then needs only its own computation plus a read of the cache.

</details>
