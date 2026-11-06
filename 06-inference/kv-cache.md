---
id: inference.kv-cache
title: KV Cache
layer: 06-inference
pathways: [builder, practitioner]
level: intermediate
prereqs: [foundations.attention, hardware.memory-hbm]
status: evolving
last_reviewed: 2026-11-06
review_every_days: 120
---

# KV Cache

## TL;DR

While generating, a model stores the keys and values of every earlier token so it doesn't recompute them. That store, the KV cache, grows with every token and every user. It often decides how many users fit on a GPU, how long a context can be, and how much a long agent conversation costs.

## Intuition

Generating text is a loop: predict one token, append it, predict the next. Each new token needs to attend to all earlier tokens (see [Attention](../03-foundations/attention.md)), which means it needs their keys and values.

Recomputing them every step would redo all past work for every new token. Instead the model computes each token's keys and values once and keeps them. It's like taking notes during a meeting instead of replaying the recording every time someone asks a question. The catch is that the notebook keeps growing, and it lives in scarce, fast GPU memory.

This splits inference into two phases:

- **Prefill:** process the whole prompt at once, filling the cache. Lots of math in parallel, so it is usually **compute-bound**.
- **Decode:** generate one token at a time, reading the entire cache at each step. Little math per byte read, so it is usually **memory-bandwidth-bound**.

## How it works

For each layer, each new token produces a key and a value vector per KV head. Per token, across the whole model:

$$
\text{bytes per token} = 2 \times L \times H_{kv} \times d_{\text{head}} \times b
$$

where $L$ is the number of layers, $H_{kv}$ the number of KV heads, $d_{\text{head}}$ the head dimension, $b$ bytes per number, and the 2 covers K and V. Total cache = bytes per token × tokens in context × concurrent sequences.

Each decode step has to read **all the model weights** plus **every sequence's whole KV cache** from GPU memory. Weights are shared across a batch, but each user's cache is their own. So as batches and contexts grow, cache reads can dominate.

## Code

- [`s2a.kv_cache`](../src/s2a/kv_cache.py): sizing functions, tested in CI.
- [KV-cache calculator notebook](../calculators/kv-cache.ipynb): runs on CPU or in Colab.

## Numbers that matter

Llama-3-8B-style model (32 layers, 8 KV heads, head dimension 128, BF16), from the calculator:

| Quantity | Value |
|---|---|
| KV cache per token | 128 KiB |
| One 8k-token conversation | 1 GiB |
| One 128k-token conversation | 16 GiB, about as large as the model's weights |
| 8k-token conversations that fit on an 80 GB GPU (after ~16 GB weights, ~4 GB overhead) | about 55 in BF16, about 111 in FP8 |

**Why decode slows down as you add users.** On a GPU with about 3.35 TB/s of memory bandwidth (H100 SXM):

| Batch (8k tokens each) | Bytes read per decode step | Lower bound per step |
|---|---|---|
| 1 user | 16 GB weights + 1.1 GB cache ≈ 17 GB | ≈ 5 ms |
| 32 users | 16 GB weights + 34 GB cache ≈ 50 GB | ≈ 15 ms |

Batching still raises total throughput (32 tokens per step instead of 1), but every user's next token arrives more slowly. At this point cache reads, not weights, are most of the traffic.

**Why it matters for agents.** Agents carry long, growing contexts: tool definitions, tool results and history. Every extra token in context costs cache memory for as long as the conversation lives, and is read again at every step.

## How the cache is made smaller or cheaper

| Technique | Idea | Trade-off |
|---|---|---|
| **GQA / MQA** | Fewer KV heads, shared by query heads | Small quality cost; built into the model |
| **MLA** | Store a compressed latent instead of full K and V; DeepSeek-V2 reported cutting its KV cache by 93.3% | Built into the model; more complex kernels |
| **Hybrid architectures** | Most layers use linear attention or state-space blocks with a fixed-size state; only some layers keep a growing cache | Built into the model; long-range recall is still being studied |
| **KV quantization** | Store the cache in FP8 or lower (KIVI goes to 2 bits) | Some accuracy loss, mostly at very low bits |
| **PagedAttention** | Allocate the cache in fixed-size blocks, like OS virtual memory, instead of one big reserved chunk per request | Mostly a win; needs paging-aware kernels |
| **Prefix caching** | Reuse the cache for shared prompt prefixes (system prompts, tool definitions, documents) across requests | Only helps when prefixes repeat exactly |
| **Eviction / windows** | Keep only recent tokens plus a few "attention sink" tokens (StreamingLLM), or drop low-importance tokens (H2O) | Loses information; risky for retrieval-heavy tasks |
| **Offloading** | Move cold cache blocks to CPU memory or SSD and bring them back when needed (e.g. LMCache) | Transfer latency |
| **Disaggregated serving** | Run prefill and decode on different GPUs and ship the cache between them | Network cost; see [Inference](inference.md) |

**Connection you see as a user:** API "prompt caching" discounts are prefix caching made visible. The provider keeps your repeated prefix's KV cache around, so it doesn't recompute it.

## Current consensus (as of 2026-11)

- PagedAttention-style block allocation and automatic prefix caching are standard in serving engines such as vLLM and SGLang.
- New models increasingly cut the cache at the architecture level (GQA as a baseline, MLA, or hybrid linear/full attention) rather than relying only on serving tricks.
- FP8 KV caches are common in production. Lower precisions and eviction are used more cautiously.
- Large deployments move the KV cache between GPUs and memory tiers (disaggregated prefill/decode, KV-aware routing, offload). Treating the cache as a resource you manage at cluster level is one of the main serving trends.

## What changed

- 2026-11: First version of this page.

## Open questions

- The simple formula doesn't fit MLA or hybrid models. How should the calculator model them? (Planned: per-layer cache types.)
- For agents, when is it cheaper to keep a session's cache warm versus recomputing from a summary?

## Resources

- Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (vLLM, 2023) (B, P)
- Zheng et al., [*SGLang: Efficient Execution of Structured Language Model Programs*](https://arxiv.org/abs/2312.07104) (RadixAttention prefix caching) (P)
- Ainslie et al., [*GQA*](https://arxiv.org/abs/2305.13245) (2023) (P)
- DeepSeek-AI, [*DeepSeek-V2*](https://arxiv.org/abs/2405.04434) (MLA, 2024) (P)
- Liu et al., [*KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*](https://arxiv.org/abs/2402.02750) (2024) (P)
- Xiao et al., [*Efficient Streaming Language Models with Attention Sinks*](https://arxiv.org/abs/2309.17453) (StreamingLLM, 2023) (P)
- Zhang et al., [*H2O: Heavy-Hitter Oracle*](https://arxiv.org/abs/2306.14048) (2023) (P)
- Google DeepMind, [*How to Scale Your Model*, "All About Transformer Inference"](https://jax-ml.github.io/scaling-book/inference/) (P)

## Check yourself

1. Why is prefill usually compute-bound but decode memory-bound?
2. With the calculator, find the context length at which one user's cache equals the model's weights for your favourite open model.
3. An agent's system prompt and tool definitions total 6,000 tokens and are identical on every request. What does prefix caching save per request, in memory and in prefill compute?
