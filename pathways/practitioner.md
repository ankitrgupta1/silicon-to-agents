# Practitioner path

For people who want the deep internals. Builds a model from scratch, then follows it down to kernels and hardware and back up to serving at scale.

**You'll be able to:** train a small chat model end to end, write and profile a GPU kernel, reason about parallelism plans, and serve a model efficiently.

| # | Step | Where | Status |
|---|---|---|---|
| 1 | Attention & the transformer | [Attention](../03-foundations/attention.md) · [04 Architectures](../04-architectures/architectures.md) | draft |
| 2 | GPT from scratch | [05 Training](../05-training/training.md) | planned |
| 3 | MoE, state-space & hybrid models | [04 Architectures](../04-architectures/architectures.md) | planned |
| 4 | Data & scaling laws | [05 Training](../05-training/training.md) | planned |
| 5 | Post-training & RL for reasoning | [05 Training](../05-training/training.md) | planned |
| 6 | GPU architecture & rooflines | [00 Hardware](../00-hardware/hardware.md) | planned |
| 7 | CUDA & Triton kernels | [01 Systems](../01-systems/systems.md) | planned |
| 8 | FlashAttention | [01 Systems](../01-systems/systems.md) | planned |
| 9 | Collectives & parallelism | [02 Distributed](../02-distributed/distributed.md) | planned |
| 10 | KV cache, quantization, speculative decoding | [KV cache](../06-inference/kv-cache.md) · [06 Inference](../06-inference/inference.md) · [calculator](../calculators/kv-cache.ipynb) | planned |
| 11 | Serving at scale | [06 Inference](../06-inference/inference.md) | planned |
| 12 | Wrap your model in an agent | [08 Agents](../08-agents/agents.md) | planned |

**Capstone:** train a tiny chat model, quantize it, serve it, expose it over MCP, wrap it in an agent and evaluate it, comparing measured speed against roofline predictions.
