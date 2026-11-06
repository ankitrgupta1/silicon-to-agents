# 06 · Inference

Running a trained model fast and cheaply: from a single forward pass to serving at cluster scale.

## Topics

| Topic | What it covers | Status |
|---|---|---|
| Inference fundamentals | Prefill vs decode, batching, TTFT and throughput, sampling, structured output | planned |
| KV cache | Sizing math, PagedAttention, prefix caching, KV quantization and offload | planned |
| Quantization | GPTQ, AWQ, FP8/FP4, GGUF | planned |
| Speculative decoding | Draft models, Medusa, EAGLE, multi-token prediction heads | planned |
| Serving engines | vLLM, SGLang, TensorRT-LLM, llama.cpp/Ollama, MLX | planned |
| Disaggregated serving | Splitting prefill and decode; KV-aware routing | planned |
| Inference cost | Tokens per dollar, latency targets | planned |

## Try it

- [KV-cache calculator](../calculators/kv-cache.ipynb)

## Start here

- vLLM's PagedAttention paper; the FlashAttention papers (B, P)
- *How to Scale Your Model*, "All About Transformer Inference" chapter (P)
