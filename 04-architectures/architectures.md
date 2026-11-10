# 04 · Architectures

What the network is: how the building blocks are arranged, and the alternatives to the standard transformer.

## Topics

| Topic | What it covers | Status |
|---|---|---|
| [The transformer block](transformer.md) | Residual stream, pre-norm RMSNorm, GQA + RoPE, SwiGLU, where the parameters live | **draft** |
| Mixture-of-Experts | Routing, shared experts, load balancing, fine-grained experts | planned |
| State-space & linear attention | Mamba, RWKV, DeltaNet and Gated DeltaNet | planned |
| Hybrid models | Mixing linear and full attention layers | planned |
| Other directions | Diffusion LMs, recurrent depth, byte-level models, long context | planned |
| Multimodal | Vision encoders, connectors, native multimodal models | planned |

## Start here

- Vaswani et al., *Attention Is All You Need* (2017) (P)
- Sebastian Raschka, *The Big LLM Architecture Comparison* and the LLM Architecture Gallery (B, P)
