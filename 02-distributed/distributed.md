# 02 · Distributed

Training and serving models too big for one device: how work and memory are split across many accelerators.

## Topics

| Topic | What it covers | Status |
|---|---|---|
| Collectives | All-reduce, all-gather, all-to-all; NCCL; ring vs tree | planned |
| Parallelism | Data, tensor, sequence/context, pipeline and expert parallelism | planned |
| ZeRO & FSDP | Sharding optimizer state, gradients and weights | planned |
| Memory techniques | Activation checkpointing, gradient accumulation, mixed precision, FP8 training | planned |
| Frameworks | Megatron-Core, DeepSpeed, TorchTitan, JAX/MaxText | planned |
| Reliability at scale | Checkpointing, fault tolerance, loss spikes | planned |
| RL infrastructure | Separating rollouts from learning; inference engines in the loop | planned |

## Start here

- Hugging Face, *The Ultra-Scale Playbook: Training LLMs on GPU Clusters* (P)
- Megatron-LM, ZeRO and PyTorch FSDP papers (P)
