# 05 · Training

How a model gets its behavior: data, pretraining, post-training and reinforcement learning for reasoning.

## Topics

| Topic | What it covers | Status |
|---|---|---|
| [Train a tiny GPT](tiny-gpt.md) | The training loop: batches, backprop, AdamW, held-out loss, sampling | **draft** |
| Data | Web crawls, deduplication, quality filtering, mixing, synthetic data | planned |
| Pretraining & scaling laws | Chinchilla, over-training, μP, LR schedules, stability | planned |
| Post-training | SFT, chat templates, LoRA/QLoRA, preference optimization (RLHF, DPO) | planned |
| RL for reasoning | Verifiable rewards, GRPO, reasoning models, reward hacking | planned |
| Test-time compute | Chain of thought, best-of-N, verifier-guided search, thinking budgets | planned |
| Distillation & merging | Teaching small models from large ones; combining models | planned |
| Alignment & interpretability | Constitutional approaches, refusals, probing, sparse autoencoders | planned |

## Start here

- Nathan Lambert, *RLHF Book* (rlhfbook.com) (B, P)
- Sebastian Raschka, *Build a Reasoning Model (From Scratch)* (P)
- DeepSeekMath (GRPO) and DeepSeek-R1 papers (P)
