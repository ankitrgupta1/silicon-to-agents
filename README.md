# Silicon to Agents

A living knowledge base, with runnable code, that covers how modern AI works across the whole stack: from the silicon that does the math, through kernels, training and inference, up to the agents built on top.

> **Living document.** The field moves fast, so each topic page records when it was last reviewed and what changed. See the [changelog](CHANGELOG.md) for updates.

## The stack

```text
08  Agents          tool use · MCP / A2A · planning · memory · multi-agent
07  Applications    prompting · RAG · context engineering · LLMOps
06  Inference       KV cache · quantization · speculative decoding · serving
05  Training        data · pretraining · post-training · RL & reasoning
04  Architectures   transformer · MoE · state-space & hybrid models
03  Foundations     tokenization · embeddings · attention
02  Distributed     collectives · data/tensor/pipeline/expert parallelism
01  Systems         CUDA · Triton · kernels · compilers · profiling
00  Hardware        GPUs · TPUs · HBM · interconnects · datacenters
──────────────────────────────────────────────────────────────────────
    Cross-cutting   evals · safety & security · cost
    Prereqs         math · Python & PyTorch · ML basics
```

| Layer | Folder |
|---|---|
| Prerequisites | [prereqs](prereqs/prereqs.md) |
| 00 Hardware | [00-hardware](00-hardware/hardware.md) |
| 01 Systems | [01-systems](01-systems/systems.md) |
| 02 Distributed | [02-distributed](02-distributed/distributed.md) |
| 03 Foundations | [03-foundations](03-foundations/foundations.md) |
| 04 Architectures | [04-architectures](04-architectures/architectures.md) |
| 05 Training | [05-training](05-training/training.md) |
| 06 Inference | [06-inference](06-inference/inference.md) |
| 07 Applications | [07-applications](07-applications/applications.md) |
| 08 Agents | [08-agents](08-agents/agents.md) |
| Cross-cutting | [cross-cutting](cross-cutting/cross-cutting.md) |

## Choose your path

The folders follow the order of the stack, but you don't have to learn in that order. Each pathway is an ordered route through the same topic pages.

| Path | For | Starts at |
|---|---|---|
| [Fundamentals](pathways/fundamentals.md) | New to ML; wants the math and intuition first | Prerequisites → attention → a tiny GPT |
| [Builder](pathways/builder.md) | Builds apps and agents; wants just enough internals | Applications and agents, then down to inference costs |
| [Practitioner](pathways/practitioner.md) | Wants the deep internals | GPT from scratch → kernels → distributed → RL → serving at scale |

## Read it and run it

- **Website:** this repo is published as a site with Jupyter Book on GitHub Pages.
- **Notebooks:** each notebook has an *Open in Colab* button. CPU examples run anywhere; GPU examples use Colab's GPUs.
- **Code:** reusable code lives in [`src/s2a`](src/s2a/__init__.py) and is tested in CI.

```bash
pip install -e ".[dev]"
pytest
```

## What makes this different

Excellent single-layer resources already exist, and this repo links to them rather than rebuilding them. It focuses on the connections between layers:

- **Traces** follow one request across the stack, for example an agent's tool call down to the bytes it moves through HBM and back.
- **Calculators** for KV-cache size, rooflines, training memory and cost per token, driven by hardware spec cards.
- **Dated synthesis.** Every topic has a "Current consensus (as of …)" box and a "What changed" log.

## Repository layout

```text
pathways/        learning routes (fundamentals, builder, practitioner)
prereqs/ … 08-agents/   one folder per layer of the stack
cross-cutting/   evals, safety & security, cost
calculators/     interactive calculators (KV cache, …)
labs/            hands-on notebooks (tokenizer, …)
src/s2a/         tested, reusable Python code
traces/          vertical slices through the stack (planned)
capstone/        train → serve → MCP → agent → eval (planned)
_templates/      template for new topic pages
```

Contributions follow the [topic template](_templates/topic.md); AI coding agents should read [AGENTS.md](AGENTS.md).

## License

Code is under the [MIT License](https://github.com/ankitrgupta1/silicon-to-agents/blob/main/LICENSE). Written content is under [CC BY 4.0](https://github.com/ankitrgupta1/silicon-to-agents/blob/main/LICENSE-CONTENT): reuse it freely with credit.
