---
id: agents.agent-loop
title: The Agent Loop
layer: 08-agents
pathways: [builder]
level: intro
prereqs: [applications.prompting, applications.rag]
status: evolving
last_reviewed: 2026-11-15
review_every_days: 120
---

# The Agent Loop

## TL;DR

An agent is a model in a loop with tools. You describe the tools; the model replies either with an answer or with a **tool call** (a tool name plus arguments); your code runs the tool and sends the result back; repeat until the model answers. The model never executes anything itself, so the code around it decides what's safe: validating arguments, turning failures into messages the model can react to, asking a human before risky actions, and capping the number of steps. Each step re-sends the whole conversation, so long runs get expensive fast without prompt caching.

## Intuition

On its own, a model can only produce text from what it already knows. Give it tools and it can look things up, calculate exactly, and act: search documents, run code, call an API, edit a file.

Picture a very capable colleague working through a ticket, where every action goes through you. They say "please search the docs for X". You do it and hand back the result. They read it and say "now run this calculation", and so on, until they write the final answer. That back-and-forth is the agent loop. "Agent" doesn't mean anything more mysterious than this; the interesting part is how well the model chooses its next step.

## How it works

```text
messages = [system, user question]
loop (at most N steps):
    response = model(messages, tool_descriptions)
    if response has no tool calls:  return response.text      # done
    for each tool call:
        result = run the tool (or an error message)
        messages += [the call, the result]
```

### Tool calling

1. **Describe each tool**: a name, a description of what it does and when to use it, and a JSON schema for its arguments. This is all the model sees.
2. **The model emits a structured call**, such as `{"name": "search_docs", "arguments": {"query": "KV cache size"}}`. Models are fine-tuned to produce these in a dedicated format, often with structured-output constraints underneath (see [Prompting](../07-applications/prompting.md)).
3. **Your code executes it**: look up the tool, validate the arguments, run it.
4. **The result goes back** as a tool message, linked to the call by an ID, and the model takes its next step.

Models can request several calls in one turn when they're independent, such as three searches at once, which saves round trips.

### Reason, then act

The ReAct pattern interleaves reasoning with actions: think about what's needed, call a tool, read the result, think again. Modern models do this natively, often writing a short plan before each call. It lets the model use one result to decide the next step, which a single retrieval pass (plain [RAG](../07-applications/rag.md)) can't.

### What the surrounding code must handle

- **Errors as messages.** An unknown tool, a bad argument or a crashing tool should come back to the model as an error result, not stop the program. Models are good at reading an error and retrying correctly.
- **Validation.** Check arguments against the schema before running anything.
- **Approval for consequential actions.** Sending email, spending money, deleting data: require a human's OK, or limit the tool's power.
- **A step limit.** Models sometimes loop. Always cap the steps (and time and cost).
- **Never `eval` model output.** Treat arguments like untrusted user input. The lab's calculator parses arithmetic itself rather than running Python.

### Workflows versus agents

Not every task needs an open-ended loop. A **workflow** runs model calls along a path you designed (retrieve, then summarize, then check); an **agent** lets the model decide the path. Workflows are cheaper, faster and more predictable; agents handle open-ended tasks where the steps can't be known in advance. A common recommendation is to start with the simplest design that works and add autonomy only when it pays for itself.

### What it costs

Every step sends the full conversation again: system prompt, tool definitions, the question and every earlier call and result. Without caching, the tokens processed grow with the square of the number of steps. With [prefix caching](../07-applications/prompting.md), each step only processes what's new, because the earlier conversation is an unchanged prefix whose [KV cache](../06-inference/kv-cache.md) can be reused.

## Code

[`s2a.agent`](../src/s2a/agent.py) is a complete agent loop in about 150 lines, with tool schemas, argument validation, error handling, an approval gate, a step limit, a safe calculator tool and cost arithmetic; it's tested in CI. The [agent loop lab](../labs/agent-loop.ipynb) gives an agent three tools built from earlier labs (calculator, search over this repo, KV-cache sizing) and runs it with stand-in models, no API key needed.

```python
from s2a.agent import run_agent, RuleBasedModel, CALCULATOR

result = run_agent(RuleBasedModel(), [CALCULATOR], "What is 4096 * 14336?")
print(result.trace())
#   user   :  What is 4096 * 14336?
#   model  -> call calculator({"expression": "4096 * 14336"})
#   tool   -> 58720256
#   model  -> answer: Based on the tool result: 58720256
```

To use a real model, pass a function that calls your provider's API and converts its tool-call format into `ModelResponse`.

## Numbers that matter

**Tokens processed over one agent run**, from the lab: a 10,000-token base (system prompt, tools, question), with each step's call and result adding about 1,000 tokens.

| Steps | Without caching | With prefix caching | Ratio |
|---|---|---|---|
| 1 | 10,000 | 10,000 | 1.0× |
| 10 | 145,000 | 19,000 | 7.6× |
| 20 | 390,000 | 29,000 | 13.4× |
| 50 | 1,725,000 | 59,000 | 29.2× |

The gap widens with every step. This is why prompt caching matters more for agents than for single questions, and why long-running agents also compact or summarize old steps (see context engineering in [07 Applications](../07-applications/applications.md)).

**Tool definitions take space too.** Every tool's name, description and schema is in the prompt on every step. Dozens of detailed tools can add thousands of tokens before the conversation even starts.

## Current consensus (as of 2026-11)

- Native tool calling is a standard feature of model APIs, and models are trained specifically to use tools well.
- Clear tool names and descriptions matter about as much as prompts do: they're how the model decides what to call.
- The Model Context Protocol (MCP) has become a common way to package tools so any agent can use them (see the protocols topic in [08 Agents](agents.md)).
- Start with simple workflows, add agentic loops where tasks are open-ended, and invest in evaluation and guardrails before autonomy.

## What changed

- 2026-11: First version of this page.

## Open questions

- How should agents handle very long tasks (hours, thousands of steps) without losing track, while keeping cost under control?

## Resources

- Yao et al., [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629) (2022) (B, P)
- Schick et al., [*Toolformer: Language Models Can Teach Themselves to Use Tools*](https://arxiv.org/abs/2302.04761) (2023) (P)
- Anthropic, [*Building effective agents*](https://www.anthropic.com/engineering/building-effective-agents) (2024) (B)
- Anthropic, [*Tool use*](https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview) documentation (B)
- OpenAI, [*Function calling*](https://platform.openai.com/docs/guides/function-calling) guide (B)
- Hugging Face, [*Agents Course*](https://huggingface.co/learn/agents-course) (B)

## Check yourself

Try each one before opening the answer.

**1. A tool raises an exception. Why send the error back to the model instead of stopping the run?**

<details>
<summary>Answer</summary>

Many failures are recoverable, and the model is usually the best placed to recover: it can fix a malformed argument, try a different tool, or tell the user what went wrong. In the lab, a model that calls a tool by the wrong name, then passes the wrong type, then divides by zero still reaches the right answer, because each error comes back as a message it can read. Stopping the program throws all that away. (Errors in your own loop code are different; those are bugs to fix.)

</details>

**2. An agent run takes 30 steps and costs far more than expected, even though each step adds only a little text. Why?**

<details>
<summary>Answer</summary>

Each step re-sends the entire conversation so far. With a large base (system prompt and tool definitions) and growing history, step 30 processes the base plus 29 steps' worth of calls and results, and the total over the run grows with the square of the number of steps. Fixes: make sure prefix caching applies (keep the start of the prompt unchanged), trim or summarize old tool results, reduce the number and size of tool definitions, and cap the steps.

</details>

**3. You're building an agent that can read a user's inbox and send email. Name three safeguards.**

<details>
<summary>Answer</summary>

Any three of these:

- Require human approval before sending.
- Give it only the permissions it needs, such as drafting without sending, or sending only to the user.
- Treat email contents as untrusted data, since a message can contain injected instructions (see [Prompting](../07-applications/prompting.md)).
- Validate tool arguments.
- Cap steps and rate-limit actions.
- Log every tool call for review.

The common thread: the model proposes, the surrounding code decides what actually happens.

</details>
