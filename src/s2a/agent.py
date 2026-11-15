"""The agent loop: a model, some tools, and a loop that connects them.

The model never runs anything itself. It sees tool descriptions, and when it
wants one it returns a structured *tool call* (a name plus arguments). This
code validates the call, runs the tool, and sends the result back as a new
message. The loop repeats until the model answers in plain text, or a step
limit is reached.

Any model can plug in: it's a function `model(messages, tool_schemas) ->
ModelResponse`. `ScriptedModel` and `RuleBasedModel` are stand-ins so the
loop can be explored without an API key.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable

# --- tools ------------------------------------------------------------------

_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict              # JSON Schema: {"type": "object", "properties": ..., "required": [...]}
    fn: Callable[..., Any]
    requires_approval: bool = False

    def schema(self) -> dict:
        """What the model is shown: name, description and parameter schema (not the code)."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def validate_args(parameters: dict, args: dict) -> list[str]:
    """Check arguments against a (simple) JSON schema. Returns a list of problems."""
    props, problems = parameters.get("properties", {}), []
    for name in parameters.get("required", []):
        if name not in args:
            problems.append(f"missing required argument '{name}'")
    for name, value in args.items():
        if name not in props:
            problems.append(f"unexpected argument '{name}'")
            continue
        expected = _TYPES.get(props[name].get("type"))
        bad_bool = isinstance(value, bool) and props[name].get("type") in ("number", "integer")
        if expected and (not isinstance(value, expected) or bad_bool):
            problems.append(f"argument '{name}' should be {props[name]['type']}, got {type(value).__name__}")
    return problems


# --- messages ---------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    args: dict
    id: str = ""


@dataclass
class ModelResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class AgentResult:
    answer: str | None
    messages: list[dict]
    steps: int
    stopped_because: str          # "answered" or "max_steps"

    def trace(self) -> str:
        """A readable log of the run."""
        lines = []
        for m in self.messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                if m.get("content"):
                    lines.append(f"  model  :  {m['content'][:120]}")
                for c in m["tool_calls"]:
                    lines.append(f"  model  -> call {c['name']}({json.dumps(c['arguments'])})")
            elif m["role"] == "assistant":
                lines.append(f"  model  -> answer: {m['content'][:200]}")
            elif m["role"] == "tool":
                lines.append(f"  tool   -> {m['content'][:120]}")
            else:
                lines.append(f"  {m['role']:<6} :  {m['content'][:120]}")
        return "\n".join(lines)


# --- the loop ---------------------------------------------------------------

Model = Callable[[list[dict], list[dict]], ModelResponse]


def run_agent(model: Model, tools: list[Tool], user_message: str, system: str | None = None,
              max_steps: int = 8, approve: Callable[[ToolCall], bool] | None = None) -> AgentResult:
    """Call the model, run any tools it asks for, feed results back; repeat until it answers."""
    registry = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    messages: list[dict] = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": user_message})

    for step in range(1, max_steps + 1):
        response = model(messages, schemas)
        if not response.tool_calls:                       # plain answer: we're done
            messages.append({"role": "assistant", "content": response.text or ""})
            return AgentResult(response.text, messages, step, "answered")

        calls = [ToolCall(c.name, c.args, c.id or f"call_{step}_{i}") for i, c in enumerate(response.tool_calls)]
        messages.append({"role": "assistant", "content": response.text,
                         "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.args} for c in calls]})
        for call in calls:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": _execute(call, registry, approve)})

    return AgentResult(None, messages, max_steps, "max_steps")


def _execute(call: ToolCall, registry: dict[str, Tool], approve) -> str:
    """Run one tool call. Every failure becomes a message the model can read and react to."""
    tool = registry.get(call.name)
    if tool is None:
        return f"Error: unknown tool '{call.name}'. Available: {', '.join(registry)}"
    problems = validate_args(tool.parameters, call.args)
    if problems:
        return "Error: " + "; ".join(problems)
    if tool.requires_approval and not (approve and approve(call)):
        return f"Error: the user did not approve running '{call.name}'."
    try:
        result = tool.fn(**call.args)
    except Exception as e:                                # report, don't crash the loop
        return f"Error: {type(e).__name__}: {e}"
    return result if isinstance(result, str) else json.dumps(result)


# --- stand-in models --------------------------------------------------------

class ScriptedModel:
    """Replays a fixed list of responses, one per call. Useful for testing the loop itself."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses, self.calls = list(responses), 0

    def __call__(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        self.calls += 1
        return self.responses.pop(0) if self.responses else ModelResponse(text="(script ended)")


class RuleBasedModel:
    """A crude stand-in for an LLM's tool-use decisions, using simple rules.

    - First turn: arithmetic in the question -> `calculator`; otherwise -> `search_docs`.
    - After a tool result: answer from it. After a tool error: answer that it failed.
    A real model makes these choices from the tool descriptions and the conversation.
    """

    _MATH = re.compile(r"[\d.]+(?:\s*[-+*/^]\s*[\d.()]+)+")

    def __call__(self, messages: list[dict], tools: list[dict]) -> ModelResponse:
        names = {t["name"] for t in tools}
        last = messages[-1]
        if last["role"] == "tool":
            if last["content"].startswith("Error"):
                return ModelResponse(text=f"I couldn't complete that: {last['content']}")
            return ModelResponse(text=f"Based on the tool result: {last['content'][:300]}")
        question = last["content"]
        math = self._MATH.search(question)
        if math and "calculator" in names:
            return ModelResponse(tool_calls=[ToolCall("calculator", {"expression": math.group().replace("^", "**")})])
        if "search_docs" in names:
            return ModelResponse(tool_calls=[ToolCall("search_docs", {"query": question})])
        return ModelResponse(text="I don't have a tool for that.")


# --- a safe calculator tool -------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod}


def calculator(expression: str) -> float:
    """Evaluate +, -, *, /, **, % on numbers. Anything else is refused (never use eval on model output)."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression: {ast.dump(node)[:40]}")
    return ev(ast.parse(expression, mode="eval"))


CALCULATOR = Tool(
    name="calculator",
    description="Evaluate an arithmetic expression with + - * / ** %. Use for any exact calculation.",
    parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    fn=calculator,
)


# --- cost of the loop -------------------------------------------------------

def tokens_processed(base: int, per_step: int, steps: int, cached: bool) -> int:
    """Input tokens the model processes over an agent run.

    Every step re-sends the whole conversation: a fixed `base` (system prompt,
    tools, question) plus `per_step` tokens added by each previous step's call
    and result. Without caching, step k processes base + (k-1)*per_step tokens;
    with a perfect prefix cache, only the new tokens are processed.
    """
    if min(base, per_step, steps) < 0:
        raise ValueError("token counts must be non-negative")
    if cached:
        return base + per_step * max(steps - 1, 0)
    return sum(base + (k - 1) * per_step for k in range(1, steps + 1))
