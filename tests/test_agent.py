from pathlib import Path

import pytest

from s2a.agent import (
    CALCULATOR,
    ModelResponse,
    RuleBasedModel,
    ScriptedModel,
    Tool,
    ToolCall,
    calculator,
    run_agent,
    tokens_processed,
    validate_args,
)
from s2a.rag import Retriever, load_markdown_corpus

ROOT = Path(__file__).resolve().parent.parent


def test_calculator_is_safe():
    assert calculator("2 + 3 * 4") == 14
    assert calculator("-(2 ** 10) % 7") == -(2 ** 10) % 7
    for bad in ["__import__('os')", "open('x')", "[1, 2]", "2 ** 1000000", "True + 1"]:
        with pytest.raises((ValueError, SyntaxError)):
            calculator(bad)


def test_validate_args():
    params = CALCULATOR.parameters
    assert validate_args(params, {"expression": "1+1"}) == []
    assert "missing required argument 'expression'" in validate_args(params, {})
    assert any("should be string" in p for p in validate_args(params, {"expression": 5}))
    assert any("unexpected" in p for p in validate_args(params, {"expression": "1", "x": 1}))
    num = {"properties": {"n": {"type": "integer"}}}
    assert validate_args(num, {"n": True}) != []           # booleans aren't integers here


def test_multi_step_run_feeds_results_back():
    model = ScriptedModel([
        ModelResponse(tool_calls=[ToolCall("calculator", {"expression": "128 * 1024"})]),
        ModelResponse(tool_calls=[ToolCall("calculator", {"expression": "131072 * 8192"})]),
        ModelResponse(text="1 GiB"),
    ])
    result = run_agent(model, [CALCULATOR], "How big is an 8k-token KV cache?")
    assert result.answer == "1 GiB" and result.steps == 3 and result.stopped_because == "answered"
    tool_msgs = [m["content"] for m in result.messages if m["role"] == "tool"]
    assert tool_msgs == ["131072", "1073741824"]
    assert result.messages[1]["tool_calls"][0]["id"] == result.messages[2]["tool_call_id"]


def test_errors_become_messages_not_crashes():
    model = ScriptedModel([
        ModelResponse(tool_calls=[
            ToolCall("weather", {"city": "Paris"}),                  # unknown tool
            ToolCall("calculator", {"expr": "1+1"}),                 # wrong argument name
            ToolCall("calculator", {"expression": "1/0"}),           # tool raises
        ]),
        ModelResponse(text="done"),
    ])
    result = run_agent(model, [CALCULATOR], "q")
    errors = [m["content"] for m in result.messages if m["role"] == "tool"]
    assert errors[0].startswith("Error: unknown tool 'weather'")
    assert "missing required argument 'expression'" in errors[1]
    assert "ZeroDivisionError" in errors[2]
    assert result.answer == "done"


def test_approval_gate():
    sent = []
    send = Tool("send_email", "Send an email.", {"type": "object", "properties": {"to": {"type": "string"}},
                "required": ["to"]}, fn=lambda to: sent.append(to) or "sent", requires_approval=True)
    script = [ModelResponse(tool_calls=[ToolCall("send_email", {"to": "a@example.com"})]), ModelResponse(text="ok")]
    denied = run_agent(ScriptedModel(list(script)), [send], "q")
    assert "did not approve" in denied.messages[-2]["content"] and sent == []
    run_agent(ScriptedModel(list(script)), [send], "q", approve=lambda call: True)
    assert sent == ["a@example.com"]


def test_step_limit_stops_runaway_loops():
    looping = ScriptedModel([ModelResponse(tool_calls=[ToolCall("calculator", {"expression": "1+1"})])] * 20)
    result = run_agent(looping, [CALCULATOR], "q", max_steps=4)
    assert result.stopped_because == "max_steps" and result.answer is None and looping.calls == 4


def test_rule_based_model_with_real_tools():
    retriever = Retriever(load_markdown_corpus(ROOT))
    search = Tool("search_docs", "Search the knowledge base.",
                  {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                  fn=lambda query: [c.source for c in retriever.search(query, k=3)])
    math = run_agent(RuleBasedModel(), [CALCULATOR, search], "What is 4096 * 14336?")
    assert "58720256" in math.answer and math.steps == 2
    docs = run_agent(RuleBasedModel(), [CALCULATOR, search], "How does byte pair encoding work?")
    assert "tokenization.md" in docs.answer


def test_tokens_processed():
    assert tokens_processed(10_000, 1_000, 10, cached=False) == 145_000
    assert tokens_processed(10_000, 1_000, 10, cached=True) == 19_000
    assert tokens_processed(500, 0, 1, cached=False) == 500
