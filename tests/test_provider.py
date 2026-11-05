"""Tests for the model provider seam (no network, no key)."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent.backend import SqliteBackend  # noqa: E402
from agent.loop import answer_question  # noqa: E402
from agent.orchestrator import merge_answers, plan_datasets  # noqa: E402
from agent.provider import (  # noqa: E402
    _chat_messages,
    _chat_tools,
    build_client,
    chat_round,
    estimate_cost_usd,
    get_provider_name,
)


class _FakeCompletions:
    """Scripted stand-in for client.chat.completions."""

    def __init__(self, script: list):
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeChatClient:
    def __init__(self, script: list):
        self.chat = SimpleNamespace(completions=_FakeCompletions(script))


class _BadRequest(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _chat_call(call_id: str, name: str, arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
        ),
    )


def _chat_response(
    content: Any = "",
    tool_calls: list | None = None,
    prompt: int = 0,
    completion: int = 0,
    total: int = 0,
    cached: int = 0,
    reasoning: str = "secret chain of thought",
    usage: Any = "present",
) -> SimpleNamespace:
    message = SimpleNamespace(
        content=content,
        tool_calls=list(tool_calls or []),
        refusal=None,
        reasoning_content=reasoning,
    )
    if usage == "present":
        usage_obj = SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        )
    else:
        usage_obj = None
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
    )
    if usage_obj is not None:
        response.usage = usage_obj
    return response


def _no_env(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "baseten")
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.delenv("BASETEN_MODEL", raising=False)
    monkeypatch.setattr("agent.provider._read_env_file_value", lambda name: None)


def test_chat_messages_translation() -> None:
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "How many?"},
        {"role": "assistant", "content": "Working on it."},
        {"type": "function_call", "call_id": "c1", "name": "run_sql",
         "arguments": '{"query": "SELECT 1;"}'},
        {"type": "function_call", "call_id": "c2", "name": "list_tables",
         "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": '{"rows": [[1]]}'},
        {"type": "function_call_output", "call_id": "c2", "output": '{"tables": []}'},
    ]
    messages = _chat_messages("SYS", history)
    assert messages[0] == {"role": "system", "content": "SYS"}
    assert messages[1] == {"role": "user", "content": "How many?"}
    assert messages[2] == {"role": "assistant", "content": "Working on it."}
    assert messages[3]["role"] == "assistant"
    assert messages[3]["tool_calls"][0]["id"] == "c1"
    assert messages[3]["tool_calls"][0]["function"]["name"] == "run_sql"
    assert messages[4]["tool_calls"][0]["id"] == "c2"
    assert messages[5] == {"role": "tool", "tool_call_id": "c1",
                           "content": '{"rows": [[1]]}'}
    assert messages[6]["tool_call_id"] == "c2"


def test_chat_tool_schema_translation_keeps_strict() -> None:
    schemas = [{
        "type": "function",
        "name": "run_sql",
        "description": "Run SQL.",
        "parameters": {"type": "object", "properties": {}},
        "strict": True,
    }]
    tools = _chat_tools(schemas)
    assert tools == [{
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run SQL.",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        },
    }]


def test_malformed_tool_arguments_pass_through() -> None:
    client = _FakeChatClient([
        _chat_response(tool_calls=[_chat_call("c1", "run_sql", "not json{")])
    ])
    result = chat_round(client, "m", "baseten", instructions="SYS",
                        tools=[], history=[])
    assert result.calls == [{"id": "c1", "name": "run_sql", "arguments": "not json{"}]


def test_usage_mapping_with_cached_tokens() -> None:
    client = _FakeChatClient([
        _chat_response(content="hi", prompt=100, completion=20, total=120, cached=30)
    ])
    result = chat_round(client, "m", "baseten", instructions="SYS",
                        tools=[], history=[])
    assert result.text == "hi"
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (100, 20, 120)
    assert result.cached_input_tokens == 30
    assert result.usage_known is True


def test_missing_usage_is_unknown() -> None:
    client = _FakeChatClient([_chat_response(content="hi", usage="absent")])
    result = chat_round(client, "m", "baseten", instructions="SYS",
                        tools=[], history=[])
    assert result.usage_known is False
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (0, 0, 0)


def test_reasoning_content_never_leaks() -> None:
    client = _FakeChatClient([_chat_response(content="done")])
    result = chat_round(client, "m", "baseten", instructions="SYS",
                        tools=[], history=[])
    assert "reasoning" not in dataclasses.asdict(result)
    assert "secret chain of thought" not in json.dumps(dataclasses.asdict(result))


def test_strict_fallback_retries_without_strict() -> None:
    err = _BadRequest("400 strict mode is not supported for structured output")
    client = _FakeChatClient([err, _chat_response(content="ok")])
    tools = [{
        "type": "function", "name": "list_tables", "description": "List.",
        "parameters": {"type": "object", "properties": {}}, "strict": True,
    }]
    result = chat_round(client, "m", "baseten", instructions="SYS",
                        tools=tools, history=[])
    assert result.text == "ok"
    assert result.strict_fallback is True
    first, second = client.chat.completions.calls
    assert first["tools"][0]["function"]["strict"] is True
    assert "strict" not in second["tools"][0]["function"]


def test_unrelated_error_is_not_retried() -> None:
    client = _FakeChatClient([_BadRequest("404 model not found", status_code=404)])
    with pytest.raises(_BadRequest):
        chat_round(client, "m", "baseten", instructions="SYS", tools=[], history=[])


def test_unknown_provider_rejected(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "nowhere")
    monkeypatch.setattr("agent.provider._read_env_file_value", lambda name: None)
    with pytest.raises(RuntimeError, match="unknown MODEL_PROVIDER"):
        get_provider_name()


def test_default_provider_is_meta(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.setattr("agent.provider._read_env_file_value", lambda name: None)
    assert get_provider_name() == "meta"


def test_build_client_baseten(monkeypatch) -> None:
    _no_env(monkeypatch)
    monkeypatch.setenv("BASETEN_API_KEY", "bk-fake-key")
    client, model, provider = build_client()
    assert provider == "baseten"
    assert "inference.baseten.co" in str(client.base_url)
    assert model == "meta-models/muse-spark-open-1026"


def test_build_client_baseten_missing_key(monkeypatch) -> None:
    _no_env(monkeypatch)
    with pytest.raises(RuntimeError, match="BASETEN_API_KEY is missing"):
        build_client()


def test_build_client_meta_unchanged(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "meta")
    monkeypatch.setenv("META_API_KEY", "mk-fake-key")
    monkeypatch.delenv("META_MODEL", raising=False)
    monkeypatch.setattr("agent.provider._read_env_file_value", lambda name: None)
    client, model, provider = build_client()
    assert provider == "meta"
    assert "api.meta.ai" in str(client.base_url)
    assert model == "muse-spark-1.3-contributor"


def test_estimate_cost_math() -> None:
    assert estimate_cost_usd(1000, 200, 500) == pytest.approx(0.003155)
    assert estimate_cost_usd(0, 0, 0) == 0.0
    assert estimate_cost_usd(100, 500, 0) == pytest.approx(500 * 0.15 / 1_000_000)


def test_orchestrator_plan_and_merge_over_baseten() -> None:
    plan_text = json.dumps({"datasets": ["a"]})
    merge_args = json.dumps({"merged_text": "total 5", "use_separate": False})
    client = _FakeChatClient([
        _chat_response(content=plan_text),
        _chat_response(tool_calls=[_chat_call("m1", "record_merge", merge_args)]),
    ])
    mapping = {"a": {"description": "A data.", "backend": object(), "instructions": "I"}}
    names, _tokens = plan_datasets("q?", mapping, client, "m", "baseten")
    assert names == ["a"]
    merged, verification, _t2 = merge_answers(
        "How many?", [{"dataset": "a", "text": "There are 5 rows."}],
        client, "m", "baseten",
    )
    assert merged == "total 5"
    assert verification["status"] == "verified"
    for call in client.chat.completions.calls:
        assert "reasoning" not in json.dumps(call.get("messages", []))


def _stub_backend(tmp: str) -> SqliteBackend:
    path = Path(tmp) / "prov.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY);")
        conn.executemany("INSERT INTO items (id) VALUES (?)", [(1,), (2,), (3,)])
        conn.commit()
    finally:
        conn.close()
    return SqliteBackend(path)


def _tool_round(name: str, args: dict, prompt: int = 10, cached: int = 0):
    return _chat_response(
        content=None,
        tool_calls=[_chat_call("c-" + name, name, args)],
        prompt=prompt, completion=5, total=prompt + 5, cached=cached,
    )


def test_answer_question_full_turn_over_baseten() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = [
                _tool_round("set_checklist", {"items": [
                    {"id": "n", "question": "How many?",
                     "evidence_needed": "count"}]}),
                _tool_round("plan_action", {"plan": [
                    {"item_id": "n", "steps": ["count"]}]}),
                _tool_round("run_sql", {"query": "SELECT count(*) FROM items;"},
                            cached=7),
                _tool_round("record_finding", {
                    "item_id": "n", "claim": "3 items",
                    "query_id": "q1", "value": 3}),
                _tool_round("submit_answer", {
                    "answer": "There are 3 items.",
                    "decisions": [{"what": "scope", "choice": "everything",
                                   "source": "user_stated", "proof": "How many?"}],
                    "gaps": []}),
            ]
            history: list = []
            result = answer_question(
                _FakeChatClient(script), "m", "instr", backend, history,
                "How many?", runs_dir=Path(tmp) / "runs", provider="baseten",
            )
            assert result.text == (
                "There are 3 items.\n\nDecisions:\n- scope: everything "
                "(you stated: \"How many?\")\n\nGaps:\n- none found\n\n"
                "Want the evidence behind these decisions?"
            )
            assert result.verification == "verification: 1 numbers matched, 0 unmatched"
            assert result.cached_input_tokens == 7
            assert "secret chain of thought" not in json.dumps(history)
        finally:
            backend.close()
