"""Provider registry tests (no network, no keys)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent import provider as prov
from agent.provider import (
    CHAT_PROVIDERS,
    PROVIDERS,
    _chat_completions_round,
    build_client,
    chat_round,
    privacy_note,
)


def _clean_env(monkeypatch) -> None:
    monkeypatch.setattr(prov, "_read_env_file_value", lambda name: None)
    for spec in CHAT_PROVIDERS.values():
        for var in (spec.key_env, spec.model_env, spec.base_url_env):
            if var:
                monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)


def test_registry_lists_every_provider() -> None:
    assert PROVIDERS[0] == "meta"
    assert {"baseten", "gemini", "groq", "openrouter", "cerebras", "ollama",
            "openai_compatible"} <= set(CHAT_PROVIDERS)
    assert set(PROVIDERS) == {"meta", *CHAT_PROVIDERS}


@pytest.mark.parametrize("name", [n for n in CHAT_PROVIDERS if n not in ("ollama", "openai_compatible")])
def test_build_client_each_hosted_provider(monkeypatch, name) -> None:
    _clean_env(monkeypatch)
    spec = CHAT_PROVIDERS[name]
    monkeypatch.setenv("MODEL_PROVIDER", name)
    monkeypatch.setenv(spec.key_env, "k-123")
    client, model, provider = build_client()
    assert provider == name
    assert model == spec.default_model
    assert spec.base_url.rstrip("/") in str(client.base_url)


@pytest.mark.parametrize("name", [n for n in CHAT_PROVIDERS if CHAT_PROVIDERS[n].needs_key])
def test_missing_key_error_never_shows_key(monkeypatch, name) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", name)
    with pytest.raises(RuntimeError, match=CHAT_PROVIDERS[name].key_env):
        build_client()


def test_model_override(monkeypatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GROQ_MODEL", "custom-model")
    assert build_client()[1] == "custom-model"


def test_ollama_needs_no_key(monkeypatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    client, model, _ = build_client()
    assert "localhost:11434" in str(client.base_url)
    assert model == "llama3.1"
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://box:11434/v1")
    assert "box:11434" in str(build_client()[0].base_url)


def test_openai_compatible_requires_url_and_model(monkeypatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("MODEL_PROVIDER", "openai_compatible")
    with pytest.raises(RuntimeError, match="OPENAI_COMPAT_BASE_URL"):
        build_client()
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://x.test/v1")
    with pytest.raises(RuntimeError, match="OPENAI_COMPAT_MODEL"):
        build_client()
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "m")
    assert build_client()[1] == "m"


def test_privacy_note() -> None:
    assert privacy_note("gemini")
    assert privacy_note("meta") == ""
    assert privacy_note("ollama") == ""


# --- strict schemas and 429 backoff -------------------------------------


class _Completions:
    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(script):
    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions(script)))


def _ok(text="hi"):
    msg = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


class _RateLimit(Exception):
    status_code = 429

    def __init__(self, retry_after=None):
        super().__init__("rate limited")
        self.response = SimpleNamespace(
            headers={"retry-after": retry_after} if retry_after else {}
        )


TOOLS = [{"type": "function", "name": "t", "strict": True,
          "parameters": {"type": "object", "properties": {}}}]


def test_non_strict_providers_never_send_strict() -> None:
    client = _client([_ok()])
    chat_round(client, "m", "groq", instructions="s", tools=TOOLS, history=[])
    sent = client.chat.completions.calls[0]["tools"][0]["function"]
    assert "strict" not in sent


def test_baseten_still_sends_strict() -> None:
    client = _client([_ok()])
    chat_round(client, "m", "baseten", instructions="s", tools=TOOLS, history=[])
    assert client.chat.completions.calls[0]["tools"][0]["function"]["strict"] is True


def test_429_retries_and_honours_retry_after() -> None:
    client = _client([_RateLimit("7"), _RateLimit(), _ok("done")])
    waits: list[float] = []
    result = _chat_completions_round(
        client, "m", CHAT_PROVIDERS["groq"], "s", [], [], sleep=waits.append
    )
    assert result.text == "done"
    assert waits == [7.0, 2.0]


def test_429_gives_up_after_limit() -> None:
    client = _client([_RateLimit()] * (prov.RATE_LIMIT_RETRIES + 1))
    waits: list[float] = []
    with pytest.raises(_RateLimit):
        _chat_completions_round(
            client, "m", CHAT_PROVIDERS["groq"], "s", [], [], sleep=waits.append
        )
    assert len(waits) == prov.RATE_LIMIT_RETRIES


def test_retry_after_is_capped() -> None:
    assert prov._retry_after_seconds(_RateLimit("9999"), 0) == prov.RATE_LIMIT_MAX_WAIT_S


def test_other_errors_not_retried() -> None:
    client = _client([RuntimeError("boom")])
    with pytest.raises(RuntimeError):
        _chat_completions_round(
            client, "m", CHAT_PROVIDERS["groq"], "s", [], [], sleep=lambda s: None
        )
    assert len(client.chat.completions.calls) == 1


def test_describe_api_error_shows_origin_without_secrets() -> None:
    response = SimpleNamespace(
        request=SimpleNamespace(url="https://x.test/v1/chat/completions?key=SECRET"),
        headers={"server": "ESF"},
        text="",
    )
    exc = RuntimeError("Error code: 404")
    exc.response = response
    out = prov.describe_api_error(exc)
    assert "url=https://x.test/v1/chat/completions" in out
    assert "server=ESF" in out and "body=<empty>" in out
    assert "SECRET" not in out


def test_describe_api_error_without_response() -> None:
    assert prov.describe_api_error(ValueError("x")) == "ValueError: x"


@pytest.mark.parametrize("status", [502, 503, 504])
def test_overload_statuses_are_retried(status) -> None:
    class Overloaded(Exception):
        status_code = status

    client = _client([Overloaded("busy"), _ok("fine")])
    waits: list[float] = []
    result = _chat_completions_round(
        client, "m", CHAT_PROVIDERS["gemini"], "s", [], [], sleep=waits.append
    )
    assert result.text == "fine" and waits == [1.0]


def test_client_errors_are_not_retried() -> None:
    class NotFound(Exception):
        status_code = 404

    client = _client([NotFound("gone")])
    with pytest.raises(NotFound):
        _chat_completions_round(
            client, "m", CHAT_PROVIDERS["gemini"], "s", [], [], sleep=lambda s: None
        )
    assert len(client.chat.completions.calls) == 1


def test_extra_content_round_trips_through_history() -> None:
    sig = {"google": {"thought_signature": "SIG"}}
    call = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(name="list_tables", arguments="{}"),
        extra_content=sig,
    )
    msg = SimpleNamespace(content=None, tool_calls=[call])
    response = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)
    result = prov._normalize_chat_response(response)
    assert result.calls[0]["extra_content"] == sig
    history = [{"type": "function_call", "call_id": "c1", "name": "list_tables",
                "arguments": "{}", "extra_content": sig}]
    sent = prov._chat_messages("SYS", history)[1]["tool_calls"][0]
    assert sent["extra_content"] == sig


def test_extra_content_read_from_pydantic_model_extra() -> None:
    sig = {"google": {"thought_signature": "SIG"}}
    call = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(name="t", arguments="{}"),
        model_extra={"extra_content": sig},
    )
    assert prov._call_extra_content(call) == sig


def test_calls_without_extra_content_unchanged() -> None:
    sent = prov._chat_messages(
        "SYS", [{"type": "function_call", "call_id": "c", "name": "t", "arguments": "{}"}]
    )[1]["tool_calls"][0]
    assert "extra_content" not in sent
