"""Model provider seam: Meta (Responses API) or any Chat Completions provider.

The loop and orchestrator above this module see only RoundResult; every
provider difference (endpoint, key, message format, tool schema shape,
usage fields) lives here. The Meta path is the original behaviour. Every
other provider is one entry in CHAT_PROVIDERS and shares one code path.

Chat Completions notes: the OpenAI SDK points at the provider's base URL
with Bearer auth. `reasoning_content` is deliberately ignored everywhere:
it is never read into the result, history, records, logs or errors.

Default model names and free-tier limits change often. Treat every default
below as a starting point to check against the provider's current docs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

META_BASE_URL = "https://api.meta.ai/v1"
META_MODEL_DEFAULT = "muse-spark-1.3-contributor"
BASETEN_BASE_URL = "https://inference.baseten.co/v1"
BASETEN_MODEL_DEFAULT = "meta-models/muse-spark-open-1026"


@dataclass(frozen=True)
class ChatProvider:
    """One Chat Completions provider."""

    name: str
    label: str
    base_url: str
    key_env: str
    model_env: str
    default_model: str
    #: False for local servers (Ollama) that need no key.
    needs_key: bool = True
    #: True only where strict tool schemas are known to work (with a
    #: one-time fallback if the server rejects them).
    strict_tools: bool = False
    #: Env variable that overrides base_url ("" = not overridable).
    base_url_env: str = ""
    #: Shown in the console; free tiers may use prompts for training.
    privacy_note: str = ""


CHAT_PROVIDERS: dict[str, ChatProvider] = {
    p.name: p
    for p in (
        ChatProvider(
            name="baseten",
            label="Baseten",
            base_url=BASETEN_BASE_URL,
            key_env="BASETEN_API_KEY",
            model_env="BASETEN_MODEL",
            default_model=BASETEN_MODEL_DEFAULT,
            strict_tools=True,
        ),
        ChatProvider(
            name="gemini",
            label="Google Gemini (OpenAI-compatible endpoint)",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            key_env="GEMINI_API_KEY",
            model_env="GEMINI_MODEL",
            default_model="gemini-3.8-flash",
            privacy_note="Free tier may use prompts to improve products.",
        ),
        ChatProvider(
            name="groq",
            label="Groq",
            base_url="https://api.groq.com/openai/v1",
            key_env="GROQ_API_KEY",
            model_env="GROQ_MODEL",
            default_model="llama-3.3-70b-versatile",
        ),
        ChatProvider(
            name="openrouter",
            label="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            key_env="OPENROUTER_API_KEY",
            model_env="OPENROUTER_MODEL",
            default_model="meta-llama/llama-3.3-70b-instruct:free",
            privacy_note="Free models may log prompts; check each model.",
        ),
        ChatProvider(
            name="cerebras",
            label="Cerebras",
            base_url="https://api.cerebras.ai/v1",
            key_env="CEREBRAS_API_KEY",
            model_env="CEREBRAS_MODEL",
            default_model="llama-3.3-70b",
        ),
        ChatProvider(
            name="ollama",
            label="Ollama (local)",
            base_url="http://localhost:11434/v1",
            key_env="OLLAMA_API_KEY",
            model_env="OLLAMA_MODEL",
            default_model="llama3.1",
            needs_key=False,
            base_url_env="OLLAMA_BASE_URL",
        ),
        ChatProvider(
            name="openai_compatible",
            label="OpenAI-compatible endpoint",
            base_url="",
            key_env="OPENAI_COMPAT_API_KEY",
            model_env="OPENAI_COMPAT_MODEL",
            default_model="",
            needs_key=False,
            base_url_env="OPENAI_COMPAT_BASE_URL",
        ),
    )
}

PROVIDERS = ("meta", *CHAT_PROVIDERS)

# Baseten list-price estimate (USD per 1M tokens). Easy to correct; never
# presented as anything but an estimate, and never used for other providers.
BASETEN_PRICE_AS_OF = "4 Nov 2026"
BASETEN_PRICE_INPUT_PER_M = 1.25
BASETEN_PRICE_CACHED_INPUT_PER_M = 0.15
BASETEN_PRICE_OUTPUT_PER_M = 4.25

#: Transient HTTP statuses worth retrying: rate limit and overload/gateway.
RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})

#: Retries for those statuses on Chat Completions providers.
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_MAX_WAIT_S = 30.0


def estimate_cost_usd(
    prompt_tokens: int, cached_tokens: int, completion_tokens: int
) -> float:
    """Baseten list-price estimate. Output tokens already include reasoning."""
    uncached = max(prompt_tokens - cached_tokens, 0)
    return (
        uncached * BASETEN_PRICE_INPUT_PER_M
        + cached_tokens * BASETEN_PRICE_CACHED_INPUT_PER_M
        + completion_tokens * BASETEN_PRICE_OUTPUT_PER_M
    ) / 1_000_000


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_env_file_value(name: str) -> str | None:
    """Read one KEY=value line from .env without sourcing the whole file."""
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return None
    prefix = name + "="
    try:
        for line in env_file.read_text().splitlines():
            if line.startswith(prefix):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        return None
    return None


def get_provider_name() -> str:
    """Selected provider: MODEL_PROVIDER (env, then .env), default meta."""
    name = (
        os.environ.get("MODEL_PROVIDER")
        or _read_env_file_value("MODEL_PROVIDER")
        or "meta"
    ).strip().lower()
    if name not in PROVIDERS:
        raise RuntimeError(
            f"unknown MODEL_PROVIDER {name!r}; use one of: {', '.join(PROVIDERS)}"
        )
    return name


def _setting(name: str) -> str | None:
    """Environment first, then .env."""
    return os.environ.get(name) or _read_env_file_value(name)


def build_client() -> tuple[OpenAI, str, str]:
    """Build the API client for the selected provider.

    Returns (client, model, provider). Raises a clear error if a required
    setting is missing. Keys and DSNs are never printed, logged or put in
    errors.
    """
    provider = get_provider_name()
    spec = CHAT_PROVIDERS.get(provider)
    if spec is not None:
        base_url = (spec.base_url_env and _setting(spec.base_url_env)) or spec.base_url
        if not base_url:
            raise RuntimeError(
                f"{spec.base_url_env} is missing: set it to the endpoint URL "
                f"for provider {spec.name!r}."
            )
        api_key = _setting(spec.key_env)
        if not api_key and spec.needs_key:
            raise RuntimeError(
                f"{spec.key_env} is missing: add it to .env as "
                f"{spec.key_env}=<key> or export it in the environment. "
                "The key itself is never printed."
            )
        model = _setting(spec.model_env) or spec.default_model
        if not model:
            raise RuntimeError(
                f"{spec.model_env} is missing: set the model name for "
                f"provider {spec.name!r}."
            )
        # Local servers ignore the key, but the SDK requires a non-empty one.
        client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        return client, model, provider
    api_key = _setting("META_API_KEY")
    if not api_key:
        raise RuntimeError(
            "META_API_KEY is missing: add it to .env as META_API_KEY=<key> "
            "or export it in the environment. The key itself is never printed."
        )
    model = _setting("META_MODEL") or META_MODEL_DEFAULT
    return OpenAI(base_url=META_BASE_URL, api_key=api_key), model, provider


def describe_api_error(exc: BaseException) -> str:
    """One-line description of an API failure, safe to print.

    Adds where the answer came from (URL without query string, the Server
    header) and the start of the response body, which is what tells a
    provider error apart from a proxy or wrong-endpoint answer. Request
    headers (the key) are never touched.
    """
    text = f"{type(exc).__name__}: {exc}"
    response = getattr(exc, "response", None)
    if response is None:
        return text
    parts = []
    try:
        url = getattr(getattr(response, "request", None), "url", None)
        if url is not None:
            parts.append("url=" + str(url).split("?", 1)[0])
        server = response.headers.get("server")
        if server:
            parts.append(f"server={server}")
        body = (response.text or "").strip().replace("\n", " ")[:300]
        parts.append(f"body={body!r}" if body else "body=<empty>")
    except Exception:
        pass
    return text + (" [" + ", ".join(parts) + "]" if parts else "")


def privacy_note(provider: str) -> str:
    """One-line data-privacy caution for a provider ("" if none)."""
    spec = CHAT_PROVIDERS.get(provider)
    return spec.privacy_note if spec else ""


@dataclass
class RoundResult:
    """One normalized model round: final text plus tool calls."""

    text: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    usage_known: bool = True
    strict_fallback: bool = False


def chat_round(
    client: Any,
    model: str,
    provider: str,
    *,
    instructions: str,
    tools: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> RoundResult:
    """One model call plus normalization. Raises on API errors."""
    spec = CHAT_PROVIDERS.get(provider)
    if spec is not None:
        return _chat_completions_round(
            client, model, spec, instructions, tools, history
        )
    return _meta_round(client, model, instructions, tools, history)


def _meta_round(
    client: Any,
    model: str,
    instructions: str,
    tools: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> RoundResult:
    """Original Responses-API behaviour, unchanged."""
    response = client.responses.create(
        model=model,
        instructions=instructions,
        tools=tools,
        input=history,
    )
    response_usage = getattr(response, "usage", None)
    calls = [
        {
            "id": getattr(item, "call_id", "") or "",
            "name": getattr(item, "name", "") or "",
            "arguments": getattr(item, "arguments", "") or "{}",
        }
        for item in getattr(response, "output", [])
        if getattr(item, "type", None) == "function_call"
    ]
    text = "".join(
        block.text
        for item in getattr(response, "output", [])
        if getattr(item, "type", None) == "message"
        for block in getattr(item, "content", [])
        if getattr(block, "type", None) == "output_text"
    ) or getattr(response, "output_text", "") or ""
    if response_usage is None:
        return RoundResult(text=text, calls=calls, usage_known=False)
    return RoundResult(
        text=text,
        calls=calls,
        input_tokens=getattr(response_usage, "input_tokens", 0) or 0,
        output_tokens=getattr(response_usage, "output_tokens", 0) or 0,
        total_tokens=getattr(response_usage, "total_tokens", 0) or 0,
    )


def _chat_tools(
    schemas: list[dict[str, Any]], strict: bool = True
) -> list[dict[str, Any]]:
    """Responses tool schemas translated to the Chat Completions shape.

    strict=False drops the strict flag (providers that do not accept it).
    """
    translated = []
    for schema in schemas:
        function: dict[str, Any] = {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema.get(
                "parameters", {"type": "object", "properties": {}}
            ),
        }
        if strict and schema.get("strict"):
            function["strict"] = True
        translated.append({"type": "function", "function": function})
    return translated


def _chat_messages(
    instructions: str, history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Shared history translated to Chat Completions messages."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": instructions}
    ]
    for item in history:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id", ""),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "") or "{}",
                            },
                        }
                    ],
                }
            )
        elif kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": str(item.get("output", "")),
                }
            )
        elif "role" in item:
            messages.append(
                {"role": item["role"], "content": item.get("content", "") or ""}
            )
        # Anything else is never sent (and never logged).
    return messages


def _is_strict_rejection(exc: BaseException) -> bool:
    """A 400 that complains about strict/structured output."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if "strict" not in text and "structured" not in text:
        return False
    return (
        "400" in text
        or "bad request" in text
        or getattr(exc, "status_code", None) == 400
    )


def _normalize_chat_response(response: Any) -> RoundResult:
    """Chat Completions response normalized. reasoning_content is dropped."""
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    text = ""
    calls: list[dict[str, Any]] = []
    if message is not None:
        text = getattr(message, "content", None) or ""
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            arguments = (
                getattr(function, "arguments", "") if function is not None else ""
            )
            calls.append(
                {
                    "id": getattr(call, "id", ""),
                    "name": getattr(function, "name", "")
                    if function is not None
                    else "",
                    "arguments": arguments or "{}",
                }
            )
    # NOTE: message.reasoning_content is deliberately never read here: it
    # stays out of the result, history, records, logs and errors.
    usage = getattr(response, "usage", None)
    if usage is None:
        return RoundResult(text=text, calls=calls, usage_known=False)
    details = getattr(usage, "prompt_tokens_details", None)
    return RoundResult(
        text=text,
        calls=calls,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        cached_input_tokens=getattr(details, "cached_tokens", 0) or 0,
    )


def _retry_after_seconds(exc: BaseException, attempt: int) -> float:
    """Wait before a retry: Retry-After if given, else 1s, 2s, 4s..."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    value = None
    if headers is not None:
        try:
            value = headers.get("retry-after")
        except Exception:
            value = None
    try:
        wait = float(value) if value is not None else 2.0**attempt
    except (TypeError, ValueError):
        wait = 2.0**attempt
    return max(0.0, min(wait, RATE_LIMIT_MAX_WAIT_S))


def _create_with_backoff(
    client: Any, kwargs: dict[str, Any], sleep: Callable[[float], None]
) -> Any:
    """chat.completions.create, retrying transient statuses a few times."""
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if (
                getattr(exc, "status_code", None) not in RETRYABLE_STATUSES
                or attempt >= RATE_LIMIT_RETRIES
            ):
                raise
            sleep(_retry_after_seconds(exc, attempt))
            attempt += 1


def _chat_completions_round(
    client: Any,
    model: str,
    spec: ChatProvider,
    instructions: str,
    tools: list[dict[str, Any]],
    history: list[dict[str, Any]],
    sleep: Callable[[float], None] = time.sleep,
) -> RoundResult:
    """Chat Completions round; strict schemas only where the provider takes them."""
    chat_tools = _chat_tools(tools, strict=spec.strict_tools)
    messages = _chat_messages(instructions, history)
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if chat_tools:
        kwargs["tools"] = chat_tools
    try:
        response = _create_with_backoff(client, kwargs, sleep)
    except Exception as exc:
        if _is_strict_rejection(exc) and any(
            tool["function"].get("strict") for tool in chat_tools
        ):
            kwargs["tools"] = _chat_tools(tools, strict=False)
            response = _create_with_backoff(client, kwargs, sleep)
            result = _normalize_chat_response(response)
            result.strict_fallback = True
            return result
        raise
    return _normalize_chat_response(response)
