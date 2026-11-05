"""Live Baseten smoke test. Run by hand with a key; never commit output.

Requires in the environment (or .env):
    MODEL_PROVIDER=baseten
    BASETEN_API_KEY=<key>        (never printed by this script)
    BASETEN_MODEL=<slug>         (optional; defaults to the verified slug)

What it does: builds the Baseten client, runs one tool-calling round
(list_tables) against data/olist.sqlite through the real provider seam,
dispatches the call, runs a second round, and prints tokens, the cost
estimate, and whether the strict fallback fired. Exits non-zero on failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.backend import SqliteBackend  # noqa: E402
from agent.provider import (  # noqa: E402
    build_client,
    chat_round,
    estimate_cost_usd,
    get_provider_name,
)
from agent.tools import ToolContext, dispatch, schemas_for  # noqa: E402


def fail(message: str) -> int:
    print(f"SMOKE FAIL: {message}")
    return 1


def main() -> int:
    try:
        provider = get_provider_name()
    except RuntimeError as exc:
        return fail(str(exc))
    if provider != "baseten":
        return fail(f"MODEL_PROVIDER is {provider!r}; set MODEL_PROVIDER=baseten")
    try:
        client, model, _provider = build_client()
    except RuntimeError as exc:
        return fail(str(exc))
    print(f"provider=baseten model={model}")

    backend = SqliteBackend(REPO_ROOT / "data" / "olist.sqlite")
    try:
        ctx = ToolContext(
            backend=backend, question="smoke test", dataset="olist"
        )
        tools = schemas_for(ctx)
        history: list = [{"role": "user", "content": "List the tables."}]
        try:
            first = chat_round(
                client, model, "baseten", instructions="Answer with tools.",
                tools=tools, history=history,
            )
        except Exception as exc:
            return fail(f"round 1 raised {type(exc).__name__}: {exc}")
        names = [call["name"] for call in first.calls]
        if "list_tables" not in names:
            return fail(f"round 1 made no list_tables call (got {names})")
        print(f"round 1 tool calls: {names}")
        call = next(c for c in first.calls if c["name"] == "list_tables")
        history.append({
            "type": "function_call", "call_id": call["id"],
            "name": call["name"], "arguments": call["arguments"],
        })
        history.append({
            "type": "function_call_output", "call_id": call["id"],
            "output": dispatch(ctx, call["name"], json.loads(call["arguments"] or "{}")),
        })
        try:
            second = chat_round(
                client, model, "baseten", instructions="Answer with tools.",
                tools=tools, history=history,
            )
        except Exception as exc:
            return fail(f"round 2 raised {type(exc).__name__}: {exc}")
        print(f"round 2 text: {second.text[:120]!r}")
        prompt = first.input_tokens + second.input_tokens
        cached = first.cached_input_tokens + second.cached_input_tokens
        completion = first.output_tokens + second.output_tokens
        total = first.total_tokens + second.total_tokens
        cost = estimate_cost_usd(prompt, cached, completion)
        print(f"tokens: {prompt} in / {completion} out / {total} total")
        print(f"cost estimate, Baseten list price: ${cost:.6f}")
        print(f"strict fallback fired: {first.strict_fallback or second.strict_fallback}")
        blob = json.dumps([first.text, second.text, history])
        if "reasoning_content" in blob:
            return fail("reasoning content leaked into results/history")
    finally:
        backend.close()
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
