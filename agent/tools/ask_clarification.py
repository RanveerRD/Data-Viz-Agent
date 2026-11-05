"""Pause the turn for one grounded user clarification.

Every option must cite a successful query from this turn's record, and every
number in the question or labels must already appear in a recorded result.
At most five clarifications per question; further calls are refused.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from agent.numbers import verify_text
from agent.preliminary_analysis import (
    DEFAULT_DATASET,
    MEMORY_DIR,
    DefinitionStore,
)

if TYPE_CHECKING:
    from agent.tools import ToolContext

NAME = "ask_clarification"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": NAME,
    "description": (
        "Ask the user to choose between readings you already queried. Each "
        "option cites its query id, one evidence value from that query's "
        "stored result, and a one-sentence explanation; options never "
        "invented. Call at most five times per question, after running "
        "the queries, before final figures."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The ambiguous choice, in one sentence.",
            },
            "options": {
                "type": "array",
                "description": "Two to four options grounded in query results.",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "What this reading means.",
                        },
                        "query_id": {
                            "type": "string",
                            "description": "Recorded successful query id, e.g. q1.",
                        },
                        "evidence": {
                            "description": "One value from that query's stored result.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "One short sentence on what this reading means, max 120 characters.",
                        },
                    },
                    "required": ["label", "query_id", "evidence", "explanation"],
                    "additionalProperties": False,
                },
            },
            "definition_key": {
                "type": ["string", "null"],
                "description": "Optional snake_case key to save the choice under.",
            },
        },
        "required": ["question", "options", "definition_key"],
        "additionalProperties": False,
    },
    "strict": True,
}

_MISSING_RECORD_ERROR = "ask_clarification is refused: call set_checklist first."
_BUDGET_ERROR = (
    "ask_clarification is refused: five clarifications already used for "
    "this question; proceed with the choice already made, or with your best "
    "reading stated in the decisions."
)
_NO_CALLBACK_ERROR = (
    "ask_clarification is unavailable in this session: no user to ask. "
    "State your decisions in submit_answer instead."
)

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _refuse(ctx: ToolContext, reason: str) -> str:
    assert ctx.record is not None
    return json.dumps(ctx.record.refuse_clarification(reason))


def _mark_last_used(
    ctx: ToolContext, cleaned: list[dict[str, Any]], definition_key: str
) -> None:
    """Flag the option matching the saved default, in tool code, never the model."""
    memory_dir = ctx.memory_dir if ctx.memory_dir is not None else MEMORY_DIR
    try:
        store = DefinitionStore.load(DEFAULT_DATASET, memory_dir)
    except OSError:
        return
    saved = ""
    for entry in store.approved_entries():
        if entry["key"] == definition_key:
            saved = entry["text"]
            break
    if not saved:
        return
    for option in cleaned:
        if saved.startswith(option["label"]):
            option["last_used"] = True
            return


def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    record = ctx.record
    if record is None:
        return json.dumps({"error": _MISSING_RECORD_ERROR})
    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        return _refuse(ctx, "ask_clarification needs a non-empty question")
    options = args.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 4:
        return _refuse(ctx, "ask_clarification needs 2 to 4 options")
    definition_key = args.get("definition_key")
    if definition_key is not None and (
        not isinstance(definition_key, str) or not _KEY_RE.match(definition_key)
    ):
        return _refuse(ctx, f"definition_key {definition_key!r} must be snake_case")
    cleaned: list[dict[str, Any]] = []
    failed_ids = {f["id"] for f in record.failed_queries}
    known_ids = {q["id"] for q in record.queries}
    for i, option in enumerate(options):
        if not isinstance(option, dict):
            return _refuse(ctx, f"option {i} must be an object")
        label = option.get("label")
        if not isinstance(label, str) or not label.strip():
            return _refuse(ctx, f"option {i} needs a non-empty label")
        query_id = option.get("query_id")
        if query_id in failed_ids:
            return _refuse(
                ctx,
                f"query {query_id!r} failed and has no citable result; "
                "cite a successful q-id instead",
            )
        if query_id not in known_ids:
            return _refuse(ctx, f"unknown query id: {query_id!r}")
        evidence = option.get("evidence")
        if not record.value_in_query(query_id, evidence):
            return _refuse(
                ctx,
                f"evidence {evidence!r} does not appear in {query_id}'s stored result",
            )
        explanation = option.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            return _refuse(ctx, f"option {i} needs a non-empty explanation")
        if len(explanation.strip()) > 120:
            return _refuse(
                ctx, f"option {i} explanation must be at most 120 characters"
            )
        cleaned.append(
            {
                "label": label.strip(),
                "query_id": query_id,
                "evidence": evidence,
                "explanation": explanation.strip(),
            }
        )
    cells = [v for q in record.queries for row in q["rows"] for v in row]
    for text in [question] + [o["label"] for o in cleaned]:
        _, unmatched = verify_text(text, record.question, cells)
        if unmatched:
            return _refuse(ctx, f"unmatched numbers: {unmatched}")
    if definition_key is not None:
        _mark_last_used(ctx, cleaned, definition_key)
    if record.clarification_calls >= 5:
        return json.dumps(record.refuse_clarification(_BUDGET_ERROR))
    if ctx.ask_user is None:
        return json.dumps({"error": _NO_CALLBACK_ERROR})
    reply = ctx.ask_user(question.strip(), cleaned)
    if isinstance(reply, dict) and "user_question" in reply:
        return json.dumps(record.note_clarification_question(reply["user_question"]))
    choice = reply["choice"] if isinstance(reply, dict) else reply
    return json.dumps(
        record.store_clarification(question.strip(), cleaned, choice, definition_key)
    )
