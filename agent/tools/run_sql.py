"""Run exactly one read SQL statement, capped at 200 rows.

Refused until a checklist and a covering plan exist. Every successful call
is recorded in the turn's AnalysisRecord as q1, q2, ... Failures are
recorded as f1, f2, ... and can never be cited by record_finding.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.tools import ToolContext

NAME = "run_sql"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": NAME,
    "description": (
        "Run exactly one SQL SELECT statement. At most 200 rows are "
        "returned; the result says whether it was truncated and names its "
        "query id (q1, q2, ...). Refused until set_checklist and a covering "
        "plan_action exist. Statements that write, change the schema, or "
        "take longer than 10 seconds are refused. Failures are recorded "
        "with a failed-query id (f1, f2, ...) that record_finding refuses."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A single SQL statement, ending without a second statement.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

_GATING_ERROR = (
    "run_sql is refused: call set_checklist first, then plan_action "
    "covering every checklist item."
)

_REFUSAL_MARKERS = (
    "not authorized",
    "readonly",
    "read-only",
    "no permission",
    "permission denied",
)


def _failure_kind(exc: BaseException) -> str:
    """Classify a run_sql failure without importing a database driver."""
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ValueError):
        return "refused"
    message = str(exc).lower()
    if any(marker in message for marker in _REFUSAL_MARKERS):
        return "refused"
    return "error"


def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    record = ctx.record
    if record is None or not record.plan_covers_checklist():
        return json.dumps({"error": _GATING_ERROR})
    sql = args["query"]
    try:
        result = ctx.backend.run_sql(sql)
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        kind = _failure_kind(exc)
        failed_id = record.record_failed_query(sql, error_text, kind)
        return json.dumps(
            {"error": error_text, "failed_query_id": failed_id, "kind": kind}
        )
    query_id = record.record_query(sql, result)
    return json.dumps({**result, "query_id": query_id})
