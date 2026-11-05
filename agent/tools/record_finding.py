"""Record one verified number for a checklist item."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.tools import ToolContext

NAME = "record_finding"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": NAME,
    "description": (
        "Record one key number for a checklist item. The value must appear "
        "in the named query's stored result (numbers within 0.005, strings "
        "exact). A verified finding marks the item proven. At most 3 "
        "findings per item. Refused findings are kept with their reason."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "item_id": {
                "type": "string",
                "description": "Checklist id this finding supports.",
            },
            "claim": {
                "type": "string",
                "description": "One sentence stating what the number shows.",
            },
            "query_id": {
                "type": "string",
                "description": "Recorded query id, e.g. q1.",
            },
            "value": {
                "description": "The number (or string) as it appears in the stored result.",
            },
        },
        "required": ["item_id", "claim", "query_id", "value"],
        "additionalProperties": False,
    },
    "strict": True,
}

_MISSING_RECORD_ERROR = "record_finding is refused: call set_checklist first."


def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    if ctx.record is None:
        return json.dumps({"error": _MISSING_RECORD_ERROR})
    return json.dumps(
        ctx.record.record_finding(
            args.get("item_id"),
            args.get("claim"),
            args.get("query_id"),
            args.get("value", None),
        )
    )
