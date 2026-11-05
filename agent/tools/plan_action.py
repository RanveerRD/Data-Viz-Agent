"""Store the plan; rejected unless it covers every checklist item."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.tools import ToolContext

NAME = "plan_action"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": NAME,
    "description": (
        "Store the query plan: steps per checklist item. Rejected unless "
        "it covers every checklist item. Call after set_checklist, before "
        "run_sql."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "plan": {
                "type": "array",
                "description": "One entry per checklist item.",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {
                            "type": "string",
                            "description": "Checklist id this entry advances.",
                        },
                        "steps": {
                            "type": "array",
                            "description": "Non-empty list of query steps.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["item_id", "steps"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["plan"],
        "additionalProperties": False,
    },
    "strict": True,
}

_MISSING_CHECKLIST_ERROR = "plan_action is refused: call set_checklist first."


def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    if ctx.record is None:
        return json.dumps({"error": _MISSING_CHECKLIST_ERROR})
    ctx.record.plan_action(args.get("plan"))
    return json.dumps({"ok": True, "planned_items": len(ctx.record.plan)})
