"""Set the turn's checklist; lazily creates the AnalysisRecord."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent.record import AnalysisRecord

if TYPE_CHECKING:
    from agent.tools import ToolContext

NAME = "set_checklist"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": NAME,
    "description": (
        "Set the checklist for this turn: one item per question to answer, "
        "each with the evidence it needs. Call this before plan_action and "
        "any run_sql call. Calling it again replaces the checklist."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "Non-empty list of checklist items.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "snake_case identifier, unique within the checklist.",
                        },
                        "question": {
                            "type": "string",
                            "description": "The sub-question this item answers.",
                        },
                        "evidence_needed": {
                            "type": "string",
                            "description": "What query result would settle it.",
                        },
                    },
                    "required": ["id", "question", "evidence_needed"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
    "strict": True,
}


def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    items = args.get("items")
    if ctx.record is None:
        record = AnalysisRecord.create(question=ctx.question, runs_dir=ctx.runs_dir)
        record.set_checklist(items)
        ctx.record = record
    else:
        ctx.record.set_checklist(items)
    return json.dumps(
        {"ok": True, "checklist_items": len(ctx.record.checklist), "run_id": ctx.record.run_id}
    )
