"""Propose a business definition or data quirk for the dataset.

The entry is stored as "proposed" only. The model can never approve or
reject; approval happens solely in console code the model cannot call.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent.preliminary_analysis import (
    MEMORY_DIR,
    DefinitionStore,
)

if TYPE_CHECKING:
    from agent.tools import ToolContext

NAME = "propose_note"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": NAME,
    "description": (
        "Propose a reusable business definition (for example revenue) or a "
        "recurring data quirk. The note is stored as proposed and needs the "
        "user's approval in the console; you cannot approve it yourself. "
        "Evidence, when given, must cite this turn's successful query ids."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "description": "Either 'definition' or 'quirk'.",
            },
            "key": {
                "type": "string",
                "description": "snake_case key, for example revenue.",
            },
            "text": {
                "type": "string",
                "description": "Plain text, max 300 characters.",
            },
            "query_id": {
                "type": ["string", "null"],
                "description": "Successful query id backing this note, e.g. q1.",
            },
        },
        "required": ["kind", "key", "text", "query_id"],
        "additionalProperties": False,
    },
    "strict": True,
}


def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    memory_dir = ctx.memory_dir if ctx.memory_dir is not None else MEMORY_DIR
    store = DefinitionStore.load(ctx.dataset, memory_dir)
    evidence = [args["query_id"]] if args.get("query_id") else None
    entry = store.propose(
        args.get("kind"),
        args.get("key"),
        args.get("text"),
        evidence,
        record=ctx.record,
    )
    return json.dumps({"ok": True, "entry_id": entry["id"], "status": "proposed"})
