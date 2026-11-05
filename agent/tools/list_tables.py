"""List the tables and views available to query."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.tools import ToolContext

NAME = "list_tables"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": NAME,
    "description": "List the names of all tables and views available to query.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "strict": True,
}


def run(ctx: ToolContext, _args: dict[str, Any]) -> str:
    return json.dumps({"tables": ctx.backend.list_tables()})
