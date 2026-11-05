"""Describe one table: columns, types, and sample values."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.tools import ToolContext

NAME = "get_schema"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": NAME,
    "description": (
        "Describe a table: column names, declared types, and 3 sample "
        "values per column. The table name must come from list_tables."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "table": {
                "type": "string",
                "description": "Exact table name from list_tables.",
            }
        },
        "required": ["table"],
        "additionalProperties": False,
    },
    "strict": True,
}


def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    return json.dumps(ctx.backend.get_schema(args["table"]))
