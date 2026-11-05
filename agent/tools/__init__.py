"""Tool registry: schemas for the API call, dispatch of function calls.

Adding a tool later means adding one file in this package and one line in
_TOOL_MODULES below. Tools never import a database driver; they only call
the injected backend and record through the context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.record import AnalysisRecord

if TYPE_CHECKING:
    from agent.backend import DatabaseBackend


@dataclass
class ToolContext:
    """What tools may touch: the database backend and this turn's record.

    The record starts as None and is created lazily by the first successful
    set_checklist call, so a turn with no tool calls stores nothing.
    """

    backend: DatabaseBackend
    question: str = ""
    runs_dir: Path = field(default_factory=lambda: Path("workspace") / "runs")
    record: AnalysisRecord | None = None
    memory_dir: Path | None = None
    ask_user: Any = None
    dataset: str = "olist"
    allow_charts: bool = True


def _tool_modules() -> tuple:
    from agent.tools import (
        ask_clarification,
        get_schema,
        list_tables,
        make_chart,
        plan_action,
        propose_note,
        record_finding,
        run_sql,
        set_checklist,
        submit_answer,
    )

    return (
        list_tables,
        get_schema,
        set_checklist,
        plan_action,
        propose_note,
        ask_clarification,
        record_finding,
        run_sql,
        make_chart,
        submit_answer,
    )


_TOOL_MODULES = _tool_modules()

TOOL_SCHEMAS: list[dict[str, Any]] = [m.SCHEMA for m in _TOOL_MODULES]
_HANDLERS = {m.NAME: m.run for m in _TOOL_MODULES}


def schemas_for(ctx: ToolContext) -> list[dict[str, Any]]:
    """Schemas offered to this turn; make_chart is single-dataset only."""
    if ctx.allow_charts:
        return TOOL_SCHEMAS
    return [m.SCHEMA for m in _TOOL_MODULES if m.NAME != "make_chart"]


def dispatch(ctx: ToolContext, name: str, args: dict[str, Any]) -> str:
    """Run the named tool and return its JSON result string."""
    if name == "make_chart" and not ctx.allow_charts:
        return json.dumps({"error": "make_chart is not available in cross-dataset mode"})
    try:
        handler = _HANDLERS[name]
    except KeyError:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        return handler(ctx, args)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
