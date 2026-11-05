"""Agentic loop: one question in, grounded answer out.

Uses the OpenAI Python SDK against a Responses-compatible API with function
tools. Conversation history is kept across questions in the same session.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.backend import DatabaseBackend
from agent.provider import chat_round
from agent.tools import ToolContext, dispatch, schemas_for
from agent.tools.make_chart import chart_candidates
MAX_TURNS = 30
REMINDER_TURNS = 3
CAP_MESSAGE = f"Reached the {MAX_TURNS}-turn limit without a final answer."
SUBMIT_REMINDER = (
    "You replied in plain text without submitting. Record a finding for "
    "each key number with record_finding, then call submit_answer. "
    "Do not answer in plain text."
)
WRAP_UP_NUDGE = (
    "Only 4 rounds remain. Stop running new queries and call submit_answer "
    "now using the findings already recorded."
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "workspace" / "runs"


def _read_env_file_value(name: str) -> str | None:
    """Read one KEY=value line from .env without sourcing the whole file."""
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return None
    prefix = name + "="
    try:
        for line in env_file.read_text().splitlines():
            if line.startswith(prefix):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        return None
    return None


@dataclass
class TurnResult:
    """Everything one console turn needs to print."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    usage_known: bool = True
    record_path: str | None = None
    verification: str | None = None
    clarification: dict[str, Any] | None = None
    submission: dict[str, Any] | None = None
    charts: list[str] = field(default_factory=list)
    chart_notes: list[str] = field(default_factory=list)
    chart_titles: list[str] = field(default_factory=list)
    cached_input_tokens: int = 0
    chart_candidates: list[dict[str, Any]] = field(default_factory=list)
    chart_drawn: list[str] = field(default_factory=list)


@dataclass
class _Usage:
    """Running token totals for one turn."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    known: bool = True


def _short_args(args: dict[str, Any], limit: int = 120) -> str:
    text = json.dumps(args)
    return text if len(text) <= limit else text[:limit] + "..."


def _summarize(name: str, result: str) -> str:
    """One-line summary of a tool's JSON result for console display."""
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return str(result)[:100]
    if not isinstance(data, dict):
        return str(result)[:100]
    if "error" in data:
        return f"error: {str(data['error'])[:120]}"
    if "rows" in data:
        count = data.get("row_count", len(data["rows"]))
        suffix = " (truncated)" if data.get("truncated") else ""
        query_id = f" [{data['query_id']}]" if "query_id" in data else ""
        return f"{name}: {count} rows{suffix}{query_id}"
    if "tables" in data:
        return f"{name}: {len(data['tables'])} tables"
    if "columns" in data:
        return f"{name}: {len(data['columns'])} columns"
    if data.get("ok"):
        status = f" ({data['status']})" if "status" in data else ""
        return f"{name}: ok{status}"
    if name == "make_chart" and "description" in data:
        return f"make_chart: {data['description']}"
    return str(result)[:100]


def _record_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


EVIDENCE_QUESTION = "Want the evidence behind these decisions?"


def _render_decision(
    decision: dict[str, Any], clarifications: list[dict[str, Any]]
) -> str:
    """Render one decision with its proof; failed proofs show UNCONFIRMED."""
    base = f"- {decision['what']}: {decision['choice']}"
    if decision.get("unconfirmed"):
        return f"{base} (UNCONFIRMED)"
    source = decision["source"]
    if source == "user_stated":
        return f"{base} (you stated: \"{decision['proof']}\")"
    if source == "user_chosen":
        for clarification in clarifications:
            if clarification["id"] == decision["proof"]:
                key = (clarification.get("definition_key") or "").replace("_", " ")
                label = clarification.get("reply", "")
                for j, option in enumerate(clarification.get("options", []), 1):
                    if option.get("label") == clarification.get("reply"):
                        if key:
                            return f"- {key}: {label} (you chose: option {j})"
                        return f"- {label} (you chose: option {j})"
                return f"{base} (you chose: \"{label}\")"
        return f"{base} (you chose: \"{decision['proof']}\")"
    return f"{base} (not asked: no alternative found, query {decision['proof']})"


def render_answer(
    submission: dict[str, Any],
    clarifications: list[dict[str, Any]],
    charts: list[str] | None = None,
    chart_notes: list[str] | None = None,
) -> str:
    """Render the accepted answer with its decisions and gaps.

    The Gaps section is always present; with no gaps found it reads
    "- none found". Chart and chart-note lines are appended by code after
    acceptance, so they never pass through number extraction.
    """
    lines = [submission["answer"], "", "Decisions:"]
    lines.extend(
        _render_decision(d, clarifications) for d in submission["decisions"]
    )
    lines.append("")
    lines.append("Gaps:")
    gaps = [g for g in submission["gaps"] if g.get("note", "").strip()]
    if gaps:
        for gap in gaps:
            item_id = (gap.get("item_id") or "").strip()
            if item_id:
                lines.append(f"- {item_id}: {gap['note'].strip()}")
            else:
                lines.append(f"- {gap['note'].strip()}")
    else:
        lines.append("- none found")
    lines.extend(["", EVIDENCE_QUESTION])
    notes = list(chart_notes or [])
    for i, path in enumerate(charts or []):
        lines.append(f"Chart: {path}")
        if i < len(notes) and notes[i]:
            lines.append(f"Chart note: {notes[i]}")
    return "\n".join(lines)


def _build_work_summary(ctx: ToolContext) -> str:
    """Compact summary of this turn's verified work for future context."""
    record = ctx.record
    assert record is not None
    lines = [f"Work summary (record {record.run_id}):", "Checklist:"]
    for item in record.checklist:
        lines.append(f"- {item['id']}: {item['question']} [{item['status']}]")
    lines.append("Queries:")
    for query in record.queries:
        sql = query["sql"]
        if len(sql) > 300:
            sql = sql[:300] + "..."
        lines.append(f"- {query['id']}: {sql} ({query['row_count']} rows)")
    lines.append("Findings:")
    for finding in record.findings:
        lines.append(f"- {finding['item_id']}: {finding['claim']} [{finding['query_id']}]")
    return "\n".join(lines)


def _partial_summary(ctx: ToolContext) -> str:
    """What the turn established before the cap hit: items, findings, counts."""
    assert ctx.record is not None
    record = ctx.record
    lines = ["", "Partial progress:"]
    for item in record.checklist:
        lines.append(f"- {item['id']}: {item['status']}")
    if record.findings:
        lines.append("Findings:")
        for finding in record.findings:
            lines.append(f"- {finding['claim']} ({finding['value']})")
    lines.append(f"Queries run: {len(record.queries)}")
    lines.append(f"Record: {_record_display_path(record.path)}")
    return "\n".join(lines)


def _compact_turn_history(
    history: list[dict[str, Any]],
    snapshot: int,
    ctx: ToolContext,
    text: str,
) -> None:
    """Replace this turn's tool traffic with question, summary, and answer.

    Raw schema outputs and result rows are dropped; the full detail stays in
    the record on disk. The rendered answer keeps the decisions in context
    for follow-up questions.
    """
    del history[snapshot:]
    history.append({"role": "user", "content": ctx.question})
    history.append({"role": "assistant", "content": _build_work_summary(ctx)})
    history.append({"role": "assistant", "content": text})


def _round(
    client: OpenAI,
    model: str,
    instructions: str,
    ctx: ToolContext,
    history: list[dict[str, Any]],
    usage: _Usage,
    provider: str = "meta",
) -> tuple[str, bool, bool]:
    """One model call plus tool execution. Returns (text, had_calls, api_error)."""
    try:
        result = chat_round(
            client,
            model,
            provider,
            instructions=instructions,
            tools=schemas_for(ctx),
            history=history,
        )
    except Exception as exc:
        print(f"api error: {type(exc).__name__}: {exc}")
        return "", False, True
    if not result.usage_known:
        usage.known = False
    else:
        usage.input_tokens += result.input_tokens
        usage.output_tokens += result.output_tokens
        usage.total_tokens += result.total_tokens
        usage.cached_input_tokens += result.cached_input_tokens
    for call in result.calls:
        history.append(
            {
                "type": "function_call",
                "call_id": call["id"],
                "name": call["name"],
                "arguments": call["arguments"],
            }
        )
        try:
            args = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError as exc:
            args = {}
            result_json = json.dumps({"error": f"JSONDecodeError: {exc}"})
        else:
            print(f"  tool: {call['name']}({_short_args(args)})")
            result_json = dispatch(ctx, call["name"], args)
        print(f"    -> {_summarize(call['name'], result_json)}")
        try:
            warning = json.loads(result_json).get("warning", "")
        except (json.JSONDecodeError, TypeError, AttributeError):
            warning = ""
        if isinstance(warning, str) and warning.startswith(
            ("UNVERIFIED", "UNCONFIRMED")
        ):
            print(f"  !! {warning}")
        history.append(
            {
                "type": "function_call_output",
                "call_id": call["id"],
                "output": result_json,
            }
        )
    return result.text, bool(result.calls), False


def answer_question(
    client: OpenAI,
    model: str,
    instructions: str,
    backend: DatabaseBackend,
    history: list[dict[str, Any]],
    question: str,
    runs_dir: Path = RUNS_DIR,
    ask_user: Any = None,
    dataset: str = "olist",
    allow_charts: bool = True,
    provider: str = "meta",
) -> TurnResult:
    """Ask one question, keeping shared history.

    At most MAX_TURNS counting rounds; rounds that pause for a user
    clarification do not count because they are waiting for the human.
    """
    ctx = ToolContext(
        backend=backend,
        question=question,
        runs_dir=runs_dir,
        ask_user=ask_user,
        dataset=dataset,
        allow_charts=allow_charts,
    )
    history.append({"role": "user", "content": question})
    snapshot = len(history) - 1
    usage = _Usage()
    last_text = ""
    capped = False
    api_failed = False
    budget = 0
    nudged = False
    while True:
        if budget >= MAX_TURNS:
            capped = True
            break
        if budget == MAX_TURNS - 4 and not nudged:
            history.append({"role": "user", "content": WRAP_UP_NUDGE})
            nudged = True
        calls_before = (
            ctx.record.clarification_calls if ctx.record is not None else 0
        )
        last_text, had_calls, api_error = _round(
            client, model, instructions, ctx, history, usage, provider
        )
        if api_error:
            del history[snapshot:]
            api_failed = True
            break
        if ctx.record is not None and ctx.record.submission is not None:
            break
        calls_after = (
            ctx.record.clarification_calls if ctx.record is not None else 0
        )
        if calls_after == calls_before:
            budget += 1
        if not had_calls:
            break
    submission = ctx.record.submission if ctx.record is not None else None
    if (
        not capped
        and not api_failed
        and ctx.record is not None
        and submission is None
        and last_text
    ):
        history.append({"role": "user", "content": SUBMIT_REMINDER})
        for _ in range(REMINDER_TURNS):
            last_text, had_calls, api_error = _round(
                client, model, instructions, ctx, history, usage, provider
            )
            if api_error:
                del history[snapshot:]
                api_failed = True
                break
            submission = ctx.record.submission if ctx.record is not None else None
            if submission is not None or not had_calls:
                break
    submission = ctx.record.submission if ctx.record is not None else None
    record_path = _record_display_path(ctx.record.path) if ctx.record else None
    clarification = ctx.record.clarification if ctx.record is not None else None
    if capped and submission is None:
        text = CAP_MESSAGE
        if ctx.record is not None:
            text += _partial_summary(ctx)
        return TurnResult(
            text=text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            usage_known=usage.known,
            record_path=record_path,
            verification="verification: no submitted answer"
            if ctx.record
            else None,
            clarification=clarification,
        )
    verification: str | None = None
    chart_entries = ctx.record.charts if ctx.record is not None else []
    chart_paths = [c["path"] for c in chart_entries]
    chart_notes = [c.get("note", "") for c in chart_entries]
    chart_titles = [c.get("title", "") for c in chart_entries]
    candidates = (
        chart_candidates(ctx.record)
        if ctx.record is not None and ctx.allow_charts and submission is not None
        else []
    )
    drawn = (
        sorted({c.get("query_id", "") for c in ctx.record.charts})
        if ctx.record is not None and submission is not None
        else []
    )
    if ctx.record is not None:
        if submission is not None:
            text = render_answer(
                submission, ctx.record.clarifications, chart_paths, chart_notes
            )
            verification = (
                f"verification: {len(submission['matched'])} numbers matched, "
                f"{len(submission['unmatched'])} unmatched"
            )
            _compact_turn_history(history, snapshot, ctx, text)
        else:
            text = last_text
            verification = "verification: no submitted answer"
            if text:
                history.append({"role": "assistant", "content": text})
    else:
        text = last_text
        if text:
            history.append({"role": "assistant", "content": text})
    return TurnResult(
        text=text,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        usage_known=usage.known,
        record_path=record_path,
        verification=verification,
        clarification=clarification,
        submission=submission,
        charts=chart_paths,
        chart_notes=chart_notes,
        chart_titles=chart_titles,
        chart_candidates=candidates,
        chart_drawn=drawn,
    )
