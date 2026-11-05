"""Console /chart flow: offers and drawing with no model call (no API)."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.__main__ import _handle_command, _offer_charts  # noqa: E402
from agent.backend import SqliteBackend  # noqa: E402
from agent.loop import answer_question  # noqa: E402
from agent.record import AnalysisRecord  # noqa: E402
from agent.tools.make_chart import chart_candidates  # noqa: E402

ROWS = [["SP", 498440.43], ["RJ", 12.345], ["MG", 100.0]]


def _record(tmp: Path, rows: list | None = None) -> AnalysisRecord:
    rows = [list(r) for r in (rows if rows is not None else ROWS)]
    record = AnalysisRecord.create("chart?", tmp / "runs")
    record.set_checklist(
        [{"id": "rev", "question": "Revenue?", "evidence_needed": "sum"}]
    )
    record.plan_action([{"item_id": "rev", "steps": ["sum"]}])
    qid = record.record_query(
        "SELECT region, revenue FROM t",
        {"columns": ["region", "revenue"], "rows": rows,
         "row_count": len(rows), "truncated": False},
    )
    record.record_finding("rev", "revenue is big", qid, rows[0][1])
    return record


def _pending(record: AnalysisRecord) -> dict[str, Any]:
    return {"record_abs": record.path, "candidates": chart_candidates(record)}


def _drawn_paths(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("chart saved: ")]


def test_chart_draws_png_and_records(tmp_path, capsys):
    record = _record(tmp_path)
    assert _handle_command("/chart", dataset="olist",
                           pending_charts=_pending(record)) is True
    out = capsys.readouterr().out
    saved = _drawn_paths(out)
    assert len(saved) == 1
    assert Path(saved[0].removeprefix("chart saved: ")).read_bytes()[:4] == b"\x89PNG"
    fresh = AnalysisRecord.load(record.path)
    (entry,) = fresh.charts
    assert entry["title"] == "revenue by region"
    assert not any(ch.isdigit() for ch in entry["title"])
    assert any(e["action"] == "record_chart" for e in fresh.audit)


def test_chart_bar_override_on_date_x(tmp_path, capsys):
    rows = [[f"2016-{m:02d}", float(m)] for m in (1, 2, 3)]
    record = AnalysisRecord.create("date?", tmp_path / "runs")
    record.set_checklist(
        [{"id": "rev", "question": "Revenue?", "evidence_needed": "sum"}]
    )
    record.plan_action([{"item_id": "rev", "steps": ["sum"]}])
    qid = record.record_query(
        "SELECT month, revenue FROM t",
        {"columns": ["month", "revenue"], "rows": rows,
         "row_count": 3, "truncated": False},
    )
    record.record_finding("rev", "jan big", qid, 1.0)
    assert _handle_command("/chart 1 bar", dataset="olist",
                           pending_charts=_pending(record)) is True
    assert len(_drawn_paths(capsys.readouterr().out)) == 1


def test_chart_line_on_category_x_refused(tmp_path, capsys):
    record = _record(tmp_path)
    assert _handle_command("/chart 1 line", dataset="olist",
                           pending_charts=_pending(record)) is True
    out = capsys.readouterr().out
    assert "use bar instead" in out
    assert _drawn_paths(out) == []
    assert AnalysisRecord.load(record.path).charts == []


def test_chart_empty_and_bad_number(tmp_path, capsys):
    assert _handle_command("/chart", dataset="olist",
                           pending_charts=None) is True
    assert "No chart available" in capsys.readouterr().out
    record = _record(tmp_path)
    assert _handle_command("/chart 5", dataset="olist",
                           pending_charts=_pending(record)) is True
    assert "choose 1 to 1" in capsys.readouterr().out


def test_offer_lines_and_already_drawn(tmp_path, capsys):
    record = _record(tmp_path)
    candidates = chart_candidates(record)
    _offer_charts(candidates, [])
    out = capsys.readouterr().out
    assert (
        "Chart available: 1. bar chart of revenue by region "
        "(3 points). Type /chart 1" in out
    )
    assert "(already drawn)" not in out
    _offer_charts(candidates, ["q1"])
    assert "(already drawn)" in capsys.readouterr().out


def _stubbed_turn(tmp: Path, **kwargs: Any):
    path = Path(tmp) / "stub.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE sales (region TEXT, revenue REAL);")
        conn.executemany("INSERT INTO sales VALUES (?, ?)", ROWS)
        conn.commit()
    finally:
        conn.close()
    backend = SqliteBackend(path)

    def call(cid: str, name: str, args: dict) -> Any:
        return types.SimpleNamespace(
            type="function_call", call_id=cid, name=name,
            arguments=json.dumps(args))

    def response(*items: Any) -> Any:
        return types.SimpleNamespace(
            output=list(items),
            usage=types.SimpleNamespace(
                input_tokens=1, output_tokens=1, total_tokens=2))

    def message(text: str) -> Any:
        return types.SimpleNamespace(
            type="message",
            content=[types.SimpleNamespace(type="output_text", text=text)])

    script = [
        response(call("c1", "set_checklist", {"items": [
            {"id": "rev", "question": "Revenue?", "evidence_needed": "sum"}]})),
        response(call("c2", "plan_action", {"plan": [
            {"item_id": "rev", "steps": ["sum"]}]})),
        response(call("c3", "run_sql",
                      {"query": "SELECT region, revenue FROM sales;"})),
        response(call("c4", "record_finding", {
            "item_id": "rev", "claim": "SP biggest",
            "query_id": "q1", "value": 498440.43})),
        response(call("c5", "submit_answer", {
            "answer": "SP leads with 498440.43 overall.",
            "decisions": [{"what": "scope", "choice": "all rows",
                           "source": "user_stated", "proof": "overall"}],
            "gaps": []})),
        response(message("extra")),
    ]

    class StubClient:
        def __init__(self) -> None:
            self.responses = self
            self.calls = 0

        def create(self, **create_kwargs: Any) -> Any:
            item = script[min(self.calls, len(script) - 1)]
            self.calls += 1
            return item

    history: list = []
    return answer_question(
        StubClient(), "m", "instr", backend, history,
        "Revenue by region overall?", runs_dir=Path(tmp) / "runs",
        ask_user=None, dataset="olist", **kwargs)


def test_full_turn_yields_candidate_and_all_mode_yields_none(tmp_path):
    with tempfile.TemporaryDirectory() as tmp:
        result = _stubbed_turn(Path(tmp))
        assert len(result.chart_candidates) == 1
        assert result.chart_candidates[0]["sql"] == (
            "SELECT region, revenue FROM sales;")
    with tempfile.TemporaryDirectory() as tmp:
        result = _stubbed_turn(Path(tmp), allow_charts=False)
        assert result.chart_candidates == []
