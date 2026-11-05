"""Regression tests for the agent/tools/ registry.

All tools are exercised through dispatch() with a ToolContext on a
temporary fixture database. The real data/olist.sqlite is never touched.
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.backend import SqliteBackend  # noqa: E402
from agent.tools import TOOL_SCHEMAS, ToolContext, dispatch  # noqa: E402

ITEMS = [
    {"id": "count_items", "question": "How many items?", "evidence_needed": "count of items"},
]

PLAN = [{"item_id": "count_items", "steps": ["count rows in items"]}]


def _make_backend(tmp: str) -> SqliteBackend:
    path = Path(tmp) / "fixture.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL);")
        conn.executemany(
            "INSERT INTO items (name) VALUES (?)", [("a",), ("b",), ("c",)]
        )
        conn.commit()
    finally:
        conn.close()
    return SqliteBackend(path)


def _ctx(backend: SqliteBackend, tmp: str, question: str = "how many?") -> ToolContext:
    return ToolContext(
        backend=backend,
        question=question,
        runs_dir=Path(tmp) / "runs",
        memory_dir=Path(tmp) / "memory",
    )


def test_registry_lists_all_tools():
    assert sorted(s["name"] for s in TOOL_SCHEMAS) == [
        "ask_clarification",
        "get_schema",
        "list_tables",
        "make_chart",
        "plan_action",
        "propose_note",
        "record_finding",
        "run_sql",
        "set_checklist",
        "submit_answer",
    ]


def test_list_tables_through_registry():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            assert json.loads(dispatch(_ctx(backend, tmp), "list_tables", {})) == {
                "tables": ["items"]
            }
        finally:
            backend.close()


def test_get_schema_through_registry():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            schema = json.loads(
                dispatch(_ctx(backend, tmp), "get_schema", {"table": "items"})
            )
        finally:
            backend.close()
    assert schema["table"] == "items"
    assert [c["name"] for c in schema["columns"]] == ["id", "name"]


def test_no_record_without_set_checklist():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            dispatch(ctx, "list_tables", {})
            dispatch(ctx, "get_schema", {"table": "items"})
            assert ctx.record is None
            assert not (Path(tmp) / "runs").exists()
        finally:
            backend.close()


def test_run_sql_refused_before_checklist_and_plan():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            refused = json.loads(
                dispatch(ctx, "run_sql", {"query": "SELECT count(*) FROM items;"})
            )
            assert "set_checklist" in refused["error"]
            ok = json.loads(dispatch(ctx, "set_checklist", {"items": ITEMS}))
            assert ok["ok"] is True
            refused = json.loads(
                dispatch(ctx, "run_sql", {"query": "SELECT count(*) FROM items;"})
            )
            assert "plan_action" in refused["error"]
        finally:
            backend.close()


def test_plan_action_rejected_unless_covering():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            dispatch(ctx, "set_checklist", {"items": ITEMS + [
                {"id": "other", "question": "q?", "evidence_needed": "e"}
            ]})
            partial = json.loads(dispatch(ctx, "plan_action", {"plan": PLAN}))
            assert "missing" in partial["error"]
        finally:
            backend.close()


def test_queries_recorded_with_ids():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            dispatch(ctx, "set_checklist", {"items": ITEMS})
            dispatch(ctx, "plan_action", {"plan": PLAN})
            first = json.loads(
                dispatch(ctx, "run_sql", {"query": "SELECT count(*) FROM items;"})
            )
            second = json.loads(
                dispatch(ctx, "run_sql", {"query": "SELECT count(*) FROM items;"})
            )
            assert (first["query_id"], second["query_id"]) == ("q1", "q2")
            assert first["rows"] == [[3]]
            assert [q["id"] for q in ctx.record.queries] == ["q1", "q2"]
            assert ctx.record.path.is_file()
        finally:
            backend.close()


def test_set_checklist_validation():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            for bad in (
                {"items": []},
                {"items": [{"id": "a", "question": "q"}]},
                {"items": [
                    {"id": "a", "question": "q", "evidence_needed": "e"},
                    {"id": "a", "question": "q", "evidence_needed": "e"},
                ]},
            ):
                result = json.loads(dispatch(ctx, "set_checklist", bad))
                assert "error" in result, f"expected error for {bad!r}"
            assert ctx.record is None
        finally:
            backend.close()


def test_run_sql_failure_recorded_with_f_id():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            dispatch(ctx, "set_checklist", {"items": ITEMS})
            dispatch(ctx, "plan_action", {"plan": PLAN})
            bad = json.loads(
                dispatch(ctx, "run_sql", {"query": "SELECT * FROM no_such_table;"})
            )
            assert "error" in bad
            assert bad["failed_query_id"] == "f1"
            assert bad["kind"] == "error"
            assert [q["id"] for q in ctx.record.queries] == []
            stored = ctx.record.failed_queries
            assert [f["id"] for f in stored] == ["f1"]
            assert stored[0]["kind"] == "error"
            assert "no_such_table" in stored[0]["sql"]
            assert stored[0]["error"] == bad["error"]
            assert "timestamp" in stored[0]
            assert ctx.record.audit[-1]["action"] == "record_failed_query"
        finally:
            backend.close()


def test_run_sql_refused_statement_recorded_as_refused():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            dispatch(ctx, "set_checklist", {"items": ITEMS})
            dispatch(ctx, "plan_action", {"plan": PLAN})
            bad = json.loads(
                dispatch(ctx, "run_sql", {"query": "SELECT 1; SELECT 2;"})
            )
            assert "error" in bad
            assert bad["kind"] == "refused"
            assert [f["id"] for f in ctx.record.failed_queries] == ["f1"]
        finally:
            backend.close()


def test_run_sql_timeout_recorded_as_timeout():
    class _TimeoutBackend:
        def run_sql(self, query):
            raise TimeoutError("query exceeded the 10s limit and was cancelled")

    with tempfile.TemporaryDirectory() as tmp:
        ctx = _ctx(_TimeoutBackend(), tmp)
        dispatch(ctx, "set_checklist", {"items": ITEMS})
        dispatch(ctx, "plan_action", {"plan": PLAN})
        bad = json.loads(dispatch(ctx, "run_sql", {"query": "SELECT 1;"}))
        assert bad["kind"] == "timeout"
        assert [f["id"] for f in ctx.record.failed_queries] == ["f1"]


def test_record_finding_refuses_failed_query_id():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            dispatch(ctx, "set_checklist", {"items": ITEMS})
            dispatch(ctx, "plan_action", {"plan": PLAN})
            dispatch(ctx, "run_sql", {"query": "SELECT * FROM no_such_table;"})
            refused = json.loads(
                dispatch(
                    ctx,
                    "record_finding",
                    {
                        "item_id": "count_items",
                        "claim": "three items",
                        "query_id": "f1",
                        "value": 3,
                    },
                )
            )
            assert "no citable result" in refused["error"]
            unknown = json.loads(
                dispatch(
                    ctx,
                    "record_finding",
                    {
                        "item_id": "count_items",
                        "claim": "three items",
                        "query_id": "f99",
                        "value": 3,
                    },
                )
            )
            assert "unknown query id" in unknown["error"]
        finally:
            backend.close()


def _check_schema_shape(schema: dict, path: str) -> None:
    assert schema.get("strict") is True, f"{path}: strict must be true"
    params = schema["parameters"]
    assert params.get("additionalProperties") is False, f"{path}: sealed object required"
    properties = params.get("properties", {})
    required = params.get("required", [])
    for key in properties:
        assert key in required, f"{path}: {key!r} missing from required"
    for key, subschema in properties.items():
        items = subschema.get("items")
        if isinstance(items, dict) and items.get("type") == "object":
            _check_schema_shape(
                {"strict": True, "parameters": items}, f"{path}.{key}[]"
            )


def test_all_tool_schemas_list_every_property_as_required():
    assert len(TOOL_SCHEMAS) == 10
    for schema in TOOL_SCHEMAS:
        _check_schema_shape(schema, schema["name"])


def test_optional_fields_accept_explicit_null():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            noted = json.loads(
                dispatch(ctx, "propose_note", {
                    "kind": "definition", "key": "revenue",
                    "text": "sum of price", "query_id": None,
                })
            )
            assert noted == {"ok": True, "entry_id": "e1", "status": "proposed"}
            dispatch(ctx, "set_checklist", {"items": ITEMS})
            dispatch(ctx, "plan_action", {"plan": PLAN})
            dispatch(ctx, "run_sql", {"query": "SELECT count(*) FROM items;"})
            ctx.ask_user = lambda question, options: {"choice": options[0]["label"]}
            clarified = json.loads(
                dispatch(ctx, "ask_clarification", {
                    "question": "Which count?",
                    "options": [
                        {"label": "all rows", "query_id": "q1", "evidence": 3,
                         "explanation": "counts every row"},
                        {"label": "all rows again", "query_id": "q1", "evidence": 3,
                         "explanation": "same count again"},
                    ],
                    "definition_key": None,
                })
            )
            assert clarified == {"ok": True, "choice": "all rows"}
        finally:
            backend.close()


def test_unknown_tool():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            result = json.loads(dispatch(_ctx(backend, tmp), "drop_database", {}))
        finally:
            backend.close()
    assert "error" in result


def test_tools_never_import_a_database_driver():
    tools_dir = Path(__file__).resolve().parent.parent / "agent" / "tools"
    for module in (
        "__init__.py",
        "list_tables.py",
        "get_schema.py",
        "run_sql.py",
        "set_checklist.py",
        "plan_action.py",
        "propose_note.py",
        "ask_clarification.py",
        "record_finding.py",
        "submit_answer.py",
    ):
        text = (tools_dir / module).read_text()
        assert "sqlite3" not in text, f"{module} touches a database driver"
        assert "psycopg" not in text, f"{module} touches a database driver"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"all {len(tests)} tests passed")
