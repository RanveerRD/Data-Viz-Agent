"""Regression tests for agent/record.py.

Run with the project environment:

    .venv/bin/python -m pytest tests/test_record.py -v
    .venv/bin/python tests/test_record.py

All records live in temporary directories; nothing touches workspace/.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.record import MAX_FAILED_QUERIES, AnalysisRecord  # noqa: E402

ITEMS = [
    {"id": "total_orders", "question": "How many orders?", "evidence_needed": "count of orders"},
    {"id": "top_state", "question": "Which state is top?", "evidence_needed": "orders by state"},
]

PLAN = [
    {"item_id": "total_orders", "steps": ["count rows in orders"]},
    {"item_id": "top_state", "steps": ["group orders by customer state"]},
]


def _record(tmp: str) -> AnalysisRecord:
    return AnalysisRecord.create(question="q?", runs_dir=Path(tmp) / "runs")


def test_set_checklist_validation():
    with tempfile.TemporaryDirectory() as tmp:
        record = _record(tmp)
        for bad in ([], [{"id": "a", "question": "q"}]):
            try:
                record.set_checklist(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected ValueError for {bad!r}")
        try:
            record.set_checklist(
                [
                    {"id": "a", "question": "q", "evidence_needed": "e"},
                    {"id": "a", "question": "q", "evidence_needed": "e"},
                ]
            )
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for duplicate ids")
        try:
            record.set_checklist(
                [{"id": "Bad-Id", "question": "q", "evidence_needed": "e"}]
            )
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for non-snake_case id")


def test_plan_must_cover_every_item():
    with tempfile.TemporaryDirectory() as tmp:
        record = _record(tmp)
        record.set_checklist(ITEMS)
        try:
            record.plan_action([{"item_id": "total_orders", "steps": ["count"]}])
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for partial plan")
        try:
            record.plan_action([{"item_id": "nope", "steps": ["x"]}])
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for unknown item")
        record.plan_action(PLAN)
        assert [p["item_id"] for p in record.plan] == ["total_orders", "top_state"]


def test_queries_recorded_with_ids():
    with tempfile.TemporaryDirectory() as tmp:
        record = _record(tmp)
        record.set_checklist(ITEMS)
        record.plan_action(PLAN)
        q1 = record.record_query(
            "SELECT 1;", {"columns": ["x"], "row_count": 1, "truncated": False, "rows": [[1]]}
        )
        q2 = record.record_query(
            "SELECT 2;", {"columns": ["x"], "row_count": 1, "truncated": False, "rows": [[2]]}
        )
        assert (q1, q2) == ("q1", "q2")
        queries = record.queries
        assert [q["id"] for q in queries] == ["q1", "q2"]
        assert queries[0]["sql"] == "SELECT 1;"
        assert queries[0]["rows"] == [[1]]
        assert "timestamp" in queries[0]


def test_audit_only_grows():
    with tempfile.TemporaryDirectory() as tmp:
        record = _record(tmp)
        before = len(record.audit)
        record.set_checklist(ITEMS)
        assert len(record.audit) > before
        seqs = [a["seq"] for a in record.audit]
        assert seqs == sorted(seqs)
        record.plan_action(PLAN)
        record.record_query("SELECT 1;", {"columns": [], "row_count": 0, "truncated": False, "rows": []})
        seqs = [a["seq"] for a in record.audit]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
        actions = [a["action"] for a in record.audit]
        assert actions == ["created", "set_checklist", "plan_action", "record_query"]


def test_state_changes_only_through_methods():
    with tempfile.TemporaryDirectory() as tmp:
        record = _record(tmp)
        record.set_checklist(ITEMS)
        record.checklist.append({"id": "hack", "question": "q", "evidence_needed": "e"})
        record.audit.clear()
        assert [c["id"] for c in record.checklist] == ["total_orders", "top_state"]
        assert len(record.audit) == 2  # created + set_checklist


def test_atomic_save_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        record = _record(tmp)
        record.set_checklist(ITEMS)
        record.plan_action(PLAN)
        path = record.path
        assert path.is_file()
        assert not any(p.suffix == ".tmp" for p in path.parent.iterdir())
        saved = json.loads(path.read_text())
        assert saved["question"] == "q?"
        assert saved["findings"] == []
        loaded = AnalysisRecord.load(path)
        assert loaded.run_id == record.run_id
        assert [c["id"] for c in loaded.checklist] == ["total_orders", "top_state"]


def test_failed_queries_capped_at_fifty():
    with tempfile.TemporaryDirectory() as tmp:
        record = _record(tmp)
        record.set_checklist(ITEMS)
        record.plan_action(PLAN)
        audit_before = len(record.audit)
        for i in range(55):
            fid = record.record_failed_query(f"SELECT {i};", "OperationalError: boom", "error")
            assert fid == f"f{i + 1}"
        failed = record.failed_queries
        assert len(failed) == MAX_FAILED_QUERIES
        assert failed[0]["id"] == "f6"
        assert failed[-1]["id"] == "f55"
        assert len(record.audit) == audit_before + 55
        actions = [a["action"] for a in record.audit[-55:]]
        assert set(actions) == {"record_failed_query"}
        assert "timestamp" in failed[0]


def test_record_finding_refuses_failed_query_id():
    with tempfile.TemporaryDirectory() as tmp:
        record = _record(tmp)
        record.set_checklist(ITEMS)
        record.plan_action(PLAN)
        record.record_failed_query("SELECT bad;", "OperationalError: boom", "error")
        refused = record.record_finding("total_orders", "c", "f1", 5)
        assert "no citable result" in refused["error"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"all {len(tests)} tests passed")
