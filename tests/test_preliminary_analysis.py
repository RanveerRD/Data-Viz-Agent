"""Regression tests for agent/preliminary_analysis.py.

No API calls. All stores live in temporary directories; the real
workspace/memory is never touched.
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.preliminary_analysis import (  # noqa: E402
    DEFAULT_DATASET,
    DefinitionStore,
    approve_definition,
)
from agent.__main__ import _handle_command  # noqa: E402
from agent.tools import TOOL_SCHEMAS, ToolContext, dispatch  # noqa: E402
from agent.backend import SqliteBackend  # noqa: E402


def _make_backend(tmp: str) -> SqliteBackend:
    path = Path(tmp) / "fixture.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL);")
        conn.commit()
    finally:
        conn.close()
    return SqliteBackend(path)


def _ctx(backend: SqliteBackend, tmp: str) -> ToolContext:
    return ToolContext(
        backend=backend,
        question="how many?",
        runs_dir=Path(tmp) / "runs",
        memory_dir=Path(tmp) / "memory",
    )


def test_only_approved_entries_load_into_prompt():
    with tempfile.TemporaryDirectory() as tmp:
        store = DefinitionStore.load(DEFAULT_DATASET, tmp)
        assert store.build_prompt_section() == ""
        store.propose("definition", "revenue", "sum of price for delivered orders")
        store.propose("quirk", "late_dates", "some timestamps are null")
        assert store.build_prompt_section() == ""
        decided = store.decide("e1", True)
        assert decided["status"] == "approved"
        section = DefinitionStore.load(DEFAULT_DATASET, tmp).build_prompt_section()
        assert "revenue" in section and "sum of price" in section
        assert "late_dates" not in section


def test_rejected_entries_stay_out_of_prompt():
    with tempfile.TemporaryDirectory() as tmp:
        store = DefinitionStore.load(DEFAULT_DATASET, tmp)
        store.propose("definition", "revenue", "sum of price")
        store.decide("e1", False)
        assert store.build_prompt_section() == ""


def test_model_tool_can_only_propose():
    names = [s["name"] for s in TOOL_SCHEMAS]
    assert "propose_note" in names
    assert "approve_definition" not in names
    assert "retire" not in names
    assert "forget" not in names
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            result = json.loads(
                dispatch(
                    ctx,
                    "propose_note",
                    {"kind": "definition", "key": "revenue", "text": "sum of price"},
                )
            )
            assert result == {"ok": True, "entry_id": "e1", "status": "proposed"}
            store = DefinitionStore.load(DEFAULT_DATASET, Path(tmp) / "memory")
            assert [e["status"] for e in store.entries] == ["proposed"]
        finally:
            backend.close()


def test_propose_note_evidence_must_cite_record_queries():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            dispatch(ctx, "set_checklist", {"items": [
                {"id": "n", "question": "how many?", "evidence_needed": "count"}
            ]})
            dispatch(ctx, "plan_action", {"plan": [{"item_id": "n", "steps": ["count"]}]})
            ok_query = json.loads(dispatch(ctx, "run_sql", {"query": "SELECT 1;"}))
            assert ok_query["query_id"] == "q1"
            good = json.loads(
                dispatch(ctx, "propose_note", {
                    "kind": "definition", "key": "revenue",
                    "text": "sum of price", "query_id": "q1",
                })
            )
            assert good["ok"] is True
            bad = json.loads(
                dispatch(ctx, "propose_note", {
                    "kind": "definition", "key": "other",
                    "text": "other text", "query_id": "q99",
                })
            )
            assert "unknown query ids" in bad["error"]
        finally:
            backend.close()


def test_propose_note_validation():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            for args in (
                {"kind": "rule", "key": "revenue", "text": "sum"},
                {"kind": "definition", "key": "Bad-Key", "text": "sum"},
                {"kind": "definition", "key": "revenue", "text": "x" * 301},
                {"kind": "definition", "key": "revenue", "text": "   "},
            ):
                result = json.loads(dispatch(ctx, "propose_note", args))
                assert "error" in result, f"expected error for {args!r}"
        finally:
            backend.close()


def test_rejected_key_blocks_same_proposal():
    with tempfile.TemporaryDirectory() as tmp:
        store = DefinitionStore.load(DEFAULT_DATASET, tmp)
        store.propose("definition", "revenue", "sum of price")
        store.decide("e1", False)
        try:
            store.propose("definition", "revenue", "sum of price")
        except ValueError as exc:
            assert "rejected" in str(exc)
        else:
            raise AssertionError("expected ValueError for rejected key+text")
        entry = store.propose("definition", "revenue", "sum of price and freight")
        assert entry["id"] == "e2"


def test_saving_new_definition_retires_old():
    with tempfile.TemporaryDirectory() as tmp:
        first = approve_definition("revenue", "sum of price", memory_dir=tmp)
        assert first["status"] == "approved"
        second = approve_definition(
            "revenue", "sum of price and freight", memory_dir=tmp
        )
        assert second["status"] == "approved"
        reloaded = DefinitionStore.load(DEFAULT_DATASET, tmp)
        assert [e["status"] for e in reloaded.entries] == ["retired", "approved"]
        assert [e["key"] for e in reloaded.approved_entries()] == ["revenue"]
        assert reloaded.approved_entries()[0]["text"] == "sum of price and freight"
        actions = [a["action"] for a in reloaded.audit]
        assert actions[-2:] == ["note_retired", "note_approved"]
        try:
            reloaded.propose("definition", "revenue", "yet another text")
            reloaded.decide("e3", True)
        except ValueError as exc:
            assert "already has approved" in str(exc)
        else:
            raise AssertionError("expected ValueError when approving over an approval")


def test_approve_definition_creates_approved_entry():
    with tempfile.TemporaryDirectory() as tmp:
        entry = approve_definition(
            "revenue", "sum of price", ["q1"], memory_dir=tmp
        )
        assert entry["kind"] == "definition"
        assert entry["evidence"] == ["q1"]
        assert entry["decided_by"] == "user"
        assert entry["decided_at"]
        reloaded = DefinitionStore.load(DEFAULT_DATASET, tmp)
        assert [e["key"] for e in reloaded.approved_entries()] == ["revenue"]


def test_audit_only_grows():
    with tempfile.TemporaryDirectory() as tmp:
        store = DefinitionStore.load(DEFAULT_DATASET, tmp)
        store.propose("definition", "revenue", "sum of price")
        store.decide("e1", True)
        seqs = [a["seq"] for a in store.audit]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
        actions = [a["action"] for a in store.audit]
        assert actions == ["note_proposed", "note_approved"]
        assert all("timestamp" in a for a in store.audit)


def test_define_command_is_gone(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        assert _handle_command("/define revenue = sum of price", tmp) is True
        assert not (Path(tmp) / f"{DEFAULT_DATASET}.json").exists()
        assert "unknown command" in capsys.readouterr().out


def test_notes_still_lists_approved(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        approve_definition("revenue", "sum of price", memory_dir=tmp)
        assert _handle_command("/notes", tmp) is True
        out = capsys.readouterr().out
        assert "revenue" in out and "sum of price" in out


def test_forget_retires_entry_and_frees_the_key(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        approve_definition("revenue", "sum of price", memory_dir=tmp)
        assert _handle_command("/forget revenue", tmp) is True
        assert "retired revenue" in capsys.readouterr().out
        reloaded = DefinitionStore.load(DEFAULT_DATASET, tmp)
        assert [e["status"] for e in reloaded.entries] == ["retired"]
        assert reloaded.build_prompt_section() == ""
        assert reloaded.approved_entries() == []
        assert reloaded.audit[-1]["action"] == "note_retired"
        assert "timestamp" in reloaded.audit[-1]
        second = approve_definition("revenue", "sum of everything", memory_dir=tmp)
        assert second["status"] == "approved"
        assert [e["key"] for e in DefinitionStore.load(
            DEFAULT_DATASET, tmp).approved_entries()] == ["revenue"]


def test_forget_without_approval_errors(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        assert _handle_command("/forget revenue", tmp) is True
        assert "no approved entry" in capsys.readouterr().out
        assert not (Path(tmp) / f"{DEFAULT_DATASET}.json").exists()


def test_entries_are_copies():
    with tempfile.TemporaryDirectory() as tmp:
        store = DefinitionStore.load(DEFAULT_DATASET, tmp)
        store.propose("definition", "revenue", "sum of price")
        store.entries.append({"id": "hack"})
        store.audit.clear()
        assert [e["id"] for e in store.entries] == ["e1"]
        assert len(store.audit) == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"all {len(tests)} tests passed")
