"""Regression tests for agent/backend.py.

Run with the project environment:

    .venv/bin/python -m pytest tests/test_agent_backend.py -v
    .venv/bin/python tests/test_agent_backend.py

Every test builds its own fixture database in a temporary directory; the
real data/olist.sqlite is never touched. The timeout test takes ~10s.
"""

import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.backend import MAX_ROWS, SqliteBackend  # noqa: E402


def _make_fixture_db(directory: str | Path, n_rows: int = 3) -> Path:
    path = Path(directory) / "fixture.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL);")
        conn.executemany(
            "INSERT INTO items (name) VALUES (?)", [(f"item-{i}",) for i in range(n_rows)]
        )
        conn.commit()
    finally:
        conn.close()
    return path


def test_list_tables():
    with tempfile.TemporaryDirectory() as tmp:
        backend = SqliteBackend(_make_fixture_db(tmp))
        try:
            assert backend.list_tables() == ["items"]
            assert backend.dialect == "SQLite"
        finally:
            backend.close()


def test_get_schema():
    with tempfile.TemporaryDirectory() as tmp:
        backend = SqliteBackend(_make_fixture_db(tmp))
        try:
            schema = backend.get_schema("items")
        finally:
            backend.close()
    assert schema["table"] == "items"
    by_name = {c["name"]: c for c in schema["columns"]}
    assert set(by_name) == {"id", "name"}
    assert by_name["name"]["samples"] == ["item-0", "item-1", "item-2"]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            backend = SqliteBackend(_make_fixture_db(tmp))
            try:
                backend.get_schema("nope")
            finally:
                backend.close()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown table")


def test_run_sql_select():
    with tempfile.TemporaryDirectory() as tmp:
        backend = SqliteBackend(_make_fixture_db(tmp))
        try:
            result = backend.run_sql("SELECT id, name FROM items ORDER BY id;")
        finally:
            backend.close()
    assert result["columns"] == ["id", "name"]
    assert result["row_count"] == 3
    assert result["truncated"] is False
    assert result["rows"][0] == [1, "item-0"]


def test_run_sql_truncates_at_200_rows():
    with tempfile.TemporaryDirectory() as tmp:
        backend = SqliteBackend(_make_fixture_db(tmp, n_rows=250))
        try:
            result = backend.run_sql("SELECT id FROM items ORDER BY id;")
        finally:
            backend.close()
    assert result["row_count"] == MAX_ROWS == 200
    assert result["truncated"] is True


def test_run_sql_rejects_multiple_statements():
    with tempfile.TemporaryDirectory() as tmp:
        backend = SqliteBackend(_make_fixture_db(tmp))
        try:
            try:
                backend.run_sql("SELECT 1; SELECT 2;")
            except ValueError:
                pass
            else:
                raise AssertionError("expected ValueError for two statements")
            try:
                backend.run_sql("SELECT 1; DROP TABLE items;")
            except ValueError:
                pass
            else:
                raise AssertionError("expected ValueError for statement + DDL")
        finally:
            backend.close()


def test_run_sql_timeout():
    with tempfile.TemporaryDirectory() as tmp:
        backend = SqliteBackend(_make_fixture_db(tmp))
        try:
            start = time.monotonic()
            try:
                backend.run_sql(
                    "WITH RECURSIVE c(x) AS ("
                    " SELECT 1 UNION ALL SELECT x + 1 FROM c"
                    ") SELECT count(*) FROM c;"
                )
            except TimeoutError as exc:
                elapsed = time.monotonic() - start
                assert "10" in str(exc), f"unexpected message: {exc}"
            else:
                raise AssertionError("expected TimeoutError for runaway query")
            assert elapsed < 60, "timeout did not stop the query"
        finally:
            backend.close()


def test_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "does_not_exist.sqlite"
        try:
            SqliteBackend(missing)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError")
        assert not missing.exists()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"all {len(tests)} tests passed")
