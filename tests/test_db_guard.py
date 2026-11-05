"""Regression tests for agent/db_guard.py.

Run with the project environment:

    .venv/bin/python -m pytest tests/test_db_guard.py -v
    .venv/bin/python tests/test_db_guard.py

Every test builds its own fixture database in a temporary directory; the
real data/olist.sqlite is never touched.
"""

import sqlite3
import sys
import tempfile
from pathlib import Path
from urllib.request import pathname2url

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import db_guard  # noqa: E402


def _make_fixture_db(directory: str | Path) -> Path:
    path = Path(directory) / "fixture.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL);")
        conn.executemany("INSERT INTO items (name) VALUES (?)", [("a",), ("b",), ("c",)])
        conn.commit()
    finally:
        conn.close()
    return path


def _assert_refused(conn: sqlite3.Connection, sql: str) -> None:
    """Assert sql fails for a read-only reason, not just any error."""
    try:
        conn.execute(sql)
    except sqlite3.Error as exc:
        msg = str(exc).lower()
        assert "not authorized" in msg or "readonly" in msg, f"unexpected error for {sql!r}: {exc!r}"
    else:
        raise AssertionError(f"expected {sql!r} to be refused")


def _readonly_conn(tmp: str) -> sqlite3.Connection:
    return db_guard.open_sqlite_readonly(_make_fixture_db(tmp))


def test_select_works():
    with tempfile.TemporaryDirectory() as tmp:
        conn = _readonly_conn(tmp)
        try:
            rows = conn.execute("SELECT name FROM items ORDER BY id;").fetchall()
        finally:
            conn.close()
    assert rows == [("a",), ("b",), ("c",)]


def test_recursive_cte_works():
    with tempfile.TemporaryDirectory() as tmp:
        conn = _readonly_conn(tmp)
        try:
            (total,) = conn.execute(
                "WITH RECURSIVE cnt(x) AS ("
                " SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 5"
                ") SELECT sum(x) FROM cnt;"
            ).fetchone()
        finally:
            conn.close()
    assert total == 15


def test_insert_refused():
    with tempfile.TemporaryDirectory() as tmp:
        conn = _readonly_conn(tmp)
        try:
            _assert_refused(conn, "INSERT INTO items (name) VALUES ('d');")
        finally:
            conn.close()


def test_update_refused():
    with tempfile.TemporaryDirectory() as tmp:
        conn = _readonly_conn(tmp)
        try:
            _assert_refused(conn, "UPDATE items SET name = 'z' WHERE id = 1;")
        finally:
            conn.close()


def test_delete_refused():
    with tempfile.TemporaryDirectory() as tmp:
        conn = _readonly_conn(tmp)
        try:
            _assert_refused(conn, "DELETE FROM items WHERE id = 1;")
        finally:
            conn.close()


def test_drop_table_refused():
    with tempfile.TemporaryDirectory() as tmp:
        conn = _readonly_conn(tmp)
        try:
            _assert_refused(conn, "DROP TABLE items;")
        finally:
            conn.close()


def test_create_table_refused():
    with tempfile.TemporaryDirectory() as tmp:
        conn = _readonly_conn(tmp)
        try:
            _assert_refused(conn, "CREATE TABLE evil (id INTEGER);")
        finally:
            conn.close()


def test_attach_refused():
    with tempfile.TemporaryDirectory() as tmp:
        conn = _readonly_conn(tmp)
        try:
            _assert_refused(conn, "ATTACH DATABASE ':memory:' AS aux;")
        finally:
            conn.close()


def test_pragma_query_only_off_denied():
    with tempfile.TemporaryDirectory() as tmp:
        conn = _readonly_conn(tmp)
        try:
            _assert_refused(conn, "PRAGMA query_only = OFF;")
        finally:
            conn.close()


def test_pragma_allowlist_reads():
    with tempfile.TemporaryDirectory() as tmp:
        conn = _readonly_conn(tmp)
        try:
            info = conn.execute("PRAGMA table_info(items);").fetchall()
            assert len(info) == 2
            _assert_refused(conn, "PRAGMA journal_mode;")
        finally:
            conn.close()


def test_missing_path_raises_without_creating():
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "does_not_exist.sqlite"
        try:
            db_guard.open_sqlite_readonly(missing)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError")
        assert not missing.exists()


def test_mode_ro_layer_only():
    """URI mode=ro alone (no query_only, no authorizer) refuses writes."""
    with tempfile.TemporaryDirectory() as tmp:
        fixture = _make_fixture_db(tmp)
        uri = "file:" + pathname2url(str(fixture.resolve())) + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            assert conn.execute("SELECT count(*) FROM items;").fetchone() == (3,)
            _assert_refused(conn, "INSERT INTO items (name) VALUES ('d');")
        finally:
            conn.close()


def test_authorizer_layer_only():
    """Authorizer alone on a read-write connection refuses writes."""
    with tempfile.TemporaryDirectory() as tmp:
        fixture = _make_fixture_db(tmp)
        conn = sqlite3.connect(str(fixture))
        try:
            db_guard.install_readonly_authorizer(conn)
            assert conn.execute("SELECT count(*) FROM items;").fetchone() == (3,)
            try:
                conn.execute("INSERT INTO items (name) VALUES ('d');")
            except sqlite3.Error as exc:
                assert "not authorized" in str(exc).lower(), f"unexpected error: {exc!r}"
            else:
                raise AssertionError("expected INSERT to be refused")
        finally:
            conn.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"all {len(tests)} tests passed")
