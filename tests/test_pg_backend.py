"""Tests for the PostgreSQL backend (agent/pg_backend.py).

Unit tests need no database. Integration tests run against the real
AdventureWorks as analyst_reader and skip automatically when Postgres or
the ANALYST_READER_PASSWORD is unavailable.
"""

import math
import os
import sqlite3
import sys
import tempfile
import types
import uuid
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

import pytest
from psycopg import errors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.__main__ import _instructions_for_question, parse_args  # noqa: E402
from agent.backend import SqliteBackend  # noqa: E402
from agent.pg_backend import (  # noqa: E402
    PostgresBackend,
    _map_pg_error,
    _pg_settings,
    _pg_value,
)

CANARY = "canary-secret-for-error-text-check"


class _FakePgError(Exception):
    def __init__(self, message: str, sqlstate: str | None):
        super().__init__(message)
        self.sqlstate = sqlstate


def test_pg_value_conversions():
    assert _pg_value(None) is None
    assert _pg_value(True) is True
    assert _pg_value(7) == 7
    assert _pg_value("x") == "x"
    assert _pg_value(Decimal("498440.43")) == 498440.43
    whole = _pg_value(Decimal("100"))
    assert whole == 100 and isinstance(whole, int)
    assert _pg_value(Decimal("NaN")) is None
    assert _pg_value(float("nan")) is None
    assert _pg_value(float("inf")) is None
    assert _pg_value(date(2024, 1, 2)) == "2024-01-02"
    assert _pg_value(datetime(2024, 1, 2, 3, 4, 5)) == "2024-01-02T03:04:05"
    assert _pg_value(time(3, 4, 5)) == "03:04:05"
    sample = uuid.uuid4()
    assert _pg_value(sample) == str(sample)
    assert _pg_value(b"ab") == "<binary 2 bytes>"
    assert _pg_value(memoryview(b"abc")) == "<binary 3 bytes>"
    assert isinstance(_pg_value([1, 2]), str)


class _SyntaxWithDiag(errors.SyntaxError):
    """SyntaxError carrying a stubbed diagnostic primary message."""

    def __init__(self, message: str, primary: str):
        super().__init__(message)
        self._primary = primary

    @property
    def diag(self):  # type: ignore[override]
        return types.SimpleNamespace(message_primary=self._primary)


def test_pg_error_mapping():
    refused = _map_pg_error(_FakePgError('permission denied for table t', "42501"))
    assert isinstance(refused, ValueError) and "read-only" in str(refused)
    refused = _map_pg_error(_FakePgError("cannot execute INSERT in a read-only transaction", "25006"))
    assert isinstance(refused, ValueError) and "read-only" in str(refused)
    timed_out = _map_pg_error(_FakePgError("canceling statement due to statement timeout", "57014"))
    assert isinstance(timed_out, TimeoutError)
    assert str(timed_out) == "query exceeded the 10s limit and was cancelled"


def test_syntax_error_keeps_primary_message():
    exc = _SyntaxWithDiag(
        'syntax error at or near "SELEC"\nLINE 1: DECLARE "agent_fetch" CURSOR FOR SELEC 1',
        'syntax error at or near "SELEC"',
    )
    mapped = _map_pg_error(exc)
    assert isinstance(mapped, ValueError)
    assert str(mapped) == (
        'SQL syntax error: syntax error at or near "SELEC". '
        "Only read-only SELECT queries are allowed."
    )
    assert "DECLARE" not in str(mapped)


def test_write_refused_with_exact_message():
    for state in ("42501", "25006"):
        refused = _map_pg_error(_FakePgError("whatever the database said", state))
        assert isinstance(refused, ValueError)
        assert str(refused) == "Only read-only SELECT queries are allowed."


def test_pg_other_errors_keep_database_message_only():
    mapped = _map_pg_error(errors.UndefinedTable('relation "nope" does not exist'))
    assert isinstance(mapped, ValueError)
    assert 'relation "nope" does not exist' in str(mapped)
    untouched = _FakePgError("boom", None)
    assert _map_pg_error(untouched) is untouched


def test_password_never_appears_in_error(monkeypatch):
    monkeypatch.setenv("ANALYST_READER_PASSWORD", CANARY)
    monkeypatch.setenv("POSTGRES_PORT", "54399")
    try:
        PostgresBackend(connect_timeout=2)
    except RuntimeError as exc:
        assert CANARY not in str(exc)
    else:
        raise AssertionError("expected a connection error on a closed port")


def test_missing_password_is_clear(monkeypatch):
    monkeypatch.delenv("ANALYST_READER_PASSWORD", raising=False)
    monkeypatch.setattr("agent.pg_backend._read_env_file_value", lambda name: None)
    try:
        PostgresBackend()
    except RuntimeError as exc:
        assert "ANALYST_READER_PASSWORD" in str(exc)
        assert CANARY not in str(exc)
    else:
        raise AssertionError("expected a missing-password error")


def test_dialect_notes_in_prompt():
    base = "You are an analyst. The SQL dialect is {dialect}."
    rendered = _instructions_for_question(
        base.replace("{dialect}", "PostgreSQL"), "adventureworks",
        "Use schema-qualified names.",
    )
    assert "Dialect notes:\nUse schema-qualified names." in rendered
    plain = _instructions_for_question(base, "olist", "")
    assert "Dialect notes" not in plain


def test_sqlite_dialect_notes():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "note.sqlite"
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE t (d TEXT);")
        conn.commit()
        conn.close()
        assert "strftime" in SqliteBackend(path).dialect_notes


def test_parse_args_dataset():
    assert parse_args([]).dataset == "olist"
    assert parse_args(["--dataset", "adventureworks"]).dataset == "adventureworks"
    assert parse_args(["--dataset", "olist"]).dataset == "olist"
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "nope"])


def _live_backend() -> PostgresBackend:
    pytest.importorskip("psycopg")
    try:
        return PostgresBackend(connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"postgres unavailable: {exc}")


def test_integration_list_tables_qualified():
    backend = _live_backend()
    try:
        tables = backend.list_tables()
        assert tables == sorted(tables)
        assert "sales.salesorderheader" in tables
        assert not any(t.startswith("pg_catalog.") for t in tables)
        assert not any(t.startswith("information_schema.") for t in tables)
        assert not any(
            t.split(".", 1)[0] in ("pr", "sa", "pe", "hr", "pu") for t in tables
        )
    finally:
        backend.close()


def test_integration_get_schema_header():
    backend = _live_backend()
    try:
        schema = backend.get_schema("sales.salesorderheader")
        assert schema["table"] == "sales.salesorderheader"
        names = [c["name"] for c in schema["columns"]]
        assert "salesorderid" in names
        assert all(len(c["samples"]) <= 3 for c in schema["columns"])
        assert isinstance(schema["foreign_keys"], list)
    finally:
        backend.close()


def test_integration_select_works():
    backend = _live_backend()
    try:
        result = backend.run_sql("SELECT COUNT(*) FROM sales.salesorderheader;")
        assert result["row_count"] == 1
        assert result["truncated"] is False
        assert isinstance(result["rows"][0][0], int)
    finally:
        backend.close()


def test_integration_writes_are_refused():
    backend = _live_backend()
    try:
        for bad in (
            "INSERT INTO sales.salesorderheader (SalesOrderID) VALUES (1);",
            "UPDATE sales.salesorderheader SET Status = 1 WHERE SalesOrderID = -1;",
            "DELETE FROM sales.salesorderheader WHERE SalesOrderID = -1;",
            "DROP TABLE sales.salesorderheader;",
            "SET statement_timeout = 0;",
            "RESET statement_timeout;",
        ):
            try:
                backend.run_sql(bad)
            except ValueError as exc:
                assert "Only read-only SELECT queries are allowed." in str(exc), bad
            else:
                raise AssertionError(f"expected refusal for {bad!r}")
    finally:
        backend.close()


def test_integration_settings_do_not_persist():
    backend = _live_backend()
    try:
        loosened = backend.run_sql("SELECT set_config('statement_timeout', '0', false);")
        assert loosened["row_count"] == 1
        try:
            backend.run_sql("SELECT pg_sleep(11);")
        except TimeoutError as exc:
            assert "10s limit" in str(exc)
        else:
            raise AssertionError("expected a timeout after set_config")
        assert backend.run_sql("SELECT 1;")["rows"] == [[1]]
    finally:
        backend.close()


def test_role_layer_refuses_insert_without_agent():
    """Direct analyst_reader connection (never the agent path)."""
    psycopg = pytest.importorskip("psycopg")
    try:
        settings = _pg_settings()
    except RuntimeError as exc:
        pytest.skip(f"no password: {exc}")
    try:
        conn = psycopg.connect(**settings)
    except Exception as exc:
        pytest.skip(f"postgres unavailable: {exc}")
    try:
        try:
            conn.execute(
                "INSERT INTO sales.salesorderheader (SalesOrderID) VALUES (1);"
            )
        except Exception as exc:
            assert getattr(exc, "sqlstate", None) in ("42501", "25006")
        else:
            raise AssertionError("expected the role to refuse INSERT")
    finally:
        conn.rollback()
        conn.close()


def test_integration_timeout_then_usable():
    backend = _live_backend()
    try:
        try:
            backend.run_sql("SELECT pg_sleep(11);")
        except TimeoutError as exc:
            assert "10s limit" in str(exc)
        else:
            raise AssertionError("expected a timeout")
        again = backend.run_sql("SELECT 1;")
        assert again["rows"] == [[1]]
    finally:
        backend.close()


def test_integration_truncation_at_200_rows():
    backend = _live_backend()
    try:
        result = backend.run_sql("SELECT * FROM generate_series(1, 300) AS g(n);")
        assert result["row_count"] == 200
        assert result["truncated"] is True
    finally:
        backend.close()


def test_integration_large_result_streams():
    import time

    backend = _live_backend()
    try:
        start = time.monotonic()
        result = backend.run_sql("SELECT * FROM generate_series(1, 5000000) AS g(n);")
        elapsed = time.monotonic() - start
        assert result["row_count"] == 200
        assert result["truncated"] is True
        assert elapsed < 8, f"took {elapsed:.1f}s; rows are not streaming"
    finally:
        backend.close()


def test_integration_numeric_sum_is_float():
    backend = _live_backend()
    try:
        result = backend.run_sql("SELECT SUM(unitprice) FROM sales.salesorderdetail;")
        assert isinstance(result["rows"][0][0], float)
    finally:
        backend.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"all {len(tests)} tests passed")
