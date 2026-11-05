"""PostgreSQL backend for the AdventureWorks database (read-only).

The agent connects only as analyst_reader. Safety is layered: the role
itself carries default_transaction_read_only and statement_timeout, and this
backend additionally opens the session read-only with an explicit 10-second
statement timeout. Each run_sql runs in its own transaction block with a
server-side named cursor, so only MAX_ROWS + 1 rows cross the wire and the
connection is cleanly reusable after any error.
"""

from __future__ import annotations

import math
import os
import uuid
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

import psycopg
from psycopg import errors, sql

from agent.backend import MAX_ROWS, STATEMENT_TIMEOUT_S, _split_statements
from agent.loop import _read_env_file_value

#: Short-name alias schemas for the five main ones (pr/production,
#: sa/sales, pe/person, hr/humanresources, pu/purchasing). They hold the
#: same data under abbreviated table names, so they are hidden from
#: list_tables (and therefore from get_schema validation).
EXCLUDED_SCHEMAS: frozenset[str] = frozenset({"pr", "sa", "pe", "hr", "pu"})


class PostgresBackend:
    """Read-only PostgreSQL backend over a single analyst_reader connection."""

    def __init__(self, **overrides: Any) -> None:
        settings = _pg_settings()
        settings.update(overrides)
        try:
            self._conn = psycopg.connect(**settings, autocommit=True)
        except Exception as exc:
            raise RuntimeError(
                "could not connect to Postgres database "
                f"{settings.get('dbname')!r} as {settings.get('user')!r} "
                f"on {settings.get('host')!r}:{settings.get('port')!r}: {exc}"
            ) from None
        try:
            self._lock_session()
        except Exception as exc:
            self._conn.close()
            raise RuntimeError(
                "could not lock Postgres session to read-only "
                f"as {settings.get('user')!r}: {exc}"
            ) from None

    @property
    def dialect(self) -> str:
        return "PostgreSQL"

    @property
    def dialect_notes(self) -> str:
        return (
            "Use schema-qualified names exactly as list_tables shows them. "
            "Unquoted identifiers fold to lowercase, so quote \"MixedCase\" names. "
            "Use EXTRACT(YEAR FROM col) or date_trunc('month', col) for date parts. "
            "Cast to numeric before rounding: ROUND(x::numeric, 2)."
        )

    def list_tables(self) -> list[str]:
        rows = self._fetchall(
            "SELECT n.nspname || '.' || c.relname "
            "FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f') "
            "AND n.nspname NOT IN ('pg_catalog', 'information_schema') "
            "ORDER BY 1;"
        )
        names = [row[0] for row in rows]
        return sorted(
            name
            for name in names
            if name.split(".", 1)[0] not in EXCLUDED_SCHEMAS
        )

    def get_schema(self, table: str) -> dict[str, Any]:
        if table not in self.list_tables():
            raise ValueError(f"unknown table: {table}")
        schema, _, name = table.rpartition(".")
        cols = self._fetchall(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s "
            "ORDER BY ordinal_position;",
            (schema, name),
        )
        columns = []
        for colname, coltype in cols:
            samples = self._fetchall(
                sql.SQL("SELECT {} FROM {}.{} WHERE {} IS NOT NULL LIMIT 3;").format(
                    sql.Identifier(colname),
                    sql.Identifier(schema),
                    sql.Identifier(name),
                    sql.Identifier(colname),
                ),
            )
            columns.append(
                {
                    "name": colname,
                    "type": coltype or "UNKNOWN",
                    "samples": [_pg_value(row[0]) for row in samples],
                }
            )
        fk_rows = self._fetchall(
            "SELECT kcu.column_name, ccu.table_schema, ccu.table_name, "
            "ccu.column_name FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "ON tc.constraint_name = kcu.constraint_name "
            "AND tc.table_schema = kcu.table_schema "
            "JOIN information_schema.constraint_column_usage ccu "
            "ON ccu.constraint_name = tc.constraint_name "
            "WHERE tc.constraint_type = 'FOREIGN KEY' "
            "AND tc.table_schema = %s AND tc.table_name = %s;",
            (schema, name),
        )
        foreign_keys = [
            f"{col} -> {fschema}.{ftable}.{fcol}"
            for col, fschema, ftable, fcol in fk_rows
        ]
        return {"table": table, "columns": columns, "foreign_keys": foreign_keys}

    def run_sql(self, query: str) -> dict[str, Any]:
        statements = _split_statements(query)
        if len(statements) != 1:
            raise ValueError("run_sql accepts exactly one SQL statement")
        self._lock_session()
        try:
            with self._conn.transaction():
                with self._conn.cursor(name="agent_fetch") as cur:
                    cur.execute(statements[0])
                    columns = [d[0] for d in (cur.description or [])]
                    fetched = cur.fetchmany(MAX_ROWS + 1)
        except Exception as exc:
            mapped = _map_pg_error(exc)
            if mapped is exc:
                raise
            raise mapped from exc
        truncated = len(fetched) > MAX_ROWS
        rows = fetched[:MAX_ROWS]
        return {
            "columns": columns,
            "rows": [[_pg_value(v) for v in row] for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }

    def close(self) -> None:
        self._conn.close()

    def _lock_session(self) -> None:
        """Re-apply read-only mode and the statement timeout.

        Called on connect and before every run_sql, so nothing a previous
        query did through set_config() can persist.
        """
        self._conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        self._conn.execute(f"SET statement_timeout = '{STATEMENT_TIMEOUT_S:g}s'")

    def _fetchall(self, query: Any, params: Any = None) -> list[tuple]:
        with self._conn.transaction():
            with self._conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()


def _pg_settings() -> dict[str, Any]:
    """Connection settings; the password never appears in logs or errors."""
    password = os.environ.get("ANALYST_READER_PASSWORD") or _read_env_file_value(
        "ANALYST_READER_PASSWORD"
    )
    if not password:
        raise RuntimeError(
            "ANALYST_READER_PASSWORD is missing: add it to .env as "
            "ANALYST_READER_PASSWORD=<password> or export it in the "
            "environment. The value itself is never printed."
        )
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "dbname": os.environ.get("POSTGRES_DB", "adventureworks"),
        "user": os.environ.get("POSTGRES_USER", "analyst_reader"),
        "password": password,
        "connect_timeout": 10,
    }


def _map_pg_error(exc: Exception) -> Exception:
    """Map a Postgres failure to a clear agent-facing error."""
    state = getattr(exc, "sqlstate", None)
    if state in ("42501", "25006"):
        return ValueError("Only read-only SELECT queries are allowed.")
    if isinstance(exc, errors.SyntaxError):
        return ValueError(
            f"SQL syntax error: {_primary_message(exc)}. "
            "Only read-only SELECT queries are allowed."
        )
    if state == "57014":
        return TimeoutError(
            f"query exceeded the {STATEMENT_TIMEOUT_S:g}s limit and was cancelled"
        )
    if isinstance(exc, errors.Error):
        return ValueError(str(exc))
    return exc


def _primary_message(exc: Exception) -> str:
    """The database's primary message only, never the full error text."""
    diag = getattr(exc, "diag", None)
    primary = getattr(diag, "message_primary", None) if diag is not None else None
    if primary:
        return primary
    return str(exc).split("\n")[0]


def _pg_value(value: Any) -> Any:
    """Convert one Postgres cell to plain JSON-safe values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Decimal):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return int(number) if number.is_integer() else number
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, memoryview)):
        return f"<binary {len(value)} bytes>"
    return str(value)
