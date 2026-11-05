"""Database backends for the analyst agent.

Only SQLite exists for now. A future Postgres backend implements the same
DatabaseBackend protocol, so tools and the loop stay untouched.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Protocol

from agent.db_guard import open_sqlite_readonly

MAX_ROWS = 200
STATEMENT_TIMEOUT_S = 10.0


class DatabaseBackend(Protocol):
    """The interface every database backend must implement."""

    @property
    def dialect(self) -> str:
        """Short dialect name, e.g. "SQLite". Inserted into the system prompt."""
        ...

    @property
    def dialect_notes(self) -> str:
        """A few lines of dialect guidance, inserted under "Dialect notes:"."""
        ...

    def list_tables(self) -> list[str]:
        """Names of queryable tables and views, sorted."""
        ...

    def get_schema(self, table: str) -> dict[str, Any]:
        """Columns, types, and sample values for one table."""
        ...

    def run_sql(self, query: str) -> dict[str, Any]:
        """Run exactly one statement; at most MAX_ROWS rows are returned."""
        ...


def _split_statements(sql: str) -> list[str]:
    """Split SQL on top-level semicolons, respecting quotes and comments."""
    statements: list[str] = []
    buf: list[str] = []
    i, n = 0, len(sql)
    quote: str | None = None
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if quote is not None:
            buf.append(ch)
            if ch == quote:
                if nxt == quote:
                    buf.append(nxt)
                    i += 1
                else:
                    quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
            buf.append(ch)
        elif ch == "-" and nxt == "-":
            while i < n and sql[i] != "\n":
                i += 1
            continue
        elif ch == "/" and nxt == "*":
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        elif ch == ";":
            statements.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf)
    if tail.strip():
        statements.append(tail)
    return [s.strip() for s in statements if s.strip()]


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


class SqliteBackend:
    """SQLite backend over a read-only connection (see agent/db_guard.py)."""

    def __init__(self, path: str | Path) -> None:
        self._conn = open_sqlite_readonly(path)
        self.path = Path(path)

    @property
    def dialect(self) -> str:
        return "SQLite"

    @property
    def dialect_notes(self) -> str:
        return (
            "Dates are text like 'YYYY-MM-DD HH:MM:SS'; "
            "use strftime('%Y-%m', col) for parts."
        )

    def list_tables(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name;"
        ).fetchall()
        return [row[0] for row in rows]

    def get_schema(self, table: str) -> dict[str, Any]:
        if table not in self.list_tables():
            raise ValueError(f"unknown table: {table}")
        quoted = _quote_ident(table)
        cols = self._conn.execute(f"PRAGMA table_info({quoted});").fetchall()
        columns = []
        for _cid, name, coltype, _notnull, _dflt, _pk in cols:
            qcol = _quote_ident(name)
            samples = self._conn.execute(
                f"SELECT {qcol} FROM {quoted} WHERE {qcol} IS NOT NULL LIMIT 3;"
            ).fetchall()
            columns.append(
                {
                    "name": name,
                    "type": coltype or "UNKNOWN",
                    "samples": [_json_safe(row[0]) for row in samples],
                }
            )
        return {"table": table, "columns": columns}

    def run_sql(self, query: str) -> dict[str, Any]:
        statements = _split_statements(query)
        if len(statements) != 1:
            raise ValueError("run_sql accepts exactly one SQL statement")
        deadline = time.monotonic() + STATEMENT_TIMEOUT_S

        def _check_timeout() -> int:
            return 1 if time.monotonic() >= deadline else 0

        conn = self._conn
        conn.set_progress_handler(_check_timeout, 1000)
        try:
            cursor = conn.execute(statements[0])
            try:
                columns = [d[0] for d in (cursor.description or [])]
                fetched = cursor.fetchmany(MAX_ROWS + 1)
            finally:
                cursor.close()
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                raise TimeoutError(
                    f"query exceeded the {STATEMENT_TIMEOUT_S:g}s limit and was cancelled"
                ) from exc
            raise
        finally:
            conn.set_progress_handler(None, 0)
        truncated = len(fetched) > MAX_ROWS
        rows = fetched[:MAX_ROWS]
        return {
            "columns": columns,
            "rows": [[_json_safe(v) for v in row] for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }

    def close(self) -> None:
        self._conn.close()
