"""Read-only SQLite connections for analytics queries.

open_sqlite_readonly() opens an existing database file through three
independent layers, so analytics code cannot modify data by accident:

1. URI ``mode=ro`` -- the file is opened read-only and never created.
2. ``PRAGMA query_only = ON`` -- the SQL layer rejects writes.
3. A ``set_authorizer`` allowlist -- only SELECT/READ/FUNCTION/RECURSIVE
   operations and an explicit list of read-only PRAGMAs are permitted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.request import pathname2url

#: PRAGMA names that are pure read-only introspection. Every other PRAGMA,
#: including writes such as ``query_only=OFF``, is denied by the authorizer.
READ_ONLY_PRAGMAS = frozenset(
    {
        "table_info",
        "table_xinfo",
        "table_list",
        "index_list",
        "index_info",
        "foreign_key_list",
    }
)

_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)


def _authorizer(action, arg1, _arg2, _db_name, _source):
    """Allow reads (plus RECURSIVE for CTEs) and listed PRAGMAs; deny the rest."""
    if action in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA:
        if (arg1 or "").lower() in READ_ONLY_PRAGMAS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


def install_readonly_authorizer(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Install the read-only authorizer on an already-open connection."""
    conn.set_authorizer(_authorizer)
    return conn


def open_sqlite_readonly(path: str | Path) -> sqlite3.Connection:
    """Open an existing SQLite file read-only.

    Raises FileNotFoundError if the file does not exist; the file is never
    created.
    """
    db_path = Path(path)
    if not db_path.is_file():
        raise FileNotFoundError(f"SQLite database not found (not creating): {db_path}")
    uri = "file:" + pathname2url(str(db_path.resolve())) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.enable_load_extension(False)
        conn.execute("PRAGMA query_only = ON;")
        install_readonly_authorizer(conn)
    except Exception:
        conn.close()
        raise
    return conn
