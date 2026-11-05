"""Dataset registry: the single place where available databases are listed.

Adding a database later means adding one entry here and nothing else:
name -> {description, label, prompt, make_backend}.

Optional keys: "bi" (BI connector) and "optional": True with a
"configured" callable. `--dataset all` skips optional entries whose
configured() is False, so one unset warehouse never blocks the others.

Factories import their backend lazily so importing this module never
requires a database driver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent


def _olist_backend() -> Any:
    from agent.backend import SqliteBackend

    return SqliteBackend(REPO_ROOT / "data" / "olist.sqlite")


def _adventureworks_backend() -> Any:
    from agent.pg_backend import PostgresBackend

    return PostgresBackend()


def _bigquery_backend() -> Any:
    from agent.bq_backend import BigQueryBackend

    return BigQueryBackend()


def _bigquery_configured() -> bool:
    from agent.bq_backend import bigquery_configured

    return bigquery_configured()


REGISTRY: dict[str, dict[str, Any]] = {
    "olist": {
        "description": (
            "Brazilian e-commerce marketplace orders, items, products, "
            "payments, and reviews."
        ),
        "label": "Olist",
        "prompt": "olist> ",
        "make_backend": _olist_backend,
        "bi": {"connector": "datasette", "database": "olist"},
    },
    "adventureworks": {
        "description": (
            "Bicycle company sales, products, purchasing, and HR."
        ),
        "label": "AdventureWorks",
        "prompt": "adventureworks> ",
        "make_backend": _adventureworks_backend,
        "bi": {"connector": "metabase", "database": "adventureworks"},
    },
    "bigquery": {
        "description": (
            "A BigQuery dataset chosen by BIGQUERY_DATASET (read-only, "
            "cost-capped)."
        ),
        "label": "BigQuery",
        "prompt": "bigquery> ",
        "make_backend": _bigquery_backend,
        "optional": True,
        "configured": _bigquery_configured,
    },
}


def active_datasets() -> dict[str, dict[str, Any]]:
    """Registry entries to use for `--dataset all`: skip unconfigured optionals."""
    return {
        name: entry
        for name, entry in REGISTRY.items()
        if not entry.get("optional") or entry["configured"]()
    }
