"""BigQuery backend (read-only, cost-capped). Optional dependency.

Install with ``pip install -r requirements-bigquery.txt``.

Read-only layers, in order:

1. Credentials: use a service account holding only BigQuery Data Viewer
   and BigQuery Job User. That is the real guarantee; the rest is defence
   in depth.
2. Statement check: every query is dry-run first. Only statement type
   SELECT is allowed, which rejects DML, DDL, scripts and procedure calls.
3. Cost and time limits: the dry run's byte estimate is checked against
   BIGQUERY_MAX_BYTES, the real job carries the same cap as
   maximum_bytes_billed, and results are fetched with a row cap and timeout.

Settings (environment, then .env):

    BIGQUERY_DATASET     required, "project.dataset" to expose
    BIGQUERY_PROJECT     project billed for queries (default: the dataset's
                         project; public datasets need your own project here)
    BIGQUERY_MAX_BYTES   scan cap per query (default 5 GB; a guess, tune it)
    BIGQUERY_TIMEOUT_S   per-query wait limit (default 60)
    GOOGLE_APPLICATION_CREDENTIALS  service account key file (or use ADC)
"""

from __future__ import annotations

import math
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from agent.backend import MAX_ROWS, _split_statements
from agent.provider import _setting

DEFAULT_MAX_BYTES = 5 * 1024**3
DEFAULT_TIMEOUT_S = 60.0


def bigquery_configured() -> bool:
    """True when a BigQuery dataset has been named in the settings."""
    return bool(_setting("BIGQUERY_DATASET"))


def _make_job_config(**kwargs: Any) -> Any:
    """QueryJobConfig, imported lazily so the library stays optional."""
    try:
        from google.cloud import bigquery
    except ImportError:
        raise RuntimeError(
            "google-cloud-bigquery is not installed: "
            "pip install -r requirements-bigquery.txt"
        ) from None
    return bigquery.QueryJobConfig(**kwargs)


def _make_client(project: str | None) -> Any:
    try:
        from google.cloud import bigquery
    except ImportError:
        raise RuntimeError(
            "google-cloud-bigquery is not installed: "
            "pip install -r requirements-bigquery.txt"
        ) from None
    try:
        return bigquery.Client(project=project)
    except Exception as exc:
        raise RuntimeError(
            f"could not create BigQuery client: {str(exc).splitlines()[0]}"
        ) from None


class BigQueryBackend:
    """Read-only BigQuery backend over one dataset."""

    def __init__(
        self,
        dataset: str | None = None,
        *,
        project: str | None = None,
        max_bytes: int | None = None,
        timeout_s: float | None = None,
        client: Any = None,
    ) -> None:
        dataset = dataset or _setting("BIGQUERY_DATASET")
        if not dataset or dataset.count(".") != 1:
            raise RuntimeError(
                "BIGQUERY_DATASET is missing or malformed: set it to "
                "project.dataset (for example bigquery-public-data.thelook_ecommerce)."
            )
        self.dataset = dataset
        data_project = dataset.split(".", 1)[0]
        self.project = project or _setting("BIGQUERY_PROJECT") or data_project
        self.max_bytes = int(
            max_bytes or _setting("BIGQUERY_MAX_BYTES") or DEFAULT_MAX_BYTES
        )
        self.timeout_s = float(
            timeout_s or _setting("BIGQUERY_TIMEOUT_S") or DEFAULT_TIMEOUT_S
        )
        self._client = client if client is not None else _make_client(self.project)

    @property
    def dialect(self) -> str:
        return "BigQuery"

    @property
    def dialect_notes(self) -> str:
        return (
            f"GoogleSQL. Tables live in `{self.dataset}`; use fully qualified "
            "names in backticks exactly as list_tables shows them. "
            "Use DATE_TRUNC(col, MONTH) and EXTRACT(YEAR FROM col) for date "
            "parts. Cost is bytes scanned: select only needed columns and "
            "filter early; SELECT * on a large table is rejected when it "
            f"would scan more than {self.max_bytes / 1024**3:g} GB."
        )

    def list_tables(self) -> list[str]:
        return sorted(
            f"{self.dataset}.{item.table_id}"
            for item in self._client.list_tables(self.dataset)
        )

    def get_schema(self, table: str) -> dict[str, Any]:
        if table not in self.list_tables():
            raise ValueError(f"unknown table: {table}")
        meta = self._client.get_table(table)
        sample_rows: list[Any] = []
        try:
            sample_rows = list(self._client.list_rows(meta, max_results=3))
        except Exception:
            # Views and some external tables cannot be read directly.
            sample_rows = []
        columns = []
        for field in meta.schema:
            type_name = field.field_type
            if getattr(field, "mode", None) == "REPEATED":
                type_name = f"ARRAY<{type_name}>"
            samples = [
                _bq_value(row[field.name])
                for row in sample_rows
                if row[field.name] is not None
            ]
            columns.append(
                {"name": field.name, "type": type_name, "samples": samples}
            )
        return {"table": table, "columns": columns}

    def run_sql(self, query: str) -> dict[str, Any]:
        statements = _split_statements(query)
        if len(statements) != 1:
            raise ValueError("run_sql accepts exactly one SQL statement")
        statement = statements[0]
        try:
            return self._run_checked(statement)
        except (ValueError, TimeoutError):
            raise
        except Exception as exc:
            if type(exc).__module__.startswith("google"):
                message = getattr(exc, "message", None) or str(exc)
                raise ValueError(
                    f"BigQuery error: {message.splitlines()[0]}"
                ) from None
            raise

    def _run_checked(self, statement: str) -> dict[str, Any]:
        dry = self._client.query(
            statement,
            job_config=_make_job_config(dry_run=True, use_query_cache=False),
        )
        if (dry.statement_type or "").upper() != "SELECT":
            raise ValueError("Only read-only SELECT queries are allowed.")
        estimate = int(dry.total_bytes_processed or 0)
        if estimate > self.max_bytes:
            raise ValueError(
                f"query would scan {estimate / 1024**3:.2f} GB, over the "
                f"{self.max_bytes / 1024**3:g} GB limit; select fewer columns "
                "or add a filter."
            )
        job = self._client.query(
            statement,
            job_config=_make_job_config(
                maximum_bytes_billed=self.max_bytes, use_query_cache=True
            ),
        )
        try:
            iterator = job.result(max_results=MAX_ROWS + 1, timeout=self.timeout_s)
            fetched = [tuple(row.values()) for row in iterator]
            columns = [field.name for field in (iterator.schema or [])]
        except TimeoutError:
            self._cancel(job)
            raise TimeoutError(
                f"query exceeded the {self.timeout_s:g}s limit and was cancelled"
            ) from None
        truncated = len(fetched) > MAX_ROWS
        rows = fetched[:MAX_ROWS]
        return {
            "columns": columns,
            "rows": [[_bq_value(v) for v in row] for row in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }

    @staticmethod
    def _cancel(job: Any) -> None:
        try:
            job.cancel()
        except Exception:
            pass

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


def _bq_value(value: Any) -> Any:
    """Convert one BigQuery cell to plain JSON-safe values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, Decimal):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return int(number) if number.is_integer() else number
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, (bytes, memoryview)):
        return f"<binary {len(value)} bytes>"
    if isinstance(value, (list, tuple)):
        return [_bq_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _bq_value(v) for k, v in value.items()}
    return str(value)
