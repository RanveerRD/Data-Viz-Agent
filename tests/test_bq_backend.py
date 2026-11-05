"""BigQuery backend tests against a fake client (no network, no library)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from agent import bq_backend
from agent.bq_backend import BigQueryBackend, _bq_value, bigquery_configured
from agent.datasets import REGISTRY, active_datasets


@pytest.fixture(autouse=True)
def _fake_job_config(monkeypatch):
    monkeypatch.setattr(
        bq_backend, "_make_job_config", lambda **kw: SimpleNamespace(**kw)
    )
    monkeypatch.setattr(bq_backend, "_setting", lambda name: None)


class _Row:
    def __init__(self, mapping: dict[str, Any]):
        self._m = mapping

    def values(self):
        return list(self._m.values())

    def __getitem__(self, key):
        return self._m[key]


class _Iter:
    def __init__(self, names, rows):
        self.schema = [SimpleNamespace(name=n) for n in names]
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _Job:
    def __init__(self, statement_type="SELECT", bytes_=100, names=("n",), rows=(), error=None):
        self.statement_type = statement_type
        self.total_bytes_processed = bytes_
        self._names, self._rows, self._error = names, rows, error
        self.cancelled = False
        self.result_kwargs: dict[str, Any] = {}

    def result(self, **kwargs):
        self.result_kwargs = kwargs
        if self._error:
            raise self._error
        return _Iter(self._names, [_Row(dict(zip(self._names, r))) for r in self._rows])

    def cancel(self):
        self.cancelled = True


class _Client:
    def __init__(self, dry=None, real=None):
        self.dry = dry or _Job()
        self.real = real or _Job()
        self.queries: list[tuple[str, Any]] = []
        self.tables = ["b", "a"]

    def query(self, sql, job_config):
        self.queries.append((sql, job_config))
        return self.dry if getattr(job_config, "dry_run", False) else self.real

    def list_tables(self, dataset):
        return [SimpleNamespace(table_id=t) for t in self.tables]

    def get_table(self, name):
        return SimpleNamespace(schema=[
            SimpleNamespace(name="id", field_type="INTEGER", mode="NULLABLE"),
            SimpleNamespace(name="tags", field_type="STRING", mode="REPEATED"),
        ])

    def list_rows(self, table, max_results):
        return [_Row({"id": 1, "tags": ["x"]}), _Row({"id": None, "tags": []})]


def _backend(client=None, **kw):
    return BigQueryBackend("proj.ds", client=client or _Client(), **kw)


def test_dataset_must_be_configured() -> None:
    with pytest.raises(RuntimeError, match="BIGQUERY_DATASET"):
        BigQueryBackend(client=_Client())
    with pytest.raises(RuntimeError, match="malformed"):
        BigQueryBackend("justone", client=_Client())


def test_list_tables_sorted_and_qualified() -> None:
    assert _backend().list_tables() == ["proj.ds.a", "proj.ds.b"]


def test_get_schema_types_and_samples() -> None:
    schema = _backend().get_schema("proj.ds.a")
    cols = {c["name"]: c for c in schema["columns"]}
    assert cols["id"]["samples"] == [1]
    assert cols["tags"]["type"] == "ARRAY<STRING>"
    with pytest.raises(ValueError, match="unknown table"):
        _backend().get_schema("proj.ds.nope")


def test_select_runs_with_cost_cap_and_timeout() -> None:
    client = _Client(real=_Job(names=("n",), rows=[(1,), (2,)]))
    out = _backend(client, max_bytes=1000, timeout_s=5).run_sql("SELECT n FROM t;")
    assert out == {"columns": ["n"], "rows": [[1], [2]], "row_count": 2, "truncated": False}
    dry_cfg, real_cfg = client.queries[0][1], client.queries[1][1]
    assert dry_cfg.dry_run is True and not hasattr(real_cfg, "dry_run")
    assert real_cfg.maximum_bytes_billed == 1000
    assert client.real.result_kwargs["timeout"] == 5


@pytest.mark.parametrize("kind", ["INSERT", "DELETE", "CREATE_TABLE", "SCRIPT", "DROP_TABLE", None])
def test_non_select_rejected_before_real_job(kind) -> None:
    client = _Client(dry=_Job(statement_type=kind))
    with pytest.raises(ValueError, match="read-only SELECT"):
        _backend(client).run_sql("DELETE FROM t WHERE true")
    assert len(client.queries) == 1  # only the dry run


def test_multiple_statements_rejected() -> None:
    client = _Client()
    with pytest.raises(ValueError, match="exactly one"):
        _backend(client).run_sql("SELECT 1; SELECT 2")
    assert client.queries == []


def test_over_budget_rejected_with_guidance() -> None:
    client = _Client(dry=_Job(bytes_=10 * 1024**3))
    with pytest.raises(ValueError, match="over the 5 GB limit"):
        _backend(client).run_sql("SELECT * FROM big")
    assert len(client.queries) == 1


def test_row_cap_marks_truncated() -> None:
    rows = [(i,) for i in range(bq_backend.MAX_ROWS + 1)]
    client = _Client(real=_Job(rows=rows))
    out = _backend(client).run_sql("SELECT n FROM t")
    assert out["truncated"] is True and out["row_count"] == bq_backend.MAX_ROWS


def test_timeout_cancels_job() -> None:
    client = _Client(real=_Job(error=TimeoutError()))
    with pytest.raises(TimeoutError, match="cancelled"):
        _backend(client, timeout_s=1).run_sql("SELECT 1")
    assert client.real.cancelled


def test_google_errors_become_short_value_errors() -> None:
    class BadRequest(Exception):
        message = "Unrecognized name: x at [1:8]"

    BadRequest.__module__ = "google.api_core.exceptions"
    client = _Client(real=_Job(error=BadRequest("400 long\nstack")))
    with pytest.raises(ValueError, match="Unrecognized name"):
        _backend(client).run_sql("SELECT x")


def test_value_conversion() -> None:
    assert _bq_value(Decimal("2.50")) == 2.5
    assert _bq_value(Decimal("3")) == 3
    assert _bq_value(date(2026, 1, 2)) == "2026-01-02"
    assert _bq_value(float("nan")) is None
    assert _bq_value([Decimal("1"), b"ab"]) == [1, "<binary 2 bytes>"]
    assert _bq_value({"a": date(2026, 1, 2)}) == {"a": "2026-01-02"}


def test_notes_mention_dataset_and_cap() -> None:
    notes = _backend(max_bytes=2 * 1024**3).dialect_notes
    assert "proj.ds" in notes and "2 GB" in notes


def test_all_mode_skips_unconfigured_optional(monkeypatch) -> None:
    monkeypatch.setattr(bq_backend, "_setting", lambda name: None)
    assert bigquery_configured() is False
    assert "bigquery" in REGISTRY
    assert "bigquery" not in active_datasets()
    monkeypatch.setattr(bq_backend, "_setting", lambda name: "p.d")
    assert "bigquery" in active_datasets()
