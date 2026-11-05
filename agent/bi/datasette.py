"""Datasette connector: pure link builder, no network, no writes."""

from __future__ import annotations

import os
from urllib.parse import quote, quote_plus, urlparse

from agent.bi import PublishResult

_BASE_ENV = "DATASETTE_URL"
_DEFAULT_BASE = "http://127.0.0.1:8001"
_MAX_URL = 8000


class DatasetteConnector:
    """Build a Datasette+vega chart link for the exact recorded SQL."""

    requires_approval = False

    def __init__(self, database: str) -> None:
        self.database = database

    def publish(
        self,
        sql: str,
        title: str,
        chart_type: str,
        x: str,
        y: str,
        x_values: list[str] | None = None,
    ) -> PublishResult:
        _ = title
        _ = x_values
        if chart_type not in ("bar", "line"):
            raise ValueError('chart_type must be "bar" or "line".')
        base = os.environ.get(_BASE_ENV, _DEFAULT_BASE).rstrip("/")
        scheme = urlparse(base).scheme
        if scheme not in ("http", "https"):
            raise ValueError(
                f"{_BASE_ENV} must be an http(s) URL, got {base!r}."
            )
        url = (
            f"{base}/{self.database}"
            f"?sql={quote_plus(sql, safe='*')}"
            f"#g.mark={chart_type}"
            f"&g.x_column={quote(x)}"
            f"&g.x_type=ordinal"
            f"&g.y_column={quote(y)}"
            f"&g.y_type=quantitative"
        )
        if len(url) > _MAX_URL:
            raise ValueError("query too long to share as a link")
        return PublishResult(
            url=url,
            note="Datasette stops queries after 1 second by default.",
        )
