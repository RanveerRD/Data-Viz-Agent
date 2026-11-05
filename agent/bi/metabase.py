"""Metabase connector: publish a chart candidate as a saved question.

Reached only through the console /publish command (no model call, no
tokens). Standard library only. The API key travels solely in the
x-api-key header and never appears in output, audit entries, or errors.
Redirects are never followed, so the key is never sent to another host.
"""

from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

from agent.bi import PublishResult
from agent.loop import _read_env_file_value

_URL_ENV = "METABASE_URL"
_KEY_ENV = "METABASE_API_KEY"
_DEFAULT_BASE = "http://localhost:3000"
_TIMEOUT_S = 10
_YEARLIKE_RE = re.compile(r"^(\d{4})$")


class _Redirected(Exception):
    """Raised instead of following an HTTP redirect."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _Redirected()


def _opener() -> urllib.request.OpenerDirector:
    """Opener with no redirect follower: redirects raise _Redirected."""
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPHandler())
    opener.add_handler(urllib.request.HTTPSHandler())
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.HTTPErrorProcessor())
    opener.add_handler(_NoRedirect())
    return opener


class MetabaseConnector:
    """Save a native-SQL question in Metabase; approval happens in console."""

    requires_approval = True

    def __init__(self, database: str) -> None:
        self.database = database

    def _base(self) -> str:
        base = (
            os.environ.get(_URL_ENV)
            or _read_env_file_value(_URL_ENV)
            or _DEFAULT_BASE
        ).rstrip("/")
        if urlparse(base).scheme not in ("http", "https"):
            raise ValueError(f"{_URL_ENV} must be an http(s) URL, got {base!r}.")
        return base

    def _key(self) -> str:
        key = os.environ.get(_KEY_ENV) or _read_env_file_value(_KEY_ENV)
        if not key:
            raise RuntimeError(f"{_KEY_ENV} is not set in .env")
        return key

    def _call(
        self, base: str, key: str, method: str, url: str, body: object = None
    ) -> tuple[int, str]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "x-api-key": key},
        )
        try:
            with _opener().open(request, timeout=_TIMEOUT_S) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except _Redirected:
            raise RuntimeError(
                "Metabase redirected the request; check METABASE_URL"
            )
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError("Metabase rejected the API key")
            if exc.code == 400:
                detail = exc.read().decode("utf-8", "replace")[:200]
                raise RuntimeError(f"Metabase refused the request: {detail}")
            raise RuntimeError(f"Metabase returned HTTP {exc.code}")
        except (TimeoutError, socket.timeout):
            raise RuntimeError("Metabase request timed out after 10 seconds")
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise RuntimeError("Metabase request timed out after 10 seconds")
            raise RuntimeError(
                f"Metabase is not reachable at {base}; is the container running?"
            )
        except OSError:
            raise RuntimeError(
                f"Metabase is not reachable at {base}; is the container running?"
            )

    def _database_id(self, base: str, key: str) -> int:
        _status, text = self._call(base, key, "GET", f"{base}/api/database")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            raise RuntimeError("unexpected response from Metabase")
        items = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise RuntimeError("unexpected response from Metabase")
        for entry in items:
            if (
                isinstance(entry, dict)
                and entry.get("name") == self.database
                and entry.get("id") is not None
            ):
                return entry["id"]
        names = sorted(
            str(entry["name"])
            for entry in items
            if isinstance(entry, dict) and "name" in entry
        )
        raise ValueError(
            f"unknown Metabase database {self.database!r}; "
            f"available: {', '.join(names)}"
        )

    def publish(
        self,
        sql: str,
        title: str,
        chart_type: str,
        x: str,
        y: str,
        x_values: list[str] | None = None,
    ) -> PublishResult:
        """Save the exact recorded SQL as a Metabase question."""
        if chart_type not in ("bar", "line"):
            raise ValueError('chart_type must be "bar" or "line".')
        base = self._base()
        key = self._key()
        database_id = self._database_id(base, key)
        name = f"Agent: {title}"
        viz: dict[str, object] = {
            "graph.dimensions": [x],
            "graph.metrics": [y],
        }
        if x_values and all(
            _YEARLIKE_RE.match(v) and 1900 <= int(v) <= 2100 for v in x_values
        ):
            # Best effort, cosmetic only: show year-like x without a
            # thousands separator. Visualization settings are stored
            # opaquely and ignored when unknown, so a wrong value here
            # cannot make Metabase reject the card.
            viz["column_settings"] = {
                json.dumps(["name", x], separators=(",", ":")): {
                    "number_separators": "."
                }
            }
        body = {
            "name": name,
            "dataset_query": {
                "type": "native",
                "database": database_id,
                "native": {"query": sql},
            },
            "display": chart_type,
            "visualization_settings": viz,
            "collection_id": None,
        }
        _status, text = self._call(base, key, "POST", f"{base}/api/card", body)
        try:
            payload = json.loads(text)
            card_id = payload.get("id") if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            card_id = None
        if card_id is None:
            raise RuntimeError("Metabase did not return a card id")
        return PublishResult(
            url=f"{base}/question/{card_id}", note="saved question created"
        )
