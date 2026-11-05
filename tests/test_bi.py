"""BI connectors and /publish (no API calls, no real network)."""

from __future__ import annotations

import json
import shlex
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.__main__ import _handle_command, _render_all_outcome  # noqa: E402
from agent.bi import CONNECTORS  # noqa: E402
from agent.bi.datasette import DatasetteConnector  # noqa: E402
from agent.bi.metabase import MetabaseConnector  # noqa: E402
from agent.datasets import REGISTRY  # noqa: E402
from agent.record import AnalysisRecord  # noqa: E402
from agent.tools.make_chart import chart_candidates  # noqa: E402

EXAMPLE_SQL = (
    "   select strftime('%Y-%m', order_purchase_timestamp) as month,"
    " count(*) as orders\r\n   from orders group by 1 order by 1"
)
EXAMPLE_URL = (
    "http://127.0.0.1:8001/olist?sql="
    "+++select+strftime%28%27%25Y-%25m%27%2C+order_purchase_timestamp%29"
    "+as+month%2C+count%28*%29+as+orders%0D%0A"
    "+++from+orders+group+by+1+order+by+1"
    "#g.mark=line&g.x_column=month&g.x_type=ordinal"
    "&g.y_column=orders&g.y_type=quantitative"
)


def test_datasette_matches_working_example():
    result = DatasetteConnector("olist").publish(
        EXAMPLE_SQL, "orders by month", "line", "month", "orders")
    assert result.url == EXAMPLE_URL


def test_datasette_bar_mark_and_round_trip():
    sql = "select a, b from t"
    result = DatasetteConnector("db").publish(sql, "t", "bar", "a", "b")
    assert "#g.mark=bar&" in result.url
    assert parse_qs(urlparse(result.url).query)["sql"] == [sql]


def test_datasette_column_encoding():
    result = DatasetteConnector("db").publish(
        "select 1", "t", "bar", "my col", "my_col")
    assert "g.x_column=my%20col" in result.url
    assert "g.y_column=my_col" in result.url


def test_datasette_overlong_sql_refused():
    with pytest.raises(ValueError, match="too long"):
        DatasetteConnector("db").publish("select " + "x" * 8000, "t", "bar", "a", "b")


def test_datasette_url_override_and_scheme(monkeypatch):
    monkeypatch.setenv("DATASETTE_URL", "https://bi.example.com:8002/")
    result = DatasetteConnector("db").publish("select 1", "t", "bar", "a", "b")
    assert result.url.startswith("https://bi.example.com:8002/db?sql=")
    monkeypatch.setenv("DATASETTE_URL", "ftp://x/y")
    with pytest.raises(ValueError, match="http"):
        DatasetteConnector("db").publish("select 1", "t", "bar", "a", "b")


def test_metabase_bad_chart_type_no_network(monkeypatch):
    monkeypatch.setenv("METABASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("METABASE_API_KEY", "k")
    with pytest.raises(ValueError, match="bar"):
        MetabaseConnector("adventureworks").publish(
            "select 1", "t", "pie", "a", "b")


def test_connector_registry():
    assert CONNECTORS["datasette"] is DatasetteConnector
    assert CONNECTORS["metabase"] is MetabaseConnector


def _record(tmp_path: Path) -> AnalysisRecord:
    record = AnalysisRecord.create("bi?", tmp_path / "runs")
    record.set_checklist(
        [{"id": "rev", "question": "Revenue?", "evidence_needed": "sum"}]
    )
    record.plan_action([{"item_id": "rev", "steps": ["sum"]}])
    qid = record.record_query(
        "SELECT region, revenue FROM t",
        {"columns": ["region", "revenue"],
         "rows": [["SP", 10.0], ["RJ", 20.0]],
         "row_count": 2, "truncated": False},
    )
    record.record_finding("rev", "big", qid, 10.0)
    return record


def _pending(record: AnalysisRecord, sqlite_path: str = "/tmp/my db.sqlite"):
    return {"record_abs": record.path, "sqlite_path": sqlite_path,
            "candidates": chart_candidates(record)}


def test_sql_is_recorded_bytes(monkeypatch):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        record = _record(Path(tmp))
        (candidate,) = chart_candidates(record)
        monkeypatch.delenv("DATASETTE_URL", raising=False)
        result = DatasetteConnector("olist").publish(
            candidate["sql"], "t", candidate["type"],
            candidate["x"], candidate["y"])
        assert parse_qs(urlparse(result.url).query)["sql"] == [
            "SELECT region, revenue FROM t"]


def test_publish_end_to_end_olist(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DATASETTE_URL", raising=False)
    record = _record(tmp_path)
    (candidate,) = chart_candidates(record)
    expected = DatasetteConnector("olist").publish(
        candidate["sql"], "revenue by region", candidate["type"],
        candidate["x"], candidate["y"])
    assert _handle_command("/publish 1", dataset="olist",
                           pending_charts=_pending(record)) is True
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == expected.url
    assert lines[1] == (
        "Start Datasette first: datasette serve --immutable "
        + shlex.quote("/tmp/my db.sqlite")
    )
    assert lines[2] == expected.note
    fresh = AnalysisRecord.load(record.path)
    assert {"connector": "datasette", "query_id": "q1", "url": expected.url} in [
        {k: e["detail"].get(k) for k in ("connector", "query_id", "url")}
        for e in fresh.audit if e["action"] == "publish"
    ]


def test_publish_out_of_range_and_usage(tmp_path, capsys):
    record = _record(tmp_path)
    pending = _pending(record)
    assert _handle_command("/publish 9", dataset="olist",
                           pending_charts=pending) is True
    assert "choose 1 to 1" in capsys.readouterr().out
    assert _handle_command("/publish abc", dataset="olist",
                           pending_charts=pending) is True
    assert "usage: /publish [n]" in capsys.readouterr().out


def test_publish_empty_and_unconfigured(tmp_path, capsys, monkeypatch):
    assert _handle_command("/publish", dataset="olist",
                           pending_charts=None) is True
    assert "No chart available" in capsys.readouterr().out
    record = _record(tmp_path)
    monkeypatch.setitem(
        REGISTRY, "plain",
        {"description": "d", "label": "P", "prompt": "p> ",
         "make_backend": lambda: None},
    )
    assert _handle_command("/publish", dataset="plain",
                           pending_charts=_pending(record)) is True
    assert "No BI connector configured for plain" in capsys.readouterr().out


def test_all_mode_prints_nothing(capsys):
    outcome = {"routing": ["olist"], "workers": [], "failed": [],
               "merged": "", "verification": {"status": "no-answers", "unmatched": []},
               "total_tokens": 0, "record_paths": [], "orchestration_path": "x"}
    _render_all_outcome(outcome)
    out = capsys.readouterr().out
    assert "Chart available" not in out
    assert "publish" not in out.lower()


def test_help_mentions_publish(capsys):
    assert _handle_command("/help", dataset="olist") is True
    assert "/publish [n]" in capsys.readouterr().out


class _FakeMetabase:
    """Local HTTP stand-in for Metabase; logs every request it receives."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.requests: list[dict] = []
        handler = self._handler()
        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                outer.requests.append({
                    "method": self.command,
                    "path": self.path.split("?")[0],
                    "api_key": self.headers.get("x-api-key"),
                    "body": self.rfile.read(length) if length else b"",
                })
                route = outer.routes.get(
                    (self.command, self.path.split("?")[0]), (404, b"nope"))
                if len(route) == 3:
                    status, body, extra = route
                else:
                    status, body = route
                    extra = {}
                raw = body.encode() if isinstance(body, str) else body
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for key, value in extra.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self._answer()

            def do_POST(self):
                self._answer()

            def log_message(self, *args):
                pass

        return Handler

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


def _metabase_env(monkeypatch, base, key="TESTKEY-abc123"):
    monkeypatch.setenv("METABASE_URL", base)
    monkeypatch.setenv("METABASE_API_KEY", key)
    return key


def test_metabase_publishes_card(monkeypatch):
    from agent.bi.metabase import MetabaseConnector

    bodies: list[dict] = []

    with _FakeMetabase({
        ("GET", "/api/database"): (200, json.dumps({"data": [
            {"id": 7, "name": "adventureworks"}, {"id": 8, "name": "other"}]})),
        ("POST", "/api/card"): (200, json.dumps({"id": 77})),
    }) as fake:
        key = _metabase_env(monkeypatch, fake.base)
        result = MetabaseConnector("adventureworks").publish(
            "SELECT 1", "revenue by region", "bar", "region", "revenue")
    assert result.url == f"{fake.base}/question/77"
    assert result.note == "saved question created"
    assert len(fake.requests) == 2
    assert all(r["api_key"] == key for r in fake.requests)
    post = next(r for r in fake.requests if r["method"] == "POST")
    bodies.append(json.loads(post["body"]))
    assert bodies[0] == {
        "name": "Agent: revenue by region",
        "dataset_query": {
            "type": "native", "database": 7, "native": {"query": "SELECT 1"}},
        "display": "bar",
        "visualization_settings": {
            "graph.dimensions": ["region"], "graph.metrics": ["revenue"]},
        "collection_id": None,
    }


def test_metabase_plain_list_shape(monkeypatch):
    from agent.bi.metabase import MetabaseConnector

    with _FakeMetabase({
        ("GET", "/api/database"): (200, json.dumps([{"id": 5, "name": "adventureworks"}])),
        ("POST", "/api/card"): (200, json.dumps({"id": 9})),
    }) as fake:
        _metabase_env(monkeypatch, fake.base)
        result = MetabaseConnector("adventureworks").publish(
            "SELECT 1", "t", "line", "x", "y")
    assert result.url == f"{fake.base}/question/9"


def test_metabase_unknown_database(monkeypatch):
    from agent.bi.metabase import MetabaseConnector

    with _FakeMetabase({
        ("GET", "/api/database"): (200, json.dumps({"data": [
            {"id": 1, "name": "sample"}, {"id": 2, "name": "other"}]})),
    }) as fake:
        _metabase_env(monkeypatch, fake.base)
        with pytest.raises(ValueError, match="available: other, sample"):
            MetabaseConnector("adventureworks").publish(
                "SELECT 1", "t", "bar", "x", "y")


def test_metabase_401_and_refused(monkeypatch):
    from agent.bi.metabase import MetabaseConnector

    with _FakeMetabase({("GET", "/api/database"): (401, "bad key")}) as fake:
        _metabase_env(monkeypatch, fake.base)
        with pytest.raises(RuntimeError, match="rejected the API key"):
            MetabaseConnector("adventureworks").publish(
                "SELECT 1", "t", "bar", "x", "y")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    dead = f"http://127.0.0.1:{sock.getsockname()[1]}"
    sock.close()
    _metabase_env(monkeypatch, dead)
    with pytest.raises(RuntimeError, match="not reachable"):
        MetabaseConnector("adventureworks").publish(
            "SELECT 1", "t", "bar", "x", "y")


def test_metabase_missing_key_makes_no_request(monkeypatch):
    import agent.bi.metabase as metabase_mod
    from agent.bi.metabase import MetabaseConnector

    with _FakeMetabase({}) as fake:
        monkeypatch.setenv("METABASE_URL", fake.base)
        monkeypatch.delenv("METABASE_API_KEY", raising=False)
        monkeypatch.setattr(metabase_mod, "_read_env_file_value",
                            lambda name: None)
        with pytest.raises(RuntimeError, match="not set"):
            MetabaseConnector("adventureworks").publish(
                "SELECT 1", "t", "bar", "x", "y")
    assert fake.requests == []


def test_metabase_never_follows_redirects(monkeypatch):
    from agent.bi.metabase import MetabaseConnector

    with _FakeMetabase({}) as second:
        routes = {("GET", "/api/database"): (
            302, b"", {"Location": f"{second.base}/api/database"})}
        with _FakeMetabase(routes) as first:
            key = _metabase_env(monkeypatch, first.base)
            with pytest.raises(RuntimeError, match="redirected"):
                MetabaseConnector("adventureworks").publish(
                    "SELECT 1", "t", "bar", "x", "y")
    assert len(first.requests) == 1
    assert first.requests[0]["api_key"] == key
    assert second.requests == []


def test_metabase_key_never_leaks(tmp_path, monkeypatch, capsys):
    from agent.bi.metabase import MetabaseConnector

    key = "SECRET-xyz-999"
    with _FakeMetabase({
        ("GET", "/api/database"): (200, json.dumps([{"id": 1, "name": "adventureworks"}])),
        ("POST", "/api/card"): (400, "card blew up: bad viz"),
    }) as fake:
        _metabase_env(monkeypatch, fake.base, key)
        with pytest.raises(RuntimeError) as excinfo:
            MetabaseConnector("adventureworks").publish(
                "SELECT 1", "t", "bar", "x", "y")
        record = _record(tmp_path)
        record.note_publish("metabase", "q1", f"{fake.base}/question/1")
        print(f"error: {excinfo.value}")
        audit_text = json.dumps(
            AnalysisRecord.load(record.path).audit[-1]["detail"])
    assert key not in str(excinfo.value)
    assert key not in capsys.readouterr().out
    assert key not in audit_text


def test_metabase_nonhttp_refused(monkeypatch):
    from agent.bi.metabase import MetabaseConnector

    monkeypatch.setenv("METABASE_URL", "ftp://x/y")
    monkeypatch.setenv("METABASE_API_KEY", "k")
    with pytest.raises(ValueError, match="http"):
        MetabaseConnector("adventureworks").publish(
            "SELECT 1", "t", "bar", "x", "y")


def test_requires_approval_flags():
    from agent.bi.datasette import DatasetteConnector
    from agent.bi.metabase import MetabaseConnector

    assert DatasetteConnector.requires_approval is False
    assert MetabaseConnector.requires_approval is True


def _metabase_routes(card_id=42):
    return {
        ("GET", "/api/database"): (200, json.dumps([{"id": 3, "name": "adventureworks"}])),
        ("POST", "/api/card"): (200, json.dumps({"id": card_id})),
    }


def _post_body(fake):
    return json.loads(next(r for r in fake.requests if r["method"] == "POST")["body"])


def _post_bodies(fake):
    return [json.loads(r["body"]) for r in fake.requests if r["method"] == "POST"]


def test_publish_line_override_on_date(tmp_path, capsys, monkeypatch):
    rows = [[f"2016-{m:02d}", float(m)] for m in (1, 2, 3)]
    with _FakeMetabase(_metabase_routes()) as fake:
        _metabase_env(monkeypatch, fake.base)
        pending = _rows_pending(tmp_path, "date?", ("month", "revenue"), rows, 1.0)
        assert pending["candidates"][0]["type"] == "line"
        monkeypatch.setattr("builtins.input", lambda _: "y")
        assert _handle_command("/publish 1 bar", dataset="adventureworks",
                               pending_charts=pending) is True
        assert _handle_command("/publish 1", dataset="adventureworks",
                               pending_charts=pending) is True
        assert [b["display"] for b in _post_bodies(fake)] == ["bar", "line"]


def test_publish_line_on_category_refused(tmp_path, capsys, monkeypatch):
    with _FakeMetabase(_metabase_routes()) as fake:
        _metabase_env(monkeypatch, fake.base)
        pending = _adventure_pending(tmp_path)
        assert _handle_command("/publish 1 line", dataset="adventureworks",
                               pending_charts=pending) is True
        out = capsys.readouterr().out
        assert "use bar instead" in out
        assert [r for r in fake.requests if r["method"] == "POST"] == []


def test_publish_junk_token_usage(tmp_path, capsys):
    pending = _adventure_pending(tmp_path)
    assert _handle_command("/publish 1 pie", dataset="adventureworks",
                           pending_charts=pending) is True
    assert "usage: /publish [n] [bar|line]" in capsys.readouterr().out


def test_publish_year_like_x_has_no_grouping(tmp_path, capsys, monkeypatch):
    rows = [["2015", 1.0], ["2016", 2.0], ["2017", 3.0]]
    with _FakeMetabase(_metabase_routes()) as fake:
        _metabase_env(monkeypatch, fake.base)
        pending = _rows_pending(tmp_path, "yr?", ("year", "revenue"), rows, 1.0)
        monkeypatch.setattr("builtins.input", lambda _: "y")
        assert _handle_command("/publish 1", dataset="adventureworks",
                               pending_charts=pending) is True
        viz = _post_body(fake)["visualization_settings"]
        assert viz["column_settings"] == {
            '["name","year"]': {"number_separators": "."}
        }


def test_publish_plain_numeric_x_has_no_column_settings(
    tmp_path, capsys, monkeypatch
):
    rows = [[5, 1.0], [6, 2.0], [7, 3.0]]
    with _FakeMetabase(_metabase_routes()) as fake:
        _metabase_env(monkeypatch, fake.base)
        pending = _rows_pending(tmp_path, "n?", ("n", "revenue"), rows, 1.0)
        monkeypatch.setattr("builtins.input", lambda _: "y")
        assert _handle_command("/publish 1", dataset="adventureworks",
                               pending_charts=pending) is True
        assert "column_settings" not in _post_body(fake)["visualization_settings"]


def _adventure_pending(tmp_path):
    record = AnalysisRecord.create("adv?", tmp_path / "runs")
    record.set_checklist(
        [{"id": "rev", "question": "Revenue?", "evidence_needed": "sum"}]
    )
    record.plan_action([{"item_id": "rev", "steps": ["sum"]}])
    qid = record.record_query(
        "SELECT category, revenue FROM t",
        {"columns": ["category", "revenue"],
         "rows": [["a", 1.0], ["b", 2.0]],
         "row_count": 2, "truncated": False},
    )
    record.record_finding("rev", "big", qid, 1.0)
    return {"record_abs": record.path, "sqlite_path": None,
            "candidates": chart_candidates(record)}


def _rows_pending(tmp_path, name, columns, rows, value):
    record = AnalysisRecord.create(name, tmp_path / "runs")
    record.set_checklist(
        [{"id": "rev", "question": "Revenue?", "evidence_needed": "sum"}]
    )
    record.plan_action([{"item_id": "rev", "steps": ["sum"]}])
    qid = record.record_query(
        f"SELECT {columns[0]}, {columns[1]} FROM t",
        {"columns": list(columns), "rows": [list(r) for r in rows],
         "row_count": len(rows), "truncated": False},
    )
    record.record_finding("rev", "big", qid, value)
    return {"record_abs": record.path, "sqlite_path": None,
            "candidates": chart_candidates(record)}


def test_publish_console_y_and_n(tmp_path, capsys, monkeypatch):
    with _FakeMetabase({
        ("GET", "/api/database"): (200, json.dumps([{"id": 3, "name": "adventureworks"}])),
        ("POST", "/api/card"): (200, json.dumps({"id": 42})),
    }) as fake:
        key = _metabase_env(monkeypatch, fake.base)
        pending = _adventure_pending(tmp_path)
        monkeypatch.setattr("builtins.input", lambda _: "Y ")
        assert _handle_command("/publish 1", dataset="adventureworks",
                               pending_charts=pending) is True
        out = capsys.readouterr().out
        assert f"{fake.base}/question/42" in out
        assert "saved question created" in out
        assert len([r for r in fake.requests if r["method"] == "POST"]) == 1
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert _handle_command("/publish 1", dataset="adventureworks",
                               pending_charts=pending) is True
        out = capsys.readouterr().out
        assert "Not published." in out
        assert len([r for r in fake.requests if r["method"] == "POST"]) == 1
    assert all(r["api_key"] == key for r in fake.requests)
