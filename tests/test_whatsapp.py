"""WhatsApp front end: webhook, sessions, clarifications (no real network).

Most tests drive handle_get/handle_post/dispatch directly with a fake
client and stub answers, so they run unprivileged. The two tests that
bind loopback sockets skip when binding is unavailable and run under
one escalated invocation (same precedent as the live-Postgres tests).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib import request as urlrequest

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.backend import SqliteBackend  # noqa: E402
from agent.whatsapp.client import WhatsAppClient, WhatsAppError, split_body  # noqa: E402
from agent.whatsapp.config import (  # noqa: E402
    ConfigError,
    WhatsAppConfig,
    load_config,
)
from agent.whatsapp.server import WhatsAppServer, format_turn_result  # noqa: E402

SENDER = "15551234567"
OTHER = "15557654321"
SECRET = "shh-secret"
OPTIONS = [
    {"label": "Alpha", "query_id": "q1", "evidence": ["a"], "explanation": "first"},
    {"label": "Beta", "query_id": "q1", "evidence": ["b"], "explanation": "second"},
]


def _config(**over: Any) -> WhatsAppConfig:
    base: dict[str, Any] = {
        "token": "tok-abc",
        "phone_number_id": "pid-1",
        "app_secret": SECRET,
        "verify_token": "vt-123",
        "allowed_numbers": frozenset({SENDER}),
        "api_version": "v22.0",
        "dataset": "olist",
    }
    base.update(over)
    return WhatsAppConfig(**base)


class _FakeClient:
    def __init__(self) -> None:
        self.sends: list[tuple[str, str]] = []
        self.images: list[tuple[str, str, str]] = []
        self.lock = threading.Lock()
        self.fail: Exception | None = None

    def send_text(self, to: str, body: str) -> None:
        if self.fail is not None:
            raise self.fail
        with self.lock:
            self.sends.append((to, body))

    def send_image(self, to: str, path: str, caption: str = "") -> None:
        if self.fail is not None:
            raise self.fail
        with self.lock:
            self.images.append((to, path, caption))

    def bodies(self) -> list[str]:
        with self.lock:
            return [body for _to, body in self.sends]


def _answer(text: str, calls: list | None = None):
    def fn(history: list, question: str, ask_user: Any) -> str:
        if calls is not None:
            calls.append(question)
        return text

    return fn


def _payload(
    sender: str = SENDER,
    body: str = "hi",
    mid: str = "wamid.1",
    mtype: str = "text",
) -> dict:
    message: dict[str, Any] = {"from": sender, "id": mid, "type": mtype}
    if mtype == "text":
        message["text"] = {"body": body}
    return {"entry": [{"changes": [{"value": {"messages": [message]}}]}]}


def _sign(raw: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


def _post(
    server: WhatsAppServer,
    payload: Any,
    style: str = "good",
    secret: str = SECRET,
) -> int:
    raw = json.dumps(payload).encode()
    if style == "good":
        sig = _sign(raw, secret)
    elif style == "noprefix":
        sig = _sign(raw, secret).removeprefix("sha256=")
    elif style == "wrong":
        sig = "sha256=" + "0" * 64
    else:
        sig = None
    return server.handle_post(raw, sig)


def _wait_until(fn, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return True
        time.sleep(0.05)
    return False


def _server(answer=None, timeout: float = 300.0, factory=None,
            datasets=None, default_dataset: str | None = "olist",
            **over: Any):
    client = _FakeClient()
    server = WhatsAppServer(
        _config(**over),
        client,
        lambda session: answer or _answer("ok"),
        clarify_timeout=timeout,
        backend_factory=factory,
        datasets=datasets,
        default_dataset=default_dataset,
    )
    return server, client


def test_handshake():
    server, _client = _server()
    status, _ctype, body = server.handle_get(
        "hub.mode=subscribe&hub.verify_token=vt-123&hub.challenge=abc"
    )
    assert (status, body) == (200, "abc")
    assert server.handle_get(
        "hub.mode=subscribe&hub.verify_token=nope&hub.challenge=abc"
    )[0] == 403
    assert server.handle_get("hub.mode=subscribe")[0] == 403
    assert server.dispatch_get("/nope", "")[0] == 404


def test_signed_post_processed():
    server, client = _server()
    calls: list = []
    server.make_answer = lambda session: _answer("done", calls)
    assert _post(server, _payload(body="how many?")) == 200
    assert _wait_until(lambda: client.bodies() == ["done"])
    assert calls == ["how many?"]


def test_signature_variants_rejected():
    for style in ("noprefix", "wrong", "missing"):
        server, client = _server()
        calls: list = []
        server.make_answer = lambda session: _answer("done", calls)
        assert _post(server, _payload(), style=style) == 403
        assert calls == [] and client.bodies() == []
    server, _client = _server()
    assert server.handle_post(b"{}", None) == 403


def test_non_allowlisted_sender_ignored():
    server, client = _server()
    calls: list = []
    server.make_answer = lambda session: _answer("done", calls)
    assert _post(server, _payload(sender="19998887777")) == 200
    assert calls == [] and client.bodies() == []


def test_duplicate_message_processed_once():
    server, client = _server()
    calls: list = []
    server.make_answer = lambda session: _answer("done", calls)
    assert _post(server, _payload(mid="wamid.9")) == 200
    assert _post(server, _payload(mid="wamid.9")) == 200
    assert _wait_until(lambda: len(client.bodies()) == 1)
    time.sleep(0.3)
    assert calls == ["hi"] and len(client.bodies()) == 1


def test_non_text_and_statuses_ignored():
    server, client = _server()
    calls: list = []
    server.make_answer = lambda session: _answer("done", calls)
    assert _post(server, _payload(mtype="image")) == 200
    payload = {"entry": [{"changes": [{"value": {"statuses": [{"id": "x"}]}}]}]}
    assert _post(server, payload) == 200
    assert _post(server, {"entry": []}) == 200
    assert calls == [] and client.bodies() == []


def test_command_message_rejected():
    server, client = _server()
    calls: list = []
    server.make_answer = lambda session: _answer("done", calls)
    assert _post(server, _payload(body="/chart 1")) == 200
    assert _wait_until(
        lambda: client.bodies() == ["Commands are not available on WhatsApp yet."]
    )
    assert calls == []


def test_split_body():
    text = "".join(f"line {i}\n" for i in range(1000))
    assert len(text) > 4000
    chunks = split_body(text)
    assert len(chunks) > 1
    assert all(len(c) <= 4000 for c in chunks)
    assert "".join(chunks) == text
    assert split_body("x" * 5000) == ["x" * 4000, "x" * 1000]


def _ask_stub(seen: list, reply: str = "picked"):
    def fn(history: list, question: str, ask_user: Any) -> str:
        result = ask_user("Pick one?", OPTIONS)
        seen.append(result)
        return f"{reply}:{json.dumps(result, sort_keys=True)}"

    return fn


def test_clarification_round_trip():
    server, client = _server()
    seen: list = []
    server.make_answer = lambda session: _ask_stub(seen)
    assert _post(server, _payload(body="which?", mid="wamid.a")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    prompt = client.bodies()[0]
    assert "1." in prompt and "2." in prompt and "0" in prompt
    assert _post(server, _payload(body="2", mid="wamid.b")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2)
    assert seen == [{"choice": "Beta"}]
    assert client.bodies()[-1].endswith('"choice": "Beta"}')


def test_zero_then_typed():
    server, client = _server()
    seen: list = []
    server.make_answer = lambda session: _ask_stub(seen)
    assert _post(server, _payload(body="q", mid="wamid.a")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert _post(server, _payload(body="0", mid="wamid.b")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2)
    assert client.bodies()[1] == "type your answer:"
    assert _post(server, _payload(body="hello typed", mid="wamid.c")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 3)
    assert seen == [{"choice": "hello typed"}]


def test_question_about_options():
    server, client = _server()
    seen: list = []
    server.make_answer = lambda session: _ask_stub(seen)
    assert _post(server, _payload(body="q", mid="wamid.a")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert _post(server, _payload(body="why these?", mid="wamid.b")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2)
    assert seen == [{"user_question": "why these?"}]


def test_clarification_timeout_abandons_turn():
    server, client = _server(timeout=0.2)

    def fn(history: list, question: str, ask_user: Any) -> str:
        history.append({"role": "note"})
        return ask_user("Pick?", OPTIONS)

    server.make_answer = lambda session: fn
    assert _post(server, _payload(body="q", mid="wamid.a")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2, timeout=10)
    assert client.bodies()[1] == (
        "I did not get an answer, so I stopped this question."
    )
    time.sleep(0.4)
    assert len(client.bodies()) == 2
    assert server.sessions[SENDER]["history"] == []


def test_overtake_next_message_first():
    server, _client = _server()
    session = {"history": [], "overtake": deque([("t", "m0", "old")])}
    from queue import Queue

    session["inbox"] = Queue()
    session["inbox"].put(("t2", "m1", "new"))
    assert server._next_message(session) == ("t", "m0", "old")
    assert server._next_message(session) == ("t2", "m1", "new")


def test_missing_config(monkeypatch):
    import agent.whatsapp.config as config_mod

    monkeypatch.setattr(config_mod, "_read_env_file_value", lambda name: "")
    with pytest.raises(ConfigError, match="WHATSAPP_TOKEN"):
        load_config({})
    env = {
        "WHATSAPP_TOKEN": "t",
        "WHATSAPP_PHONE_NUMBER_ID": "p",
        "WHATSAPP_APP_SECRET": "s",
        "WHATSAPP_VERIFY_TOKEN": "v",
        "WHATSAPP_ALLOWED_NUMBERS": "abc",
        "WHATSAPP_API_VERSION": "v22.0",
    }
    with pytest.raises(ConfigError, match="digits only"):
        load_config(env)
    env["WHATSAPP_ALLOWED_NUMBERS"] = "1555"
    with pytest.raises(ConfigError, match="unknown WHATSAPP_DATASET"):
        load_config({**env, "WHATSAPP_DATASET": "nope"})
    with pytest.raises(ConfigError, match="not supported"):
        load_config({**env, "WHATSAPP_DATASET": "all"})
    config = load_config(env)
    assert config.dataset == "ask"
    assert config.allowed_numbers == frozenset({"1555"})


def test_no_blocking_input_reachable(monkeypatch):
    def raising_input(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("input() reached from WhatsApp path")

    monkeypatch.setattr("builtins.input", raising_input)
    server, client = _server()
    seen: list = []
    options = [
        {**OPTIONS[0], "definition_key": "revenue", "last_used": True},
        OPTIONS[1],
    ]

    def fn(history: list, question: str, ask_user: Any) -> str:
        result = ask_user("Pick?", options)
        seen.append(result)
        return "final"

    server.make_answer = lambda session: fn
    assert _post(server, _payload(body="q", mid="wamid.a")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert _post(server, _payload(body="1", mid="wamid.b")) == 200
    assert _wait_until(lambda: client.bodies()[-1] == "final")
    assert seen == [{"choice": "Alpha"}]
    assert not any("Save this" in body for body in client.bodies())
    for name in ("server.py", "__main__.py", "client.py", "config.py"):
        source = (
            Path(__file__).resolve().parent.parent / "agent" / "whatsapp" / name
        ).read_text()
        assert "input(" not in source
        assert "offer_definition" not in source


def test_secrets_never_logged(tmp_path, capsys, caplog):
    import agent.whatsapp.client as client_mod

    caplog.set_level(logging.INFO, logger="agent.whatsapp.server")
    server, client = _server()
    server.make_answer = lambda session: _answer("ok")
    raw = json.dumps(_payload(body="hi")).encode()
    assert server.handle_post(raw, _sign(raw)) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert server.handle_post(raw, "sha256=" + "0" * 64) == 403
    client.fail = client_mod.WhatsAppError("boom")
    assert _post(server, _payload(body="again", mid="wamid.z")) == 200
    assert _wait_until(
        lambda: "boom" in caplog.text or client.bodies() == [], timeout=10
    )
    assert "tok-abc" not in capsys.readouterr().out
    assert "tok-abc" not in capsys.readouterr().err
    for secret in ("tok-abc", SECRET, "vt-123"):
        assert secret not in caplog.text


def test_oversize_body_rejected():
    server, client = _server()
    calls: list = []
    server.make_answer = lambda session: _answer("done", calls)
    raw = b"x" * (64 * 1024 + 1)
    status, _ctype, _body = server.dispatch_post("/webhook", len(raw), raw, _sign(raw))
    assert status == 413
    assert calls == []


def test_startup_line_and_logging():
    import logging as std_logging

    from agent.whatsapp.__main__ import (
        HOST,
        PORT,
        _LOG_FORMAT,
        configure_logging,
        startup_line,
    )

    assert HOST == "127.0.0.1" and PORT == 8080
    assert "asctime" in _LOG_FORMAT and "levelname" in _LOG_FORMAT
    config = _config()
    assert startup_line(config) == (
        "WhatsApp webhook ready on 127.0.0.1:8080, "
        "dataset=olist, 1 allowed number(s)"
    )
    root = std_logging.getLogger()
    saved = root.handlers[:]
    saved_level = root.level
    try:
        configure_logging()
        installed = [
            h for h in root.handlers
            if isinstance(h, std_logging.StreamHandler)
            and getattr(h.formatter, "_fmt", "") == _LOG_FORMAT
        ]
        assert installed
        assert root.level <= std_logging.INFO
    finally:
        root.handlers = saved
        root.level = saved_level


def test_accepted_message_logs_short_id_only(tmp_path, caplog):
    import logging as std_logging

    caplog.set_level(std_logging.INFO, logger="agent.whatsapp.server")
    server, client = _server()
    server.make_answer = lambda session: _answer("ok")
    assert _post(server, _payload(body="hello there")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    accepted = [
        r for r in caplog.records if "accepted" in r.getMessage()
    ]
    assert len(accepted) == 1
    assert accepted[0].getMessage() == "message from \u20264567 accepted"
    assert SENDER not in accepted[0].getMessage()
    assert "hello there" not in caplog.text
    for secret in ("tok-abc", SECRET, "vt-123"):
        assert secret not in caplog.text


def test_handler_timeout_default():
    from agent.whatsapp.server import _WebhookHandler

    assert _WebhookHandler.timeout == 10


def test_stalled_connection_closed(monkeypatch):
    import socket as socket_mod

    from agent.whatsapp.server import _WebhookHandler

    try:
        probe = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    except OSError:
        pytest.skip("loopback unavailable")
    probe.server_close()
    monkeypatch.setattr(_WebhookHandler, "timeout", 0.5)
    server, _client = _server()
    thread = threading.Thread(
        target=server.serve, kwargs={"port": 0}, daemon=True
    )
    thread.start()
    try:
        assert _wait_until(lambda: server.httpd is not None)
        port = server.httpd.server_port
        sock = socket_mod.create_connection(("127.0.0.1", port), timeout=5)
        sock.settimeout(5)
        try:
            assert sock.recv(1) == b""
        finally:
            sock.close()
    finally:
        if server.httpd is not None:
            server.httpd.shutdown()


def test_format_turn_result():
    submission = {"answer": "x"}
    fake = type("R", (), {"text": "A\nChart: workspace/charts/a.png\nChart note: n\nB", "verification": "verification: 1 matched", "submission": submission})()
    assert format_turn_result(fake) == "A\nChart note: n\nB\nverification: 1 matched"
    assert format_turn_result(type("R", (), {"submission": None, "text": "t", "verification": "v"})()) is None
    assert format_turn_result(None) is None


class _FakeGraph:
    """Loopback stand-in for the Graph API; skipped when bind is denied."""

    def __init__(self) -> None:
        self.posts: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                outer.posts.append(
                    {
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "body": json.loads(raw),
                    }
                )
                payload = b'{"messages": [{"id": "wamid.x"}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                pass

        try:
            self.server = HTTPServer(("127.0.0.1", 0), Handler)
        except OSError:
            pytest.skip("loopback unavailable")
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


def test_graph_client_posts(monkeypatch):
    import agent.whatsapp.client as client_mod

    fake = _FakeGraph()
    try:
        monkeypatch.setattr(client_mod, "_GRAPH_ROOT", fake.base)
        client = WhatsAppClient("v22.0", "pid-9", "tok-xyz")
        long_body = "".join(f"line {i}\n" for i in range(600))
        client.send_text("15550001111", long_body)
    finally:
        fake.close()
    assert len(fake.posts) > 1
    assert all(len(p["body"]["text"]["body"]) <= 4000 for p in fake.posts)
    assert "".join(p["body"]["text"]["body"] for p in fake.posts) == long_body
    assert all(p["auth"] == "Bearer tok-xyz" for p in fake.posts)
    assert all(p["path"] == "/v22.0/pid-9/messages" for p in fake.posts)


def _sqlite_db(tmp_path):
    import sqlite3

    path = tmp_path / "wa.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT);")
        conn.executemany(
            "INSERT INTO items (name) VALUES (?)", [("a",), ("b",)]
        )
        conn.commit()
    finally:
        conn.close()
    return path


def _db_answer(session):
    """Answer through the session's own worker-thread backend."""
    backend = session["backend"]

    def answer(history, question, ask_user):
        tables = backend.list_tables()
        rows = backend.run_sql("SELECT name FROM items ORDER BY id;")["rows"]
        return f"{tables}:{rows}"

    return answer


def test_real_sqlite_backend_in_worker_thread(tmp_path):
    import threading

    from agent.backend import SqliteBackend

    path = _sqlite_db(tmp_path)
    factory_threads: list[int] = []

    def factory(_name=None):
        factory_threads.append(threading.get_ident())
        return SqliteBackend(path)

    server, client = _server(factory=factory, timeout=30)
    server.make_answer = lambda session: _db_answer(session)
    assert _post(server, _payload(body="q", mid="wamid.r1")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1, timeout=10)
    assert client.bodies() == ["['items']:[['a'], ['b']]"]
    assert factory_threads and factory_threads[0] != threading.get_ident()


def test_two_senders_query_simultaneously(tmp_path):
    import threading

    from agent.backend import SqliteBackend

    path = _sqlite_db(tmp_path)
    barrier = threading.Barrier(2, timeout=10)

    def factory(_name=None):
        return SqliteBackend(path)

    def make_answer(session):
        backend = session["backend"]

        def answer(history, question, ask_user):
            barrier.wait()
            rows = backend.run_sql("SELECT name FROM items ORDER BY id;")["rows"]
            return f"{rows}"

        return answer

    server, client = _server(
        factory=factory, timeout=30, allowed_numbers=frozenset({SENDER, OTHER})
    )
    server.make_answer = make_answer
    assert _post(server, _payload(sender=SENDER, body="q", mid="wamid.s1")) == 200
    assert _post(server, _payload(sender=OTHER, body="q", mid="wamid.o1")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2, timeout=10)
    assert sorted(client.bodies()) == ["[['a'], ['b']]", "[['a'], ['b']]"]
    by_sender = {}
    with client.lock:
        for to, body in client.sends:
            by_sender[to] = body
    assert by_sender == {SENDER: "[['a'], ['b']]", OTHER: "[['a'], ['b']]"}
    server.close_sessions()


def test_factory_failure_then_success(tmp_path):
    from agent.backend import SqliteBackend

    path = _sqlite_db(tmp_path)
    attempts: list[int] = []

    def factory(_name=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("db down")
        return SqliteBackend(path)

    server, client = _server(factory=factory, timeout=30)
    server.make_answer = lambda session: _db_answer(session)
    assert _post(server, _payload(body="q", mid="wamid.f1")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1, timeout=10)
    assert "Sorry" in client.bodies()[0]
    assert _post(server, _payload(body="q", mid="wamid.f2")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2, timeout=10)
    assert client.bodies()[1] == "['items']:[['a'], ['b']]"
    assert len(attempts) == 2
    server.close_sessions()


class _TrackedSqlite(SqliteBackend):
    """Real read-only backend that records close calls.

    Connects lazily so the sqlite connection is created in whichever thread
    first uses it (the worker thread), never on the creating thread.
    """

    def __init__(self, path, closed):
        self._lazy_path = str(path)
        self._closed_flag = closed

    def _real(self):
        real = self.__dict__.get("_real_backend")
        if real is None:
            real = SqliteBackend(self._lazy_path)
            self.__dict__["_real_backend"] = real
        return real

    def __getattr__(self, name):
        return getattr(self._real(), name)

    def close(self):
        real = self.__dict__.get("_real_backend")
        if real is not None:
            real.close()
        self._closed_flag.append(True)


def _picker_setup(tmp_path):
    """Two datasets: real temp SQLite under olist, stub sentinel under zz."""
    from types import SimpleNamespace

    closed: list = []
    real = _TrackedSqlite(_sqlite_db(tmp_path), closed)
    sentinel = SimpleNamespace(close=lambda: closed.append("zz"))
    made: dict[str, int] = {}

    def factory(name: str):
        made[name] = made.get(name, 0) + 1
        return {"olist": real, "zz": sentinel}[name]

    datasets = {
        "olist": {"label": "Olist", "description": "Brazilian orders."},
        "zz": {"label": "Zed", "description": "Stub data."},
    }
    return datasets, factory, made, closed, real


def _ds_answer(seen: list):
    def make_answer(session):
        def answer(history, question, ask_user):
            seen.append(
                {
                    "dataset": session.get("dataset"),
                    "backend": session.get("backend"),
                    "hist_len": len(history),
                    "question": question,
                }
            )
            history.append(question)
            backend = session["backend"]
            tables = backend.list_tables() if hasattr(backend, "list_tables") else []
            return f"ok:{session.get('dataset')}:{tables}"

        return answer

    return make_answer


def test_picker_picks_dataset_and_runs_held_question(tmp_path):
    import threading

    datasets, factory, made, closed, real = _picker_setup(tmp_path)
    seen: list = []
    server, client = _server(datasets=datasets, default_dataset=None, factory=factory)
    server.make_answer = _ds_answer(seen)
    assert _post(server, _payload(body="rev?", mid="wamid.p1")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert "1. Olist" in client.bodies()[0] and "2. Zed" in client.bodies()[0]
    assert _post(server, _payload(body="2", mid="wamid.p2")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2)
    assert seen[0]["dataset"] == "zz"
    assert made == {"zz": 1}
    assert client.bodies()[1].startswith("ok:zz:")


def test_picker_invalid_then_valid(tmp_path):
    datasets, factory, made, closed, real = _picker_setup(tmp_path)
    seen: list = []
    server, client = _server(datasets=datasets, default_dataset=None, factory=factory)
    server.make_answer = _ds_answer(seen)
    assert _post(server, _payload(body="q", mid="wamid.i1")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert _post(server, _payload(body="xyz", mid="wamid.i2")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2)
    assert client.bodies()[1].startswith("Which dataset?")
    assert _post(server, _payload(body="1", mid="wamid.i3")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 3)
    assert seen[0]["dataset"] == "olist"
    assert seen[0]["backend"] is real
    assert "items" in client.bodies()[2]
    assert made == {"olist": 1}


def test_picker_factory_runs_off_main_thread(tmp_path):
    import threading

    factory_threads: list[int] = []

    def factory(name: str):
        from agent.backend import SqliteBackend

        factory_threads.append(threading.get_ident())
        return SqliteBackend(_sqlite_db(tmp_path))

    datasets = {
        "olist": {"label": "Olist", "description": "Brazilian orders."},
        "zz": {"label": "Zed", "description": "Stub data."},
    }
    server, client = _server(datasets=datasets, default_dataset=None, factory=factory)
    server.make_answer = _ds_answer([])
    assert _post(server, _payload(body="q", mid="wamid.t1")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert _post(server, _payload(body="1", mid="wamid.t2")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2)
    assert factory_threads and factory_threads[0] != threading.get_ident()


def test_picker_invalid_twice_stops(tmp_path):
    datasets, factory, made, closed, real = _picker_setup(tmp_path)
    seen: list = []
    server, client = _server(datasets=datasets, default_dataset=None, factory=factory)
    server.make_answer = _ds_answer(seen)
    assert _post(server, _payload(body="q", mid="wamid.s1")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert _post(server, _payload(body="nope", mid="wamid.s2")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2)
    assert _post(server, _payload(body="nah", mid="wamid.s3")) == 200
    assert _wait_until(
        lambda: any("couldn't tell which dataset" in b for b in client.bodies())
    )
    assert seen == [] and made == {}


def test_second_question_does_not_ask_again(tmp_path):
    datasets, factory, made, closed, real = _picker_setup(tmp_path)
    seen: list = []
    server, client = _server(datasets=datasets, default_dataset=None, factory=factory)
    server.make_answer = _ds_answer(seen)
    assert _post(server, _payload(body="q1", mid="wamid.n1")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert _post(server, _payload(body="1", mid="wamid.n2")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2)
    assert _post(server, _payload(body="q2", mid="wamid.n3")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 3)
    assert [s["question"] for s in seen] == ["q1", "q2"]
    assert sum(1 for b in client.bodies() if b.startswith("Which dataset?")) == 1


def test_dataset_command_repicks_and_closes(tmp_path):
    datasets, factory, made, closed, real = _picker_setup(tmp_path)
    seen: list = []
    server, client = _server(datasets=datasets, default_dataset=None, factory=factory)
    server.make_answer = _ds_answer(seen)
    assert _post(server, _payload(body="q1", mid="wamid.d1")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert _post(server, _payload(body="1", mid="wamid.d2")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2)
    assert _post(server, _payload(body="/dataset", mid="wamid.d3")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 3)
    assert client.bodies()[2].startswith("Which dataset?")
    assert closed == [True]
    assert _post(server, _payload(body="2", mid="wamid.d4")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 4)
    assert client.bodies()[3] == "Using Zed. Ask your question. Send /dataset to switch."
    assert _post(server, _payload(body="q2", mid="wamid.d5")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 5)
    assert [s["dataset"] for s in seen] == ["olist", "zz"]
    assert [s["hist_len"] for s in seen] == [0, 0]
    assert made == {"olist": 1, "zz": 1}


def test_dataset_pick_timeout_starts_nothing(tmp_path):
    datasets, factory, made, closed, real = _picker_setup(tmp_path)
    seen: list = []
    server, client = _server(
        datasets=datasets, default_dataset=None, factory=factory, timeout=0.2
    )
    server.make_answer = _ds_answer(seen)
    assert _post(server, _payload(body="/dataset", mid="wamid.e1")) == 200
    assert _wait_until(
        lambda: any("nothing was started" in b for b in client.bodies()), timeout=10
    )
    assert seen == []
    assert _post(server, _payload(body="1", mid="wamid.e2")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 3)
    assert client.bodies()[-1].startswith("Which dataset?")
    assert _post(server, _payload(body="1", mid="wamid.e3")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 4)
    assert seen[0]["dataset"] == "olist"


def test_fixed_dataset_skips_picker():
    server, client = _server()
    server.make_answer = lambda session: (lambda h, q, a: "fixed-ok")
    assert _post(server, _payload(body="q", mid="wamid.f1")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert client.bodies() == ["fixed-ok"]


def test_config_dataset_defaults_to_ask(monkeypatch):
    import agent.whatsapp.config as config_mod

    monkeypatch.setattr(config_mod, "_read_env_file_value", lambda name: "")
    env = {
        "WHATSAPP_TOKEN": "t",
        "WHATSAPP_PHONE_NUMBER_ID": "p",
        "WHATSAPP_APP_SECRET": "s",
        "WHATSAPP_VERIFY_TOKEN": "v",
        "WHATSAPP_ALLOWED_NUMBERS": "1555",
        "WHATSAPP_API_VERSION": "v22.0",
    }
    assert load_config(env).dataset == "ask"


def test_full_http_round_trip(monkeypatch):
    import urllib.error as urlerror

    try:
        probe = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    except OSError:
        pytest.skip("loopback unavailable")
    probe.server_close()
    server, client = _server()
    calls: list = []
    server.make_answer = lambda session: _answer("hello back", calls)
    thread = threading.Thread(
        target=server.serve, kwargs={"port": 0}, daemon=True
    )
    thread.start()
    assert _wait_until(lambda: server.httpd is not None)
    port = server.httpd.server_port
    base = f"http://127.0.0.1:{port}"
    try:
        with urlrequest.urlopen(base + "/webhook?hub.mode=subscribe"
                                "&hub.verify_token=vt-123&hub.challenge=zz") as resp:
            assert (resp.status, resp.read()) == (200, b"zz")
        raw = json.dumps(_payload(body="ping")).encode()
        req = urlrequest.Request(
            base + "/webhook", data=raw, method="POST",
            headers={"Content-Type": "application/json",
                     "X-Hub-Signature-256": _sign(raw)},
        )
        with urlrequest.urlopen(req) as resp:
            assert resp.status == 200
    except urlerror.URLError:
        pytest.skip("loopback unavailable")
    finally:
        server.httpd.shutdown()
    assert _wait_until(lambda: client.bodies() == ["hello back"])
    assert calls == ["ping"]


class _FakeOpener:
    """Injectable transport capturing requests and replaying responses."""

    def __init__(self, replies: list) -> None:
        self.requests: list = []
        self.replies = list(replies)

    def open(self, request, timeout=None):
        self.requests.append(request)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100


def _chart_file(monkeypatch, tmp_path, name="r1-1.png", data=_PNG):
    """A real PNG under a fake charts dir; returns the absolute path str."""
    import agent.whatsapp.client as client_mod

    root = Path(tmp_path).resolve()
    monkeypatch.setattr(client_mod, "_CHARTS_DIR", root)
    path = root / name
    path.write_bytes(data)
    return str(path)


def test_send_image_upload_then_message(monkeypatch, tmp_path):
    import agent.whatsapp.client as client_mod

    name = _chart_file(monkeypatch, tmp_path)
    opener = _FakeOpener([b'{"id": "mid-abc123"}', b"{}"])
    client = WhatsAppClient("v22.0", "pid-9", "tok-secret", opener=opener)
    client.send_image("15550001111", name, "orders by month")
    assert len(opener.requests) == 2
    upload, message = opener.requests
    assert upload.full_url.endswith("/v22.0/pid-9/media")
    assert upload.get_header("Authorization") == "Bearer tok-secret"
    assert upload.get_header("Content-type").startswith("multipart/form-data")
    raw = upload.data
    assert b'name="messaging_product"' in raw and b"whatsapp" in raw
    assert b'name="type"' in raw and b"image/png" in raw
    assert b'name="file"' in raw and b"Content-Type: image/png" in raw
    assert _PNG in raw
    assert b"tok-secret" not in raw
    assert message.full_url.endswith("/v22.0/pid-9/messages")
    payload = json.loads(message.data.decode())
    assert payload["image"] == {"id": "mid-abc123", "caption": "orders by month"}
    assert payload["to"] == "15550001111"


def test_send_image_caption_truncated(monkeypatch, tmp_path):
    import agent.whatsapp.client as client_mod

    name = _chart_file(monkeypatch, tmp_path)
    opener = _FakeOpener([b'{"id": "m1"}', b"{}"])
    client = WhatsAppClient("v22.0", "pid-9", "tok-secret", opener=opener)
    client.send_image("15550001111", name, "t" * 2000)
    payload = json.loads(opener.requests[1].data.decode())
    assert len(payload["image"]["caption"]) == 1024


def test_send_image_refuses_outside_charts_dir(monkeypatch, tmp_path):
    import agent.whatsapp.client as client_mod

    monkeypatch.setattr(client_mod, "_CHARTS_DIR", tmp_path / "charts")
    opener = _FakeOpener([])
    client = WhatsAppClient("v22.0", "pid-9", "tok-secret", opener=opener)
    with pytest.raises(ValueError):
        client.send_image("15550001111", "/etc/passwd")
    assert opener.requests == []


def test_send_image_refuses_symlink_escape(monkeypatch, tmp_path):
    import agent.whatsapp.client as client_mod

    charts = Path(tmp_path).resolve() / "charts"
    charts.mkdir()
    monkeypatch.setattr(client_mod, "_CHARTS_DIR", charts)
    outside = Path(tmp_path).resolve() / "evil.png"
    outside.write_bytes(_PNG)
    link = charts / "link.png"
    link.symlink_to(outside)
    opener = _FakeOpener([])
    client = WhatsAppClient("v22.0", "pid-9", "tok-secret", opener=opener)
    with pytest.raises(ValueError):
        client.send_image("15550001111", str(link))
    assert opener.requests == []


def test_send_image_refuses_non_png(monkeypatch, tmp_path):
    name = _chart_file(monkeypatch, tmp_path, name="r1-1.txt", data=b"nope")
    opener = _FakeOpener([])
    client = WhatsAppClient("v22.0", "pid-9", "tok-secret", opener=opener)
    with pytest.raises(ValueError):
        client.send_image("15550001111", name)
    assert opener.requests == []


def test_send_image_refuses_oversize(monkeypatch, tmp_path):
    import agent.whatsapp.client as client_mod

    name = _chart_file(monkeypatch, tmp_path)
    monkeypatch.setattr(client_mod, "_MAX_IMAGE_BYTES", 10)
    opener = _FakeOpener([])
    client = WhatsAppClient("v22.0", "pid-9", "tok-secret", opener=opener)
    with pytest.raises(ValueError, match="5 MB"):
        client.send_image("15550001111", name)
    assert opener.requests == []


def test_send_image_missing_media_id(monkeypatch, tmp_path):
    name = _chart_file(monkeypatch, tmp_path)
    opener = _FakeOpener([b"{}"])
    client = WhatsAppClient("v22.0", "pid-9", "tok-secret", opener=opener)
    with pytest.raises(WhatsAppError):
        client.send_image("15550001111", name)
    assert len(opener.requests) == 1


def test_send_image_http_error_hides_token(monkeypatch, tmp_path):
    import urllib.error

    import agent.whatsapp.client as client_mod

    name = _chart_file(monkeypatch, tmp_path)
    err = urllib.error.HTTPError(
        "https://graph.facebook.com/x", 401, "Unauthorized", {}, None
    )
    opener = _FakeOpener([err])
    client = WhatsAppClient("v22.0", "pid-9", "tok-secret", opener=opener)
    with pytest.raises(WhatsAppError) as caught:
        client.send_image("15550001111", name)
    assert "tok-secret" not in str(caught.value)


def test_media_id_logged_last4_only(monkeypatch, tmp_path, caplog):
    import logging

    import agent.whatsapp.client as client_mod

    name = _chart_file(monkeypatch, tmp_path)
    opener = _FakeOpener([b'{"id": "mid-abcdef12"}', b"{}"])
    client = WhatsAppClient("v22.0", "pid-9", "tok-secret", opener=opener)
    with caplog.at_level(logging.INFO, logger="agent.whatsapp.client"):
        client.send_image("15550001111", name)
    logged = caplog.text
    assert "mid-abcdef12" not in logged
    assert "ef12" in logged
    assert "tok-secret" not in logged


def _tuple_answer(text, images):
    def make_answer(session):
        def answer(history, question, ask_user):
            return text, images

        return answer

    return make_answer


def test_tuple_reply_sends_text_then_images():
    server, client = _server()
    server.make_answer = _tuple_answer(
        "answer here", [("workspace/charts/a.png", "cap a"), ("workspace/charts/b.png", "cap b")]
    )
    assert _post(server, _payload(body="q?", mid="wamid.img1")) == 200
    assert _wait_until(lambda: client.bodies() == ["answer here"])
    assert _wait_until(lambda: len(client.images) == 2)
    assert client.images[0] == (SENDER, "workspace/charts/a.png", "cap a")


def test_tuple_reply_caps_at_three_images():
    server, client = _server()
    server.make_answer = _tuple_answer(
        "t", [(f"workspace/charts/{i}.png", f"c{i}") for i in range(5)]
    )
    assert _post(server, _payload(body="q?", mid="wamid.img2")) == 200
    assert _wait_until(lambda: len(client.images) == 3)


def test_image_failure_falls_back_and_session_survives():
    from agent.whatsapp.client import WhatsAppError

    server, client = _server()

    def bad_send_image(to, path, caption=""):
        raise WhatsAppError("nope")

    client.send_image = bad_send_image  # type: ignore[method-assign]
    server.make_answer = _tuple_answer("answer", [("workspace/charts/a.png", "c")])
    assert _post(server, _payload(body="q?", mid="wamid.img3")) == 200
    assert _wait_until(
        lambda: client.bodies()
        == ["answer", "I drew the chart but could not send the image."]
    )
    server.make_answer = lambda session: _answer("fine")
    assert _post(server, _payload(body="next?", mid="wamid.img4")) == 200
    assert _wait_until(lambda: "fine" in client.bodies())


def test_unsafe_image_skipped_quietly():
    server, client = _server()
    real_images = client.images

    def bad_send_image(to, path, caption=""):
        raise ValueError("refusing")

    client.send_image = bad_send_image  # type: ignore[method-assign]
    server.make_answer = _tuple_answer("answer", [("/etc/x.png", "c")])
    assert _post(server, _payload(body="q?", mid="wamid.img5")) == 200
    assert _wait_until(lambda: client.bodies() == ["answer"])
    assert real_images == []
    assert not any("could not send" in b for b in client.bodies())


def test_format_turn_result_keeps_chart_note():
    from agent.whatsapp.server import format_turn_result

    class Result:
        submission = {"x": 1}
        text = "Revenue 5.\nChart: workspace/charts/r-1.png\nChart note: no data for June."
        verification = "verification: 1 numbers matched, 0 unmatched"

    text = format_turn_result(Result())
    assert "workspace/charts" not in text
    assert "Chart note: no data for June." in text
    assert "verification:" in text


def test_image_caption_carries_no_path():
    from agent.whatsapp.__main__ import _image_caption

    caption = _image_caption("orders by month", "Chart note: no data for June")
    assert ".png" not in caption and "workspace" not in caption
    assert _image_caption("t", "n" * 2000) != "" and len(_image_caption("t", "n" * 2000)) <= 1024


def test_digit_first_message_still_reaches_picker(tmp_path):
    datasets, factory, made, closed, real = _picker_setup(tmp_path)
    server, client = _server(datasets=datasets, default_dataset=None, factory=factory)
    server.make_answer = _ds_answer([])
    assert _post(server, _payload(body="2", mid="wamid.dg0")) == 200
    assert _wait_until(
        lambda: any(b.startswith("Which dataset?") for b in client.bodies())
    )
    assert not any("isn't answering anything" in b for b in client.bodies())


def _ask_server(tmp_path):
    """Ask-mode server on olist(real temp SQLite)/zz(sentinel) with answers."""
    datasets, factory, made, closed, real = _picker_setup(tmp_path)
    seen: list = []
    server, client = _server(datasets=datasets, default_dataset=None, factory=factory)
    server.make_answer = _ds_answer(seen)
    return server, client, seen, made, closed


def _pick_first(server, client, seen):
    assert _post(server, _payload(body="q", mid="wamid.k0")) == 200
    assert _wait_until(
        lambda: any(b.startswith("Which dataset?") for b in client.bodies())
    )
    assert _post(server, _payload(body="1", mid="wamid.k1")) == 200
    assert _wait_until(lambda: seen and seen[0]["dataset"] == "olist")


def test_phrases_trigger_picker(tmp_path):
    server, client, seen, made, closed = _ask_server(tmp_path)
    _pick_first(server, client, seen)
    phrases = [
        "change dataset",
        "Switch Dataset",
        "  change   database? ",
        "dataset",
        "datasets",
        "switch database",
        "CHANGE DATABASE!",
    ]
    for i, phrase in enumerate(phrases):
        assert _post(server, _payload(body=phrase, mid=f"wamid.pp{i}a")) == 200
        assert _wait_until(
            lambda: sum(b.startswith("Which dataset?") for b in client.bodies())
            >= i + 2
        )
        assert _post(server, _payload(body="1", mid=f"wamid.pp{i}b")) == 200
        assert _wait_until(
            lambda: client.bodies()[-1].startswith("Using Olist.")
        )
    assert client.bodies()[-1] == (
        "Using Olist. Ask your question. Send /dataset to switch."
    )
    assert [s["question"] for s in seen] == ["q"]


def test_dataset_questions_reach_agent(tmp_path):
    server, client, seen, made, closed = _ask_server(tmp_path)
    _pick_first(server, client, seen)
    questions = [
        "which dataset has more orders?",
        "how many orders per dataset",
        "how many orders in olist?",
    ]
    for i, question in enumerate(questions):
        assert _post(server, _payload(body=question, mid=f"wamid.qq{i}")) == 200
        assert _wait_until(lambda: len(seen) >= i + 2)
    assert [s["question"] for s in seen] == ["q"] + questions
    assert sum(b.startswith("Which dataset?") for b in client.bodies()) == 1


def test_phrases_fixed_mode_reach_agent():
    server, client = _server()
    calls: list = []
    server.make_answer = lambda session: _answer("ok", calls)
    assert _post(server, _payload(body="change dataset", mid="wamid.fx1")) == 200
    assert _post(server, _payload(body="olist", mid="wamid.fx2")) == 200
    assert _wait_until(lambda: client.bodies() == ["ok", "ok"])
    assert calls == ["change dataset", "olist"]


def test_direct_switch_forms(tmp_path):
    server, client, seen, made, closed = _ask_server(tmp_path)
    _pick_first(server, client, seen)
    cases = [
        ("switch to Zed", "Using Zed. Ask your question. Send /dataset to switch."),
        ("use olist", "Using Olist. Ask your question. Send /dataset to switch."),
        ("ZED", "Using Zed. Ask your question. Send /dataset to switch."),
        ("/dataset 1", "Using Olist. Ask your question. Send /dataset to switch."),
        ("switch to 2", "Using Zed. Ask your question. Send /dataset to switch."),
        ("change to zed.", "Using Zed. Ask your question. Send /dataset to switch."),
        ("  USE   OLIST  ", "Using Olist. Ask your question. Send /dataset to switch."),
    ]
    for i, (message, confirmation) in enumerate(cases):
        assert _post(server, _payload(body=message, mid=f"wamid.sw{i}")) == 200
        assert _wait_until(lambda: client.bodies()[-1] == confirmation)
    assert sum(b.startswith("Which dataset?") for b in client.bodies()) == 1
    assert _post(server, _payload(body="q2", mid="wamid.swq")) == 200
    assert _wait_until(lambda: len(seen) == 2)
    assert [s["dataset"] for s in seen] == ["olist", "olist"]
    assert [s["hist_len"] for s in seen] == [0, 0]
    assert closed == [True]
    assert made == {"olist": 2}


def test_unknown_name_falls_back_to_picker(tmp_path):
    server, client, seen, made, closed = _ask_server(tmp_path)
    _pick_first(server, client, seen)
    assert _post(server, _payload(body="/dataset xyz", mid="wamid.ux1")) == 200
    assert _wait_until(
        lambda: sum(b.startswith("Which dataset?") for b in client.bodies()) >= 2
    )
    assert _post(server, _payload(body="2", mid="wamid.ux2")) == 200
    assert _wait_until(lambda: client.bodies()[-1].startswith("Using Zed."))
    assert _post(server, _payload(body="switch dataset to nowhere", mid="wamid.ux3")) == 200
    assert _wait_until(
        lambda: sum(b.startswith("Which dataset?") for b in client.bodies()) >= 3
    )
    assert _post(server, _payload(body="1", mid="wamid.ux4")) == 200
    assert _wait_until(lambda: client.bodies()[-1].startswith("Using Olist."))
    assert [s["question"] for s in seen] == ["q"]


def test_fixed_mode_confirmation_has_no_hint():
    server, client = _server()
    server.make_answer = lambda session: _answer("ok")
    assert _post(server, _payload(body="/dataset", mid="wamid.fh1")) == 200
    assert _wait_until(
        lambda: any(b.startswith("Which dataset?") for b in client.bodies())
    )
    assert _post(server, _payload(body="1", mid="wamid.fh2")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2)
    assert client.bodies()[1] == "Using Olist. Ask your question."


_ABOUT_PHRASES_TEST = [
    "help",
    "/help",
    "What can I ask?",
    "what data do you have",
    "what datasets are there",
    "what tables are there",
    "what tables do you have?",
    "  what is in here  ",
]


def test_about_phrases_no_model_call():
    server, client = _server()
    calls: list = []
    server.make_answer = lambda session: _answer("must not run", calls)
    for i, phrase in enumerate(_ABOUT_PHRASES_TEST):
        assert _post(server, _payload(body=phrase, mid=f"wamid.ab{i}")) == 200
    assert _wait_until(lambda: len(client.bodies()) == len(_ABOUT_PHRASES_TEST))
    assert calls == []
    for body in client.bodies():
        assert "Olist" in body and "AdventureWorks" in body
        assert "what is the top product category?" in body
        assert "what were total sales in 2024?" not in body
    assert all("Tables in" not in b for b in client.bodies())


def test_about_ask_mode_before_pick_no_picker(tmp_path):
    server, client, seen, made, closed = _ask_server(tmp_path)
    assert _post(server, _payload(body="help", mid="wamid.ah0")) == 200
    assert _wait_until(lambda: bool(client.bodies()))
    text = client.bodies()[0]
    assert text.startswith("I answer questions")
    assert "what is the top product category?" in text
    assert "how many rows are there?" in text
    assert "Tables in" not in text
    assert "Which dataset?" not in text
    assert seen == []


def test_match_dataset_command_units():
    server, _client = _server()
    match = server._match_dataset_command
    assert match("switch to adventure works") == ("switch", "adventureworks")
    assert match("change dataset to banana") == ("picker", None)
    assert match("olist") == ("switch", "olist")
    assert match("use olist") == ("switch", "olist")
    assert match("dataset 2") == ("switch", "adventureworks")
    for message in [
        "change it to a bar chart",
        "switch to a line chart",
        "change the chart to bar",
        "change currency to usd",
        "switch back to monthly",
        "how many orders in olist?",
        "which dataset has more orders?",
        "switch to nowhere",
        "use blah",
    ]:
        assert match(message) == (None, None), message


def test_chart_talk_reaches_agent(tmp_path):
    server, client, seen, made, closed = _ask_server(tmp_path)
    _pick_first(server, client, seen)
    messages = [
        "change it to a bar chart",
        "switch to a line chart",
        "change the chart to bar",
        "change currency to usd",
        "switch back to monthly",
        "how many orders in olist?",
    ]
    for i, message in enumerate(messages):
        assert _post(server, _payload(body=message, mid=f"wamid.ct{i}")) == 200
        assert _wait_until(lambda: len(seen) >= i + 2)
    assert [s["question"] for s in seen] == ["q"] + messages
    assert sum(b.startswith("Which dataset?") for b in client.bodies()) == 1
    assert _post(server, _payload(body="change dataset to banana", mid="wamid.ctb")) == 200
    assert _wait_until(
        lambda: sum(b.startswith("Which dataset?") for b in client.bodies()) >= 2
    )
    assert _post(server, _payload(body="1", mid="wamid.ctc")) == 200
    assert _wait_until(lambda: client.bodies()[-1].startswith("Using Olist."))
    assert _post(server, _payload(body="switch to Zed", mid="wamid.ctd")) == 200
    assert _wait_until(lambda: client.bodies()[-1].startswith("Using Zed."))
    assert [s["question"] for s in seen] == ["q"] + messages


def test_about_lists_tables(tmp_path):
    from agent.backend import SqliteBackend

    path = _sqlite_db(tmp_path)
    server, client = _server(factory=lambda name: SqliteBackend(path))
    server.make_answer = lambda session: _answer("must not run")
    assert _post(server, _payload(body="what tables do you have?", mid="wamid.at1")) == 200
    assert _wait_until(lambda: bool(client.bodies()))
    text = client.bodies()[0]
    assert "Tables in Olist: items" in text
    assert "what is the top product category?" in text
    assert "what were total sales in 2024?" not in text


def test_about_table_cap():
    from types import SimpleNamespace

    stub = SimpleNamespace(
        list_tables=lambda: [f"t{i}" for i in range(20)], close=lambda: None
    )
    server, client = _server(factory=lambda name: stub)
    server.make_answer = lambda session: _answer("must not run")
    assert _post(server, _payload(body="what tables are there", mid="wamid.ac1")) == 200
    assert _wait_until(lambda: bool(client.bodies()))
    text = client.bodies()[0]
    assert "t0" in text and "t14" in text
    assert "t15" not in text and "t19" not in text
    assert "...and 5 more" in text


def test_about_like_question_reaches_agent():
    server, client = _server()
    calls: list = []
    server.make_answer = lambda session: _answer("ok", calls)
    question = "what tables are in the orders data?"
    assert _post(server, _payload(body=question, mid="wamid.aq1")) == 200
    assert _wait_until(lambda: client.bodies() == ["ok"])
    assert calls == [question]


def test_about_ask_mode_after_pick(tmp_path):
    server, client, seen, made, closed = _ask_server(tmp_path)
    _pick_first(server, client, seen)
    before = len(seen)
    assert _post(server, _payload(body="help", mid="wamid.ah1")) == 200
    assert _wait_until(lambda: client.bodies()[-1].startswith("I answer questions"))
    assert len(seen) == before
    text = client.bodies()[-1]
    assert "Tables in Olist: items" in text
    assert "what is the top product category?" in text


def test_no_submission_failure_line():
    server, client = _server()
    server.make_answer = lambda session: (lambda h, q, a: (None, []))
    assert _post(server, _payload(body="q?", mid="wamid.an1")) == 200
    assert _wait_until(
        lambda: any("couldn't answer that from the data" in b for b in client.bodies())
    )
    assert all("numbers from the database" in b for b in client.bodies())


def test_exception_keeps_old_line():
    server, client = _server()

    def make_answer(session):
        def answer(history, question, ask_user):
            raise RuntimeError("boom")

        return answer

    server.make_answer = make_answer
    assert _post(server, _payload(body="q?", mid="wamid.ae1")) == 200
    assert _wait_until(
        lambda: any("couldn't finish that question" in b for b in client.bodies())
    )


def test_digit_guard_replies_without_model_turn():
    server, client = _server()
    calls: list = []
    server.make_answer = lambda session: _answer("should not run", calls)
    assert _post(server, _payload(body="42", mid="wamid.dg1")) == 200
    assert _wait_until(
        lambda: any("isn't answering anything" in b for b in client.bodies())
    )
    assert calls == []
    assert _post(server, _payload(body="real question?", mid="wamid.dg2")) == 200
    assert _wait_until(lambda: "should not run" in client.bodies())
    assert calls == ["real question?"]
