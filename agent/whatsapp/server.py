"""WhatsApp webhook server: one session per sender, work in background.

The HTTP layer only verifies, parses, and enqueues; each sender has one
worker thread that runs whole turns one at a time. Clarifications pause
the worker on a queue until that sender's next message arrives.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import queue
import string
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from agent.clarify import format_options_text, interpret_clarification_reply
from agent.datasets import REGISTRY
from agent.whatsapp.client import WhatsAppClient, WhatsAppError
from agent.whatsapp.config import WhatsAppConfig

log = logging.getLogger(__name__)

_MAX_BODY = 64 * 1024
_DEDUP_KEEP = 500
_OWN_COMMAND_REPLY = "Commands are not available on WhatsApp yet."
_TIMEOUT_LINE = "I did not get an answer, so I stopped this question."
_PICK_TIMEOUT_LINE = "I did not get a dataset choice, so nothing was started."
_PICK_STOP_LINE = "I couldn't tell which dataset you meant, so I stopped this question."
_IMAGE_FAILURE_LINE = "I drew the chart but could not send the image."
_DIGIT_REPLY = (
    "That number isn't answering anything right now. Please ask your question."
)
_DATASET_PHRASES = frozenset(
    {
        "change dataset",
        "switch dataset",
        "change database",
        "switch database",
        "dataset",
        "datasets",
    }
)
_FILLER_WORDS = frozenset({"to", "the", "dataset", "database"})
_SWITCH_VERBS = ("switch", "change", "use")
_TRIGGER_WORDS = frozenset({"switch", "change", "use", "dataset", "database", "datasets"})
_DATASET_WORDS = frozenset({"dataset", "datasets", "database"})
_ABOUT_PHRASES = frozenset(
    {
        "help",
        "what can i ask",
        "what data do you have",
        "what datasets are there",
        "what tables are there",
        "what tables do you have",
        "what is in here",
    }
)
_EXAMPLE_QUESTIONS = {
    "olist": (
        "how many orders are there?",
        "show monthly orders in 2017 as a line chart",
        "what is the top product category?",
    ),
    "adventureworks": (
        "how many orders are there?",
        "what were total sales in 2024?",
        "which product has the highest profit margin?",
    ),
}
_GENERIC_EXAMPLES = (
    "how many rows are there?",
    "show a monthly trend as a line chart",
)
_MAX_ABOUT_TABLES = 15
_NO_ANSWER_LINE = (
    "I couldn't answer that from the data. I answer questions with numbers "
    "from the database, such as counts, totals or trends."
)
_FAILURE_LINE = "Sorry \u2014 I couldn't finish that question. Please try again."


def _short(value: object) -> str:
    """Last-4 masking for phone numbers and message ids in logs."""
    text = str(value)
    return "\u2026" + text[-4:] if len(text) > 4 else text


class _Abandoned(Exception):
    """A turn stopped cleanly (clarification timeout or send failure)."""


def format_turn_result(result: Any) -> str | None:
    """Compose the WhatsApp reply for a finished turn, or None on failure.

    Sends the answer text plus the verification line. "Chart:" lines name
    files the user cannot see, so they are dropped. "Chart note:" lines are
    kept: the same note also goes in the image caption, and the duplication
    is deliberate — the text is the backstop if the image send fails.
    Never raises.
    """
    if result is None or getattr(result, "submission", None) is None:
        return None
    lines = [
        line
        for line in (getattr(result, "text", "") or "").splitlines()
        if not line.startswith("Chart: ")
    ]
    text = "\n".join(lines).strip()
    verification = getattr(result, "verification", "") or ""
    if verification:
        text = f"{text}\n{verification}" if text else verification
    return text


class WhatsAppServer:
    """Loopback webhook server with per-sender sessions and workers."""

    def __init__(
        self,
        config: WhatsAppConfig,
        client: WhatsAppClient,
        make_answer: Any,
        clarify_timeout: float = 300.0,
        backend_factory: Any = None,
        datasets: dict[str, dict[str, Any]] | None = None,
        default_dataset: str | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.make_answer = make_answer
        self.clarify_timeout = clarify_timeout
        self.backend_factory = backend_factory
        self.datasets = datasets or REGISTRY
        self.default_dataset = default_dataset
        self.sessions: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.seen: deque[str] = deque(maxlen=_DEDUP_KEEP)
        self.seen_set: set[str] = set()
        self.httpd: ThreadingHTTPServer | None = None

    # -- webhook entry points (fast; agent work stays on workers) --

    def handle_get(self, query: str) -> tuple[int, str, str]:
        """Meta verification handshake."""
        params = parse_qs(query)
        mode = (params.get("hub.mode") or [""])[0]
        token = (params.get("hub.verify_token") or [""])[0]
        challenge = (params.get("hub.challenge") or [""])[0]
        if mode == "subscribe" and hmac.compare_digest(token, self.config.verify_token):
            return 200, "text/plain", challenge
        return 403, "text/plain", "forbidden"

    def handle_post(self, raw: bytes, signature: str | None) -> int:
        """Verify, parse, enqueue. Returns the HTTP status."""
        if not self._valid_signature(raw, signature):
            log.warning("rejected unsigned webhook")
            return 403
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            log.warning("rejected unparseable webhook")
            return 200
        with self.lock:
            for sender, mid, body in self._extract(payload):
                if sender not in self.config.allowed_numbers:
                    continue
                if mid and mid in self.seen_set:
                    log.info("duplicate message %s ignored", _short(mid))
                    continue
                if mid:
                    if len(self.seen) == self.seen.maxlen:
                        self.seen_set.discard(self.seen[0])
                    self.seen.append(mid)
                    self.seen_set.add(mid)
                session = self.sessions.get(sender)
                if session is None:
                    session = {
                        "history": [],
                        "inbox": queue.Queue(),
                        "overtake": deque(),
                    }
                    session["worker"] = threading.Thread(
                        target=self._serve_sender,
                        args=(sender, session),
                        daemon=True,
                        name=f"wa-{_short(sender)}",
                    )
                    self.sessions[sender] = session
                    session["worker"].start()
                session["inbox"].put((time.monotonic(), mid, body))
                log.info("message from %s accepted", _short(sender))
        return 200

    # -- internals --

    def _valid_signature(self, raw: bytes, signature: str | None) -> bool:
        if not signature or not signature.startswith("sha256="):
            return False
        expected = hmac.new(
            self.config.app_secret.encode("utf-8"), raw, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest("sha256=" + expected, signature)

    @staticmethod
    def _extract(payload: Any) -> list[tuple[str, str, str]]:
        """Pull (sender, message id, text body) triples from a webhook."""
        items: list[tuple[str, str, str]] = []
        if not isinstance(payload, dict):
            return items
        for entry in payload.get("entry", []) or []:
            if not isinstance(entry, dict):
                continue
            for change in entry.get("changes", []) or []:
                if not isinstance(change, dict):
                    continue
                value = change.get("value", {}) or {}
                if not isinstance(value, dict):
                    continue
                for message in value.get("messages", []) or []:
                    if not isinstance(message, dict):
                        continue
                    if message.get("type") != "text":
                        continue
                    body = (message.get("text") or {}).get("body", "")
                    sender = message.get("from", "")
                    if not sender or not isinstance(body, str) or not body:
                        continue
                    items.append((sender, str(message.get("id", "")), body))
        return items

    def _send(self, sender: str, body: str) -> bool:
        try:
            self.client.send_text(sender, body)
        except WhatsAppError as exc:
            log.warning("send to %s failed: %s", _short(sender), exc)
            return False
        return True

    def _send_images(self, sender: str, images: list) -> None:
        """Send up to 3 drawn charts after the text answer.

        Unsafe paths are skipped quietly (logged); a failed send gets one
        short text line. Either way the session stays usable.
        """
        failed = False
        for path, caption in list(images)[:3]:
            try:
                self.client.send_image(sender, path, caption)
            except ValueError:
                log.warning("skipped un-sendable chart for %s", _short(sender))
            except WhatsAppError as exc:
                log.warning("chart image to %s failed: %s", _short(sender), exc)
                failed = True
        if failed:
            self._send(sender, _IMAGE_FAILURE_LINE)

    def _next_message(self, session: dict[str, Any]) -> tuple[float, str, str]:
        """Oldest pending message: the overtake stash first, else the inbox."""
        if session["overtake"]:
            return session["overtake"].popleft()
        return session["inbox"].get()

    def _asker(self, sender: str, session: dict[str, Any]) -> Any:
        """ask_clarification over WhatsApp: prompt, then block for a reply.

        NOTE (accepted): any message arriving while a clarification waits —
        including "/" commands — is consumed as the clarification reply,
        exactly like a message sent while the console blocks on input.
        """

        def ask(question: str, options: list[dict[str, Any]]) -> dict[str, str]:
            if not self._send(sender, format_options_text(question, options)):
                raise _Abandoned()
            asked_at = time.monotonic()
            deadline = asked_at + self.clarify_timeout
            typed = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    arrived, _mid, body = session["inbox"].get(timeout=remaining)
                except queue.Empty:
                    break
                if arrived < asked_at:
                    session["overtake"].append((arrived, _mid, body))
                    continue
                if typed:
                    # After "0" the next message is free text, exactly like
                    # the console's "type your answer:" input.
                    return {"choice": body.strip()}
                kind, value = interpret_clarification_reply(body, options)
                if kind == "choice":
                    return {"choice": value}
                if kind == "question":
                    return {"user_question": value}
                if body.strip() == "0":
                    typed = True
                if not self._send(sender, value):
                    raise _Abandoned()
            self._send(sender, _TIMEOUT_LINE)
            raise _Abandoned()

        return ask

    def _open_backend(self, session: dict[str, Any]) -> Any:
        """Create the session backend in this worker thread. Never shared."""
        backend = self.backend_factory(session["dataset"])
        session["backend"] = backend
        return backend

    def _dataset_list_text(self) -> str:
        """Numbered dataset list for the picker."""
        lines = ["Which dataset? Reply with a number:"]
        for i, (_key, info) in enumerate(self.datasets.items(), 1):
            lines.append(f"{i}. {info['label']}: {info['description']}")
        return "\n".join(lines)

    def _parse_pick(self, body: str) -> str | None:
        """A valid option number picks that dataset; anything else is None."""
        try:
            index = int(body.strip())
        except ValueError:
            return None
        names = list(self.datasets)
        if 1 <= index <= len(names):
            return names[index - 1]
        return None

    @staticmethod
    def _normalize_command(text: str) -> str:
        """Lowercase, collapse whitespace, drop trailing punctuation."""
        return " ".join(text.strip().lower().split()).rstrip(string.punctuation).strip()

    @staticmethod
    def _normalize_identifier(value: str) -> str:
        """Compare names ignoring case, spaces and hyphens."""
        return value.lower().replace(" ", "").replace("-", "")

    def _resolve_dataset(self, candidate: str) -> str | None:
        """Registry key for a normalized identifier or 1-based number."""
        names = list(self.datasets)
        if candidate.isdigit():
            index = int(candidate) - 1
            return names[index] if 0 <= index < len(names) else None
        for name in names:
            label = str(self.datasets[name].get("label", ""))
            if candidate and (
                candidate == self._normalize_identifier(name)
                or candidate == self._normalize_identifier(label)
            ):
                return name
        return None

    def _match_dataset_command(self, text: str) -> tuple[str | None, str | None]:
        """Classify a whole message as a dataset command.

        Returns ("picker", None) for /dataset and the plain phrases,
        ("switch", name) for /dataset <id> and phrase-plus-name forms, and
        (None, None) for everything else. Whole-message match only: a longer
        question that merely contains a dataset word is never a command.
        """
        norm = self._normalize_command(text)
        if norm in _DATASET_PHRASES:
            return ("picker", None)
        if norm == "/dataset" or norm.startswith("/dataset "):
            arg = norm[len("/dataset") :].strip()
            if not arg:
                return ("picker", None)
            name = self._resolve_dataset(self._normalize_identifier(arg))
            return ("switch", name) if name is not None else ("picker", None)
        tokens = norm.split()
        trigger = any(token in _TRIGGER_WORDS for token in tokens)
        mentions_data = any(token in _DATASET_WORDS for token in tokens)
        command_shaped = norm.startswith(("dataset ", "datasets", "database"))
        core = [token for token in tokens if token not in _FILLER_WORDS]
        had_verb = False
        while core and core[0] in _SWITCH_VERBS:
            core.pop(0)
            had_verb = True
        candidate = self._normalize_identifier(" ".join(core))
        if not candidate:
            return ("picker", None) if mentions_data else (None, None)
        if candidate.isdigit() and not trigger:
            return (None, None)
        name = self._resolve_dataset(candidate)
        if name is not None:
            return ("switch", name)
        # An unknown target falls back to the picker only for dataset-shaped
        # commands ("change dataset to banana", "dataset 9"); chart talk like
        # "change it to a bar chart" and plain questions go to the agent.
        if command_shaped or (had_verb and mentions_data):
            return ("picker", None)
        return (None, None)

    def _await_pick(
        self, session: dict[str, Any], sender: str, asked_at: float, held: bool
    ) -> str | None:
        """Wait for a valid pick; resend the list once, then stop.

        NOTE (accepted): a second question sent while the picker waits is
        consumed as the pick reply, exactly like a message sent while a
        clarification waits is consumed as the clarification reply. The
        overtake stash below only protects messages that predate the list.
        """
        deadline = asked_at + self.clarify_timeout
        retries = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                arrived, _mid, body = session["inbox"].get(timeout=remaining)
            except queue.Empty:
                break
            if arrived < asked_at:
                session["overtake"].append((arrived, _mid, body))
                continue
            kind = None
            pick_name = None
            if self.default_dataset is None:
                kind, pick_name = self._match_dataset_command(body)
            if kind == "switch" and pick_name is not None:
                return pick_name
            if kind == "picker" or body.strip() == "/dataset" or body.strip().startswith("/dataset "):
                asked_at = time.monotonic()
                deadline = asked_at + self.clarify_timeout
                retries = 0
                self._send(sender, self._dataset_list_text())
                continue
            name = self._parse_pick(body)
            if name is not None:
                return name
            if retries >= 1:
                self._send(sender, _PICK_STOP_LINE)
                return None
            retries += 1
            self._send(sender, self._dataset_list_text())
        self._send(sender, _TIMEOUT_LINE if held else _PICK_TIMEOUT_LINE)
        return None

    def _ensure_dataset(
        self, session: dict[str, Any], sender: str, held: str
    ) -> str | None:
        """Pick a dataset for the held first question; None when stopped."""
        self._send(sender, self._dataset_list_text())
        name = self._await_pick(session, sender, time.monotonic(), held=True)
        if name is None:
            return None
        session["dataset"] = name
        return name

    def _is_about(self, text: str) -> bool:
        """Whole-message about phrase; one leading slash is allowed for help."""
        norm = self._normalize_command(text)
        if norm.startswith("/"):
            norm = norm[1:]
        return norm in _ABOUT_PHRASES

    def _about_text(self, session: dict[str, Any]) -> str:
        """Code-built reply describing the data; never calls the model."""
        lines = ["I answer questions with numbers from these databases:"]
        for _key, info in self.datasets.items():
            lines.append(f"- {info['label']}: {info['description']}")
        name = session.get("dataset")
        backend = session.get("backend")
        if name is not None and backend is not None:
            try:
                tables = [str(t) for t in backend.list_tables()]
            except Exception:
                tables = []
            if tables:
                shown = tables[:_MAX_ABOUT_TABLES]
                extra = len(tables) - len(shown)
                line = f"Tables in {self.datasets[name]['label']}: " + ", ".join(shown)
                if extra > 0:
                    line += f", ...and {extra} more"
                lines.append(line)
        picked = [name] if name in self.datasets else list(self.datasets)
        lines.append("Try asking:")
        for key in picked:
            for example in _EXAMPLE_QUESTIONS.get(key, _GENERIC_EXAMPLES):
                lines.append(f"- {example}")
        return "\n".join(lines)

    def _confirmation(self, label: str) -> str:
        """Post-pick line; the switch hint only exists in ask mode."""
        text = f"Using {label}. Ask your question."
        if self.default_dataset is None:
            text += " Send /dataset to switch."
        return text

    def _switch_dataset(self, session: dict[str, Any], sender: str, name: str) -> None:
        """Direct switch: same reset as /dataset but no picker; confirm."""
        backend = session.pop("backend", None)
        if backend is not None:
            try:
                backend.close()
            except Exception:
                log.exception("session backend close failed for %s", _short(sender))
        session.pop("dataset", None)
        session["history"].clear()
        session["dataset"] = name
        label = self.datasets[name]["label"]
        self._send(sender, self._confirmation(label))

    def _reset_dataset(self, session: dict[str, Any], sender: str) -> None:
        """Clear dataset, backend, history and instructions; show the picker."""
        backend = session.pop("backend", None)
        if backend is not None:
            try:
                backend.close()
            except Exception:
                log.exception("session backend close failed for %s", _short(sender))
        session.pop("dataset", None)
        session["history"].clear()
        self._send(sender, self._dataset_list_text())
        session["pick_pending"] = True
        session["pick_asked_at"] = time.monotonic()

    def _serve_sender(self, sender: str, session: dict[str, Any]) -> None:
        """One sender's turns, strictly one at a time, forever."""
        while True:
            if session.get("pick_pending"):
                name = self._await_pick(
                    session, sender, session["pick_asked_at"], held=False
                )
                session.pop("pick_pending", None)
                if name is not None:
                    session["dataset"] = name
                    label = self.datasets[name]["label"]
                    self._send(sender, self._confirmation(label))
                continue
            _ts, _mid, body = self._next_message(session)
            text = body.strip()
            picked = (
                session.get("dataset") is not None or self.default_dataset is not None
            )
            if picked and text.isascii() and text.isdigit():
                # No clarification or picker is waiting here: those consume
                # digit replies directly from the inbox. A lone number would
                # otherwise start a model turn that can never succeed. In
                # ask mode with nothing chosen yet the picker below is the
                # digit handler, so digits fall through to it.
                self._send(sender, _DIGIT_REPLY)
                continue
            if self.default_dataset is None:
                kind, switch_name = self._match_dataset_command(text)
                if kind == "switch" and switch_name is not None:
                    self._switch_dataset(session, sender, switch_name)
                    continue
                if kind == "picker":
                    self._reset_dataset(session, sender)
                    continue
            if text == "/dataset" or text.startswith("/dataset "):
                self._reset_dataset(session, sender)
                continue
            if session.get("dataset") is None and self.default_dataset is not None:
                session["dataset"] = self.default_dataset
            if self._is_about(text):
                # Code-only reply: no model turn, placed before the "/"
                # fallback so "/help" works. The backend is opened here in
                # the worker thread when a dataset is already known, so the
                # table list is fresh; otherwise that section is omitted.
                try:
                    if (
                        self.backend_factory is not None
                        and "backend" not in session
                        and session.get("dataset") is not None
                    ):
                        self._open_backend(session)
                except Exception:
                    log.exception("about backend failed for %s", _short(sender))
                    self._send(sender, _FAILURE_LINE)
                    continue
                self._send(sender, self._about_text(session))
                continue
            if text.startswith("/"):
                self._send(sender, _OWN_COMMAND_REPLY)
                continue
            if session.get("dataset") is None:
                if self._ensure_dataset(session, sender, body) is None:
                    continue
            snapshot = len(session["history"])
            try:
                if self.backend_factory is not None and "backend" not in session:
                    self._open_backend(session)
                answer_fn = self.make_answer(session)
                reply = answer_fn(
                    session["history"], body, self._asker(sender, session)
                )
            except _Abandoned:
                del session["history"][snapshot:]
                continue
            except Exception:
                log.exception("answer failed for %s", _short(sender))
                del session["history"][snapshot:]
                self._send(sender, _FAILURE_LINE)
                continue
            if isinstance(reply, tuple):
                reply_text, images = reply
            else:
                reply_text, images = reply, []
            if reply_text is None:
                del session["history"][snapshot:]
                self._send(sender, _NO_ANSWER_LINE)
                continue
            self._send(sender, reply_text)
            self._send_images(sender, images)

    # -- HTTP routing (thin wrappers; logic stays unit-testable) --


    def dispatch_get(self, path: str, query: str) -> tuple[int, str, str]:
        """Route a GET: (status, content-type, body)."""
        if path != "/webhook":
            return 404, "text/plain", "not found"
        return self.handle_get(query)

    def dispatch_post(
        self, path: str, length: int, raw: bytes, signature: str | None
    ) -> tuple[int, str, str]:
        """Route a POST: (status, content-type, body)."""
        if path != "/webhook":
            return 404, "text/plain", "not found"
        if length > _MAX_BODY:
            return 413, "text/plain", "body too large"
        status = self.handle_post(raw, signature)
        return status, "text/plain", "ok" if status == 200 else "error"

    # -- serving --

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Bind loopback and serve until interrupted."""
        self.httpd = ThreadingHTTPServer((host, port), _WebhookHandler)
        self.httpd.daemon_threads = True
        self.httpd.app = self  # type: ignore[attr-defined]
        log.info("WhatsApp webhook listening on %s:%d", host, port)
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.httpd.server_close()
            self.close_sessions()

    def close_sessions(self) -> None:
        """Defensively close every live session backend (shutdown path)."""
        with self.lock:
            sessions = list(self.sessions.values())
        for session in sessions:
            backend = session.pop("backend", None)
            if backend is None:
                continue
            try:
                backend.close()
            except Exception:
                log.exception("session backend close failed")


class _WebhookHandler(BaseHTTPRequestHandler):
    """One request per connection; a stalled peer cannot hold a thread."""

    server_version = "WhatsAppAgent/1"
    timeout = 10

    @property
    def app(self) -> "WhatsAppServer":
        return self.server.app  # type: ignore[attr-defined,no-any-return]

    def _reply(self, status: int, content_type: str, body: str) -> None:
        raw = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        status, content_type, body = self.app.dispatch_get(
            parsed.path, parsed.query
        )
        self._reply(status, content_type, body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if 0 < length <= _MAX_BODY else b""
        status, content_type, body = self.app.dispatch_post(
            urlparse(self.path).path,
            length,
            raw,
            self.headers.get("X-Hub-Signature-256"),
        )
        self._reply(status, content_type, body)

    def log_message(self, *args: object) -> None:
        log.info("%s %s", self.command, self.path.split("?")[0])
