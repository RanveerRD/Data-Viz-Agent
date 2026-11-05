"""Regression tests for grounded clarification (ask_clarification).

No API calls. The fake client replays a fixed script; databases and memory
live in temporary directories only.
"""

import json
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.__main__ import (  # noqa: E402
    _console_ask_user,
    _definition_text,
    _format_evidence,
    _offer_definition_save,
)
from agent.backend import SqliteBackend  # noqa: E402
from agent.loop import answer_question  # noqa: E402
from agent.tools import ToolContext, dispatch  # noqa: E402

ITEMS = [
    {"id": "revenue", "question": "What is revenue?", "evidence_needed": "totals by status"},
]

PLAN = [{"item_id": "revenue", "steps": ["list statuses", "total by status"]}]

OPTIONS = [
    {"label": "delivered only", "query_id": "q1", "evidence": "delivered",
     "explanation": "excludes shipping fees"},
    {"label": "delivered and shipped", "query_id": "q1", "evidence": "shipped",
     "explanation": "includes shipped orders"},
]

ASK = {
    "question": "Which statuses count as revenue?",
    "options": OPTIONS,
    "definition_key": "revenue",
}


def _make_backend(tmp: str) -> SqliteBackend:
    path = Path(tmp) / "fixture.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT NOT NULL, price REAL NOT NULL);"
        )
        conn.executemany(
            "INSERT INTO orders (status, price) VALUES (?, ?)",
            [("delivered", 100.0), ("delivered", 50.0), ("canceled", 30.0), ("shipped", 20.0)],
        )
        conn.commit()
    finally:
        conn.close()
    return SqliteBackend(path)


def _ctx(backend: SqliteBackend, tmp: str, ask_user=None) -> ToolContext:
    return ToolContext(
        backend=backend,
        question="What is revenue?",
        runs_dir=Path(tmp) / "runs",
        memory_dir=Path(tmp) / "memory",
        ask_user=ask_user,
    )


def _choice(label_index: int = 0):
    return lambda q, opts: {"choice": opts[label_index]["label"]}


def _ready_ctx(backend: SqliteBackend, tmp: str, ask_user=None) -> ToolContext:
    ctx = _ctx(backend, tmp, ask_user)
    dispatch(ctx, "set_checklist", {"items": ITEMS})
    dispatch(ctx, "plan_action", {"plan": PLAN})
    result = json.loads(dispatch(ctx, "run_sql", {"query": "SELECT DISTINCT status FROM orders;"}))
    assert result["query_id"] == "q1"
    return ctx


def _inputs(*replies):
    it = iter(replies)

    def _next(prompt=""):
        print(prompt, end="")
        return next(it)

    return _next


def _message(text: str):
    return types.SimpleNamespace(
        type="message",
        content=[types.SimpleNamespace(type="output_text", text=text)],
    )


def _call(call_id: str, name: str, args: dict):
    return types.SimpleNamespace(
        type="function_call", call_id=call_id, name=name, arguments=json.dumps(args)
    )


def _response(*items):
    return types.SimpleNamespace(
        output=list(items),
        usage=types.SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
    )


class _StubClient:
    def __init__(self, script):
        self._script = list(script)
        self.responses = self
        self.calls = 0

    def create(self, **kwargs):
        item = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return item


def test_option_evidence_not_in_result_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ready_ctx(backend, tmp, ask_user=_choice())
            bad = dict(ASK)
            bad["options"] = [
                {"label": "refunded only", "query_id": "q1", "evidence": "refunded",
                 "explanation": "includes refunded orders"},
                OPTIONS[1],
            ]
            refused = json.loads(dispatch(ctx, "ask_clarification", bad))
            assert "does not appear" in refused["error"]
            assert len(ctx.record.rejected) == 1
            assert ctx.record.audit[-1]["action"] == "clarification_rejected"
            assert ctx.record.clarification is None
            assert ctx.record.clarification_calls == 0
        finally:
            backend.close()


def test_option_unknown_query_id_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ready_ctx(backend, tmp, ask_user=_choice())
            dispatch(ctx, "run_sql", {"query": "SELECT * FROM no_such_table;"})
            assert [f["id"] for f in ctx.record.failed_queries] == ["f1"]
            bad = dict(ASK)
            bad["options"] = [
                {"label": "broken", "query_id": "f1", "evidence": "x",
                 "explanation": "cites a failure"},
                OPTIONS[1],
            ]
            refused = json.loads(dispatch(ctx, "ask_clarification", bad))
            assert "no citable result" in refused["error"]
            bad2 = dict(ASK)
            bad2["options"] = [
                {"label": "missing", "query_id": "q99", "evidence": "x",
                 "explanation": "cites nothing"},
                OPTIONS[1],
            ]
            refused2 = json.loads(dispatch(ctx, "ask_clarification", bad2))
            assert "unknown query id" in refused2["error"]
        finally:
            backend.close()


def test_option_explanation_required_and_capped():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ready_ctx(backend, tmp, ask_user=_choice())
            bad = dict(ASK)
            bad["options"] = [dict(OPTIONS[0]), dict(OPTIONS[1])]
            del bad["options"][0]["explanation"]
            refused = json.loads(dispatch(ctx, "ask_clarification", bad))
            assert "explanation" in refused["error"]
            bad["options"][0]["explanation"] = "x" * 121
            refused = json.loads(dispatch(ctx, "ask_clarification", bad))
            assert "at most 120 characters" in refused["error"]
        finally:
            backend.close()


def test_number_in_label_not_in_result_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ready_ctx(backend, tmp, ask_user=_choice())
            bad = dict(ASK)
            bad["options"] = [
                {"label": "orders over 9999", "query_id": "q1", "evidence": "delivered",
                 "explanation": "counts every row"},
                OPTIONS[1],
            ]
            refused = json.loads(dispatch(ctx, "ask_clarification", bad))
            assert "unmatched numbers" in refused["error"]
        finally:
            backend.close()


def test_accepted_choice_returned_and_audited():
    seen = {}

    def _ask(question, options):
        seen["question"] = question
        seen["options"] = options
        return {"choice": options[1]["label"]}

    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ready_ctx(backend, tmp, ask_user=_ask)
            result = json.loads(dispatch(ctx, "ask_clarification", dict(ASK)))
            assert result == {"ok": True, "choice": "delivered and shipped"}
            assert seen["question"] == ASK["question"]
            assert [o["label"] for o in seen["options"]] == [
                "delivered only", "delivered and shipped"]
            stored = ctx.record.clarification
            assert stored["reply"] == "delivered and shipped"
            assert stored["definition_key"] == "revenue"
            assert ctx.record.clarification_calls == 1
            assert ctx.record.audit[-1]["action"] == "ask_clarification"
        finally:
            backend.close()


def test_user_question_returned_with_guidance():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ready_ctx(
                backend, tmp,
                ask_user=lambda q, opts: {"user_question": "what about canceled?"},
            )
            result = json.loads(dispatch(ctx, "ask_clarification", dict(ASK)))
            assert result["user_question"] == "what about canceled?"
            assert "do not start a new checklist" in result["guidance"]
            assert "ask_clarification again" in result["guidance"]
            assert ctx.record.clarification is None
            assert ctx.record.clarification_calls == 1
            assert ctx.record.audit[-1]["action"] == "clarification_question"
        finally:
            backend.close()


def test_sixth_clarification_refused():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            calls = {"n": 0}

            def _ask(question, options):
                calls["n"] += 1
                if calls["n"] == 2:
                    return {"user_question": "why not canceled?"}
                return {"choice": options[0]["label"]}

            ctx = _ready_ctx(backend, tmp, ask_user=_ask)
            for _ in range(5):
                assert json.loads(dispatch(ctx, "ask_clarification", dict(ASK)))["ok"]
            assert ctx.record.clarification_calls == 5
            sixth = json.loads(dispatch(ctx, "ask_clarification", dict(ASK)))
            assert "five clarifications" in sixth["error"]
            assert calls["n"] == 5
        finally:
            backend.close()


def test_set_checklist_between_clarifications_keeps_budget():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ready_ctx(backend, tmp, ask_user=_choice())
            assert json.loads(dispatch(ctx, "ask_clarification", dict(ASK)))["ok"]
            dispatch(ctx, "set_checklist", {"items": ITEMS + [
                {"id": "extra", "question": "More?", "evidence_needed": "more rows"}
            ]})
            assert ctx.record.clarification_calls == 1
            for _ in range(4):
                assert json.loads(dispatch(ctx, "ask_clarification", dict(ASK)))["ok"]
            sixth = json.loads(dispatch(ctx, "ask_clarification", dict(ASK)))
            assert "five clarifications" in sixth["error"]
        finally:
            backend.close()


def test_last_used_option_marked_from_saved_definition():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            from agent.preliminary_analysis import approve_definition
            approve_definition(
                "revenue", "delivered only: excludes shipping fees",
                memory_dir=Path(tmp) / "memory",
            )
            seen = {}

            def _ask(question, options):
                seen["options"] = options
                return {"choice": options[0]["label"]}

            ctx = _ready_ctx(backend, tmp, ask_user=_ask)
            assert json.loads(dispatch(ctx, "ask_clarification", dict(ASK)))["ok"]
            assert seen["options"][0].get("last_used") is True
            assert "last_used" not in seen["options"][1]
            stored = ctx.record.clarification
            assert stored["options"][0].get("last_used") is True
        finally:
            backend.close()


def test_no_callback_returns_state_decisions_error():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ready_ctx(backend, tmp, ask_user=None)
            result = json.loads(dispatch(ctx, "ask_clarification", dict(ASK)))
            assert "State your decisions" in result["error"]
            assert ctx.record.clarification is None
            assert ctx.record.clarification_calls == 0
        finally:
            backend.close()


def test_set_checklist_refuses_dropping_unproven_item():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            items = ITEMS + [
                {"id": "status", "question": "Which status?",
                 "evidence_needed": "status list"},
            ]
            dispatch(ctx, "set_checklist", {"items": items})
            dispatch(ctx, "plan_action", {"plan": [
                {"item_id": "revenue", "steps": ["count"]},
                {"item_id": "status", "steps": ["list"]},
            ]})
            dispatch(ctx, "run_sql", {"query": "SELECT COUNT(*) FROM orders;"})
            dispatch(ctx, "record_finding", {
                "item_id": "revenue", "claim": "4 orders",
                "query_id": "q1", "value": 4})
            assert ctx.record.is_proven("revenue")
            dropped = json.loads(
                dispatch(ctx, "set_checklist", {"items": [ITEMS[0]]}))
            assert "include the existing unproven items" in dropped["error"]
            assert [c["id"] for c in ctx.record.checklist] == ["revenue", "status"]
            kept = json.loads(dispatch(ctx, "set_checklist", {"items": [items[1]]}))
            assert kept["ok"] is True
            assert [c["id"] for c in ctx.record.checklist] == ["status"]
        finally:
            backend.close()


def test_console_renders_options_with_evidence_and_explanation(capsys):
    reply = _console_ask_user(
        ASK["question"], OPTIONS, input_fn=_inputs("1"))
    assert reply == {"choice": "delivered only"}
    out = capsys.readouterr().out
    assert "1. delivered only [delivered]" in out
    assert "excludes shipping fees" in out
    assert "2. delivered and shipped [shipped]" in out
    assert "0. something else" in out


def test_console_formats_numeric_evidence(capsys):
    assert _format_evidence(1234567) == "1,234,567"
    assert _format_evidence(48899.34) == "48,899.34"
    assert _format_evidence("delivered") == "delivered"
    options = [dict(OPTIONS[0], evidence=1234567), dict(OPTIONS[1])]
    reply = _console_ask_user("Which?", options, input_fn=_inputs("1"))
    assert reply == {"choice": "delivered only"}
    assert "[1,234,567]" in capsys.readouterr().out


def test_console_reprompts_on_bad_number(capsys):
    reply = _console_ask_user(
        ASK["question"], OPTIONS, input_fn=_inputs("9", "2"))
    assert reply == {"choice": "delivered and shipped"}
    assert "pick 1 to 2" in capsys.readouterr().out


def test_console_reprompts_on_empty_reply(capsys):
    reply = _console_ask_user(
        ASK["question"], OPTIONS, input_fn=_inputs("", "1"))
    assert reply == {"choice": "delivered only"}
    assert "pick 1 to 2" in capsys.readouterr().out


def test_console_zero_takes_typed_choice(capsys):
    reply = _console_ask_user(
        ASK["question"], OPTIONS, input_fn=_inputs("0", "my own reading"))
    assert reply == {"choice": "my own reading"}
    assert "type your answer:" in capsys.readouterr().out


def test_console_text_is_a_user_question():
    reply = _console_ask_user(
        ASK["question"], OPTIONS, input_fn=_inputs("what about canceled?"))
    assert reply == {"user_question": "what about canceled?"}


def test_console_shows_last_used_and_enter_selects_it(capsys):
    options = [dict(OPTIONS[0], last_used=True), dict(OPTIONS[1])]
    reply = _console_ask_user(ASK["question"], options, input_fn=_inputs(""))
    assert reply == {"choice": "delivered only"}
    assert "(last used)" in capsys.readouterr().out


def test_console_empty_without_last_used_reprompts(capsys):
    reply = _console_ask_user(
        ASK["question"], OPTIONS, input_fn=_inputs("", "2"))
    assert reply == {"choice": "delivered and shipped"}
    assert "pick 1 to 2" in capsys.readouterr().out


def test_offer_save_skipped_for_last_used_choice():
    asked = []

    def _approve(key, text, evidence):
        raise AssertionError("approve must not be called")

    clarification = {
        "question": ASK["question"],
        "options": [dict(OPTIONS[0], last_used=True), dict(OPTIONS[1])],
        "reply": "delivered only",
        "definition_key": "revenue",
    }
    assert _offer_definition_save(
        clarification, ask=lambda p: asked.append(p) or "y", approve=_approve
    ) is False
    assert asked == []


def test_offer_save_prompt_uses_new_default_wording(capsys):
    saved = {}
    prompts = []

    def _approve(key, text, evidence):
        saved["args"] = (key, text, evidence)
        return {"key": key, "text": text}

    clarification = {
        "question": ASK["question"],
        "options": OPTIONS,
        "reply": "delivered and shipped",
        "definition_key": "revenue",
    }
    assert _offer_definition_save(
        clarification, ask=lambda p: prompts.append(p) or "y", approve=_approve
    ) is True
    assert "Save as the new default" in prompts[0]
    assert "approved revenue" in capsys.readouterr().out
    assert saved["args"][0] == "revenue"


def test_instructions_require_asking_every_time():
    text = (Path(__file__).resolve().parent.parent / "agent" / "instructions.md").read_text()
    assert "do not ask when an approved definition exists" not in text
    assert "EVERY time, even if a saved definition exists" in text
    assert "top_metric" in text
    assert "after the last clarification is answered" in text.lower()
    for key in ("revenue", "top_metric", "order_status", "year_date", "cost", "period"):
        assert key in text
    assert "every column or table that could represent it" in text
    assert "counts as stating it" in text
    assert "only_option is only for a list of distinct values" in text
    assert "FIRST, before computing any figure" in text
    assert "Never compute combinations of candidates" in text


def test_loop_continues_same_turn_after_reply():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            script = [
                _response(_call("c1", "set_checklist", {"items": ITEMS})),
                _response(_call("c2", "plan_action", {"plan": PLAN})),
                _response(_call("c3", "run_sql", {
                    "query": "SELECT DISTINCT status FROM orders;"})),
                _response(_call("c4", "ask_clarification", dict(ASK))),
                _response(_message("Noted, computing.")),
            ]
            history: list = []
            result = answer_question(
                _StubClient(script), "m", "instr", backend, history, "What is revenue?",
                runs_dir=Path(tmp) / "runs",
                ask_user=_choice(),
            )
            assert result.clarification is not None
            assert result.clarification["reply"] == "delivered only"
            assert result.text == "Noted, computing."
        finally:
            backend.close()


def test_offer_definition_save_only_with_key():
    saved = {}

    def _approve(key, text, evidence):
        saved["args"] = (key, text, evidence)
        return {"key": key, "text": text}

    clarification = {
        "question": ASK["question"],
        "options": OPTIONS,
        "reply": "delivered only",
        "definition_key": "revenue",
    }
    assert _offer_definition_save(None, ask=lambda p: "y", approve=_approve) is False
    assert _offer_definition_save(
        {**clarification, "definition_key": None},
        ask=lambda p: "y", approve=_approve,
    ) is False
    assert saved == {}
    assert _offer_definition_save(
        clarification, ask=lambda p: "n", approve=_approve
    ) is False
    assert saved == {}
    assert _offer_definition_save(
        clarification, ask=lambda p: "y", approve=_approve
    ) is True
    assert saved["args"] == (
        "revenue", "delivered only: excludes shipping fees", ["q1"])


def test_definition_text_prefers_option_explanation():
    assert _definition_text({
        "reply": "delivered only",
        "options": OPTIONS,
    }) == "delivered only: excludes shipping fees"
    assert _definition_text({
        "reply": "my own reading",
        "options": OPTIONS,
    }) == "my own reading"


def test_offer_definition_save_stores_typed_answer_verbatim():
    saved = {}

    def _approve(key, text, evidence):
        saved["args"] = (key, text, evidence)
        return {"key": key, "text": text}

    clarification = {
        "question": ASK["question"],
        "options": OPTIONS,
        "reply": "only high-value delivered",
        "definition_key": "revenue",
    }
    assert _offer_definition_save(
        clarification, ask=lambda p: "y", approve=_approve
    ) is True
    assert saved["args"] == ("revenue", "only high-value delivered", ["q1"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"all {len(tests)} tests passed")
