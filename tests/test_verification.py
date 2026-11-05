"""Regression tests for findings, submit_answer, and the submit loop.

Run with the project environment:

    .venv/bin/python -m pytest tests/test_verification.py -v
    .venv/bin/python tests/test_verification.py

No API calls: the loop tests use a stubbed client. Temporary directories
only; the real data/olist.sqlite is never touched.
"""

import json
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.backend import SqliteBackend  # noqa: E402
from agent.loop import (  # noqa: E402
    MAX_TURNS,
    WRAP_UP_NUDGE,
    answer_question,
    render_answer,
)
from agent.numbers import extract_numbers, verify_text  # noqa: E402
from agent.tools import TOOL_SCHEMAS, ToolContext, dispatch  # noqa: E402

ITEMS = [
    {"id": "total_revenue", "question": "Total revenue?", "evidence_needed": "sum of revenue"},
    {"id": "top_region", "question": "Top region?", "evidence_needed": "revenue by region"},
]

PLAN = [
    {"item_id": "total_revenue", "steps": ["sum revenue"]},
    {"item_id": "top_region", "steps": ["group by region"]},
]

DECISIONS = [
    {"what": "scope", "choice": "all rows", "source": "only_option", "proof": "q2"}
]

LOOP_DECISIONS = [
    {"what": "scope", "choice": "everything", "source": "user_stated", "proof": "How many?"}
]


def _make_backend(tmp: str) -> SqliteBackend:
    path = Path(tmp) / "fixture.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE sales (region TEXT NOT NULL, revenue REAL NOT NULL);")
        conn.executemany(
            "INSERT INTO sales (region, revenue) VALUES (?, ?)",
            [("SP", 498440.43), ("RJ", 12.345)],
        )
        conn.commit()
    finally:
        conn.close()
    return SqliteBackend(path)


def _ctx(backend: SqliteBackend, tmp: str, question: str = "q?") -> ToolContext:
    return ToolContext(backend=backend, question=question, runs_dir=Path(tmp) / "runs")


def _flowed_ctx(backend: SqliteBackend, tmp: str) -> ToolContext:
    """Context with checklist, plan, q1 (two regions) and q2 (one region)."""
    ctx = _ctx(backend, tmp)
    assert json.loads(dispatch(ctx, "set_checklist", {"items": ITEMS}))["ok"] is True
    assert json.loads(dispatch(ctx, "plan_action", {"plan": PLAN}))["ok"] is True
    result = json.loads(
        dispatch(ctx, "run_sql", {"query": "SELECT region, revenue FROM sales ORDER BY region;"})
    )
    assert result["query_id"] == "q1"
    result = json.loads(
        dispatch(ctx, "run_sql", {"query": "SELECT DISTINCT region FROM sales WHERE region = 'RJ';"})
    )
    assert result["query_id"] == "q2"
    return ctx


def test_finding_refused_unknown_query_id():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            result = json.loads(
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": "total_revenue", "claim": "c", "query_id": "q9", "value": 1},
                )
            )
            assert "unknown query id" in result["error"]
            assert ctx.record.rejected and "unknown query id" in ctx.record.rejected[-1]["reason"]
        finally:
            backend.close()


def test_finding_refused_value_not_in_result():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            result = json.loads(
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": "total_revenue", "claim": "c", "query_id": "q1", "value": 777},
                )
            )
            assert "does not appear" in result["error"]
            assert not ctx.record.is_proven("total_revenue")
        finally:
            backend.close()


def test_finding_accepted_within_tolerance_and_max_three():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            ok = json.loads(
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": "top_region", "claim": "RJ is 12.345", "query_id": "q1", "value": 12.347},
                )
            )
            assert ok.get("ok") is True
            assert ctx.record.is_proven("top_region")
            for i in range(2):
                ok = json.loads(
                    dispatch(
                        ctx,
                        "record_finding",
                        {"item_id": "top_region", "claim": f"more {i}", "query_id": "q1", "value": "RJ"},
                    )
                )
                assert ok.get("ok") is True
            fourth = json.loads(
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": "top_region", "claim": "fourth", "query_id": "q1", "value": "RJ"},
                )
            )
            assert "already has 3 findings" in fourth["error"]
        finally:
            backend.close()


def test_repeat_set_checklist_keeps_queries_drops_stale_findings():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            dispatch(
                ctx,
                "record_finding",
                {"item_id": "top_region", "claim": "RJ", "query_id": "q1", "value": "RJ"},
            )
            assert ctx.record.is_proven("top_region")
            dispatch(ctx, "set_checklist", {"items": [ITEMS[0]]})
            assert [q["id"] for q in ctx.record.queries] == ["q1", "q2"]
            assert ctx.record.findings == []
            assert ctx.record.plan == []
            assert [c["id"] for c in ctx.record.checklist] == ["total_revenue"]
            assert not ctx.record.is_proven("total_revenue")
        finally:
            backend.close()


def test_submit_refused_uncovered_item():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            dispatch(
                ctx,
                "record_finding",
                {"item_id": "total_revenue", "claim": "sum", "query_id": "q1", "value": 498440.43},
            )
            result = json.loads(
                dispatch(
                    ctx,
                    "submit_answer",
                    {"answer": "Total is 498440.43.", "decisions": DECISIONS, "gaps": []},
                )
            )
            assert "uncovered" in result["error"] and "top_region" in result["error"]
            assert ctx.record.submission is None
        finally:
            backend.close()


def test_submit_refused_unmatched_number_then_accepted():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            for item_id, value in (("total_revenue", 498440.43), ("top_region", "RJ")):
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": item_id, "claim": "c", "query_id": "q1", "value": value},
                )
            refused = json.loads(
                dispatch(
                    ctx,
                    "submit_answer",
                    {"answer": "Total is 498440.43 plus 7 bonus.", "decisions": DECISIONS, "gaps": []},
                )
            )
            assert refused["unmatched"] == ["7"]
            fixed = json.loads(
                dispatch(
                    ctx,
                    "submit_answer",
                    {
                        "answer": "Total is 498,440.",
                        "decisions": DECISIONS,
                        "gaps": [],
                    },
                )
            )
            assert fixed["ok"] is True
            assert fixed["status"] == "answered"
            assert fixed["rejections"] == 1
            assert fixed["unmatched"] == []
            assert ctx.record.submission["answer"] == "Total is 498,440."
        finally:
            backend.close()


def test_submit_escalated_after_three_rejections():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            for item_id, value in (("total_revenue", 498440.43), ("top_region", "RJ")):
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": item_id, "claim": "c", "query_id": "q1", "value": value},
                )
            last = {}
            submitted_decisions = DECISIONS
            submitted_gaps = [{"item_id": "", "note": "Date range unclear."}]
            for _ in range(3):
                last = json.loads(
                    dispatch(
                        ctx,
                        "submit_answer",
                        {
                            "answer": "Total is 9.",
                            "decisions": submitted_decisions,
                            "gaps": submitted_gaps,
                        },
                    )
                )
            assert last["ok"] is True
            assert last["status"] == "escalated"
            assert last["warning"].startswith("UNVERIFIED NUMBERS:")
            assert "9" in last["warning"]
            assert "UNCONFIRMED" not in last["warning"]
            assert ctx.record.submission["status"] == "escalated"
            assert ctx.record.rejections == 3
            assert ctx.record.submission["decisions"] == submitted_decisions
            assert ctx.record.submission["gaps"] == submitted_gaps
        finally:
            backend.close()


def test_years_and_list_numbering_ignored():
    assert extract_numbers("1. First\n2. Second") == []
    matched, unmatched = verify_text(
        "In 2024:\n1. Total\nThe total is 5.", "How many?", [5]
    )
    assert matched == ["5"]
    assert unmatched == []


def test_iso_dates_and_times_produce_no_numbers():
    assert extract_numbers("from 2017-01-01 to 2017-12-31") == []
    assert extract_numbers("in 2017-01") == []
    assert extract_numbers("at 10:30:00") == []
    assert extract_numbers("at 10:30") == []
    matched, unmatched = verify_text(
        "From 2017-01-01 to 2017-12-31.", "Over what period?", [99441]
    )
    assert matched == []
    assert unmatched == []


def test_hyphen_between_digits_is_a_range_separator():
    assert extract_numbers("2017-2018") == ["2017", "2018"]
    assert extract_numbers("10-20") == ["10", "20"]
    assert extract_numbers("-5.2") == ["-5.2"]
    assert extract_numbers("(-5.2)") == ["-5.2"]
    matched, unmatched = verify_text(
        "Total is 5 from 2017-01-01 to 2017-12-31.", "How many?", [5]
    )
    assert matched == ["5"]
    assert unmatched == []


def test_identifiers_are_not_number_claims():
    assert extract_numbers("order d6160fb7873f shipped") == []
    assert extract_numbers("sku123") == []
    assert extract_numbers("the 3rd item") == []
    assert extract_numbers("6160fb78") == []


def test_single_letter_suffix_is_extracted():
    assert extract_numbers("13.6M") == ["13.6"]
    matched, unmatched = verify_text("It is 13.6M.", "How big?", [5])
    assert matched == []
    assert unmatched == ["13.6"]


def test_real_numbers_next_to_words_still_count():
    assert extract_numbers("35 units") == ["35"]
    assert extract_numbers("8.1%") == ["8.1%"]
    assert extract_numbers("$48,899.34") == ["48,899.34"]


def test_quoted_result_identifier_passes():
    matched, unmatched = verify_text(
        'The product_id is "d6160fb7873f".', "Which product?", ["d6160fb7873f", 5]
    )
    assert matched == []
    assert unmatched == []


def test_made_up_id_yields_no_numbers():
    assert extract_numbers("order deadbeef99") == []
    matched, unmatched = verify_text("order deadbeef99 shipped", "Q?", [5])
    assert matched == []
    assert unmatched == []


def test_quoted_result_string_with_digits_is_masked():
    matched, unmatched = verify_text("Code 12 AB done.", "Q?", ["12 AB"])
    assert matched == []
    assert unmatched == []


def test_touching_occurrence_is_not_masked():
    matched, unmatched = verify_text("Code 12 ABx done.", "Q?", ["12 AB"])
    assert matched == []
    assert unmatched == ["12"]


def test_short_text_cell_does_not_swallow_numbers():
    matched, unmatched = verify_text("Total is 1,234.", "How much?", ["1", 1234])
    assert matched == ["1,234"]
    assert unmatched == []


def test_render_answer_without_gaps():
    submission = {
        "answer": "A.",
        "decisions": [
            {"what": "scope", "choice": "all", "source": "user_stated", "proof": "all rows"},
            {"what": "metric", "choice": "revenue", "source": "only_option", "proof": "q1"},
        ],
        "gaps": [],
    }
    assert render_answer(submission, []) == (
        "A.\n\nDecisions:\n- scope: all (you stated: \"all rows\")\n"
        "- metric: revenue (not asked: no alternative found, query q1)\n\nGaps:\n- none found\n\n"
        "Want the evidence behind these decisions?"
    )


def test_render_answer_with_gaps():
    submission = {
        "answer": "A.",
        "decisions": [
            {"what": "metric", "choice": "delivered only", "source": "user_chosen", "proof": "c1"},
            {"what": "bad", "choice": "x", "source": "user_stated", "proof": "y", "unconfirmed": True},
        ],
        "gaps": [
            {"item_id": "n", "note": "Need cost."},
            {"item_id": "", "note": "General."},
        ],
    }
    clarifications = [
        {
            "id": "c1",
            "definition_key": "revenue",
            "options": [
                {"label": "delivered only"},
                {"label": "all statuses"},
            ],
            "reply": "delivered only",
        }
    ]
    assert render_answer(submission, clarifications) == (
        "A.\n\nDecisions:\n- revenue: delivered only (you chose: option 1)\n"
        "- bad: x (UNCONFIRMED)\n\nGaps:\n- n: Need cost.\n- General.\n\n"
        "Want the evidence behind these decisions?"
    )


def test_render_answer_user_chosen_typed_reply():
    submission = {
        "answer": "A.",
        "decisions": [
            {"what": "metric", "choice": "my own", "source": "user_chosen", "proof": "c2"},
        ],
        "gaps": [],
    }
    clarifications = [{"id": "c2", "options": [{"label": "a"}, {"label": "b"}], "reply": "my own"}]
    assert "(you chose: \"my own\")" in render_answer(submission, clarifications)


CLARIFY_OPTIONS = [
    {"label": "delivered only", "query_id": "q1", "evidence": "SP",
     "explanation": "counts delivered orders"},
    {"label": "all statuses", "query_id": "q1", "evidence": "RJ",
     "explanation": "counts every order"},
]


def _decided_ctx(backend: SqliteBackend, tmp: str) -> ToolContext:
    """Flowed context with findings plus one clarification (c1)."""
    ctx = _ctx(backend, tmp, question="Revenue for delivered orders by region today?")
    dispatch(ctx, "set_checklist", {"items": ITEMS})
    dispatch(ctx, "plan_action", {"plan": PLAN})
    result = json.loads(
        dispatch(ctx, "run_sql", {"query": "SELECT region, revenue FROM sales ORDER BY region;"})
    )
    assert result["query_id"] == "q1"
    result = json.loads(
        dispatch(ctx, "run_sql", {"query": "SELECT DISTINCT region FROM sales WHERE region = 'RJ';"})
    )
    assert result["query_id"] == "q2"
    for item_id, value in (("total_revenue", 498440.43), ("top_region", "RJ")):
        dispatch(
            ctx,
            "record_finding",
            {"item_id": item_id, "claim": "c", "query_id": "q1", "value": value},
        )
    ctx.ask_user = lambda q, opts: {"choice": opts[0]["label"]}
    clarified = json.loads(
        dispatch(ctx, "ask_clarification", {
            "question": "Which statuses count?",
            "options": CLARIFY_OPTIONS,
            "definition_key": "revenue",
        })
    )
    assert clarified["ok"] is True
    return ctx


def _submit(ctx, decisions, answer="Total is 498,440."):
    return json.loads(
        dispatch(ctx, "submit_answer", {"answer": answer, "decisions": decisions, "gaps": []})
    )


def test_decisions_each_source_accepted():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _decided_ctx(backend, tmp)
            result = _submit(ctx, [
                {"what": "status scope", "choice": "delivered orders",
                 "source": "user_stated", "proof": "delivered orders"},
                {"what": "status scope", "choice": "delivered",
                 "source": "user_chosen", "proof": "c1"},
                {"what": "date range", "choice": "all years",
                 "source": "only_option", "proof": "q2"},
            ])
            assert result["ok"] is True
            assert result["status"] == "answered"
            stored = ctx.record.submission["decisions"]
            assert [d["proof"] for d in stored] == ["c1", "delivered orders", "q2"]
            assert stored[0]["source"] == "user_chosen"
            assert stored[0]["choice"] == "delivered only"
        finally:
            backend.close()


def test_only_option_accepted_without_clarification():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            for item_id, value in (("total_revenue", 498440.43), ("top_region", "RJ")):
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": item_id, "claim": "c", "query_id": "q1", "value": value},
                )
            result = _submit(ctx, [
                {"what": "metric", "choice": "RJ",
                 "source": "only_option", "proof": "q2"},
            ])
            assert result["ok"] is True
        finally:
            backend.close()


def test_only_option_refused_when_clarified():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _decided_ctx(backend, tmp)
            result = _submit(ctx, [
                {"what": "status scope", "choice": "delivered orders",
                 "source": "user_stated", "proof": "delivered orders"},
                {"what": "revenue definition", "choice": "delivered only",
                 "source": "only_option", "proof": "q2"},
            ])
            assert "covered by clarification c1" in result["error"]
            assert "user_chosen" in result["error"]
            assert ctx.record.submission is None
        finally:
            backend.close()


def _proven_ctx(backend: SqliteBackend, tmp: str) -> ToolContext:
    """Flowed context with both items proven, ready to submit."""
    ctx = _flowed_ctx(backend, tmp)
    for item_id, value in (("total_revenue", 498440.43), ("top_region", "RJ")):
        dispatch(
            ctx,
            "record_finding",
            {"item_id": item_id, "claim": "c", "query_id": "q1", "value": value},
        )
    return ctx


def test_only_option_matches_definition_key_only():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _proven_ctx(backend, tmp)
            ctx.record.store_clarification(
                "Which orders should count toward the 2017 top category?",
                [{"label": "delivered only", "query_id": "q1", "evidence": "SP"}],
                "delivered only",
                "order_status",
            )
            for what in (
                "year date column",
                "category names",
                "only items with a matching product",
            ):
                result = _submit(ctx, [
                    {"what": what, "choice": "order_purchase_timestamp",
                     "source": "only_option", "proof": "q2"},
                ])
                assert result["ok"] is True, what
            for key, what, choice in (
                ("revenue", "revenue definition", "revenue"),
                ("year_date", "year date column", "order_purchase_timestamp"),
                ("top_metric", "top metric", "items sold"),
            ):
                fresh = _proven_ctx(backend, tmp)
                fresh.record.store_clarification("Which?", [], choice, key)
                refused = _submit(fresh, [
                    {"what": what, "choice": choice,
                     "source": "only_option", "proof": "q2"},
                ])
                assert "covered by clarification c1" in refused["error"], what
        finally:
            backend.close()


def test_only_option_needs_listing_query():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _proven_ctx(backend, tmp)
            agg = json.loads(dispatch(ctx, "run_sql", {"query": "SELECT SUM(revenue) FROM sales;"}))
            assert agg["query_id"] == "q3"
            refused = _submit(ctx, [
                {"what": "total", "choice": "sum",
                 "source": "only_option", "proof": "q3"},
            ])
            assert "lists the possible values" in refused["error"]
            grouped = json.loads(dispatch(ctx, "run_sql", {"query": "SELECT region, SUM(revenue) FROM sales GROUP BY region;"}))
            assert grouped["query_id"] == "q4"
            refused = _submit(ctx, [
                {"what": "region", "choice": "RJ",
                 "source": "only_option", "proof": "q4"},
            ])
            assert "shows exactly one" in refused["error"]
            single = json.loads(dispatch(ctx, "run_sql", {"query": "SELECT region, SUM(revenue) FROM sales WHERE region = 'RJ' GROUP BY region;"}))
            assert single["query_id"] == "q5"
            ok = _submit(ctx, [
                {"what": "region", "choice": "RJ",
                 "source": "only_option", "proof": "q5"},
            ])
            assert ok["ok"] is True
        finally:
            backend.close()


def test_keyless_clarifications_never_block_only_option():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            for item_id, value in (("total_revenue", 498440.43), ("top_region", "RJ")):
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": item_id, "claim": "c", "query_id": "q1", "value": value},
                )
            ctx.ask_user = lambda q, opts: {"choice": opts[0]["label"]}
            dispatch(ctx, "ask_clarification", {
                "question": "Which statuses count?",
                "options": CLARIFY_OPTIONS,
            })
            assert ctx.record.clarifications[0]["definition_key"] is None
            ctx.ask_user = lambda q, opts: {"user_question": "what about canceled?"}
            dispatch(ctx, "ask_clarification", {
                "question": "Which statuses count?",
                "options": CLARIFY_OPTIONS,
            })
            result = _submit(ctx, [
                {"what": "delivered scope", "choice": "delivered",
                 "source": "only_option", "proof": "q2"},
            ])
            assert result["ok"] is True
        finally:
            backend.close()


def test_user_stated_whole_question_refused():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _decided_ctx(backend, tmp)
            whole = "Revenue for delivered orders by region today?"
            result = _submit(ctx, [
                {"what": "scope", "choice": "everything",
                 "source": "user_stated", "proof": whole},
            ])
            assert "whole question" in result["error"]
            short = _ctx(backend, tmp, question="Total sales?")
            dispatch(short, "set_checklist", {"items": ITEMS})
            dispatch(short, "plan_action", {"plan": PLAN})
            dispatch(short, "run_sql", {"query": "SELECT 1;"})
            ok = json.loads(
                dispatch(short, "submit_answer", {
                    "answer": "Total is 1.",
                    "decisions": [
                        {"what": "scope", "choice": "everything",
                         "source": "user_stated", "proof": "Total sales?"},
                    ],
                    "gaps": [{"item_id": "total_revenue", "note": "x"},
                             {"item_id": "top_region", "note": "y"}],
                })
            )
            assert ok["ok"] is True
        finally:
            backend.close()


def test_decision_proof_failures_rejected():
    cases = [
        ([{"what": "s", "choice": "x", "source": "user_stated", "proof": "xyz"}],
         "quote"),
        ([{"what": "s", "choice": "x", "source": "user_stated", "proof": "abc"}],
         "at least 4 characters"),
        ([{"what": "s", "choice": "x", "source": "user_chosen", "proof": "c99"}],
         "unknown clarification"),
        ([{"what": "s", "choice": "x", "source": "only_option", "proof": "q99"}],
         "unknown query"),
        ([{"what": "s", "choice": "x", "source": "vibes", "proof": "q1"}],
         "unknown source"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            for decisions, hint in cases:
                ctx = _decided_ctx(backend, tmp)
                result = _submit(ctx, decisions)
                assert hint in result["error"], f"{decisions}: {result}"
                assert ctx.record.submission is None
        finally:
            backend.close()


def test_user_chosen_matches_user_question_reply():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            for item_id, value in (("total_revenue", 498440.43), ("top_region", "RJ")):
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": item_id, "claim": "c", "query_id": "q1", "value": value},
                )
            ctx.ask_user = lambda q, opts: {"user_question": "should canceled orders count?"}
            asked = json.loads(
                dispatch(ctx, "ask_clarification", {
                    "question": "Which statuses count?",
                    "options": CLARIFY_OPTIONS,
                })
            )
            assert asked["user_question"] == "should canceled orders count?"
            bad = _submit(ctx, [
                {"what": "status scope", "choice": "delivered",
                 "source": "user_chosen", "proof": "c1"},
            ])
            assert "does not match" in bad["error"]
            good = _submit(ctx, [
                {"what": "status scope", "choice": "canceled orders",
                 "source": "user_chosen", "proof": "c1"},
            ])
            assert good["ok"] is True
            assert good["rejections"] == 1
        finally:
            backend.close()


def test_escalation_marks_unconfirmed_decisions():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _decided_ctx(backend, tmp)
            bad = [
                {"what": "status scope", "choice": "x",
                 "source": "user_stated", "proof": "xyz"},
                {"what": "date range", "choice": "all years",
                 "source": "only_option", "proof": "q2"},
            ]
            last = {}
            for _ in range(3):
                last = _submit(ctx, bad)
            assert last["status"] == "escalated"
            assert "UNCONFIRMED DECISIONS: ['status scope']" in last["warning"]
            stored = ctx.record.submission["decisions"]
            assert stored[0]["proof"] == "c1"
            assert "unconfirmed" not in stored[0]
            assert stored[1].get("unconfirmed") is True
            assert "unconfirmed" not in stored[2]
        finally:
            backend.close()


def test_user_chosen_renders_key_with_spaces():
    submission = {
        "answer": "A.",
        "decisions": [
            {"what": "m", "choice": "sold", "source": "user_chosen", "proof": "c1"},
        ],
        "gaps": [],
    }
    clarifications = [
        {
            "id": "c1",
            "definition_key": "top_metric",
            "options": [{"label": "revenue"}, {"label": "items sold"}],
            "reply": "items sold",
        }
    ]
    assert render_answer(submission, clarifications) == (
        "A.\n\nDecisions:\n- top metric: items sold (you chose: option 2)\n\n"
        "Gaps:\n- none found\n\nWant the evidence behind these decisions?"
    )


def test_empty_decisions_accepted_with_clarifications():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _decided_ctx(backend, tmp)
            result = _submit(ctx, [])
            assert result["ok"] is True
            stored = ctx.record.submission["decisions"]
            assert len(stored) == 1
            assert stored[0]["source"] == "user_chosen"
            assert stored[0]["proof"] == "c1"
            assert stored[0]["choice"] == "delivered only"
        finally:
            backend.close()


def test_submit_rejects_empty_decisions():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            for bad in ([], [{"what": "scope"}]):
                result = json.loads(
                    dispatch(
                        ctx,
                        "submit_answer",
                        {"answer": "Total is 498440.43.", "decisions": bad, "gaps": []},
                    )
                )
                assert "decision" in result["error"]
            assert ctx.record.submission is None
        finally:
            backend.close()


def test_gap_named_item_passes_without_proof():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            dispatch(
                ctx,
                "record_finding",
                {"item_id": "total_revenue", "claim": "sum", "query_id": "q1", "value": 498440.43},
            )
            result = json.loads(
                dispatch(
                    ctx,
                    "submit_answer",
                    {
                        "answer": "Total is 498440.43.",
                        "decisions": DECISIONS,
                        "gaps": [{"item_id": "top_region", "note": "Need cost data."}],
                    },
                )
            )
            assert result["ok"] is True
        finally:
            backend.close()


def _message(text: str):
    return types.SimpleNamespace(
        type="message",
        content=[types.SimpleNamespace(type="output_text", text=text)],
    )


def _call(call_id: str, name: str, args: dict) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        type="function_call", call_id=call_id, name=name, arguments=json.dumps(args)
    )


def _response(*items) -> types.SimpleNamespace:
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


def _stub_backend(tmp: str) -> SqliteBackend:
    path = Path(tmp) / "stub.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY);")
        conn.executemany("INSERT INTO items (id) VALUES (?)", [(1,), (2,), (3,)])
        conn.commit()
    finally:
        conn.close()
    return SqliteBackend(path)


def test_loop_shows_accepted_submit_text():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = [
                _response(_call("c1", "set_checklist", {"items": [
                    {"id": "n", "question": "How many?", "evidence_needed": "count"}]})),
                _response(_call("c2", "plan_action", {"plan": [
                    {"item_id": "n", "steps": ["count"]}]})),
                _response(_call("c3", "run_sql", {"query": "SELECT count(*) FROM items;"})),
                _response(_call("c4", "record_finding", {
                    "item_id": "n", "claim": "3 items", "query_id": "q1", "value": 3})),
                _response(_call("c5", "submit_answer", {
                    "answer": "There are 3 items.",
                    "decisions": LOOP_DECISIONS,
                    "gaps": []})),
                _response(_message("WRONG TEXT")),
            ]
            history: list = []
            result = answer_question(
                _StubClient(script), "m", "instr", backend, history, "How many?",
                runs_dir=Path(tmp) / "runs",
            )
            expected = (
                "There are 3 items.\n\nDecisions:\n- scope: everything (you stated: \"How many?\")\n\n"
                "Gaps:\n- none found\n\n"
                "Want the evidence behind these decisions?"
            )
            assert result.text == expected
            assert any(
                m.get("content", "") == expected
                for m in history
                if m.get("role") == "assistant"
            )
            assert result.verification == "verification: 1 numbers matched, 0 unmatched"
            assert result.record_path is not None
        finally:
            backend.close()


def test_loop_reminds_when_plain_text_without_submit():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = [
                _response(
                    _call("c1", "set_checklist", {"items": [
                        {"id": "n", "question": "How many?", "evidence_needed": "count"}]}),
                    _call("c2", "plan_action", {"plan": [
                        {"item_id": "n", "steps": ["count"]}]})),
                _response(_message("The answer is 3 items.")),
                _response(_call("c3", "run_sql", {"query": "SELECT count(*) FROM items;"})),
                _response(_call("c4", "record_finding", {
                    "item_id": "n", "claim": "3 items", "query_id": "q1", "value": 3})),
                _response(_call("c5", "submit_answer", {
                    "answer": "There are 3 items.",
                    "decisions": LOOP_DECISIONS,
                    "gaps": []})),
                _response(_message("done")),
            ]
            history: list = []
            result = answer_question(
                _StubClient(script), "m", "instr", backend, history, "How many?",
                runs_dir=Path(tmp) / "runs",
            )
            expected = (
                "There are 3 items.\n\nDecisions:\n- scope: everything (you stated: \"How many?\")\n\n"
                "Gaps:\n- none found\n\n"
                "Want the evidence behind these decisions?"
            )
            assert result.text == expected
            # The reminder did its job (submit was accepted), then the turn's
            # tool traffic — including the reminder — was compacted away.
            assert all(
                m.get("type") not in ("function_call", "function_call_output")
                for m in history
            )
            assert len(history) == 3
            assert history[0] == {"role": "user", "content": "How many?"}
            assert "Work summary" in history[1].get("content", "")
            assert history[2].get("content", "") == expected
        finally:
            backend.close()


def test_loop_plain_text_without_record():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = [_response(_message("Yes, all statuses."))]
            result = answer_question(
                _StubClient(script), "m", "instr", backend, [], "yes?",
                runs_dir=Path(tmp) / "runs",
            )
            assert result.text == "Yes, all statuses."
            assert result.record_path is None
            assert result.verification is None
        finally:
            backend.close()




def test_acceptance_on_final_round_returns_answer_not_cap():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = [
                _response(
                    _call("c1", "set_checklist", {"items": [
                        {"id": "n", "question": "How many?", "evidence_needed": "count"}]}),
                    _call("c2", "plan_action", {"plan": [
                        {"item_id": "n", "steps": ["count"]}]})),
            ]
            for i in range(1, 28):
                script.append(_response(_call(f"g{i}", "get_schema", {"table": "items"})))
            script += [
                _response(
                    _call("c29a", "run_sql", {"query": "SELECT count(*) FROM items;"}),
                    _call("c29b", "record_finding", {
                        "item_id": "n", "claim": "3 items", "query_id": "q1", "value": 3}),
                ),
                _response(_call("c30", "submit_answer", {
                    "answer": "There are 3 items.",
                    "decisions": LOOP_DECISIONS,
                    "gaps": []})),
            ]
            assert len(script) == MAX_TURNS
            client = _StubClient(script)
            result = answer_question(
                client, "m", "instr", backend, [], "How many?",
                runs_dir=Path(tmp) / "runs",
            )
            assert result.text == (
                "There are 3 items.\n\nDecisions:\n- scope: everything (you stated: \"How many?\")\n\n"
                "Gaps:\n- none found\n\n"
                "Want the evidence behind these decisions?"
            )
            assert "Reached the" not in result.text
            assert client.calls == MAX_TURNS
            assert result.verification == "verification: 1 numbers matched, 0 unmatched"
        finally:
            backend.close()


def test_no_calls_after_early_acceptance():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = [
                _response(_call("c1", "set_checklist", {"items": [{"id": "n", "question": "How many?", "evidence_needed": "count"}]})),
                _response(_call("c2", "plan_action", {"plan": [{"item_id": "n", "steps": ["count"]}]})),
                _response(_call("c3", "get_schema", {"table": "items"})),
                _response(_call("c4", "run_sql", {"query": "SELECT count(*) FROM items;"})),
                _response(_call("c5", "record_finding", {"item_id": "n", "claim": "3 items", "query_id": "q1", "value": 3})),
                _response(_call("c6", "submit_answer", {"answer": "There are 3 items.", "decisions": LOOP_DECISIONS, "gaps": []})),
            ]
            assert len(script) == 6
            client = _StubClient(script)
            result = answer_question(
                client, "m", "instr", backend, [], "How many?",
                runs_dir=Path(tmp) / "runs",
            )
            assert result.text == (
                "There are 3 items.\n\nDecisions:\n- scope: everything (you stated: \"How many?\")\n\n"
                "Gaps:\n- none found\n\n"
                "Want the evidence behind these decisions?"
            )
            assert client.calls == 6
        finally:
            backend.close()


def _accepted_answer_script(prefix: str, question: str = "How many?"):
    return [
        _response(_call(prefix + "1", "set_checklist", {"items": [
            {"id": "n", "question": question, "evidence_needed": "count"}]})),
        _response(_call(prefix + "2", "plan_action", {"plan": [
            {"item_id": "n", "steps": ["count"]}]})),
        _response(_call(prefix + "3", "run_sql", {"query": "SELECT count(*) FROM items;"})),
        _response(_call(prefix + "4", "record_finding", {
            "item_id": "n", "claim": "3 items", "query_id": "q1", "value": 3})),
        _response(_call(prefix + "5", "submit_answer", {
            "answer": "There are 3 items.",
            "decisions": [
                {"what": "scope", "choice": "everything",
                 "source": "user_stated", "proof": question},
            ],
            "gaps": []})),
    ]


def test_history_compacted_after_answer():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            history: list = []
            result = answer_question(
                _StubClient(_accepted_answer_script("c")), "m", "instr",
                backend, history, "How many?",
                runs_dir=Path(tmp) / "runs",
            )
            assert all(
                m.get("type") not in ("function_call", "function_call_output")
                for m in history
            )
            assert len(history) == 3
            assert history[0] == {"role": "user", "content": "How many?"}
            summary = history[1].get("content", "")
            assert "q1" in summary
            assert "SELECT count(*) FROM items;" in summary
            assert "3 items" in summary
            assert history[2].get("content", "") == result.text
            assert "Decisions:" in result.text
        finally:
            backend.close()


def test_history_stays_small_across_questions():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = (
                _accepted_answer_script("a", "First?")
                + _accepted_answer_script("b", "Second?")
                + _accepted_answer_script("c", "Third?")
            )
            client = _StubClient(script)
            history: list = []
            for question in ("First?", "Second?", "Third?"):
                answer_question(
                    client, "m", "instr", backend, history, question,
                    runs_dir=Path(tmp) / "runs",
                )
            assert len(history) == 9
            assert all(
                m.get("type") not in ("function_call", "function_call_output")
                for m in history
            )
            assert [m.get("content") for m in history if m.get("role") == "user"] == [
                "First?",
                "Second?",
                "Third?",
            ]
        finally:
            backend.close()


def test_clarification_rounds_do_not_use_cap(monkeypatch):
    monkeypatch.setattr("agent.loop.MAX_TURNS", 5)
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            ask = {
                "question": "Which rows count?",
                "options": [
                    {"label": "all counted rows", "query_id": "q1", "evidence": 3,
                     "explanation": "counts every row"},
                    {"label": "only these rows", "query_id": "q1", "evidence": 3,
                     "explanation": "counts these rows"},
                ],
                "definition_key": "row_scope",
            }
            script = [
                _response(_call("c1", "set_checklist", {"items": [
                    {"id": "n", "question": "How many?", "evidence_needed": "count"}]})),
                _response(_call("c2", "plan_action", {"plan": [
                    {"item_id": "n", "steps": ["count"]}]})),
                _response(_call("c3", "run_sql", {"query": "SELECT count(*) FROM items;"})),
                _response(_call("c4", "ask_clarification", dict(ask))),
                _response(_call("c5", "ask_clarification", dict(ask))),
                _response(_call("c6", "record_finding", {
                    "item_id": "n", "claim": "3 items", "query_id": "q1", "value": 3})),
                _response(_call("c7", "submit_answer", {
                    "answer": "There are 3 items.",
                    "decisions": [
                        {"what": "scope", "choice": "everything",
                         "source": "user_stated", "proof": "How many?"},
                    ],
                    "gaps": []})),
            ]
            client = _StubClient(script)
            result = answer_question(
                client, "m", "instr", backend, [], "How many?",
                runs_dir=Path(tmp) / "runs",
                ask_user=lambda q, opts: {"choice": opts[0]["label"]},
            )
            assert "Reached the" not in result.text
            assert "There are 3 items." in result.text
            assert client.calls == 7
        finally:
            backend.close()


def test_wrap_up_nudge_sent_with_three_rounds_left(monkeypatch):
    monkeypatch.setattr("agent.loop.MAX_TURNS", 6)
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = [
                _response(_call("c1", "set_checklist", {"items": [
                    {"id": "n", "question": "How many?", "evidence_needed": "count"}]})),
                _response(_call("c2", "plan_action", {"plan": [
                    {"item_id": "n", "steps": ["count"]}]})),
                _response(_call("c3", "run_sql", {"query": "SELECT count(*) FROM items;"})),
            ]
            history: list = []
            result = answer_question(
                _StubClient(script), "m", "instr", backend, history, "How many?",
                runs_dir=Path(tmp) / "runs",
            )
            assert result.record_path is not None
            assert any(
                m.get("role") == "user" and m.get("content") == WRAP_UP_NUDGE
                for m in history
            )
        finally:
            backend.close()


def test_cap_shows_partial_summary():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = [
                _response(_call("c1", "set_checklist", {"items": [
                    {"id": "n", "question": "How many?", "evidence_needed": "count"}]})),
                _response(_call("c2", "plan_action", {"plan": [
                    {"item_id": "n", "steps": ["count"]}]})),
                _response(_call("c3", "run_sql", {"query": "SELECT count(*) FROM items;"})),
                _response(_call("c4", "record_finding", {
                    "item_id": "n", "claim": "3 items", "query_id": "q1", "value": 3})),
            ]
            for i in range(5, 17):
                script.append(_response(_call(f"g{i}", "run_sql", {
                    "query": "SELECT count(*) FROM items;"})))
            assert len(script) == 16
            history: list = []
            result = answer_question(
                _StubClient(script), "m", "instr", backend, [], "How many?",
                runs_dir=Path(tmp) / "runs",
            )
            assert f"Reached the {MAX_TURNS}-turn limit" in result.text
            assert "- n: proven" in result.text
            assert "- 3 items (3)" in result.text
            assert f"Queries run: {MAX_TURNS - 3}" in result.text
            assert "Record: " in result.text
            assert result.verification == "verification: no submitted answer"
        finally:
            backend.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"all {len(tests)} tests passed")
