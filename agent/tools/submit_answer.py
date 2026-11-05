"""Submit the final answer; every number must trace to recorded data."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from agent.numbers import verify_text
from agent.record import STATUS_ESCALATED

if TYPE_CHECKING:
    from agent.tools import ToolContext

NAME = "submit_answer"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": NAME,
    "description": (
        "Submit the final answer. Rejected unless every checklist item is "
        "proven or named in gaps, and every number in the answer appears in "
        "a recorded query result or the user's question. Answered "
        "clarifications are added as user_chosen decisions automatically; "
        "do not resubmit them. A rejection lists "
        "the unmatched numbers so the answer can be fixed and resubmitted. "
        "After 3 rejections the answer is accepted but marked escalated."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The full answer text for the user.",
            },
            "decisions": {
                "type": "array",
                "description": (
                    "Every choice that could change a number: what it is, "
                    "the choice made, its source and its proof."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "what": {
                            "type": "string",
                            "description": "What was decided.",
                        },
                        "choice": {
                            "type": "string",
                            "description": "The choice made.",
                        },
                        "source": {
                            "type": "string",
                            "description": "One of user_stated, user_chosen, only_option.",
                        },
                        "proof": {
                            "type": "string",
                            "description": (
                                "Quote from the question, clarification id, "
                                "or recorded query id."
                            ),
                        },
                    },
                    "required": ["what", "choice", "source", "proof"],
                    "additionalProperties": False,
                },
            },
            "gaps": {
                "type": "array",
                "description": "Items the data cannot answer, each naming its checklist item.",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {
                            "type": "string",
                            "description": "Checklist id, or empty for a general gap.",
                        },
                        "note": {
                            "type": "string",
                            "description": "What is missing and what is needed instead.",
                        },
                    },
                    "required": ["item_id", "note"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["answer", "decisions", "gaps"],
        "additionalProperties": False,
    },
    "strict": True,
}

_MISSING_RECORD_ERROR = "submit_answer is refused: call set_checklist first."
_MAX_REJECTIONS = 3

_WORD_RE = re.compile(r"[a-z0-9]+")
_LISTING_RE = re.compile(r"\bgroup\s+by\b|\bdistinct\b", re.IGNORECASE)

_IGNORED_KEY_TOKENS = frozenset({"only", "the", "and", "for"})


def _auto_decisions(record) -> list[dict[str, Any]]:
    """Build user_chosen decisions from answered clarifications, in code.

    The model must not need to supply these; a model-supplied duplicate
    (same proof id) is dropped by the caller.
    """
    auto: list[dict[str, Any]] = []
    for clarification in record.clarifications:
        if clarification.get("question") is None:
            continue
        key = clarification.get("definition_key")
        question = clarification.get("question") or ""
        what = f"{key}: {question}" if key else question
        auto.append(
            {
                "what": what,
                "choice": clarification.get("reply", ""),
                "source": "user_chosen",
                "proof": clarification["id"],
            }
        )
    return auto


def _key_tokens(text: str) -> set[str]:
    return {
        token
        for token in _WORD_RE.findall(text.lower())
        if len(token) >= 3 and token not in _IGNORED_KEY_TOKENS
    }


def _covering_clarification(record, what: str) -> str | None:
    """Id of the first keyed clarification overlapping the what text, if any.

    Only the clarification's definition_key (split on "_") is compared
    against the decision's what text. Clarifications without a key are
    skipped, as are choice, question, options and reply.
    """
    topic = _key_tokens(what)
    if not topic:
        return None
    for clarification in record.clarifications:
        key = clarification.get("definition_key") or ""
        if not key.strip():
            continue
        parts = {
            part
            for part in key.lower().split("_")
            if len(part) >= 3 and part not in _IGNORED_KEY_TOKENS
        }
        if topic & parts:
            return clarification["id"]
    return None

_SOURCES = ("user_stated", "user_chosen", "only_option")


def _validate_decisions(
    decisions: Any, record
) -> tuple[list[dict[str, Any]] | None, list[int], str | None]:
    """Check each decision's proof. Return (normalized, failed, error).

    normalized holds every submitted decision in shape; failed lists the
    indexes whose proof check failed so escalation can mark them
    UNCONFIRMED instead of storing a bare rejection.
    """
    if not isinstance(decisions, list) or not decisions:
        return [], [], "submit_answer needs a non-empty list of decisions"
    known_clarifications = {c["id"]: c for c in record.clarifications}
    known_queries = {q["id"] for q in record.queries}
    normalized: list[dict[str, Any]] = []
    failed: list[int] = []
    first_error = ""
    for i, entry in enumerate(decisions):
        if not isinstance(entry, dict):
            normalized.append({"what": "?", "choice": "?", "source": "?", "proof": "?"})
            failed.append(i)
            if len(failed) == 1:
                first_error = f"decision {i} must be an object with what, choice, source, proof"
            continue
        what = entry.get("what")
        choice = entry.get("choice")
        source = entry.get("source")
        proof = entry.get("proof")
        if (
            not isinstance(what, str)
            or not what.strip()
            or not isinstance(choice, str)
            or not choice.strip()
            or not isinstance(proof, str)
            or not proof.strip()
        ):
            normalized.append(
                {
                    "what": what if isinstance(what, str) else "?",
                    "choice": choice if isinstance(choice, str) else "?",
                    "source": source if isinstance(source, str) else "?",
                    "proof": proof if isinstance(proof, str) else "?",
                }
            )
            failed.append(i)
            if len(failed) == 1:
                first_error = f"decision {i} needs non-empty what, choice and proof"
            continue
        problem: str | None = None
        if source not in _SOURCES:
            problem = (
                f"unknown source {source!r}; use user_stated, user_chosen "
                "or only_option, or ask via ask_clarification"
            )
        elif source == "user_stated":
            quote = proof.strip()
            if len(quote) < 4 or quote.lower() not in record.question.lower():
                problem = (
                    "user_stated proof must be a quote of at least 4 characters "
                    "from the user's question"
                )
            elif (
                quote.lower() == record.question.strip().lower()
                and len(record.question.split()) > 6
            ):
                problem = (
                    "user_stated proof must be the stating words, not the whole question"
                )
        elif source == "user_chosen":
            clarification = known_clarifications.get(proof.strip())
            if clarification is None:
                problem = (
                    f"unknown clarification {proof.strip()!r}; "
                    "ask via ask_clarification"
                )
            else:
                reply = clarification["reply"].strip().lower()
                made = choice.strip().lower()
                if reply != made and reply not in made and made not in reply:
                    problem = (
                        f"choice does not match clarification {proof.strip()}'s reply; "
                        "ask via ask_clarification"
                    )
        elif proof.strip() not in known_queries:
            problem = f"unknown query {proof.strip()!r}"
        else:
            cited = next(
                q for q in record.queries if q["id"] == proof.strip()
            )
            if not _LISTING_RE.search(cited.get("sql", "")) or len(cited.get("rows", [])) != 1:
                problem = (
                    "only_option needs a query that lists the possible values "
                    "and shows exactly one; ask via ask_clarification or cite "
                    "the user's words"
                )
            else:
                covering = _covering_clarification(record, what.strip())
                if covering is not None:
                    problem = (
                        f"covered by clarification {covering}; cite it as user_chosen "
                        "instead of only_option"
                    )
        normalized.append(
            {
                "what": what.strip(),
                "choice": choice.strip(),
                "source": source if isinstance(source, str) else "?",
                "proof": proof.strip(),
            }
        )
        if problem is not None:
            failed.append(i)
            if len(failed) == 1:
                first_error = f"decision {i} ({what.strip()}): {problem}"
    if failed:
        return normalized, failed, first_error
    return normalized, [], None


def _validate_gaps(gaps: Any, item_ids: list[str]) -> list[dict[str, str]] | str:
    """Return cleaned gaps, or an error string."""
    if not isinstance(gaps, list):
        return "gaps must be a list of {item_id, note}"
    known = set(item_ids)
    cleaned: list[dict[str, str]] = []
    for i, entry in enumerate(gaps):
        if not isinstance(entry, dict):
            return f"gaps entry {i} must be an object with item_id and note"
        item_id = entry.get("item_id", "")
        if not isinstance(item_id, str):
            return f"gaps entry {i} has a bad item_id"
        if item_id and item_id not in known:
            return f"gaps entry {i} names unknown checklist item {item_id!r}"
        note = entry.get("note")
        if not isinstance(note, str) or not note.strip():
            return f"gaps entry {i} needs a non-empty note"
        cleaned.append({"item_id": item_id, "note": note.strip()})
    return cleaned


def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    record = ctx.record
    if record is None:
        return json.dumps({"error": _MISSING_RECORD_ERROR})
    answer = args.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return json.dumps({"error": "submit_answer needs a non-empty answer"})
    auto = _auto_decisions(record)
    raw = args.get("decisions", [])
    if isinstance(raw, list):
        auto_ids = {d["proof"] for d in auto}
        raw = [
            d
            for d in raw
            if not (
                isinstance(d, dict)
                and d.get("source") == "user_chosen"
                and d.get("proof") in auto_ids
            )
        ]
    if isinstance(raw, list) and not raw and auto:
        combined, failed, error = auto, [], None
    else:
        decisions, failed, error = _validate_decisions(raw, record)
        combined = auto + (decisions or [])
    if error is not None:
        return _reject(
            record, error, [], answer, combined, [],
            [i + len(auto) for i in failed],
        )
    gaps = _validate_gaps(args.get("gaps", []), record.item_ids())
    if isinstance(gaps, str):
        return json.dumps({"error": gaps})
    gap_items = {g["item_id"] for g in gaps if g["item_id"]}
    uncovered = [
        item_id
        for item_id in record.item_ids()
        if not record.is_proven(item_id) and item_id not in gap_items
    ]
    if uncovered:
        reason = f"uncovered checklist items: {uncovered}"
        return _reject(record, reason, [], answer, combined, gaps)
    cells = [v for q in record.queries for row in q["rows"] for v in row]
    matched, unmatched = verify_text(answer, record.question, cells)
    if unmatched:
        reason = f"unmatched numbers: {unmatched}"
        return _reject(record, reason, unmatched, answer, combined, gaps)
    result = record.submit(answer, combined, gaps, matched, unmatched)
    return json.dumps(result)


def _reject(
    record,
    reason: str,
    unmatched: list[str],
    answer: str,
    decisions: list[dict[str, Any]] | None,
    gaps: list[dict[str, str]],
    failed: list[int] | None = None,
) -> str:
    """Count the rejection; escalate-accept once the limit is reached."""
    total = record.note_rejection(reason, unmatched)
    if total >= _MAX_REJECTIONS:
        failed = failed or []
        stored = []
        for i, decision in enumerate(decisions or []):
            marked = dict(decision)
            if i in failed:
                marked["unconfirmed"] = True
            stored.append(marked)
        result = record.submit(
            answer,
            decisions=stored,
            gaps=list(gaps),
            matched=[],
            unmatched=list(unmatched),
            status=STATUS_ESCALATED,
        )
        warnings = [f"UNVERIFIED NUMBERS: {unmatched or reason}"]
        whats = [d["what"] for d in stored if d.get("unconfirmed")]
        if whats:
            warnings.append(f"UNCONFIRMED DECISIONS: {whats}")
        return json.dumps({**result, "warning": " ; ".join(warnings)})
    return json.dumps({"error": reason, "unmatched": unmatched, "rejections": total})
