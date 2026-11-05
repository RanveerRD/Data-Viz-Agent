"""Number extraction and matching for answer verification.

Pure functions with no dependencies, so the verification rules are testable
without the API or a database.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

# A number with optional commas, decimals, and a trailing percent sign.
# A trailing dot only counts when a digit follows it, so list numbering
# like "1." is never extracted. The sign is handled separately: a hyphen
# counts as minus only at the start or after whitespace or "(".
_NUMBER_RE = re.compile(r"(?:\d[\d,]*)(?:\.\d+)?%?")
_CURRENCY_PREFIX_RE = re.compile(r"^[$€£]")
_TRAILING_NON_DIGIT_RE = re.compile(r"[.%]$")

# ISO dates, year-months, and clock times are not number claims. Full dates
# come first so a YYYY-MM inside a date is consumed as part of it. Month,
# day, hour, and minute/second ranges are validated so values like
# "2017-2018" or "25:99" fall through to the normal number rules instead
# of being silently masked.
_MASK_RE = re.compile(
    r"\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
    r"|\b\d{4}-(?:0[1-9]|1[0-2])\b"
    r"|\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b"
)


def _mask_dates_and_times(text: str) -> str:
    """Blank out date/time spans with spaces, preserving positions."""

    def _spaces(match: re.Match[str]) -> str:
        return " " * (match.end() - match.start())

    return _MASK_RE.sub(_spaces, text)


_WORD_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
)


def _is_word_char(ch: str) -> bool:
    return ch in _WORD_CHARS


def extract_numbers(text: str) -> list[str]:
    """Raw number tokens in the order they appear, commas and % kept.

    List numbering like "1." at the start of a line is not a number claim
    and is skipped; a trailing period anywhere else is sentence punctuation.
    A hyphen between two digits is a range separator, not a minus sign.
    A digit run attached to a letter or underscore on the left is part of
    an identifier, not a number. On the right, only two or more letters or
    underscores disqualify the run, so a single suffix like M or % still
    extracts (and gets rejected downstream, pushing full numbers).
    """
    text = _mask_dates_and_times(text)
    tokens: list[str] = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        start = match.start()
        end = match.end()
        if start > 0 and _is_word_char(text[start - 1]):
            continue
        if (
            start > 0
            and text[start - 1] == "-"
            and (start - 1 == 0 or text[start - 2] in " \t\n\r\f\v(")
        ):
            token = "-" + token
        following = text[end : end + 2]
        if len(following) == 2 and all(_is_word_char(c) for c in following):
            continue
        line_start = text.rfind("\n", 0, start) + 1
        if (
            token.isdigit()
            and end < len(text)
            and text[end] == "."
            and not text[line_start:start].strip()
        ):
            continue
        tokens.append(token)
    return tokens


def _mask_recorded_strings(text: str, cells: list[object]) -> str:
    """Blank out exact occurrences of long letter-bearing string cells.

    Only cells of at least 4 characters containing a letter are masked, and
    only where the occurrence does not touch another letter, digit, or
    underscore. Positions are preserved with spaces.
    """
    strings = sorted(
        {
            c
            for c in cells
            if isinstance(c, str) and len(c) >= 4 and any(ch.isalpha() for ch in c)
        },
        key=len,
        reverse=True,
    )
    for s in strings:
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(s) + r"(?![A-Za-z0-9_])"
        )
        text = pattern.sub(lambda m: " " * (m.end() - m.start()), text)
    return text


def normalize_token(token: str) -> str | None:
    """Normalize a token to a plain numeric string, or None to ignore it.

    Integers from 1900 to 2100 are treated as years and ignored.
    """
    token = _CURRENCY_PREFIX_RE.sub("", token).replace(",", "")
    token = _TRAILING_NON_DIGIT_RE.sub("", token)
    if not token or token == "-":
        return None
    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    if value == value.to_integral_value() and 1900 <= value <= 2100:
        return None
    return format(value.normalize(), "f") if "." in token else str(value)


def _parse_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def match_recorded(answer_token: str, cells: list[object]) -> bool:
    """True when a recorded cell matches the answer number.

    The recorded value is rounded (half up) to the answer's decimal places
    first, so 498,440 matches a recorded 498440.43.
    """
    normalized = (normalize_token(answer_token) or "").replace(",", "")
    answer_value = _parse_decimal(normalized)
    if answer_value is None:
        return False
    decimals = -answer_value.as_tuple().exponent if answer_value.as_tuple().exponent < 0 else 0
    quantum = Decimal(1).scaleb(-decimals)
    for cell in cells:
        if isinstance(cell, bool):
            continue
        candidate = _parse_decimal(str(cell).replace(",", ""))
        if candidate is None:
            continue
        if candidate.quantize(quantum, rounding=ROUND_HALF_UP) == answer_value:
            return True
    return False


def match_question(answer_token: str, question_numbers: list[str]) -> bool:
    """True when the answer number is echoed from the user's question."""
    normalized = normalize_token(answer_token)
    if normalized is None:
        return False
    wanted = normalized.replace(",", "")
    for raw in question_numbers:
        candidate = normalize_token(raw)
        if candidate is not None and candidate.replace(",", "") == wanted:
            return True
    return False


def verify_text(
    answer: str, question: str, recorded_cells: list[object]
) -> tuple[list[str], list[str]]:
    """Split the answer's numbers into (matched, unmatched) raw tokens.

    Substrings that exactly equal a recorded string value are masked first,
    so identifiers quoted from results verify cleanly.
    """
    question_numbers = extract_numbers(question)
    answer = _mask_recorded_strings(answer, recorded_cells)
    matched: list[str] = []
    unmatched: list[str] = []
    for token in extract_numbers(answer):
        if normalize_token(token) is None:
            continue
        if match_recorded(token, recorded_cells) or match_question(
            token, question_numbers
        ):
            matched.append(token)
        else:
            unmatched.append(token)
    return matched, unmatched
