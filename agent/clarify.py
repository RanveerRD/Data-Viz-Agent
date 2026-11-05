"""Clarification prompting shared by the console and WhatsApp front ends.

interpret_clarification_reply is pure: the console loops it with
print/input, WhatsApp loops it across messages. Same semantics both sides:

- a valid option number picks that option's label;
- "0" means the next reply is free text (the asker shows "type your answer:");
- an empty reply picks the last-used option when one is marked, else reprompts;
- bare out-of-range digits reprompt;
- anything else is a question about the options.
"""

from __future__ import annotations

from typing import Any


def _format_evidence(value: object) -> str:
    """Format one evidence value; numbers get thousands separators."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,}"
    return str(value)


def format_options_text(question: str, options: list[dict[str, Any]]) -> str:
    """Render a clarification as numbered plain text."""
    lines = [f"clarification: {question}"]
    for i, option in enumerate(options, 1):
        marker = " (last used)" if option.get("last_used") else ""
        lines.append(
            f"  {i}. {option['label']} [{_format_evidence(option['evidence'])}]{marker}"
        )
        lines.append(f"     {option['explanation']}")
    lines.append("  0. something else (type your own)")
    return "\n".join(lines)


def interpret_clarification_reply(
    reply: str, options: list[dict[str, Any]]
) -> tuple[str, str]:
    """Classify one clarification reply.

    Returns ("choice", label), ("ask", prompt_text) meaning show the
    prompt and read again, or ("question", text).
    """
    labels = [option["label"] for option in options]
    text = reply.strip()
    if text == "0":
        return ("ask", "type your answer:")
    try:
        index = int(text)
    except ValueError:
        index = -1
    if 1 <= index <= len(labels):
        return ("choice", labels[index - 1])
    if text == "":
        for option in options:
            if option.get("last_used"):
                return ("choice", option["label"])
        return ("ask", f"pick 1 to {len(labels)}, or 0 to type your own")
    if text.isdigit():
        return ("ask", f"pick 1 to {len(labels)}, or 0 to type your own")
    return ("question", text)
