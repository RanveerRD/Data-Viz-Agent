"""Live provider smoke test on synthetic data. Run by hand with a key.

    MODEL_PROVIDER=gemini GEMINI_API_KEY=... python scripts/provider_smoke.py

(or put the settings in .env). The key is never printed. Uses a tiny
synthetic SQLite database created in a temp folder, so no real data leaves
your machine. It runs the full agent loop once, so it exercises tool
calling, the verification layer and token accounting, and checks the answer
against a known result: 3 delivered orders totalling 450.

Exit code 0 = the provider completed the question with correct figures,
1 = it failed or got the figures wrong. Read the printed notes either way:
a weak model can pass tool calling and still fail verification.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.__main__ import _base_instructions  # noqa: E402
from agent.backend import SqliteBackend  # noqa: E402
from agent.loop import answer_question  # noqa: E402
from agent.provider import build_client, get_provider_name  # noqa: E402

QUESTION = "How many orders have status delivered, and what is their total amount?"
EXPECTED = ("3", "450")


def fail(message: str) -> int:
    print(f"SMOKE FAIL: {message}")
    return 1


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE orders (id INTEGER, status TEXT, amount REAL)")
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?)",
        [(1, "delivered", 100), (2, "delivered", 150), (3, "delivered", 200),
         (4, "canceled", 80), (5, "shipped", 60)],
    )
    conn.commit()
    conn.close()


def _auto_pick_first(question: str, options: list[dict]) -> dict[str, str]:
    """Answer any clarification with option 1 so the run can finish."""
    print(f"  (clarification asked: {question!r}; picking option 1)")
    return {"choice": "1"}


def main() -> int:
    try:
        provider = get_provider_name()
        client, model, _ = build_client()
        instructions = _base_instructions("SQLite")
    except RuntimeError as exc:
        return fail(str(exc))
    print(f"provider={provider} model={model}")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "smoke.sqlite"
        _make_db(db)
        backend = SqliteBackend(db)
        try:
            result = answer_question(
                client, model, instructions, backend, [], QUESTION,
                runs_dir=Path(tmp) / "runs", ask_user=_auto_pick_first,
                dataset="smoke", allow_charts=False, provider=provider,
            )
        except Exception as exc:
            return fail(f"{type(exc).__name__}: {str(exc)[:300]}")
        finally:
            backend.close()
    print(f"answer: {result.text[:400]!r}")
    if getattr(result, "verification", None):
        print(f"verification: {result.verification}")
    if result.usage_known:
        print(
            f"tokens: {result.input_tokens} in / {result.output_tokens} out / "
            f"{result.total_tokens} total"
        )
    else:
        print("tokens: n/a")
    text = result.text.replace(",", "")
    missing = [value for value in EXPECTED if value not in text]
    if not result.text:
        return fail("empty answer")
    if missing:
        return fail(f"answer lacks expected figures {missing} (3 orders, 450)")
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
