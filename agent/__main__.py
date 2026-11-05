"""Console entry point: .venv/bin/python -m agent."""

from __future__ import annotations

import argparse
import shlex
from functools import partial
from typing import Any

from agent.clarify import (
    _format_evidence,
    format_options_text,
    interpret_clarification_reply,
)
from agent.datasets import REGISTRY, active_datasets
from agent.loop import REPO_ROOT, answer_question
from agent.provider import (
    BASETEN_PRICE_AS_OF,
    build_client,
    estimate_cost_usd,
    privacy_note,
)
from agent.record import AnalysisRecord
from agent.preliminary_analysis import (
    DEFAULT_DATASET,
    MEMORY_DIR,
    DefinitionStore,
    approve_definition,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse console flags; --dataset picks the database and memory store."""
    parser = argparse.ArgumentParser(description="Ask questions about a dataset.")
    parser.add_argument(
        "--dataset",
        choices=[*REGISTRY, "all"],
        default=next(iter(REGISTRY)),
        help="which dataset to query (default: first registry entry)",
    )
    return parser.parse_args(argv)


def _base_instructions(dialect: str) -> str:
    try:
        text = (REPO_ROOT / "agent" / "instructions.md").read_text()
    except OSError as exc:
        raise RuntimeError(f"cannot read instructions: {exc}") from exc
    return text.replace("{dialect}", dialect)


def _instructions_for_question(
    base: str, dataset: str = DEFAULT_DATASET, dialect_notes: str = ""
) -> str:
    """Base instructions plus dialect notes and this question's definitions."""
    instructions = base
    if dialect_notes:
        instructions += "\n\nDialect notes:\n" + dialect_notes
    section = DefinitionStore.load(dataset).build_prompt_section()
    if not section:
        return instructions
    return instructions + "\n\n" + section


def _print_datasets(current: str) -> None:
    """List every registry dataset; no model call."""
    for name, entry in REGISTRY.items():
        marker = " (current)" if name == current else ""
        print(f"{name} ({entry['label']}): {entry['description']}{marker}")
    if current == "all":
        print("asking across all datasets (current)")
    print("Run with --dataset <name> or --dataset all")


def _parse_n_and_type(tokens: list[str], usage: str) -> tuple[Any, str | None]:
    """Parse '[n] [bar|line]' args shared by /chart and /publish."""
    number: Any = 1
    override: str | None = None
    for token in tokens:
        if token in ("bar", "line"):
            override = token
        else:
            try:
                number = int(token)
            except ValueError:
                print(usage)
                return None, None
    return number, override


def _candidate_x_labels(record: Any, candidate: dict) -> list[str]:
    """X labels of the candidate's plottable rows, reloaded from the record."""
    query = next((q for q in record.queries if q["id"] == candidate["query_id"]), None)
    columns = list(query.get("columns", [])) if query else []
    xi = columns.index(candidate["x"]) if candidate["x"] in columns else -1
    yi = columns.index(candidate["y"]) if candidate["y"] in columns else -1
    if not query or xi < 0 or yi < 0:
        return []
    return [str(r[xi]) for r in query["rows"] if r[yi] is not None]


def _line_allowed(record: Any, candidate: dict) -> bool:
    """A line needs date-like x with at least 3 plottable rows."""
    from agent.tools.make_chart import _is_date_series

    labels = _candidate_x_labels(record, candidate)
    return len(labels) >= 3 and _is_date_series(labels)


def _run_chart_command(line: str, pending: dict | None) -> bool:
    """Draw a console-offered candidate with no model call. Always handled."""
    from agent.tools.make_chart import render_stored_chart

    if not pending or not pending["candidates"]:
        print("No chart available for the last answer.")
        return True
    number, override = _parse_n_and_type(line.split()[1:], "usage: /chart [n] [bar|line]")
    if number is None:
        return True
    candidates = pending["candidates"]
    if not 1 <= number <= len(candidates):
        print(f"choose 1 to {len(candidates)}")
        return True
    candidate = candidates[number - 1]
    chart_type = override or candidate["type"]
    if chart_type not in ("bar", "line"):
        print("usage: /chart [n] [bar|line]")
        return True
    record = AnalysisRecord.load(pending["record_abs"])
    if chart_type == "line" and not _line_allowed(record, candidate):
        print(
            "line needs date-like x values (YYYY-MM-DD, YYYY-MM or YYYY) "
            "with at least 3 rows; use bar instead."
        )
        return True
    result = render_stored_chart(record, candidate, chart_type)
    if "error" in result:
        print(f"error: {result['error']}")
    else:
        print(f"chart saved: {result['path']}")
    return True


def _handle_command(
    line: str,
    memory_dir=MEMORY_DIR,
    dataset: str = DEFAULT_DATASET,
    pending_charts: dict | None = None,
) -> bool:
    """Run a / command. Returns True when the line was a command."""
    if not line.startswith("/"):
        return False
    if line == "/datasets":
        _print_datasets(dataset)
        return True
    if line == "/help":
        print("/notes - list approved notes")
        print("/forget <key> - retire the approved note for a key")
        print("/datasets - list available datasets")
        print("/chart [n] [bar|line] - draw an offered chart for the last answer")
        print("/publish [n] [bar|line] - share an offered chart as a BI link")
        print("/help - show this list")
        return True
    if line == "/chart" or line.startswith("/chart "):
        return _run_chart_command(line, pending_charts)
    if line == "/publish" or line.startswith("/publish "):
        return _run_publish_command(line, pending_charts, dataset)
    if line == "/notes":
        approved = DefinitionStore.load(dataset, memory_dir).approved_entries()
        if not approved:
            print("no approved notes yet")
        for entry in approved:
            print(f"- {entry['key']} ({entry['kind']}): {entry['text']}")
        return True
    if line.startswith("/forget "):
        key = line[len("/forget "):].strip()
        try:
            store = DefinitionStore.load(dataset, memory_dir)
            entry = store.retire(key)
        except ValueError as exc:
            print(f"error: {exc}")
        else:
            print(f"retired {entry['key']}: {entry['text']}")
        return True
    print("unknown command (try /help)")
    return True


def _print_new_charts(text: str, charts: list[str]) -> None:
    """Announce charts not already listed under the answer (e.g. /chart)."""
    for path in charts:
        if path not in text:
            print(f"chart saved: {path}")


def _offer_charts(
    candidates: list[dict], drawn: list[str], connector: str | None = None
) -> None:
    """Print one offer line per candidate. No model call, no tokens."""
    drawn_ids = set(drawn)
    for i, candidate in enumerate(candidates, 1):
        action = f"Type /chart {i}"
        if connector:
            action += f" or /publish {i} ({connector})"
        if candidate["query_id"] in drawn_ids:
            action += " (already drawn)"
        print(
            f"Chart available: {i}. {candidate['type']} chart of "
            f"{candidate['y']} by {candidate['x']} "
            f"({candidate['points']} points). {action}"
        )


def _run_publish_command(line: str, pending: dict | None, dataset: str) -> bool:
    """Share a stored candidate as a BI link. No model call, no tokens."""
    from agent.bi import CONNECTORS
    from agent.tools.make_chart import _code_title

    if not pending or not pending["candidates"]:
        print("No chart available for the last answer.")
        return True
    bi = REGISTRY.get(dataset, {}).get("bi")
    if bi is None:
        print(f"No BI connector configured for {dataset}")
        return True
    number, override = _parse_n_and_type(line.split()[1:], "usage: /publish [n] [bar|line]")
    if number is None:
        return True
    candidates = pending["candidates"]
    if not 1 <= number <= len(candidates):
        print(f"choose 1 to {len(candidates)}")
        return True
    candidate = candidates[number - 1]
    chart_type = override or candidate["type"]
    record = AnalysisRecord.load(pending["record_abs"])
    if chart_type == "line" and not _line_allowed(record, candidate):
        print(
            "line needs date-like x values (YYYY-MM-DD, YYYY-MM or YYYY) "
            "with at least 3 rows; use bar instead."
        )
        return True
    connector_cls = CONNECTORS[bi["connector"]]
    if connector_cls.requires_approval:
        name = f"Agent: {_code_title(candidate['x'], candidate['y'])}"
        answer = input(
            f"Publish to {bi['connector']} as a saved question named "
            f"'{name}'? It creates one saved question and does not change "
            "the source data. (y/n)"
        ).strip().lower()
        if answer != "y":
            print("Not published.")
            return True
    connector = connector_cls(database=bi["database"])
    try:
        result = connector.publish(
            candidate["sql"],
            _code_title(candidate["x"], candidate["y"]),
            chart_type,
            candidate["x"],
            candidate["y"],
            x_values=_candidate_x_labels(record, candidate),
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return True
    record.note_publish(bi["connector"], candidate["query_id"], result.url)
    print(result.url)
    if bi["connector"] == "datasette" and pending.get("sqlite_path"):
        quoted = shlex.quote(pending["sqlite_path"])
        print(f"Start Datasette first: datasette serve --immutable {quoted}")
    if result.note:
        print(result.note)
    return True


def _console_ask_user(
    question: str, options: list[dict], input_fn=input
) -> dict[str, str]:
    """Numbered choice prompt for ask_clarification.

    Returns {"choice": ...} for a valid option number or 0 plus typed text,
    else {"user_question": ...}. Bare digits out of range and empty replies
    reprompt locally without touching the model or the call budget.
    """
    print(format_options_text(question, options))
    while True:
        reply = input_fn("choice (number): ").strip()
        if reply == "0":
            typed = input_fn("type your answer: ").strip()
            if typed:
                return {"choice": typed}
            continue
        kind, value = interpret_clarification_reply(reply, options)
        if kind == "choice":
            return {"choice": value}
        if kind == "question":
            return {"user_question": value}
        print(value)


def _definition_text(clarification: dict) -> str:
    """Text to save for a clarification choice.

    An option pick saves as "<label>: <explanation>"; a typed answer, which
    matches no option label, saves as typed.
    """
    reply = clarification["reply"]
    for option in clarification.get("options", []):
        if option.get("label") == reply:
            return f"{reply}: {option.get('explanation', '')}".strip()
    return reply


def _offer_definition_save(
    clarification: dict | None, ask=input, approve=approve_definition
) -> bool:
    """Offer to save a clarification choice as a default definition.

    Only runs when the user's choice carries a definition_key, and only
    console code can write the approval — never a model tool. Returns True
    when a definition was saved.
    """
    if not clarification or not clarification.get("definition_key"):
        return False
    key = clarification["definition_key"]
    for option in clarification.get("options", []):
        if option.get("last_used") and option.get("label") == clarification["reply"]:
            return False
    try:
        reply = ask(
            f"Save as the new default for {key}? (y/n) "
        ).strip().lower()
    except EOFError:
        return False
    if reply not in ("y", "yes"):
        return False
    evidence = sorted({o["query_id"] for o in clarification.get("options", [])})
    try:
        entry = approve(key, _definition_text(clarification), evidence)
    except ValueError as exc:
        print(f"error: {exc}")
        return False
    else:
        print(f"approved {entry['key']}: {entry['text']}")
        return True


def _review_proposed_notes(dataset: str = DEFAULT_DATASET) -> None:
    """Ask the user to approve or reject each proposed note."""
    store = DefinitionStore.load(dataset)
    for entry in store.entries:
        if entry["status"] != "proposed":
            continue
        print(f"proposed note [{entry['id']}] ({entry['kind']}) {entry['key']}:")
        print(f"  {entry['text']}")
        if entry["evidence"]:
            print(f"  evidence: {', '.join(entry['evidence'])}")
        try:
            reply = input("Save this note? (y/n) ").strip().lower()
        except EOFError:
            break
        if reply in ("y", "yes"):
            store.decide(entry["id"], True)
            print(f"approved {entry['key']}")
        else:
            store.decide(entry["id"], False)
            print(f"rejected {entry['key']}")


def _backend_for_dataset(dataset: str):
    """Connect to the chosen dataset; fail fast with a clear error."""
    return REGISTRY[dataset]["make_backend"]()


def _banner_and_prompt(dataset: str) -> tuple[str, str]:
    """Banner and prompt label derived from the registry entry."""
    entry = REGISTRY[dataset]
    return (
        f"Ask about the {entry['label']} database. Empty line exits.",
        entry["prompt"],
    )


def _all_banner() -> str:
    labels = [entry["label"] for entry in active_datasets().values()]
    return f"Ask across {' and '.join(labels)}. Empty line exits."


def _instructions_for_dataset(name: str, backend) -> str:
    """Full per-question instructions for one dataset worker."""
    instructions = _base_instructions(backend.dialect)
    if backend.dialect_notes:
        instructions += "\n\nDialect notes:\n" + backend.dialect_notes
    section = DefinitionStore.load(name).build_prompt_section()
    if section:
        instructions += "\n\n" + section
    return instructions


def _render_all_outcome(outcome: dict) -> None:
    """Print the merged reply, then each worker's rendered answer."""
    print(f"routing: {', '.join(outcome['routing'])}")
    if outcome["merged"]:
        print("\nCombined:\n" + outcome["merged"])
    elif outcome["verification"]["status"] == "fallback":
        print(
            "\nUNVERIFIED NUMBERS: "
            + ", ".join(outcome["verification"]["unmatched"])
        )
    else:
        print("\nNote: answers shown separately; no combined reply recorded.")
    for worker in outcome["workers"]:
        print(f"\n[{worker['dataset']}]")
        if worker["answer"] is None:
            if worker.get("partial_summary"):
                print(worker["partial_summary"])
            else:
                print(worker.get("note", "No accepted submission."))
            continue
        print(worker["answer"])
    if outcome["failed"]:
        print(
            "\nNote: no accepted answer from: "
            + ", ".join(outcome["failed"])
            + "; their numbers are excluded."
        )
    print(f"\ntokens: {outcome['total_tokens']} total")
    for path in outcome["record_paths"]:
        print(f"Record: {path}")
    print(f"Orchestration: {outcome['orchestration_path']}")


def _run_all_mode(client, model, provider: str = "meta") -> int:
    """Ask one question across every dataset via the orchestrator."""
    from agent.orchestrator import run_cross_dataset_question

    datasets: dict = {}
    try:
        for name, entry in active_datasets().items():
            backend = entry["make_backend"]()
            datasets[name] = {
                "description": entry["description"],
                "backend": backend,
                "instructions": _instructions_for_dataset(name, backend),
            }
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    print(_all_banner())
    try:
        while True:
            try:
                question = input("all> ")
            except EOFError:
                break
            if not question.strip():
                break
            if question.strip() == "/datasets":
                _print_datasets("all")
                continue
            if question.strip() == "/help":
                _handle_command("/help", dataset="all")
                continue
            if question.strip().startswith("/"):
                names = " or ".join(REGISTRY)
                print(f"Use --dataset {names} for saved definitions.")
                continue
            try:
                outcome = run_cross_dataset_question(
                    question, datasets, client, model, _console_ask_user, provider
                )
            except RuntimeError as exc:
                print(f"error: {exc}")
                continue
            _render_all_outcome(outcome)
    finally:
        for info in datasets.values():
            info["backend"].close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset
    try:
        client, model, provider = build_client()
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    note = privacy_note(provider)
    if note:
        print(f"note ({provider}): {note} Use only public or synthetic data.")
    if dataset == "all":
        return _run_all_mode(client, model, provider)
    try:
        backend = _backend_for_dataset(dataset)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    try:
        base_instructions = _base_instructions(backend.dialect)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    banner, prompt_label = _banner_and_prompt(dataset)
    save = partial(approve_definition, dataset=dataset)
    history: list = []
    pending_charts: dict | None = None
    print(banner)
    while True:
        try:
            question = input(prompt_label)
        except EOFError:
            break
        if not question.strip():
            break
        if _handle_command(
            question.strip(), dataset=dataset, pending_charts=pending_charts
        ):
            continue
        pending_charts = None
        instructions = _instructions_for_question(
            base_instructions, dataset, backend.dialect_notes
        )
        result = answer_question(
            client, model, instructions, backend, history, question,
            ask_user=_console_ask_user,
            dataset=dataset,
            provider=provider,
        )
        if result.text:
            print(result.text)
        _print_new_charts(result.text, result.charts)
        if result.chart_candidates and result.record_path:
            sqlite_path = getattr(backend, "path", None)
            pending_charts = {
                "record_abs": REPO_ROOT / result.record_path,
                "candidates": result.chart_candidates,
                "sqlite_path": str(sqlite_path) if sqlite_path else None,
            }
            connector = REGISTRY.get(dataset, {}).get("bi", {}).get("connector")
            _offer_charts(result.chart_candidates, result.chart_drawn, connector)
        if result.usage_known:
            print(
                f"tokens: {result.input_tokens} in / "
                f"{result.output_tokens} out / {result.total_tokens} total"
            )
            if provider == "baseten":
                cost = estimate_cost_usd(
                    result.input_tokens,
                    result.cached_input_tokens,
                    result.output_tokens,
                )
                print(
                    f"cost estimate, Baseten list price, {BASETEN_PRICE_AS_OF}: "
                    f"${cost:.6f}"
                )
        else:
            print("tokens: n/a")
            if provider == "baseten":
                print("cost unknown")
        if result.record_path:
            print(f"Record: {result.record_path}")
        if result.verification:
            print(result.verification)
        _offer_definition_save(result.clarification, approve=save)
        _review_proposed_notes(dataset)
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
