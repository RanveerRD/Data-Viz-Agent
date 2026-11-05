"""Approved business definitions and data quirks, one store per dataset.

Storage mirrors agent/record.py: one JSON file per dataset at
workspace/memory/<dataset>.json, atomic writes (temp file, fsync,
os.replace), and an append-only audit list. Properties return copies so
callers can never edit the stored state directly.

Entries flow one way: the model-facing propose_note tool adds "proposed"
entries only. Approval or rejection happens solely in console code the model
cannot call, via decide() or the approve_definition() helper below.
"""

from __future__ import annotations

import copy
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.datasets import REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = REPO_ROOT / "workspace" / "memory"
DEFAULT_DATASET = next(iter(REGISTRY))

KINDS = ("definition", "quirk")
STATUSES = ("proposed", "approved", "rejected", "retired")

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_TEXT_LEN = 300

PROMPT_HEADER = "Last-used definitions (offer as defaults, still confirm with ask_clarification)"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_entry_fields(kind: Any, key: Any, text: Any, evidence: Any) -> list[str]:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {list(KINDS)}, got {kind!r}")
    if not isinstance(key, str) or not _KEY_RE.match(key):
        raise ValueError(f"key {key!r} must be snake_case")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text needs a non-empty definition or quirk note")
    if len(text.strip()) > MAX_TEXT_LEN:
        raise ValueError(f"text is {len(text.strip())} characters, max is {MAX_TEXT_LEN}")
    if evidence is None:
        return []
    if not isinstance(evidence, list) or any(
        not isinstance(e, str) or not e.strip() for e in evidence
    ):
        raise ValueError("evidence must be a list of query-id strings")
    return [e.strip() for e in evidence]


class DefinitionStore:
    """One dataset's proposed/approved/rejected definitions and quirks."""

    def __init__(
        self,
        dataset: str,
        path: str | Path,
        state: dict[str, Any] | None = None,
    ) -> None:
        self._dataset = dataset
        self._path = Path(path)
        self._state = state if state is not None else _blank_store(dataset)
        self._seq = len(self._state["audit"])

    @classmethod
    def load(
        cls, dataset: str = DEFAULT_DATASET, memory_dir: str | Path = MEMORY_DIR
    ) -> DefinitionStore:
        """Load the dataset's store, starting empty when no file exists yet."""
        path = Path(memory_dir) / f"{dataset}.json"
        if path.is_file():
            state = json.loads(path.read_text())
        else:
            state = _blank_store(dataset)
        return cls(dataset, path, state)

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entries(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._state["entries"])

    @property
    def audit(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._state["audit"])

    def approved_entries(self) -> list[dict[str, Any]]:
        """Only approved entries; these are what the prompt may use."""
        return [e for e in self.entries if e["status"] == "approved"]

    def build_prompt_section(self) -> str:
        """Render approved entries for the system prompt; "" when none."""
        approved = self.approved_entries()
        if not approved:
            return ""
        lines = [PROMPT_HEADER + ":"]
        for entry in approved:
            lines.append(f"- {entry['key']} ({entry['kind']}): {entry['text']}")
        return "\n".join(lines)

    def propose(
        self,
        kind: Any,
        key: Any,
        text: Any,
        evidence: Any = None,
        record: Any = None,
    ) -> dict[str, Any]:
        """Add a "proposed" entry. Never approves; returns the entry."""
        cleaned_evidence = _validate_entry_fields(kind, key, text, evidence)
        if record is not None and cleaned_evidence:
            known = {q["id"] for q in record.queries}
            unknown = [e for e in cleaned_evidence if e not in known]
            if unknown:
                raise ValueError(f"evidence cites unknown query ids: {unknown}")
        for entry in self._state["entries"]:
            if entry["key"] == key and entry["text"] == text.strip():
                if entry["status"] == "rejected":
                    raise ValueError(
                        f"key {key!r} was already rejected with this text; "
                        "propose different text"
                    )
                raise ValueError(
                    f"key {key!r} already has this exact {entry['status']} entry"
                )
        entry = {
            "id": f"e{int(self._state.get('entry_seq', 0)) + 1}",
            "kind": kind,
            "key": key,
            "text": text.strip(),
            "status": "proposed",
            "evidence": cleaned_evidence,
            "proposed_at": _utc_now_iso(),
            "decided_at": None,
            "decided_by": None,
        }
        self._state["entry_seq"] = int(self._state.get("entry_seq", 0)) + 1
        self._state["entries"].append(entry)
        self._append_audit("note_proposed", {"id": entry["id"], "key": key})
        self.save()
        return copy.deepcopy(entry)

    def decide(
        self, entry_id: str, approve: bool, decided_by: str = "user"
    ) -> dict[str, Any]:
        """Approve or reject a proposed entry. Console use only."""
        for entry in self._state["entries"]:
            if entry["id"] == entry_id:
                break
        else:
            raise ValueError(f"unknown entry id: {entry_id!r}")
        if entry["status"] != "proposed":
            raise ValueError(f"entry {entry_id!r} is already {entry['status']}")
        if approve:
            self._check_one_approved(entry["key"], entry_id)
            entry["status"] = "approved"
        else:
            entry["status"] = "rejected"
        entry["decided_at"] = _utc_now_iso()
        entry["decided_by"] = decided_by
        self._append_audit(
            "note_approved" if approve else "note_rejected",
            {"id": entry_id, "key": entry["key"]},
        )
        self.save()
        return copy.deepcopy(entry)

    def approve_definition(
        self,
        key: Any,
        text: Any,
        evidence: Any = None,
        decided_by: str = "user",
    ) -> dict[str, Any]:
        """Create an already-approved definition. Console use only, never a tool.

        A key that already has an approved entry retires the old one first,
        so the key always ends with exactly one approved entry.
        """
        cleaned_evidence = _validate_entry_fields("definition", key, text, evidence)
        for old in self._state["entries"]:
            if old["key"] == key and old["status"] == "approved":
                old["status"] = "retired"
                old["decided_at"] = _utc_now_iso()
                old["decided_by"] = decided_by
                self._append_audit("note_retired", {"id": old["id"], "key": key})
                break
        entry = {
            "id": f"e{int(self._state.get('entry_seq', 0)) + 1}",
            "kind": "definition",
            "key": key,
            "text": text.strip(),
            "status": "approved",
            "evidence": cleaned_evidence,
            "proposed_at": _utc_now_iso(),
            "decided_at": _utc_now_iso(),
            "decided_by": decided_by,
        }
        self._state["entry_seq"] = int(self._state.get("entry_seq", 0)) + 1
        self._state["entries"].append(entry)
        self._append_audit("note_approved", {"id": entry["id"], "key": key})
        self.save()
        return copy.deepcopy(entry)

    def retire(self, key: Any, decided_by: str = "user") -> dict[str, Any]:
        """Retire the approved entry for a key, keeping it in the file.

        Console use only, never a tool. Retired entries leave the system
        prompt and free the key for a new definition.
        """
        for entry in self._state["entries"]:
            if entry["key"] == key and entry["status"] == "approved":
                break
        else:
            raise ValueError(f"no approved entry for key {key!r}")
        entry["status"] = "retired"
        entry["decided_at"] = _utc_now_iso()
        entry["decided_by"] = decided_by
        self._append_audit("note_retired", {"id": entry["id"], "key": key})
        self.save()
        return copy.deepcopy(entry)

    def _check_one_approved(self, key: str, entry_id: str | None) -> None:
        for entry in self._state["entries"]:
            if (
                entry["key"] == key
                and entry["status"] == "approved"
                and entry["id"] != entry_id
            ):
                raise ValueError(
                    f"key {key!r} already has approved entry {entry['id']!r}"
                )

    def save(self) -> Path:
        """Persist atomically: temp file, flush, fsync, rename."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        with open(tmp, "w") as handle:
            json.dump(self._to_dict(), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self._path)
        return self._path

    def _append_audit(self, action: str, detail: dict[str, Any]) -> None:
        self._seq += 1
        self._state["audit"].append(
            {"seq": self._seq, "action": action, "detail": detail, "timestamp": _utc_now_iso()}
        )

    def _to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self._dataset,
            "entries": self._state["entries"],
            "entry_seq": self._state.get("entry_seq", 0),
            "audit": self._state["audit"],
        }


def approve_definition(
    key: str,
    text: str,
    evidence: Any = None,
    dataset: str = DEFAULT_DATASET,
    memory_dir: str | Path = MEMORY_DIR,
    decided_by: str = "user",
) -> dict[str, Any]:
    """Create an already-approved definition. Console use only, never a tool."""
    store = DefinitionStore.load(dataset, memory_dir)
    return store.approve_definition(key, text, evidence, decided_by)


def _blank_store(dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "entries": [],
        "entry_seq": 0,
        "audit": [],
    }
