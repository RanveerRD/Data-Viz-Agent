"""WhatsApp front-end configuration (env first, then .env)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from agent.datasets import REGISTRY
from agent.loop import _read_env_file_value

_REQUIRED = (
    "WHATSAPP_TOKEN",
    "WHATSAPP_PHONE_NUMBER_ID",
    "WHATSAPP_APP_SECRET",
    "WHATSAPP_VERIFY_TOKEN",
    "WHATSAPP_ALLOWED_NUMBERS",
    "WHATSAPP_API_VERSION",
)
_NUMBER_RE = re.compile(r"^[0-9]+$")


class ConfigError(RuntimeError):
    """One clear line describing a missing or invalid setting."""


@dataclass(frozen=True)
class WhatsAppConfig:
    token: str
    phone_number_id: str
    app_secret: str
    verify_token: str
    allowed_numbers: frozenset[str]
    api_version: str
    dataset: str


def load_config(env: dict | None = None) -> WhatsAppConfig:
    """Read settings from env, falling back to .env per key."""
    source = os.environ if env is None else env

    def get(name: str) -> str:
        return source.get(name) or _read_env_file_value(name) or ""

    for name in _REQUIRED:
        if not get(name):
            raise ConfigError(f"{name} is not set (env or .env)")
    allowed = frozenset(
        part.strip() for part in get("WHATSAPP_ALLOWED_NUMBERS").split(",") if part.strip()
    )
    if not allowed:
        raise ConfigError("WHATSAPP_ALLOWED_NUMBERS is empty")
    for number in sorted(allowed):
        if not _NUMBER_RE.match(number):
            raise ConfigError("WHATSAPP_ALLOWED_NUMBERS must be digits only")
    dataset = get("WHATSAPP_DATASET") or "ask"
    if dataset == "all":
        raise ConfigError("WHATSAPP_DATASET 'all' is not supported yet")
    if dataset != "ask" and dataset not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise ConfigError(f"unknown WHATSAPP_DATASET {dataset!r}; known: {known} (or 'ask')")
    return WhatsAppConfig(
        token=get("WHATSAPP_TOKEN"),
        phone_number_id=get("WHATSAPP_PHONE_NUMBER_ID"),
        app_secret=get("WHATSAPP_APP_SECRET"),
        verify_token=get("WHATSAPP_VERIFY_TOKEN"),
        allowed_numbers=allowed,
        api_version=get("WHATSAPP_API_VERSION"),
        dataset=dataset,
    )
