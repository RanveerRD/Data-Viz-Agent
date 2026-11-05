"""Minimal WhatsApp Cloud API client: text and chart images, no redirects."""

from __future__ import annotations

import json
import logging
import secrets
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_TIMEOUT_S = 10
_MAX_BODY = 4000
_CAPTION_LIMIT = 1024
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CHARTS_DIR = (_REPO_ROOT / "workspace" / "charts").resolve()


def _mask_tail(value: object) -> str:
    """Last-4 masking for media ids in logs."""
    text = str(value)
    return "…" + text[-4:] if len(text) > 4 else text


class WhatsAppError(RuntimeError):
    """One friendly line; never contains the token."""


_GRAPH_ROOT = "https://graph.facebook.com"


class _Redirected(Exception):
    """Raised instead of following an HTTP redirect."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _Redirected()


def _opener() -> urllib.request.OpenerDirector:
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPHandler())
    opener.add_handler(urllib.request.HTTPSHandler())
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.HTTPErrorProcessor())
    opener.add_handler(_NoRedirect())
    return opener


def split_body(body: str, limit: int = _MAX_BODY) -> list[str]:
    """Split a long body at line breaks; hard-cut an overlong single line."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in body.splitlines(keepends=True):
        while len(line) > limit:
            if current_len:
                chunks.append("".join(current))
                current, current_len = [], 0
            chunks.append(line[:limit])
            line = line[limit:]
        if current_len + len(line) > limit and current:
            chunks.append("".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks or [""]


class WhatsAppClient:
    """Send WhatsApp messages via the Cloud API: text and chart images."""

    def __init__(
        self,
        api_version: str,
        phone_number_id: str,
        token: str,
        opener: Any = None,
    ) -> None:
        self.api_version = api_version
        self.phone_number_id = phone_number_id
        self._token = token
        self._opener = opener

    def send_text(self, to: str, body: str) -> None:
        """Send body as one or more text messages."""
        for chunk in split_body(body):
            self._post(
                {
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "text",
                    "text": {"body": chunk},
                }
            )

    def send_image(self, to: str, path: str, caption: str = "") -> None:
        """Upload a chart PNG, then send it as an image message.

        Only files inside workspace/charts are sent; anything else raises
        ValueError without any request. Caption carries the chart title,
        never a file path. Send failures raise WhatsAppError.
        """
        data = self._checked_chart(path)
        media_id = self._upload_png(data)
        log.info("chart media uploaded %s", _mask_tail(media_id))
        self._post(
            {
                "messaging_product": "whatsapp",
                "to": to,
                "type": "image",
                "image": {"id": media_id, "caption": caption[:_CAPTION_LIMIT]},
            }
        )

    def _checked_chart(self, path: str) -> bytes:
        """Read a chart file after the safety checks; ValueError if refused."""
        full = (_REPO_ROOT / path).resolve()
        try:
            full.relative_to(_CHARTS_DIR)
        except ValueError:
            raise ValueError(f"refusing to send a file outside {_CHARTS_DIR}")
        if full.suffix.lower() != ".png" or not full.is_file():
            raise ValueError("refusing to send a non-PNG chart file")
        if full.stat().st_size > _MAX_IMAGE_BYTES:
            raise ValueError("refusing to send a chart over 5 MB")
        return full.read_bytes()

    def _upload_png(self, data: bytes) -> str:
        """Upload PNG bytes; return the media id."""
        boundary = "----wachart" + secrets.token_hex(8)

        def part(name: str, value: str) -> bytes:
            return (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
                f"\r\n\r\n{value}\r\n"
            ).encode("utf-8")

        body = (
            part("messaging_product", "whatsapp")
            + part("type", "image/png")
            + (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                'filename="chart.png"\r\nContent-Type: image/png\r\n\r\n'
            ).encode("utf-8")
            + data
            + f"\r\n--{boundary}--\r\n".encode("utf-8")
        )
        raw = self._send_request(
            f"{_GRAPH_ROOT}/{self.api_version}/{self.phone_number_id}/media",
            body,
            f"multipart/form-data; boundary={boundary}",
        )
        try:
            media_id = json.loads(raw.decode("utf-8")).get("id")
        except (UnicodeDecodeError, ValueError, AttributeError):
            media_id = None
        if not media_id:
            raise WhatsAppError("WhatsApp did not return a media id")
        return str(media_id)

    def _post(self, payload: dict) -> None:
        self._send_request(
            f"{_GRAPH_ROOT}/{self.api_version}/{self.phone_number_id}/messages",
            json.dumps(payload).encode("utf-8"),
            "application/json",
        )

    def _send_request(self, url: str, data: bytes, content_type: str) -> bytes:
        """POST bytes; return the raw response. Token only in the header."""
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": content_type,
            },
        )
        opener = self._opener if self._opener is not None else _opener()
        try:
            with opener.open(request, timeout=_TIMEOUT_S) as response:
                return response.read()
        except _Redirected:
            raise WhatsAppError(
                "WhatsApp redirected the request; check configuration"
            )
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise WhatsAppError("WhatsApp rejected the API token")
            raise WhatsAppError(f"WhatsApp refused the message (HTTP {exc.code})")
        except (TimeoutError, OSError) as exc:
            if isinstance(exc, TimeoutError) or isinstance(
                getattr(exc, "reason", None), TimeoutError
            ):
                raise WhatsAppError("WhatsApp request timed out")
            raise WhatsAppError("Could not reach WhatsApp to send a message")
