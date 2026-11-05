"""BI connectors: turn a stored chart candidate into a link in a BI tool.

No model call, no tokens. Each connector publishes the EXACT recorded
SQL string for a query id. Adding a tool later is one new file in this
package and one line in CONNECTORS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class PublishResult:
    url: str
    note: str = ""


class BIConnector(Protocol):
    """Build a shareable link for one recorded query."""

    requires_approval: bool

    def publish(
        self,
        sql: str,
        title: str,
        chart_type: str,
        x: str,
        y: str,
        x_values: list[str] | None = None,
    ) -> PublishResult:
        """Return the link for this SQL. No network, no writes.

        x_values carries the plotted x labels when the caller has them;
        connectors use them only for display decisions, never as data.
        """
        ...


def _connectors() -> dict[str, type]:
    from agent.bi import datasette, metabase

    return {
        "datasette": datasette.DatasetteConnector,
        "metabase": metabase.MetabaseConnector,
    }


CONNECTORS: dict[str, type] = _connectors()
