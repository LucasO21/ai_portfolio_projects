"""
core/citations.py
=================
Derive sidebar citation cards from LangGraph message history.

Tool outputs already embed machine-readable hints (``IMAGE_URL:`` lines from
summary/details/compare/recommend, and markdown bike headers from search).
This module parses those strings so the Streamlit layer stays thin.
"""

from __future__ import annotations

import re
from typing import TypedDict

from langchain_core.messages import BaseMessage, ToolMessage

_IMAGE_URL_RE = re.compile(
    r"^IMAGE_URL:\s*(\S+)\s*\|\s*(.+?)\s*$",
    re.MULTILINE,
)
# Line after search header often: "  Price: $X ..."
_PRICE_LINE_RE = re.compile(r"Price:\s*([^|]+)", re.IGNORECASE)


class BikeCitation(TypedDict, total=False):
    """One row for the citation / sources panel."""

    name: str
    model: str
    price: str
    image_url: str
    tool_name: str


def _dedupe_key(c: BikeCitation) -> tuple[str, str, str]:
    return (c.get("image_url", ""), c.get("name", ""), c.get("model", ""))


def citations_from_tool_message(msg: ToolMessage) -> list[BikeCitation]:
    """Parse a single ToolMessage body into citation dicts."""
    name = msg.name or ""
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    out: list[BikeCitation] = []

    for url, img_name in _IMAGE_URL_RE.findall(content):
        out.append(
            {
                "name": img_name.strip(),
                "model": "",
                "price": "",
                "image_url": url.strip(),
                "tool_name": name,
            }
        )

    if name == "search_bikes":
        blocks = content.split("**")
        for i in range(1, len(blocks), 2):
            header = blocks[i]
            m_pair = re.match(r"^(.+?)\s*-\s*(.+)$", header.strip())
            if not m_pair:
                continue
            bike_name, bike_model = m_pair.group(1).strip(), m_pair.group(2).strip()
            tail = blocks[i + 1] if i + 1 < len(blocks) else ""
            pm = _PRICE_LINE_RE.search(tail)
            price = pm.group(1).strip() if pm else ""
            out.append(
                {
                    "name": bike_name,
                    "model": bike_model,
                    "price": price,
                    "image_url": "",
                    "tool_name": name,
                }
            )

    return out


def citations_from_messages(messages: list[BaseMessage]) -> list[BikeCitation]:
    """Aggregate citations from the most recent tool round-trips (chronological)."""
    ordered: list[BikeCitation] = []
    seen: set[tuple[str, str, str]] = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        for c in citations_from_tool_message(msg):
            key = _dedupe_key(c)
            if key in seen and key != ("", "", ""):
                continue
            seen.add(key)
            ordered.append(c)
    return ordered
