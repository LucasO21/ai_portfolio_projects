"""
core/tools/__init__.py
======================
Exports all LangChain tools and the canonical TOOLS list.

Import from here rather than from individual modules. If a tool moves to a
different file in the future, only this __init__.py changes — callers are
unaffected.

Usage:
    from core.tools import TOOLS                   # for agent/graph setup
    from core.tools import search_bikes            # for direct invocation in tests

The LLM selects tools by their docstrings, not by position in TOOLS.
The ordering here is for human readability only.
"""
from core.tools.search import search_bikes
from core.tools.summary import get_bike_summary
from core.tools.details import get_bike_details
from core.tools.compare import compare_bikes
from core.tools.recommend import get_recommendation

TOOLS = [
    search_bikes,
    get_bike_summary,
    get_bike_details,
    compare_bikes,
    get_recommendation,
]

__all__ = [
    "TOOLS",
    "search_bikes",
    "get_bike_summary",
    "get_bike_details",
    "compare_bikes",
    "get_recommendation",
]
