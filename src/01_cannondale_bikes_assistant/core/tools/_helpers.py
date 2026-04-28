"""
core/tools/_helpers.py
======================
Private utility functions shared across core/tools/*.

These are NOT LangChain tools. They are pure Python helpers that tools call
internally. The leading underscore signals "not part of the public API" —
external code should import tools from core/tools/__init__.py, not from here.

Functions:
  parse_price(price_str)              → float | None
    Converts "16,000" or "$5,999" strings to a float for numeric filtering.
    Lives here (not duplicated in each tool) because search_bikes and
    get_recommendation both need price filtering.

  extract_image_urls_from_docs(docs)  → list[dict]
    Extracts unique {url, name} dicts from a list of LangChain Documents.
    Every tool that retrieves docs can surface images to the UI via these dicts.

Design note — why parse_price is here and not at ingest time:
  The current CSV data stores prices as strings like "16,000". Ideally (Phase 1)
  we convert to float once at ingest and store price_usd as a numeric field
  in MongoDB. That removes this runtime parsing from the hot path entirely.
  Until Phase 1, we parse at query time and accept the overhead.
"""
from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Section 1 — Price parser
# ---------------------------------------------------------------------------

def parse_price(price_str) -> Optional[float]:
    """Convert a price string like '16,000' or '$5,999' to a float.

    Returns None if the value is missing or unparseable, so callers can
    skip filtering rather than raise—a missing price is not an error.

    Examples:
      parse_price("5,999")   → 5999.0
      parse_price("$16,000") → 16000.0
      parse_price(None)      → None
      parse_price("N/A")     → None
    """
    if not price_str:
        return None
    try:
        return float(str(price_str).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Section 2 — Image URL extractor
# ---------------------------------------------------------------------------

def extract_image_urls_from_docs(docs: list) -> list[dict]:
    """Return a deduplicated list of {url, name} dicts from retrieved documents.

    Looks for 'bike_image_url' in each document's metadata. If a valid URL is
    found and has not been seen yet, it is included once (the `seen` set
    deduplicates across multiple docs that may reference the same bike).

    The returned list is consumed by tools, which append IMAGE_URL: markers to
    their output text. The Streamlit UI parses those markers and displays the
    images in a sidebar panel, separate from the text response.

    Args:
        docs: List of LangChain Document objects returned by a retriever.

    Returns:
        List of dicts: [{"url": "https://...", "name": "SuperSix EVO"}, ...]
        Empty list if no valid image URLs are found in the documents.
    """
    results: list[dict] = []
    seen: set = set()
    for doc in docs:
        url = doc.metadata.get("bike_image_url")
        # Skip missing, non-string, or relative URLs.
        if not url or not isinstance(url, str) or not url.startswith("http"):
            continue
        if url in seen:
            continue
        name = (
            doc.metadata.get("bike_model")
            or doc.metadata.get("bike_name")
            or "Cannondale Bike"
        )
        results.append({"url": url, "name": name})
        seen.add(url)
    return results
