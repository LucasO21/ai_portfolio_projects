"""
core/tools/search.py
====================
Defines the search_bikes LangChain tool.

When the LLM should call this tool:
  User wants to browse, filter, or discover bikes. Trigger keywords:
  "show me", "list", "find", "what bikes", "under $X", "between X and Y".
  NOT for "tell me about the Scalpel" (→ use summary) or "compare X vs Y" (→ use compare).

How it works:
  1. Runs a vector search with k=10 — a broader candidate set than the default
     k=5, because we know we're about to throw some away in the filter steps.
  2. Post-filters by bike_type (keyword match across text and metadata fields).
  3. Post-filters by price range (parse_price converts string prices to floats).
  4. Returns a formatted markdown list with IMAGE_URL markers for the UI.

Phase 1 upgrade path:
  Replace step 2 with an Atlas $vectorSearch pre-filter on a 'category'
  metadata field (stored as a clean enum at ingest time). This eliminates the
  risk of k=10 not containing any results of the right type when the collection
  is large. See core/rag/retriever.py (Phase 1) for the pre-filter API.
"""
from __future__ import annotations

from typing import Optional

from langchain.tools import tool

from core.rag.vectorstore import build_vectorstore
from core.tools._helpers import extract_image_urls_from_docs, parse_price


@tool
def search_bikes(
    query: str,
    bike_type: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> str:
    """Search for Cannondale bikes matching criteria. Use this when the user wants to
    find, list, browse, or filter bikes by type, price range, or features.

    Args:
        query: Search terms describing desired bike characteristics.
        bike_type: Optional category filter (e.g. 'road', 'mountain', 'gravel', 'electric', 'hybrid').
        min_price: Optional minimum price in USD.
        max_price: Optional maximum price in USD.

    Returns:
        Formatted list of matching bikes with key details and IMAGE_URL markers.
    """
    try:
        # k=10 gives a broader candidate set before post-filtering.
        # Phase 1 will replace post-filter with Atlas pre-filter so we don't
        # lose matching docs that fell outside the top-k.
        vs = build_vectorstore()
        retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )
        docs = retriever.invoke(query)

        # --- Type filter (Python-side until Phase 1 Atlas pre-filter) ---
        # We check multiple text fields because 'mountain' might appear in
        # description_1 rather than bike_name for some models.
        if bike_type:
            bt = bike_type.lower()
            docs = [
                d for d in docs
                if bt in d.page_content.lower()
                or bt in d.metadata.get("bike_name", "").lower()
                or bt in d.metadata.get("description_1", "").lower()
                or bt in d.metadata.get("description_2", "").lower()
                or bt in d.metadata.get("highlights", "").lower()
                or bt in d.metadata.get("bike_image_url", "").lower()
            ]

        # --- Price filter ---
        # We check both bounds independently so partial filters work:
        # "under $5000" sets only max_price; "over $3000" sets only min_price.
        filtered = []
        for doc in docs:
            price_val = parse_price(doc.metadata.get("price"))
            if min_price is not None and price_val is not None and price_val < min_price:
                continue
            if max_price is not None and price_val is not None and price_val > max_price:
                continue
            filtered.append(doc)

        if not filtered:
            return "No bikes found matching your criteria. Try broadening your search."

        # --- Format results ---
        results = []
        for doc in filtered:
            m = doc.metadata
            desc = str(m.get("description_1", ""))[:150]
            img_url = m.get("bike_image_url", "")
            img_link = (
                f"  [View Image]({img_url})"
                if img_url and str(img_url).startswith("http")
                else ""
            )
            results.append(
                f"**{m.get('bike_name', 'N/A')} - {m.get('bike_model', 'N/A')}**\n"
                f"  Price: ${m.get('price', 'N/A')} | Color: {m.get('color', 'N/A')}\n"
                f"  {desc}\n"
                f"{img_link}"
            )

        return f"Found {len(filtered)} matching bikes:\n\n" + "\n\n".join(results)

    except Exception as e:
        return f"Error searching bikes: {str(e)}"
