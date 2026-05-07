"""
core/rag/retriever.py
=====================
Responsible for building the retriever used by all tools.

Where it fits in the pipeline:
  vectorstore.py  →  retriever.py  →  core/tools/*.py

Two modes depending on whether COHERE_API_KEY is set in .env:

  With Cohere (Phase 1+):
    Fetches candidate_k=20 docs via vector similarity, then passes them
    through CohereRerank (reranker.py) to return the best top_n=5.
    This is wrapped in LangChain's ContextualCompressionRetriever so that
    all tool code continues to call retriever.invoke(query) unchanged.

  Without Cohere (fallback):
    Plain vector similarity search returning retriever_k docs directly.
    Useful for local dev without a Cohere key or for cost control.

Why candidate_k=20 before reranking:
  Cross-encoders (Cohere's reranker) need a candidate set to reorder.
  Too small a set (k=5) gives the reranker nothing to work with.
  Too large (k=100+) is slow. 20 is a standard starting point that
  gives the reranker enough variety while keeping latency low.

The k parameter in build_retriever() means FINAL output count:
  - With Cohere: fetches max(20, k*4) candidates, reranks to k.
  - Without Cohere: fetches k docs directly.
  Default is settings.retriever_k (5). Most tools call build_retriever()
  with no args and get 5 results back either way.

Run directly to verify retrieval (with and without reranking if configured):
  PYTHONPATH=src poetry run python src/01_cannondale_bikes_assistant/core/rag/retriever.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Make `core` importable when this file is run directly.
_project_dir = Path(__file__).resolve().parents[2]
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from langchain_core.retrievers import BaseRetriever

from core.config import get_settings
from core.rag.vectorstore import build_vectorstore


# ---------------------------------------------------------------------------
# Section 1 — Retriever factory
# ---------------------------------------------------------------------------

def build_retriever(k: Optional[int] = None) -> BaseRetriever:
    """Return a retriever that yields k documents for a query string.

    With COHERE_API_KEY set:
      Returns a ContextualCompressionRetriever that fetches max(20, k*4)
      candidates via vector search, then reranks to top k via Cohere.
      All tools call .invoke(query) on the result — same interface either way.

    Without COHERE_API_KEY:
      Returns a plain similarity retriever fetching k docs directly.

    Args:
        k: Final number of documents to return. Defaults to settings.retriever_k (5).
    """
    s = get_settings()
    final_k = k if k is not None else s.retriever_k

    if s.cohere_key:
        # Fetch a larger candidate pool so the reranker has something to reorder.
        # max(20, final_k * 4) means: at minimum 20, or 4x the requested output.
        candidate_k = max(20, final_k * 4)

        base_retriever = build_vectorstore().as_retriever(
            search_type="similarity",
            search_kwargs={"k": candidate_k},
        )

        from langchain_classic.retrievers import ContextualCompressionRetriever

        from core.rag.reranker import build_reranker

        return ContextualCompressionRetriever(
            base_compressor=build_reranker(top_n=final_k),
            base_retriever=base_retriever,
        )

    # Fallback: plain vector search — no reranking.
    return build_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": final_k},
    )


# ---------------------------------------------------------------------------
# Run this file to verify retrieval works. Shows ranked results for several
# fixed queries and — if Cohere is configured — prints before/after rerank.
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    QUERIES = [
        "lightweight carbon road bike for racing",
        "trail mountain bike for rough terrain",
        "affordable gravel bike under $2000",
        "electric mountain bike",
        "beginner road bike",
    ]

    s = get_settings()
    reranking_enabled = bool(s.cohere_key)

    print("=== retriever.py smoke test ===")
    print(f"reranking : {'enabled (Cohere)' if reranking_enabled else 'disabled (no COHERE_API_KEY)'}\n")

    # --- With reranking: show candidate list vs final reranked list ---
    if reranking_enabled:
        from core.rag.reranker import rerank_docs

        print("--- Before/after rerank comparison (first query) ---")
        q = QUERIES[0]
        plain_retriever = build_vectorstore().as_retriever(search_kwargs={"k": 20})
        candidates = plain_retriever.invoke(q)

        print(f"\nQuery: {q!r}")
        print(f"Before rerank (top 5 of {len(candidates)}):")
        for i, doc in enumerate(candidates[:5], 1):
            m = doc.metadata
            print(f"  {i}. {m.get('bike_name', '?')} — {m.get('bike_model', '?')}")

        reranked = rerank_docs(candidates, q, top_n=5)
        print(f"After rerank (top 5):")
        for i, doc in enumerate(reranked, 1):
            m = doc.metadata
            print(f"  {i}. {m.get('bike_name', '?')} — {m.get('bike_model', '?')}")

    # --- Final retriever (reranked if Cohere, plain otherwise) for all queries ---
    print(f"\n--- Final retriever results for all queries ---")
    retriever = build_retriever()
    for q in QUERIES:
        docs = retriever.invoke(q)
        print(f"\nQuery: {q!r}")
        if not docs:
            print("  WARNING: 0 documents returned.")
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            print(f"  {i}. {m.get('bike_name', '?')} — {m.get('bike_model', '?')}  |  ${m.get('price', '?')}")

    print("\n=== OK ===")
