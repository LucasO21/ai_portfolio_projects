"""
core/rag/reranker.py
====================
Responsible for reranking retrieved documents using Cohere's reranking API.

Where it fits in the pipeline:
  retriever.py retrieves k=20 candidate docs via vector search →
  reranker.py re-scores them against the original query →
  top_n=5 best docs are returned to the tool.

Why reranking matters:
  Vector similarity is a proxy for relevance — cosine distance between
  embeddings captures semantic similarity, but not "does this document
  actually answer the question?". Cohere's reranker is a cross-encoder
  trained specifically to score (query, document) relevance, so it routinely
  reorders results — e.g. a bike mentioned by name in the query jumps from
  rank #8 to rank #1 even if its embedding wasn't the closest.

  The two-stage pattern (retrieve many, rerank few) is the standard way
  to get cross-encoder quality without paying cross-encoder latency on
  every document in the collection. Vector search is O(log N); reranking
  is O(k) where k is a small candidate set (here 20).

Why no @lru_cache:
  CohereRerank takes top_n as a parameter. Caching a fixed instance would
  lock in one value of top_n for the whole process. Construction is
  instantaneous (no network calls), so creating a fresh instance per call
  is fine. This differs from build_llm() / build_embeddings() which are
  always constructed with the same args.

Requires:
  COHERE_API_KEY set in .env — see core/config.py.

Run directly to verify Cohere connects and reorders results:
  PYTHONPATH=src poetry run python src/01_cannondale_bikes_assistant/core/rag/reranker.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Make `core` importable when this file is run directly.
_project_dir = Path(__file__).resolve().parents[2]
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from langchain_cohere import CohereRerank
from langchain_core.documents import Document

from core.config import get_settings


# ---------------------------------------------------------------------------
# Section 1 — Reranker factory
# ---------------------------------------------------------------------------

def build_reranker(top_n: int = 5) -> CohereRerank:
    """Return a CohereRerank compressor configured from Settings.

    Args:
        top_n: How many documents to keep after reranking.
               Default 5 matches retriever_k so tools always see 5 results.

    CohereRerank is a BaseDocumentCompressor in LangChain:
      - Used inside ContextualCompressionRetriever in retriever.py (auto-wired).
      - Can also be called directly via rerank_docs() below for testing.

    Raises ValueError if COHERE_API_KEY is not set.
    """
    s = get_settings()
    if not s.cohere_key:
        raise ValueError(
            "COHERE_API_KEY is not set. Add it to .env to enable reranking."
        )
    return CohereRerank(
        cohere_api_key=s.cohere_key,
        model="rerank-english-v2.0",
        top_n=top_n,
    )


# ---------------------------------------------------------------------------
# Section 2 — Standalone rerank helper
# ---------------------------------------------------------------------------

def rerank_docs(docs: List[Document], query: str, top_n: int = 5) -> List[Document]:
    """Rerank a list of documents against query, return top_n.

    Convenience wrapper around build_reranker().compress_documents() for
    callers that already hold a doc list and want to rerank in one call.
    Not used by retriever.py (which auto-wraps via ContextualCompressionRetriever),
    but useful for the test blocks below and any ad-hoc experimentation.

    Returns docs unchanged if the list is empty.
    """
    if not docs:
        return docs
    return build_reranker(top_n=top_n).compress_documents(docs, query)


# =============================================================================
# In-script test blocks — run in VS Code interactive mode (# %%) or as:
#   PYTHONPATH=src poetry run python src/01_cannondale_bikes_assistant/core/rag/reranker.py
# =============================================================================

## %% [Setup]
import sys
from pathlib import Path

_project_dir = Path(__file__).resolve().parents[2]
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from core.rag.vectorstore import build_vectorstore
from core.rag.reranker import rerank_docs


## %% [Test 1] Road bike query — before vs after rerank
query = "lightweight carbon road bike for racing"

base_retriever = build_vectorstore().as_retriever(search_kwargs={"k": 20})
candidates = base_retriever.invoke(query)

print(f"Query: {query!r}\n")
print(f"=== Before rerank ({len(candidates)} candidates) ===")
for i, doc in enumerate(candidates, 1):
    m = doc.metadata
    print(f"  {i:2}. {m.get('bike_name', '?')} — {m.get('bike_model', '?')}")

reranked = rerank_docs(candidates, query, top_n=5)
print(f"\n=== After rerank (top 5) ===")
for i, doc in enumerate(reranked, 1):
    m = doc.metadata
    print(f"  {i}. {m.get('bike_name', '?')} — {m.get('bike_model', '?')}")


## %% [Test 2] MTB query — before vs after rerank
query2 = "trail mountain bike for technical rough terrain"

candidates2 = build_vectorstore().as_retriever(search_kwargs={"k": 20}).invoke(query2)
reranked2 = rerank_docs(candidates2, query2, top_n=5)

print(f"Query: {query2!r}\n")
print(f"=== Before rerank (top 5 of {len(candidates2)} candidates) ===")
for i, doc in enumerate(candidates2[:5], 1):
    m = doc.metadata
    print(f"  {i}. {m.get('bike_name', '?')} — {m.get('bike_model', '?')}")

print(f"\n=== After rerank (top 5) ===")
for i, doc in enumerate(reranked2, 1):
    m = doc.metadata
    print(f"  {i}. {m.get('bike_name', '?')} — {m.get('bike_model', '?')}")


## %% [Test 3] Budget query — confirm price-relevant bikes surface after rerank
query3 = "affordable gravel bike under $2000"

candidates3 = build_vectorstore().as_retriever(search_kwargs={"k": 20}).invoke(query3)
reranked3 = rerank_docs(candidates3, query3, top_n=5)

print(f"Query: {query3!r}\n")
print(f"=== After rerank (top 5) ===")
for i, doc in enumerate(reranked3, 1):
    m = doc.metadata
    print(f"  {i}. {m.get('bike_name', '?')} — ${m.get('price', '?')}")

## %%
