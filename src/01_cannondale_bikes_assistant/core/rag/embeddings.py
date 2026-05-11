"""
core/rag/embeddings.py
======================
Responsible for building and caching the OpenAI embedding model.

Where it fits in the pipeline:
  config.py  →  embeddings.py  →  vectorstore.py  →  retriever.py

Why a dedicated module (instead of inlining OpenAIEmbeddings in vectorstore.py):
  - Single place to change the model name. Phase 1 migrates from
    text-embedding-ada-002 to text-embedding-3-small by changing one key
    in .env — this file and all its callers need no edits.
  - @lru_cache ensures the same embedding object is shared across vectorstore,
    ingest scripts, and any future re-embed utilities within a single process,
    so we don't create multiple OpenAI API clients unnecessarily.

When to read this file:
  - Phase 0 (now):  to understand how the embedding model is configured.
  - Phase 1:  change EMBEDDING_MODEL in .env; re-run the smoke test below
              to confirm the new model loads and the vector dimension is correct.

Run directly to verify the model loads and can embed a sample string:
  PYTHONPATH=src poetry run python src/01_cannondale_bikes_assistant/core/rag/embeddings.py
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

# Make `core` importable when this file is run directly (not via poetry run python -m).
# parents[0] = core/rag/, parents[1] = core/, parents[2] = 01_cannondale_bikes_assistant/
_project_dir = Path(__file__).resolve().parents[2]
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from langchain_openai import OpenAIEmbeddings

from core.config import get_settings


# ---------------------------------------------------------------------------
# Section 1 — Embedding factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def build_embeddings() -> OpenAIEmbeddings:
    """Return a cached OpenAI embedding model configured from Settings.

    @lru_cache(maxsize=1) means:
      - First call: reads settings, constructs OpenAIEmbeddings, stores it.
      - Every subsequent call: returns the stored object instantly (no I/O).
    This is safe because OpenAIEmbeddings is stateless—it holds no mutable
    per-call state, only the model name and API key.

    Called by:
      - core/rag/vectorstore.py  (to embed queries at search time)
      - dev/phase_01_ingest_sample.py  (to embed document text at ingest time)
      - scripts/reembed_migration.py  (Phase 1: to re-embed all docs with new model)
    """
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,      # "text-embedding-ada-002" now; "text-embedding-3-small" in Phase 1
        api_key=s.openai_key,         # type: ignore[arg-type]  (SecretStr vs str)
    )


# ---------------------------------------------------------------------------
# Run directly to verify the model loads and can embed a sample string.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    embeddings = build_embeddings()
    s = get_settings()

    print("=== embeddings.py smoke test ===\n")
    print(f"model          : {s.embedding_model}")

    # Embed a short test string and inspect the output dimension.
    # Dimension should be 1536 for both ada-002 and text-embedding-3-small.
    sample = "lightweight carbon road bike"
    vector = embeddings.embed_query(sample)

    print(f"sample query   : {sample!r}")
    print(f"embedding dim  : {len(vector)}")
    print(f"first 5 values : {[round(v, 6) for v in vector[:5]]}")
    print("\n=== OK ===")
