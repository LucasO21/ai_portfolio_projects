from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

# Make `core` importable no matter how this file is run.
_project_dir = Path(__file__).resolve().parents[2]
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

from core.config import get_settings
from core.rag.embeddings import build_embeddings


@lru_cache(maxsize=1)
def build_vectorstore() -> MongoDBAtlasVectorSearch:
    """Return a cached MongoDBAtlasVectorSearch connected to the Atlas cluster.

    @lru_cache(maxsize=1) means:
      - First call: opens a MongoClient, builds the vectorstore, stores it.
      - Every subsequent call: returns the same object — no new network connection.
    The MongoClient held inside is long-lived by design: PyMongo manages a
    connection pool and reconnects automatically if the connection drops.

    Embedding model comes from build_embeddings() (also lru_cache'd), so
    swapping the embedding model in Phase 1 requires only a .env change.
    """
    s = get_settings()

    # Step 1: open a connection to your Atlas cluster.
    client = MongoClient(s.mongo_uri)

    # Step 2: point at the right database and collection.
    # This is where your 218 bike documents live.
    collection = client[s.mongo_db_name][s.mongo_collection]

    # Step 3: wrap it all in LangChain's MongoDBAtlasVectorSearch.
    # - collection:     where the documents are
    # - embedding:      how to turn a query string into a vector (from embeddings.py)
    # - index_name:     the Atlas Vector Search index to query against
    # - text_key:       field in each document that holds the text
    # - embedding_key:  field that holds the stored vector
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=build_embeddings(),
        index_name=s.vector_index_name,
        text_key="text",
        embedding_key="embedding",
    )


# ---------------------------------------------------------------------------
# Run this file to verify the vectorstore connects and your docs are there.
# Does NOT run a search yet — that's the next file (retriever.py).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # sys, Path, MongoClient, and get_settings are already imported at module level.
    # sys.path is already patched at module level too — nothing to repeat here.
    s = get_settings()
    client = MongoClient(s.mongo_uri)
    collection = client[s.mongo_db_name][s.mongo_collection]

    count = collection.count_documents({})
    sample = collection.find_one()

    print("=== vectorstore.py smoke test ===\n")
    print(f"connected to   : {s.mongo_db_name}.{s.mongo_collection}")
    print(f"document count : {count}")

    if sample:
        print(f"fields in a doc: {list(sample.keys())}")
        if "embedding" in sample:
            print(f"embedding dim  : {len(sample['embedding'])}")
    else:
        print("WARNING: collection is empty")

    print("\n=== OK ===")
    client.close()
