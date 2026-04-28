from __future__ import annotations

import sys
from pathlib import Path

# Make `core` importable no matter how this file is run.
_project_dir = Path(__file__).resolve().parents[2]
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient

from core.config import get_settings


def build_vectorstore() -> MongoDBAtlasVectorSearch:
    s = get_settings()

    # Step 1: open a connection to your Atlas cluster
    client = MongoClient(s.mongo_uri)

    # Step 2: point at the right database and collection
    # This is where your 218 bike documents live.
    collection = client[s.mongo_db_name][s.mongo_collection]

    # Step 3: build the embedding model
    # This is what converts a text query into a vector so Atlas can
    # compare it against the stored embeddings.
    embeddings = OpenAIEmbeddings(
        model=s.embedding_model,
        api_key=s.openai_key,  # type: ignore[arg-type]
    )

    # Step 4: wrap it all in LangChain's MongoDBAtlasVectorSearch
    # - collection: where the documents are
    # - embedding: how to turn a query string into a vector
    # - index_name: the Atlas Vector Search index to query against
    # - text_key: the field in each document that holds the text ("text")
    # - embedding_key: the field that holds the stored vector ("embedding")
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=s.vector_index_name,
        text_key="text",
        embedding_key="embedding",
    )


# ---------------------------------------------------------------------------
# Run this file to verify the vectorstore connects and your docs are there.
# Does NOT run a search yet — that's the next file (retriever.py).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Tell Python where to find the `core` package.
    # parents[0] = core/rag/
    # parents[1] = core/
    # parents[2] = 01_cannondale_bikes_assistant/   <-- this is what we need on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from pymongo import MongoClient
    from core.config import get_settings

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
