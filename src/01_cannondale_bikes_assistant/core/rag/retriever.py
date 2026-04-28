from __future__ import annotations

import sys
from pathlib import Path

# Make `core` importable no matter how this file is run.
_project_dir = Path(__file__).resolve().parents[2]
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from core.config import get_settings
from core.rag.vectorstore import build_vectorstore


def build_retriever(k: int = None):
    s = get_settings()

    # Use k from config if not passed in explicitly.
    num_results = k if k is not None else s.retriever_k

    # build_vectorstore() gives us the LangChain object connected to Atlas.
    # .as_retriever() turns it into something you can call with a query string.
    # search_type="similarity" means: find documents with the closest vectors.
    return build_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": num_results},
    )


# ---------------------------------------------------------------------------
# Run this file to verify retrieval actually works against your Atlas index.
# This is the first file that does a real vector search — you should see
# actual bike names come back for the query below.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    retriever = build_retriever()

    query = "lightweight carbon road bike"
    print(f"=== retriever.py smoke test ===")
    print(f"query: {query!r}\n")

    docs = retriever.invoke(query)

    if not docs:
        print("WARNING: 0 documents returned.")
        print("This usually means the Atlas Vector Search index does not exist yet.")
        print("Go to Atlas UI -> your cluster -> Search -> Create Search Index")
        print("Type: vectorSearch, Name: vector_index, field: embedding, dims: 1536, similarity: cosine")
    else:
        print(f"got {len(docs)} documents back:\n")
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            print(f"  {i}. {m.get('bike_name')} — {m.get('bike_model')}  |  ${m.get('price')}")

    print("\n=== OK ===")
