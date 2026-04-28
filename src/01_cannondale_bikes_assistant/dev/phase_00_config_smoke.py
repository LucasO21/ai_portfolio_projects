"""
dev/phase_00_config_smoke.py
============================
Phase 0 verification script — run this to confirm the project foundation is solid
before moving on to Phase 1 (RAG upgrades).

What it checks:
  1. Settings load from .env without errors (missing required keys raise immediately).
  2. Secrets are present (we print a bool, never the value).
  3. All core/ modules are importable (catches __init__ errors, bad imports).
  4. The MongoDB Atlas connection opens and the collection has documents.
  5. Optional keys (Cohere, LangSmith) are reported but not required.

Passing criteria:
  - Script exits 0.
  - All "required" lines show True or a non-empty value.
  - No Python tracebacks.

How to run:
  cd ai_portfolio_projects
  PYTHONPATH=src poetry run python src/01_cannondale_bikes_assistant/dev/phase_00_config_smoke.py

Expected output (values will differ):
  === Phase 0 — Config Smoke Test ===

  [Settings]
  openai key loaded   : True
  mongo uri loaded    : True
  cohere key loaded   : False   ← optional, OK if False
  llm model           : gpt-4o
  embedding model     : text-embedding-ada-002
  retriever k         : 5
  mongo db            : cannondale_bikes_db
  collection          : bikes_collection
  vector index        : vector_index

  [Core imports]
  core.config         : OK
  core.llm            : OK
  core.rag.embeddings : OK
  core.rag.vectorstore: OK
  core.rag.retriever  : OK
  core.tools          : OK (5 tools)

  [Atlas connection]
  document count      : 218
  has embeddings      : True
  embedding dim       : 1536

  === PASSED ===
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — makes `core` importable regardless of CWD or PYTHONPATH.
# ---------------------------------------------------------------------------
# This script lives at:  src/01_cannondale_bikes_assistant/dev/phase_00_config_smoke.py
# parents[0] = dev/
# parents[1] = 01_cannondale_bikes_assistant/  ← we want this on sys.path
_project_dir = Path(__file__).resolve().parents[1]
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))


def run_smoke_test() -> bool:
    """Run all checks. Return True if every required check passes."""
    all_ok = True

    print("\n=== Phase 0 — Config Smoke Test ===\n")

    # ------------------------------------------------------------------
    # 1. Settings
    # ------------------------------------------------------------------
    print("[Settings]")
    try:
        from core.config import get_settings
        s = get_settings()

        # Print presence of secrets (never print the value).
        print(f"  openai key loaded   : {bool(s.openai_key)}")
        print(f"  mongo uri loaded    : {bool(s.mongo_uri)}")
        print(f"  cohere key loaded   : {bool(s.cohere_key)}   ← optional")
        print(f"  llm model           : {s.llm_model}")
        print(f"  embedding model     : {s.embedding_model}")
        print(f"  retriever k         : {s.retriever_k}")
        print(f"  mongo db            : {s.mongo_db_name}")
        print(f"  collection          : {s.mongo_collection}")
        print(f"  vector index        : {s.vector_index_name}")

        if not s.openai_key:
            print("  ERROR: OPENAI_API_KEY is missing from .env")
            all_ok = False
        if not s.mongo_uri:
            print("  ERROR: MONGO_DB_URI is missing from .env")
            all_ok = False

    except Exception as exc:
        print(f"  ERROR loading settings: {exc}")
        all_ok = False

    # ------------------------------------------------------------------
    # 2. Core module imports
    # ------------------------------------------------------------------
    print("\n[Core imports]")
    import_checks = [
        ("core.config",          "core.config",          None),
        ("core.llm",             "core.llm",             None),
        ("core.rag.embeddings",  "core.rag.embeddings",  None),
        ("core.rag.vectorstore", "core.rag.vectorstore", None),
        ("core.rag.retriever",   "core.rag.retriever",   None),
        ("core.tools",           "core.tools",           None),
    ]

    for label, module_path, _ in import_checks:
        try:
            mod = __import__(module_path, fromlist=[""])
            # For core.tools, show how many tools are exported.
            if module_path == "core.tools":
                n = len(getattr(mod, "TOOLS", []))
                print(f"  {label:<20}: OK ({n} tools)")
            else:
                print(f"  {label:<20}: OK")
        except Exception as exc:
            print(f"  {label:<20}: ERROR — {exc}")
            all_ok = False

    # ------------------------------------------------------------------
    # 3. Atlas connection
    # ------------------------------------------------------------------
    print("\n[Atlas connection]")
    try:
        from pymongo import MongoClient
        from core.config import get_settings

        s = get_settings()
        client = MongoClient(s.mongo_uri, serverSelectionTimeoutMS=5_000)

        # serverInfo() forces an actual connection attempt.
        client.admin.command("ping")

        collection = client[s.mongo_db_name][s.mongo_collection]
        count = collection.count_documents({})
        sample = collection.find_one()

        print(f"  document count      : {count}")

        if count == 0:
            print("  WARNING: collection is empty — did ingest run?")
            all_ok = False

        if sample:
            has_embedding = "embedding" in sample
            print(f"  has embeddings      : {has_embedding}")
            if has_embedding:
                dim = len(sample["embedding"])
                print(f"  embedding dim       : {dim}")
            else:
                print("  WARNING: documents have no 'embedding' field — re-embed may be needed")

        client.close()

    except Exception as exc:
        print(f"  ERROR connecting to Atlas: {exc}")
        all_ok = False

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------
    print()
    if all_ok:
        print("=== PASSED — Phase 0 foundation is solid. Ready for Phase 1. ===")
    else:
        print("=== FAILED — fix the errors above before proceeding. ===")

    return all_ok


if __name__ == "__main__":
    ok = run_smoke_test()
    sys.exit(0 if ok else 1)
