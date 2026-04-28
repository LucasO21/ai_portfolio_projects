# `dev/_archive/` — retired learning scripts

Files in this folder are kept for **diff-against-history**, not for running.
Their replacements live under `core/` and `dev/phase_*.py`.

| Archived file | Status | Replacement | Why retired |
| --- | --- | --- | --- |
| `01_create_vectorstore.py` | retired | [`core/rag/ingest.py`](../../core/rag/ingest.py) + [`dev/phase_01_ingest_sample.py`](../phase_01_ingest_sample.py) | Top-of-file `%load_ext autoreload` / `%autoreload 2` are Jupyter magics — the script is unrunnable as plain Python. Mixed in dead Chroma fallback code and bare expressions like `documents` / `len(documents)` that only "work" in a notebook. The clean ingest path is now in `core/rag/ingest.py`. |
| `02_rag_pipeline.py` | retired | superseded by `core/rag/*` + `core/tools/*` | Earlier RAG experiment; tool definitions duplicated and drifted from the ones in `app/app2.py`. Re-run-ability is unclear (uses module-global side effects). |
| `03_rag_pipeline_v2.py` | retired | superseded by `core/rag/*` + `core/tools/*` | Second iteration of the same experiment. Has known bugs (variable order around `query` / `except` / `msgs`) and contains tool code that has since been promoted to `core/tools/`. |

## Comparing old vs new

```bash
# See what moved and how it changed:
git log --oneline -- src/01_cannondale_bikes_assistant/dev/_archive/
git diff <old-commit> -- src/01_cannondale_bikes_assistant/dev/01_create_vectorstore.py
```

## Do not run these files

If you need to verify Phase 0/1 behaviour, run the canonical scripts in
[`dev/`](..) (see [`dev/README.md`](../README.md)). The archive exists so we
can show *before vs after* during the learning portion of the rebuild — once
the new code is stable and you've internalised the diff, this folder can be
deleted.
