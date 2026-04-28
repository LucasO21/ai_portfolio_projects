# dev/ — Phase Scripts

This folder is **where you develop and learn**. Every phase has at least one
runnable script that proves the corresponding `core/` layer works before you
touch Streamlit.

**Rule of thumb:** if you can't demonstrate a feature from a `dev/` script with
printed output, do not embed it in `app/` yet.

---

## How to run any script

From the **repository root** (`ai_portfolio_projects/`):

```bash
PYTHONPATH=src poetry run python src/01_cannondale_bikes_assistant/dev/<script_name>.py
```

Or set `PYTHONPATH` once for the session:

```bash
export PYTHONPATH=src
poetry run python src/01_cannondale_bikes_assistant/dev/phase_00_config_smoke.py
```

---

## Phase checklist (run in order)

### Phase 0 — Foundation

| Script | What passing looks like |
| ------ | ----------------------- |
| `phase_00_config_smoke.py` | Prints all settings (no secret values), confirms Atlas connection, reports doc count, exits 0 |

```bash
PYTHONPATH=src poetry run python src/01_cannondale_bikes_assistant/dev/phase_00_config_smoke.py
```

Gate: all `[Settings]`, `[Core imports]`, and `[Atlas connection]` checks pass.

---

### Phase 1 — RAG Upgrade

| Script | What passing looks like |
| ------ | ----------------------- |
| `phase_01_retrieval_queries.py` | For 5 fixed queries, prints ranked bike names and similarity scores |
| `phase_01_rerank_smoke.py` | Prints results before and after Cohere rerank — order visibly changes for at least one query |

> These scripts do not exist yet. Create them after the embedding migration and
> Cohere reranker are wired into `core/rag/`.

---

### Phase 2 — LangGraph Agent

| Script | What passing looks like |
| ------ | ----------------------- |
| `phase_02_graph_invoke.py` | One hard-coded user message → full `AgentState` printed (truncated) |
| `phase_02_graph_stream.py` | Token chunks print to stdout as they arrive |

> These scripts do not exist yet. Create them after `core/agent/graph.py` is built.

---

### Phase 3 — Evaluation

| Script | What passing looks like |
| ------ | ----------------------- |
| `phase_03_eval_local.py` | 5 golden queries → Ragas metrics table printed to stdout |

> Depends on `evaluation/golden_queries.yaml` and `evaluation/ragas_eval.py`.

---

## Archive

`_archive/` contains the legacy scripts that predated the `core/` refactor.
They are kept for diff reference only — do not import from them.

| File | Why retired |
| ---- | ----------- |
| `01_create_vectorstore.py` | Jupyter magics (`%load_ext`), duplicated ingest logic |
| `02_rag_pipeline.py` | Jupyter magics, known bugs in agent/history wiring |
