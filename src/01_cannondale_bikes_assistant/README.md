# 🚴‍♂️ Cannondale Bikes AI Assistant
***An Agentic RAG System on LangGraph with Streaming Tool Calls***

<div align="center">
<img src="png/app_main.png" alt="Cannondale Bike AI Assistant Homepage" width="800">
</div>

---

## 📋 Project Overview

Shopping for a high-end bicycle online means sifting through hundreds of models with dense specs spread across product pages. Comparing frames, drivetrains, and pricing across the Cannondale lineup is time-consuming — especially for riders who aren't sure what they need.

This project tackles that with an **agentic RAG system** that acts like a knowledgeable bike-shop employee. A **LangGraph** agent backed by five specialized tools searches, summarizes, details, compares, and recommends Cannondale bikes through natural conversation. The agent decides which tool to invoke based on intent, retrieves bike data from a **MongoDB Atlas** vector store (optionally **reranked by Cohere**), and streams a grounded answer into the Streamlit UI.

### 🎯 Key Concepts

- **LangGraph ReAct loop** — `StateGraph` with a single `call_model` node and the prebuilt `ToolNode`. The LLM keeps calling tools until it decides it has enough context.
- **Tool-calling RAG** — Five typed tools (price ranges, bike types, experience levels) the LLM invokes as function calls, each with its own prompt and retrieval strategy.
- **Vector search + optional rerank** — Atlas vector similarity for candidate retrieval; **Cohere Rerank** via `ContextualCompressionRetriever` when `COHERE_API_KEY` is set (`k=20` candidates → top `5`).
- **Token streaming** — Tokens are forwarded out of the model node via LangGraph's custom stream writer, so the UI can render with `st.write_stream` in real time.
- **Layered architecture** — `core/` holds all logic (config, llm, rag, tools, agent, prompts, citations); `app/streamlit_app.py` is a thin UI shell that imports `core/` only.

---

## 🏗️ How It Works

```
                  ┌──────────────────────┐
                  │   User Query         │
                  │  (Streamlit chat)    │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │   call_model node    │◀──┐
                  │  (LLM + tools bound) │   │
                  └──────────┬───────────┘   │
                  tool_calls?│               │
                             ▼               │
                  ┌──────────────────────┐   │
                  │      ToolNode        │   │
                  │  search / summary /  │───┘
                  │ details / compare /  │ loop until no tool_calls
                  │     recommend        │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  MongoDB Atlas       │
                  │  vector search       │
                  │  + Cohere rerank     │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  Streaming answer    │
                  │  + inline sources    │
                  └──────────────────────┘
```

### Five Specialized Tools

| Tool | Purpose | Notes |
| ---- | ------- | ----- |
| `search_bikes` | Browse / filter the catalog | Optional `bike_type`, `min_price`, `max_price` |
| `get_bike_summary` | 3–4 sentence overview + bullets | |
| `get_bike_details` | Full specs and components | Pulls model code / color / price from metadata |
| `compare_bikes` | Side-by-side, 2–3 bikes | Markdown table with rider recommendation |
| `get_recommendation` | Personalized suggestion | Optional `budget`, `experience_level` |

### Code — agent graph

```python
# core/agent/graph.py
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from core.agent.nodes import call_model
from core.agent.state import AgentState
from core.tools import TOOLS

def _should_continue(state):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

builder = StateGraph(AgentState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", _should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "model")
graph = builder.compile()
```

### Code — model node with token streaming

```python
# core/agent/nodes.py
from langgraph.config import get_stream_writer
from langchain_core.messages import SystemMessage
from core.llm import build_llm
from core.prompts import load_prompt
from core.tools import TOOLS

def call_model(state):
    writer = get_stream_writer()  # no-op unless stream_mode=["custom", ...]
    llm = build_llm().bind_tools(TOOLS)
    messages = [SystemMessage(content=load_prompt("system")), *state["messages"]]

    merged = None
    for chunk in llm.stream(messages):
        text = getattr(chunk, "content", "") or ""
        if isinstance(text, str) and text:
            writer(text)                        # forward delta to st.write_stream
        merged = chunk if merged is None else merged + chunk
    return {"messages": [merged]}
```

---

## 🛠️ Tech Stack

```
🧠 LLM:          OpenAI GPT-4o (configurable)
🔍 Embeddings:   OpenAI text-embedding-ada-002
📈 Reranker:     Cohere Rerank (optional)
🗄️ Data Store:   MongoDB Atlas (vectorSearch index)
🌐 Frontend:     Streamlit
🔗 Framework:    LangGraph + LangChain
📦 Tooling:      Poetry, Ruff, Pydantic Settings
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10–3.12
- OpenAI API key + MongoDB Atlas (with a vector search index named `vector_index`)
- (Optional) Cohere API key for rerank

### Install

```bash
git clone https://github.com/LucasO21/ai_portfolio_projects.git
cd ai_portfolio_projects
poetry install
```

### `.env` (repo root)

```dotenv
OPENAI_API_KEY=...
MONGO_DB_URI=...
COHERE_API_KEY=...           # optional — enables reranking

# Optional overrides
LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0.1
EMBEDDING_MODEL=text-embedding-ada-002
RETRIEVER_K=5
CANNONDALE_DB_NAME=cannondale_bikes_db
CANNONDALE_COLLECTION=bikes_collection
VECTOR_INDEX_NAME=vector_index
```

### Run the app

```bash
poetry run streamlit run src/01_cannondale_bikes_assistant/app/streamlit_app.py
```

Available at `http://localhost:8501`. The legacy `app/app2.py` (LangChain `AgentExecutor`) is kept for reference.

> Data is already in MongoDB Atlas; no re-ingest is required. The original ingestion script lives in `dev/_archive/01_create_vectorstore.py` if you need to rebuild the collection.

---

## 💡 UI features

- **Streaming responses** with a `Thinking…` / `Calling tool: <name>…` status indicator before the first token arrives.
- **Inline sources expander** under each assistant message (bike images, names, prices) parsed from tool output.
- **Sample Questions** expander at the top of the chat.
- **Sidebar** with token usage + estimated cost and a Clear Chat button.

### Side-by-side comparison

<div align="center">
<img src="png/comparison.png" alt="Side-by-side comparison of two Cannondale bikes" width="800">
</div>

### Conversational follow-ups

<div align="center">
<img src="png/follow_up.png" alt="Follow-up query reformatting the comparison into a table" width="800">
</div>

---

## 📁 Project Structure

```
src/01_cannondale_bikes_assistant/
├── app/
│   ├── streamlit_app.py         # Phase 3 UI (LangGraph + core/)
│   └── app2.py                  # legacy AgentExecutor app (reference only)
├── core/
│   ├── config.py                # pydantic-settings; reads .env
│   ├── llm.py                   # build_llm() factory (cached)
│   ├── citations.py             # tool output → citation cards
│   ├── agent/
│   │   ├── state.py             # AgentState (messages + add_messages)
│   │   ├── nodes.py             # call_model with streaming
│   │   └── graph.py             # compiled StateGraph
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── vectorstore.py       # MongoDBAtlasVectorSearch (cached)
│   │   ├── retriever.py         # vector search → optional Cohere rerank
│   │   └── reranker.py
│   ├── tools/
│   │   ├── search.py
│   │   ├── summary.py
│   │   ├── details.py
│   │   ├── compare.py
│   │   ├── recommend.py
│   │   └── _helpers.py
│   └── prompts/                 # system + per-tool prompt templates
├── dev/
│   ├── phase_00_config_smoke.py
│   └── _archive/                # old ingestion + RAG experiments
├── database/bikes_csv/          # source CSVs (used at ingest)
└── png/                         # screenshots
```
