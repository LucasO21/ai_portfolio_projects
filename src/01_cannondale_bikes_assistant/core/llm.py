"""
core/llm.py
===========
Responsible for building and caching the ChatOpenAI language model.

Where it fits in the pipeline:
  config.py  →  llm.py  →  core/tools/*.py  →  core/agent/graph.py (Phase 2)

Why a dedicated module:
  - Single place to configure model name, temperature, and any future LLM params
    (context window, seed, logit bias). Change .env, not scattered tool files.
  - @lru_cache means every module that calls build_llm() gets the same object,
    avoiding repeated httpx client initialization inside langchain_openai.

Streaming note:
  ChatOpenAI supports streaming via .astream() regardless of how it was
  constructed. We do NOT bake streaming=True into this builder because the same
  model object is shared by sync tool chains and the async agent graph.
  Streaming is requested at call time (e.g. graph.astream(...) in Phase 2).

When to read this file:
  - Phase 0 (now):  to understand LLM configuration.
  - Phase 2 (LangGraph):  the agent graph will call build_llm() and pass it to
    StateGraph nodes. No changes to this file are expected.

Run directly to verify the model connects and returns a response:
  PYTHONPATH=src poetry run python src/01_cannondale_bikes_assistant/core/llm.py
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

# Make `core` importable when this file is run directly.
# parents[0] = core/, parents[1] = 01_cannondale_bikes_assistant/
_project_dir = Path(__file__).resolve().parents[1]
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from langchain_openai import ChatOpenAI

from core.config import get_settings


# ---------------------------------------------------------------------------
# Section 1 — LLM factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def build_llm() -> ChatOpenAI:
    """Return a cached ChatOpenAI instance configured from Settings.

    Why @lru_cache here:
      Each tool function calls build_llm() independently. Without caching,
      every tool invocation would create a new ChatOpenAI object, which
      initialises an httpx client and validates auth on every construction.
      Caching keeps one client alive for the entire process lifetime.

    Temperature 0.1 (the default):
      Low temperature = more deterministic, factual responses. This is
      appropriate for a product assistant where hallucinated specs are harmful.
      Raise it (e.g. to 0.5) if you want more varied recommendation phrasing.
    """
    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,              # e.g. "gpt-4o" — override via LLM_MODEL in .env
        temperature=s.llm_temperature,  # e.g. 0.1 — override via LLM_TEMPERATURE in .env
        api_key=s.openai_key,           # type: ignore[arg-type]  (SecretStr vs str)
    )


# ---------------------------------------------------------------------------
# Run directly to verify the model connects and generates a response.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    llm = build_llm()
    s = get_settings()

    print("=== llm.py smoke test ===\n")
    print(f"model       : {s.llm_model}")
    print(f"temperature : {s.llm_temperature}")
    print("sending test message...")

    response = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
    print(f"response    : {response.content!r}")
    print("\n=== OK ===")
