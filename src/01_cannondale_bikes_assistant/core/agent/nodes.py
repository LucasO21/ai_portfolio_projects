"""
core/agent/nodes.py
===================
Node functions for the LangGraph StateGraph.

What is a node?
  A node is a plain Python function with signature (state: AgentState) -> dict.
  The dict it returns is a partial state update — LangGraph merges it into the
  full state using each field's reducer. Nodes should do one thing and be small
  enough to test in isolation.

Nodes defined here:
  call_model  — Sends the full message history (+ system prompt) to the LLM.
                The LLM may respond with text (final answer) or with tool_calls
                (requests to invoke one of the 5 tools). LangGraph's routing
                in graph.py decides which case we're in.

The ToolNode (tool execution) is built directly in graph.py using LangGraph's
prebuilt ToolNode — no custom code needed there.

Flow reminder:
  call_model → [has tool_calls?] → yes → ToolNode → call_model → ...
                                 → no  → END
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_project_dir = Path(__file__).resolve().parents[2]
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from langchain_core.messages import SystemMessage

from core.agent.state import AgentState
from core.llm import build_llm
from core.prompts import load_prompt
from core.tools import TOOLS


# ---------------------------------------------------------------------------
# Section 1 — LLM with tools bound (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _llm_with_tools():
    """Return the LLM with all tools bound, cached for the process lifetime.

    bind_tools() tells the LLM which tools are available and their schemas.
    The LLM can then respond with tool_calls instead of plain text when it
    decides a tool is needed. Caching avoids re-binding on every node call.
    """
    return build_llm().bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Section 2 — call_model node
# ---------------------------------------------------------------------------

def call_model(state: AgentState) -> dict:
    """Invoke the LLM with the current message history.

    Inputs (from state):
      state["messages"] — full conversation so far (HumanMessage, AIMessage,
                          ToolMessage in sequence).

    Returns (partial state update):
      {"messages": [AIMessage]} — the LLM's response is appended to history
      via the add_messages reducer in AgentState.

    The system prompt is prepended on every call but NOT stored in state.
    This keeps state clean (no duplicate system messages) while ensuring
    the LLM always has its instructions, even after tool round-trips.

    After this node, graph.py checks if the returned AIMessage has tool_calls.
    If yes → ToolNode runs the requested tool and loops back here.
    If no  → the message is the final answer and the graph ends.
    """
    system = SystemMessage(content=load_prompt("system"))

    # Prepend system message so it's always first, regardless of how many
    # tool round-trips have already happened.
    messages = [system] + list(state["messages"])

    response = _llm_with_tools().invoke(messages)
    return {"messages": [response]}
