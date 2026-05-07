"""
core/agent/graph.py
===================
Assembles the LangGraph StateGraph and exposes a compiled `graph` object.

Graph structure:
  START → call_model → [tool_calls?] → yes → tools → call_model (loop)
                                      → no  → END

  call_model: LLM decides what to do (answer directly or call a tool).
  tools:      LangGraph's prebuilt ToolNode executes whichever tool the LLM
              requested and appends a ToolMessage to state.
  The loop continues until the LLM responds without tool_calls.

Why this pattern (ReAct / tool-calling loop)?
  The LLM sees the full message history on each call, including all prior
  ToolMessages. This lets it chain tools — e.g. search_bikes first to find
  candidates, then get_bike_details on the top result — without any custom
  orchestration code. The LLM decides when it has enough information to
  stop calling tools and return a final answer.

Entry point for all callers:
  from core.agent import graph                          # sync invoke
  result = graph.invoke({"messages": [HumanMessage(content="...")]})

  from core.agent import graph                          # streaming
  for chunk in graph.stream({"messages": [HumanMessage(content="...")]}):
      print(chunk)

Run this file directly to test the graph end-to-end without Streamlit:
  PYTHONPATH=src/01_cannondale_bikes_assistant poetry run python \\
    src/01_cannondale_bikes_assistant/core/agent/graph.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_dir = Path(__file__).resolve().parents[2]
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from core.agent.nodes import call_model
from core.agent.state import AgentState
from core.tools import TOOLS


# ---------------------------------------------------------------------------
# Section 1 — Routing condition
# ---------------------------------------------------------------------------

def _should_continue(state: AgentState) -> str:
    """Decide whether to call a tool or end the graph.

    Called after every call_model node. Inspects the last message:
    - If the LLM included tool_calls → route to "tools" node.
    - Otherwise → the LLM produced a final answer → END.

    This is the only conditional edge in the graph. Everything else is fixed.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Section 2 — Graph assembly
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    """Construct and compile the StateGraph.

    Kept as a function (rather than module-level statements) so the graph
    can be rebuilt in tests without re-importing the module.
    """
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("model", call_model)
    # ToolNode is LangGraph's prebuilt executor: it reads tool_calls from the
    # last AIMessage, invokes the matching tool, and returns a ToolMessage.
    builder.add_node("tools", ToolNode(TOOLS))

    # Edges
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model",
        _should_continue,
        # Explicit map makes the routing readable in LangSmith traces.
        {"tools": "tools", END: END},
    )
    # After tools execute, always loop back to the LLM.
    builder.add_edge("tools", "model")

    return builder.compile()


# Module-level compiled graph — import this everywhere.
graph = _build_graph()

