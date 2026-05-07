"""
core/agent/state.py
===================
Defines AgentState — the single shared data structure that flows through
every node in the LangGraph StateGraph.

What is a StateGraph state?
  In LangGraph, every node receives the full state dict and returns a partial
  update. LangGraph merges the update back into state using a "reducer" for
  each field. The default reducer is last-write-wins (new value replaces old).
  The `messages` field uses the `add_messages` reducer instead, which APPENDS
  new messages to the existing list rather than replacing it — this is how
  conversation history is preserved across multiple node invocations.

Fields:
  messages  — Full conversation history. HumanMessage (user input) →
               AIMessage (LLM response, possibly with tool_calls) →
               ToolMessage (tool output) → AIMessage (final answer).
               The add_messages reducer means nodes just return new messages
               and LangGraph handles appending them correctly.

Why only messages?
  A minimal state keeps the graph easy to reason about. The LLM sees the full
  message history on every call, so intent, retrieved content, and prior
  answers are all implicitly available via the messages. Extra fields like
  `intent` or `retrieved_bikes` can be added in Phase 3+ if needed for
  features like citation panels or eval — they'd just be additional TypedDict
  keys with their own reducers.
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Annotated[..., add_messages] tells LangGraph to use the add_messages
    # reducer: each node's returned messages are APPENDED, not replaced.
    # Without this annotation, returning {"messages": [...]} would overwrite
    # the entire history on every node call.
    messages: Annotated[list[BaseMessage], add_messages]
