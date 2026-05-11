"""
core/agent/
===========
LangGraph-based agent for the Cannondale Bikes Assistant.

Exports:
  graph  — compiled StateGraph, the single entry point for all queries.

Usage:
  from core.agent import graph
  result = graph.invoke({"messages": [HumanMessage(content="...")]})
"""
from core.agent.graph import graph

__all__ = ["graph"]
