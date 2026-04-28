"""
core/tools/compare.py
=====================
Defines the compare_bikes LangChain tool.

When the LLM should call this tool:
  User wants to see bikes side by side. Trigger keywords:
  "compare", "vs", "versus", "difference between", "which is better",
  "SuperSix vs CAAD13".
  NOT for listing bikes (→ use search) or a single bike overview (→ use summary).

How it works:
  1. Splits the comma-separated input into 2-3 bike names (caps at 3).
  2. Retrieves the top document for each bike name individually.
  3. Builds a combined context string by concatenating the docs with headers.
  4. Loads the compare prompt from core/prompts/compare.md.
  5. Runs a chain where the context is the pre-built string (not a live retriever),
     because comparison requires interleaving docs from multiple queries.
  6. Appends deduplicated IMAGE_URL markers for each bike.

Why not use the LCEL retriever-as-context pattern here:
  The summary and details tools pass the retriever directly into the chain
  (`{"context": retriever, ...}`). That works for single-query lookups. For
  comparison, we need one document per bike name — querying a single retriever
  with a blended query would not reliably surface one doc per bike. Building
  the combined context manually ensures each bike is represented.
"""
from __future__ import annotations

from typing import List

from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from core.llm import build_llm
from core.prompts import load_prompt
from core.rag.retriever import build_retriever
from core.tools._helpers import extract_image_urls_from_docs


@tool
def compare_bikes(bike_names: str) -> str:
    """Compare 2-3 Cannondale bikes side by side.
    Use when the user wants to compare specific bike models, see differences, or decide between options.

    Args:
        bike_names: Comma-separated bike names to compare (e.g. 'SuperSix EVO, CAAD13, Synapse').

    Returns:
        Structured side-by-side comparison with specs, differences, and IMAGE_URL markers.
    """
    try:
        retriever = build_retriever()
        llm = build_llm()

        names = [n.strip() for n in bike_names.split(",") if n.strip()]
        if len(names) < 2:
            return "Please provide at least 2 bike names separated by commas to compare."
        # Cap at 3 bikes — beyond that the context gets unwieldy and the
        # comparison table becomes hard to read.
        if len(names) > 3:
            names = names[:3]

        # Retrieve the best matching document for each bike individually.
        all_docs = []
        all_image_data: List[dict] = []
        for name in names:
            docs = retriever.invoke(name)
            if docs:
                all_docs.append(docs[0])
                all_image_data.extend(extract_image_urls_from_docs(docs[:1]))

        if len(all_docs) < 2:
            return "Could not find enough bikes to compare. Please check the bike names and try again."

        # Build combined context: one labeled section per bike so the LLM
        # knows which text belongs to which bike.
        combined_context = ""
        for i, doc in enumerate(all_docs):
            combined_context += (
                f"\n\n--- Bike {i + 1}: {doc.metadata.get('bike_model', 'Unknown')} ---\n"
                + doc.page_content
            )

        prompt = ChatPromptTemplate.from_template(load_prompt("compare"))

        # Context is a fixed string (not a live retriever), so we use a lambda.
        chain = (
            {
                "context": lambda _: combined_context,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(bike_names)

        # Deduplicate image URLs across all retrieved bikes.
        seen: set = set()
        for img in all_image_data:
            if img["url"] not in seen:
                result += f"\n\nIMAGE_URL: {img['url']} | {img['name']}"
                seen.add(img["url"])

        return result

    except Exception as e:
        return f"Error comparing bikes: {str(e)}"


# =============================================================================
# Manual test blocks — run individually as VS Code interactive cells (# %%)
# How to run:
#   cd ai_portfolio_projects
#   PYTHONPATH=src poetry run python src/01_cannondale_bikes_assistant/core/tools/compare.py
# Or select a cell in VS Code and run with Shift+Enter (Python Interactive).
# =============================================================================

# %% [Setup] Path + imports
import sys
from pathlib import Path

_project_dir = Path(__file__).resolve().parents[2]  # → 01_cannondale_bikes_assistant/
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from core.tools.compare import compare_bikes  # noqa: E402

# %% [Test 1] Two bikes
result_two = compare_bikes.invoke("SuperSix EVO, CAAD13")
print(result_two)

# %% [Test 2] Three bikes
result_three = compare_bikes.invoke("SuperSix EVO, Synapse, Topstone")
print(result_three)

# %% [Test 3] Edge case — fewer than 2 names
result_edge = compare_bikes.invoke("SuperSix EVO")
print(result_edge)

# %%
