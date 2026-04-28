"""
core/tools/details.py
=====================
Defines the get_bike_details LangChain tool.

When the LLM should call this tool:
  User wants full technical specifications or in-depth analysis. Trigger keywords:
  "specs", "specifications", "detailed", "everything about", "full breakdown",
  "technical details", "components".
  NOT for a quick overview (→ use summary) or comparison (→ use compare).

How it works:
  1. Retrieves top-k docs for the bike query.
  2. Extracts image URLs and structured metadata (model code, color, price)
     from the top result.
  3. Loads the details prompt from core/prompts/details.md.
  4. Runs an LCEL chain: retriever → prompt → LLM → StrOutputParser.
  5. Appends a metadata section and IMAGE_URL markers to the result.

Why we pull metadata from docs[0] separately (not from the LLM output):
  The LLM may paraphrase or omit the model code. Extracting it directly from
  MongoDB metadata guarantees accuracy — this is important for a product
  assistant where a wrong model code is a bug, not a style issue.
"""
from __future__ import annotations

from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from core.llm import build_llm
from core.prompts import load_prompt
from core.rag.retriever import build_retriever
from core.tools._helpers import extract_image_urls_from_docs


@tool
def get_bike_details(bike_query: str) -> str:
    """Provide comprehensive, detailed specifications and analysis of a Cannondale bike.
    Use when the user wants full specs, detailed descriptions, or in-depth technical information.

    Args:
        bike_query: The bike name or descriptive query (e.g. 'SuperSix EVO', 'Jekyll 1 specs').

    Returns:
        Detailed description with specs, structured metadata, and IMAGE_URL markers.
    """
    try:
        retriever = build_retriever()
        llm = build_llm()

        docs = retriever.invoke(bike_query)
        image_data = extract_image_urls_from_docs(docs)

        # Pull structured metadata from the top document to append verbatim.
        # We don't rely on the LLM to reproduce these accurately.
        metadata_section = ""
        if docs:
            m = docs[0].metadata
            parts = []
            if m.get("model_code"):
                parts.append(f"**Model Code:** {m['model_code']}")
            if m.get("color"):
                parts.append(f"**Color:** {m['color']}")
            if m.get("price"):
                parts.append(f"**Price:** ${m['price']}")
            if parts:
                metadata_section = (
                    "\n\n**Additional Information:**\n"
                    + "\n".join(f"- {p}" for p in parts)
                )

        prompt = ChatPromptTemplate.from_template(load_prompt("details"))

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(bike_query)
        result += metadata_section

        for img in image_data:
            result += f"\n\nIMAGE_URL: {img['url']} | {img['name']}"

        return result

    except Exception as e:
        return f"Error generating details: {str(e)}"


# =============================================================================
# Manual test blocks — run individually as VS Code interactive cells (# %%)
# How to run:
#   cd ai_portfolio_projects
#   PYTHONPATH=src poetry run python src/01_cannondale_bikes_assistant/core/tools/details.py
# Or select a cell in VS Code and run with Shift+Enter (Python Interactive).
# =============================================================================

# %% [Setup] Path + imports
import sys
from pathlib import Path

_project_dir = Path(__file__).resolve().parents[2]  # → 01_cannondale_bikes_assistant/
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from core.tools.details import get_bike_details  # noqa: E402

# %% [Test 1] Specific bike by name
result_specific = get_bike_details.invoke("SuperSix EVO")
print(result_specific)

# %% [Test 2] Descriptive query (specs/technical language)
result_specs = get_bike_details.invoke("Jekyll 1 full suspension specs")
print(result_specs)

# %% [Test 3] Vague query — tests retriever fallback behavior
result_vague = get_bike_details.invoke("lightweight road bike")
print(result_vague)

# %%
