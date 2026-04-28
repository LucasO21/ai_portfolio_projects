"""
core/tools/summary.py
=====================
Defines the get_bike_summary LangChain tool.

When the LLM should call this tool:
  User wants a quick overview of a specific bike. Trigger keywords:
  "tell me about", "what is", "summary of", "quick info", "brief description".
  NOT for full technical specs (→ use details) or comparison (→ use compare).

How it works:
  1. Calls build_retriever() to fetch top-k docs for the bike query.
  2. Extracts image URLs from the retrieved documents' metadata.
  3. Loads the prompt template from core/prompts/summary.md.
  4. Runs an LCEL chain: retriever → prompt → LLM → StrOutputParser.
  5. Appends IMAGE_URL markers so the UI can display bike images.

Why the chain retrieves twice (once for image URLs, once inside the chain):
  The LCEL pattern `{"context": retriever, "question": RunnablePassthrough()}`
  makes the retriever run again when the chain is invoked. This is a known
  redundancy — we retrieve first (step 1) only to pull image URLs from metadata
  before the chain runs. Phase 2 (LangGraph) will consolidate to a single
  retrieve step inside the graph's retrieve node.
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
def get_bike_summary(bike_query: str) -> str:
    """Provide a concise summary of a Cannondale bike.
    Use when the user wants a brief overview, quick description, or summary of a specific bike.

    Args:
        bike_query: The bike name or descriptive query (e.g. 'Scalpel', 'Synapse Carbon').

    Returns:
        A 3-4 sentence summary with key bullet points and IMAGE_URL markers.
    """
    try:
        retriever = build_retriever()
        llm = build_llm()

        # Pre-fetch docs to extract image URLs from metadata.
        docs = retriever.invoke(bike_query)
        image_data = extract_image_urls_from_docs(docs)

        prompt = ChatPromptTemplate.from_template(load_prompt("summary"))

        # LCEL chain: retriever provides context, query passes through unchanged.
        # build_retriever() and build_llm() are lru_cache'd, so calling them here
        # is instant — they return the already-constructed objects.
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(bike_query)

        # IMAGE_URL markers are parsed by the UI layer (Streamlit) and displayed
        # as images in the sidebar. The markers are stripped before storing in
        # chat history so the stored text stays clean.
        for img in image_data:
            result += f"\n\nIMAGE_URL: {img['url']} | {img['name']}"

        return result

    except Exception as e:
        return f"Error generating summary: {str(e)}"
