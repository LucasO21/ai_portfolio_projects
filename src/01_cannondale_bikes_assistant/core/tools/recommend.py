"""
core/tools/recommend.py
=======================
Defines the get_recommendation LangChain tool.

When the LLM should call this tool:
  User asks for a personalised suggestion. Trigger keywords:
  "recommend", "suggest", "best for", "should I get", "what bike for me",
  "beginner road bike", "trail riding under $4000".
  NOT for browsing a category (→ use search) or specs on a known bike (→ use details).

How it works:
  1. Retrieves top-k docs for the rider's use-case query.
  2. Optionally filters out bikes over the stated budget.
  3. Extracts image URLs from the remaining docs.
  4. Loads the recommend prompt from core/prompts/recommend.md.
  5. Runs a chain with a pre-built context string (not a live retriever) so
     the budget and experience variables can be injected alongside context.
  6. Appends IMAGE_URL markers.

Why we inject budget and experience as separate template variables:
  The LCEL pattern `{"context": retriever, "question": RunnablePassthrough()}`
  only threads one input variable through. Recommendation needs three: context,
  budget, and experience level. Solving this with lambdas that capture the
  values keeps the chain pattern consistent without switching to a different
  prompt abstraction.
"""
from __future__ import annotations

from typing import Optional

from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from core.llm import build_llm
from core.prompts import load_prompt
from core.rag.retriever import build_retriever
from core.tools._helpers import extract_image_urls_from_docs, parse_price


@tool
def get_recommendation(
    query: str,
    budget: Optional[float] = None,
    experience_level: Optional[str] = None,
) -> str:
    """Recommend the best Cannondale bike for the user's needs.
    Use when the user asks for a recommendation, suggestion, or 'which bike should I get'.

    Args:
        query: Description of riding needs, terrain, and goals.
        budget: Optional maximum budget in USD.
        experience_level: Optional rider experience ('beginner', 'intermediate', 'advanced').

    Returns:
        Personalised bike recommendation with reasoning and IMAGE_URL markers.
    """
    try:
        retriever = build_retriever()
        llm = build_llm()

        docs = retriever.invoke(query)

        # Apply budget filter before building context so the LLM only sees
        # bikes the rider can actually afford. If the filter removes everything,
        # fall back to all retrieved docs rather than returning nothing.
        if budget is not None:
            filtered = [
                doc for doc in docs
                if (pv := parse_price(doc.metadata.get("price"))) is None or pv <= budget
            ]
            if filtered:
                docs = filtered

        image_data = extract_image_urls_from_docs(docs)

        # Flatten docs to a single context string for the prompt.
        context_text = "\n\n".join(doc.page_content for doc in docs)

        # Format budget/experience for the prompt (human-readable, not Python repr).
        budget_str = f"${budget:,.0f}" if budget is not None else "not specified"
        exp_str = experience_level if experience_level else "not specified"

        prompt = ChatPromptTemplate.from_template(load_prompt("recommend"))

        # Lambdas capture budget_str and exp_str from the closure above.
        # RunnablePassthrough() threads the query string through unchanged.
        chain = (
            {
                "context": lambda _: context_text,
                "question": RunnablePassthrough(),
                "budget": lambda _: budget_str,
                "experience": lambda _: exp_str,
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(query)

        for img in image_data:
            result += f"\n\nIMAGE_URL: {img['url']} | {img['name']}"

        return result

    except Exception as e:
        return f"Error generating recommendation: {str(e)}"
