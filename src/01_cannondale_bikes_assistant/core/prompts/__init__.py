"""
core/prompts/__init__.py
========================
Utility for loading prompt templates from .md files in this directory.

Each .md file is a plain-text template string with {variable} placeholders
compatible with LangChain's ChatPromptTemplate.from_template().

Why .md files instead of inline Python strings:
  - No Python string escaping — the prompt reads exactly as the LLM sees it.
  - Diffs are clean — a prompt edit shows only the changed lines, not surrounding
    Python code.
  - Reusable — multiple modules can load the same prompt without duplication.

Available prompt files:
  system.md    — Agent system prompt (tool-selection guidelines, response rules)
  summary.md   — Prompt for get_bike_summary tool
  details.md   — Prompt for get_bike_details tool
  compare.md   — Prompt for compare_bikes tool
  recommend.md — Prompt for get_recommendation tool
"""
from __future__ import annotations

from pathlib import Path

# This file's directory is core/prompts/ — prompt .md files live alongside it.
_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Read and return the contents of core/prompts/{name}.md.

    Args:
        name: Filename without extension (e.g. "summary", "details", "system").

    Returns:
        The raw template string, ready to pass to ChatPromptTemplate.from_template().

    Raises:
        FileNotFoundError: if no matching .md file exists in core/prompts/.

    Example:
        prompt = ChatPromptTemplate.from_template(load_prompt("summary"))
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"Available prompts: {[p.stem for p in _PROMPTS_DIR.glob('*.md')]}"
        )
    return path.read_text(encoding="utf-8")
