# Cannondale Bikes Assistant — Phase 3 UI (LangGraph + core/)
# Run from repo root:
#   poetry run streamlit run src/01_cannondale_bikes_assistant/app/streamlit_app.py

from __future__ import annotations

import re
import sys
from pathlib import Path

_project_dir = Path(__file__).resolve().parents[1]
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from typing import Any, Mapping, Sequence, cast

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

st.set_page_config(
    page_title="Cannondale Bike Expert",
    page_icon="🚴‍♂️",
    layout="centered",
)

from core.agent.graph import graph  # noqa: E402
from core.citations import citations_from_messages  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GPT-4o pricing (per token). Update if you switch models in .env.
INPUT_TOKEN_COST = 2.50 / 1_000_000
OUTPUT_TOKEN_COST = 10.00 / 1_000_000
IMAGE_DISPLAY_WIDTH = 320

_IMAGE_URL_LINE_RE = re.compile(r"\n*IMAGE_URL:\s*https?://\S+.*", re.MULTILINE)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

WELCOME = (
    "👋 Hi! I'm your Cannondale bike assistant. "
    "What would you like to know?"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_content(msg: BaseMessage) -> str:
    c = msg.content
    return c if isinstance(c, str) else str(c)


def _strip_image_markers(text: str) -> str:
    """Remove IMAGE_URL: lines and convert markdown images to plain links."""
    text = _IMAGE_URL_LINE_RE.sub("", text)
    text = _MD_IMAGE_RE.sub(r"[\1](\2)", text)
    return text.strip()


def _add_token_usage(msg: AIMessage) -> None:
    """Pull usage_metadata from the final AIMessage and add to running totals."""
    usage = getattr(msg, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return
    pt = int(usage.get("input_tokens", 0) or 0)
    ct = int(usage.get("output_tokens", 0) or 0)
    tt = int(usage.get("total_tokens", pt + ct) or 0)
    st.session_state.total_prompt_tokens += pt
    st.session_state.total_completion_tokens += ct
    st.session_state.total_tokens += tt
    st.session_state.total_cost += pt * INPUT_TOKEN_COST + ct * OUTPUT_TOKEN_COST


def _render_assistant_message(text: str, citations: Sequence[Mapping[str, Any]]) -> None:
    """Render a finalized assistant message: clean text + inline sources expander."""
    st.markdown(_strip_image_markers(text))
    if not citations:
        return
    has_image = any((c.get("image_url") or "").startswith("http") for c in citations)
    label = f"📷 Sources ({len(citations)})" if has_image else f"🔗 Sources ({len(citations)})"
    with st.expander(label, expanded=False):
        for c in citations:
            url = (c.get("image_url") or "").strip()
            name = (c.get("name") or "Cannondale Bike").strip()
            model = (c.get("model") or "").strip()
            price = (c.get("price") or "").strip()
            caption_bits = [name]
            if model:
                caption_bits.append(model)
            if price:
                caption_bits.append(price)
            caption = " — ".join(caption_bits)
            if url.startswith("http"):
                st.image(url, width=IMAGE_DISPLAY_WIDTH, caption=caption)
            else:
                st.markdown(f"- **{caption}**")


def _stream_assistant_turn() -> None:
    """Stream a single assistant turn into the current chat container.

    - Streams text deltas via LangGraph's ``custom`` mode into a placeholder.
    - When done, replaces the placeholder with cleaned markdown and an inline
      sources expander built from any ToolMessages produced this turn.
    """
    pre_count = len(st.session_state.messages)

    placeholder = st.empty()
    buf: list[str] = []
    last_values: dict[str, Any] | None = None

    for mode, chunk in cast(Any, graph).stream(
        {"messages": list(st.session_state.messages)},
        stream_mode=["custom", "values"],
    ):
        if mode == "custom" and isinstance(chunk, str) and chunk:
            buf.append(chunk)
            placeholder.markdown(_strip_image_markers("".join(buf)) + "▌")
        elif mode == "values" and isinstance(chunk, dict):
            last_values = chunk

    if last_values and isinstance(last_values.get("messages"), list):
        st.session_state.messages = list(last_values["messages"])

    new_msgs = st.session_state.messages[pre_count:]
    final_ai: AIMessage | None = None
    for m in reversed(new_msgs):
        if isinstance(m, AIMessage) and not m.tool_calls:
            final_ai = m
            break

    if final_ai is not None:
        _add_token_usage(final_ai)
        text = _text_content(final_ai)
    else:
        text = "".join(buf)

    citations = citations_from_messages([m for m in new_msgs if isinstance(m, ToolMessage)])
    msg_idx = len(st.session_state.messages) - 1
    if citations:
        st.session_state.message_citations[msg_idx] = citations

    placeholder.empty()
    _render_assistant_message(text, citations)


def _reset_session() -> None:
    st.session_state.messages = []
    st.session_state.message_citations = {}
    st.session_state.total_prompt_tokens = 0
    st.session_state.total_completion_tokens = 0
    st.session_state.total_tokens = 0
    st.session_state.total_cost = 0.0


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "message_citations" not in st.session_state:
    st.session_state.message_citations = {}
for k, default in (
    ("total_prompt_tokens", 0),
    ("total_completion_tokens", 0),
    ("total_tokens", 0),
    ("total_cost", 0.0),
):
    st.session_state.setdefault(k, default)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🚴‍♂️ Cannondale Bike AI Assistant")
st.markdown("**Powered by LangGraph + 5 specialized tools**: Search, Summary, Details, Compare, Recommend.")
st.write("---")

with st.expander("💡 Sample Questions - Try These!", expanded=False):
    sample_groups: list[tuple[str, list[str]]] = [
        (
            "🔍 **Search & Filter**",
            [
                "Show me mountain bikes under $5000",
                "What gravel bikes do you have?",
                "List electric bikes between $4000 and $8000",
            ],
        ),
        (
            "📝 **Quick Summaries**",
            [
                "Tell me about the Scalpel",
                "Quick summary of Synapse Carbon",
                "What is the Topstone?",
            ],
        ),
        (
            "📋 **Detailed Specs**)",
            [
                "Full specifications for Jekyll 1",
                "Detailed breakdown of SuperSix EVO",
                "Everything about the Moterra Neo",
            ],
        ),
        (
            "⚖️ **Comparisons**",
            [
                "Compare Synapse vs CAAD13",
                "Differences between Topstone and Topstone Carbon",
                "Compare Scalpel, Habit, and Jekyll",
            ],
        ),
        (
            "💡 **Recommendations**",
            [
                "Best bike for weekend trail riding under $4,000",
                "What road bike for a beginner with $2,500 budget?",
                "Recommend a commuter bike for city riding",
            ],
        ),
    ]
    for heading, questions in sample_groups:
        st.markdown(heading)
        bullets = "\n".join("- " + q.replace("$", r"\$") for q in questions)
        st.markdown(bullets)


# ---------------------------------------------------------------------------
# Greeting (rendered once when there's no history)
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(WELCOME)


# ---------------------------------------------------------------------------
# Render chat history (final assistant turns + their inline sources)
# ---------------------------------------------------------------------------

for idx, msg in enumerate(st.session_state.messages):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(_text_content(msg))
    elif isinstance(msg, AIMessage) and not msg.tool_calls and _text_content(msg).strip():
        with st.chat_message("assistant"):
            cites = st.session_state.message_citations.get(idx, [])
            _render_assistant_message(_text_content(msg), cites)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

user_text = st.chat_input("Ask about any Cannondale bike…")

if user_text:
    user_text = user_text.strip()
    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
        st.session_state.messages.append(HumanMessage(content=user_text))

        with st.chat_message("assistant"):
            try:
                _stream_assistant_turn()
            except Exception as e:  # noqa: BLE001
                st.error(f"Sorry, I encountered an error: {e}")


# ---------------------------------------------------------------------------
# Sidebar — token usage + clear chat (mirrors legacy app2.py)
# ---------------------------------------------------------------------------

with st.sidebar:
    with st.expander("📊 Token Usage & Cost", expanded=False):
        st.markdown("**Current Session:**")
        st.write(f"**Prompt Tokens:** {st.session_state.total_prompt_tokens:,}")
        st.write(f"**Completion Tokens:** {st.session_state.total_completion_tokens:,}")
        st.write(f"**Total Tokens:** {st.session_state.total_tokens:,}")
        st.write(f"**Estimated Cost:** ${st.session_state.total_cost:.4f}")

    st.write("")

    if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
        _reset_session()
        st.rerun()

st.markdown("---")
