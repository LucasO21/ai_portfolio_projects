# CANNONDALE BIKE EXPERT v2 - WITH 5 SPECIALIZED AI TOOLS
# WEBSITE: https://www.cannondale.com/en-us

# Run Streamlit ----
# poetry run streamlit run src/01_cannondale_bikes_assistant/app/app2.py

# ==============================================================================
# IMPORTS
# ==============================================================================

# LangChain ----
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.callbacks import get_openai_callback

# Standard Library ----
import streamlit as st
import re
import os
import sys
from pathlib import Path
from typing import Optional, List


# ==============================================================================
# PATH SETUP
# ==============================================================================

def find_project_root(start: Path) -> Path:
    for parent in start.resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (missing pyproject.toml)")

project_root = find_project_root(Path(__file__))
sys.path.append(str(project_root / "src"))

# Custom Imports ----
from global_utilities.general.paths import CANNONDALE_BIKES_ASSISTANT_DIR
from global_utilities.general.api_keys import get_env_key
from global_utilities.general.mongo import get_mongo_client


# ==============================================================================
# CONFIGURATION
# ==============================================================================

OPENAI_API_KEY: str = get_env_key("openai")

LLM_MODEL: str = "gpt-4o"
EMBEDDING_MODEL: str = "text-embedding-ada-002"
LLM_TEMPERATURE: float = 0.1
RETRIEVER_K: int = 5
MAX_AGENT_ITERATIONS: int = 5
IMAGE_DISPLAY_WIDTH: int = 300
VECTOR_INDEX_NAME: str = "vector_index"
TEXT_KEY: str = "text"
EMBEDDING_KEY: str = "embedding"

# GPT-4o pricing (per 1M tokens)
INPUT_TOKEN_COST: float = 2.50 / 1_000_000
OUTPUT_TOKEN_COST: float = 10.00 / 1_000_000

# MongoDB Connection ----
MONGO_CLIENT, MONGO_DB_NAME, MONGO_COLLECTION_NAME, MONGO_COLLECTION = get_mongo_client()


# ==============================================================================
# CACHED INITIALIZATION
# ==============================================================================

@st.cache_resource
def get_embedding_model() -> OpenAIEmbeddings:
    """Initialize and cache the OpenAI embedding model."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )


@st.cache_resource
def get_vectorstore() -> MongoDBAtlasVectorSearch:
    """Initialize and cache the MongoDB Atlas vector store."""
    return MongoDBAtlasVectorSearch(
        collection=MONGO_COLLECTION,
        embedding=get_embedding_model(),
        index_name=VECTOR_INDEX_NAME,
        text_key=TEXT_KEY,
        embedding_key=EMBEDDING_KEY,
    )


@st.cache_resource
def get_retriever():
    """Initialize and cache the retriever with k=5."""
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )


@st.cache_resource
def get_llm() -> ChatOpenAI:
    """Initialize and cache the ChatOpenAI model."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY,
    )


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def extract_image_urls_from_docs(docs: list) -> List[dict]:
    """Extract unique bike image URLs and names from retrieved documents."""
    results: List[dict] = []
    seen: set = set()
    for doc in docs:
        url = doc.metadata.get("bike_image_url")
        if url and url not in seen and isinstance(url, str) and url.startswith("http"):
            name = doc.metadata.get("bike_model") or doc.metadata.get("bike_name") or "Cannondale Bike"
            results.append({"url": url, "name": name})
            seen.add(url)
    return results


def strip_image_markers(text: str) -> str:
    """Remove IMAGE_URL: markers and convert markdown images to links."""
    text = re.sub(r'\n*IMAGE_URL:\s*https?://\S+.*', '', text)
    # Convert ![alt](url) to [alt](url) so agent can't force inline images
    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'[\1](\2)', text)
    return text.strip()


def extract_urls_from_text(text: str) -> List[dict]:
    """Extract IMAGE_URL markers (with optional bike name) from agent response text."""
    results: List[dict] = []
    for match in re.finditer(r'IMAGE_URL:\s*(https?://\S+?)(?:\s*\|\s*(.+?))?$', text, re.MULTILINE):
        url = match.group(1)
        name = match.group(2).strip() if match.group(2) else "Cannondale Bike"
        results.append({"url": url, "name": name})
    return results


def parse_price(price_str) -> Optional[float]:
    """Safely parse a price string like '16,000' or '$5,999' into a float."""
    if not price_str:
        return None
    try:
        return float(str(price_str).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


# ==============================================================================
# TOOL 1: SEARCH BIKES
# ==============================================================================

@tool
def search_bikes(
    query: str,
    bike_type: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> str:
    """Search for Cannondale bikes matching criteria. Use this when the user wants to
    find, list, browse, or filter bikes by type, price range, or features.

    Args:
        query: Search terms describing desired bike characteristics
        bike_type: Optional bike category filter (e.g. 'road', 'mountain', 'gravel', 'electric', 'hybrid')
        min_price: Optional minimum price in USD
        max_price: Optional maximum price in USD

    Returns:
        Formatted list of matching bikes with key details and image URLs
    """
    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )
        docs = retriever.invoke(query)

        # Post-filter by bike_type — search across all text fields, not just bike_name
        if bike_type:
            bt = bike_type.lower()
            docs = [
                d for d in docs
                if bt in d.page_content.lower()
                or bt in d.metadata.get("bike_name", "").lower()
                or bt in d.metadata.get("description_1", "").lower()
                or bt in d.metadata.get("description_2", "").lower()
                or bt in d.metadata.get("highlights", "").lower()
                or bt in d.metadata.get("bike_image_url", "").lower()
            ]

        # Post-filter by price range
        filtered = []
        for doc in docs:
            price_val = parse_price(doc.metadata.get("price"))
            if min_price and price_val is not None and price_val < min_price:
                continue
            if max_price and price_val is not None and price_val > max_price:
                continue
            filtered.append(doc)

        if not filtered:
            return "No bikes found matching your criteria. Try broadening your search."

        # Build formatted results
        results = []
        for doc in filtered:
            m = doc.metadata
            desc = str(m.get("description_1", ""))[:150]
            img_url = m.get("bike_image_url", "")
            img_link = ""
            if img_url and str(img_url).startswith("http"):
                img_link = f"  [View Image]({img_url})"
            results.append(
                f"**{m.get('bike_name', 'N/A')} - {m.get('bike_model', 'N/A')}**\n"
                f"  Price: ${m.get('price', 'N/A')} | Color: {m.get('color', 'N/A')}\n"
                f"  {desc}\n"
                f"{img_link}"
            )

        output = f"Found {len(filtered)} matching bikes:\n\n" + "\n\n".join(results)

        return output

    except Exception as e:
        return f"Error searching bikes: {str(e)}"


# ==============================================================================
# TOOL 2: BIKE SUMMARY
# ==============================================================================

@tool
def get_bike_summary(bike_query: str) -> str:
    """Provide a concise summary of a Cannondale bike.
    Use when the user wants a brief overview, quick description, or summary of a specific bike.

    Args:
        bike_query: The bike name or descriptive query (e.g. 'Scalpel', 'Synapse Carbon')

    Returns:
        A 3-4 sentence summary with key bullet points and image URL
    """
    try:
        retriever = get_retriever()
        llm = get_llm()

        docs = retriever.invoke(bike_query)
        image_data = extract_image_urls_from_docs(docs)

        summary_template = """
        You are a Cannondale bike expert. Provide a CONCISE SUMMARY (3-4 sentences max) of the bike.

            Context:
            {context}

            Query: {question}

            Instructions:
            - Keep it brief and focused on the most important features
            - Mention bike type, key technology, and target use
            - Include price if available
            - Maximum 4 sentences
            - Follow with 4-5 bullet points of the most important features and specs

            Summary:"""

        prompt = ChatPromptTemplate.from_template(summary_template)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(bike_query)

        for img in image_data:
            result += f"\n\nIMAGE_URL: {img['url']} | {img['name']}"

        return result

    except Exception as e:
        return f"Error generating summary: {str(e)}"


# ==============================================================================
# TOOL 3: BIKE DETAILS
# ==============================================================================

@tool
def get_bike_details(bike_query: str) -> str:
    """Provide comprehensive, detailed specifications and analysis of a Cannondale bike.
    Use when the user wants full specs, detailed descriptions, or in-depth technical information.

    Args:
        bike_query: The bike name or descriptive query (e.g. 'SuperSix EVO', 'Jekyll 1 specs')

    Returns:
        Detailed description with specs, components, metadata, and image URL
    """
    try:
        retriever = get_retriever()
        llm = get_llm()

        docs = retriever.invoke(bike_query)
        image_data = extract_image_urls_from_docs(docs)

        # Extract metadata from top result
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
                metadata_section = "\n\n**Additional Information:**\n" + "\n".join(f"- {p}" for p in parts)

        detail_template = """You are a Cannondale bike expert. Provide a COMPREHENSIVE,
        DETAILED description of the bike.

            Context:
            {context}

            Query: {question}

            Instructions:
            - Provide extensive technical specifications
            - Include frame details, components, and drivetrain information
            - Mention pricing, colors, and model variations if available
            - Describe the bike's intended use and performance characteristics
            - Include any special technologies or features
            - Be thorough and technical in your response

            Detailed Description:"""

        prompt = ChatPromptTemplate.from_template(detail_template)
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


# ==============================================================================
# TOOL 4: COMPARE BIKES
# ==============================================================================

@tool
def compare_bikes(bike_names: str) -> str:
    """Compare 2-3 Cannondale bikes side by side.
    Use when the user wants to compare specific bike models, see differences, or decide between options.

    Args:
        bike_names: Comma-separated bike names to compare (e.g. 'SuperSix EVO, CAAD13, Synapse')

    Returns:
        Side-by-side comparison with specs, differences, and recommendations
    """
    try:
        retriever = get_retriever()
        llm = get_llm()

        names = [n.strip() for n in bike_names.split(",") if n.strip()]
        if len(names) < 2:
            return "Please provide at least 2 bike names separated by commas to compare."
        if len(names) > 3:
            names = names[:3]

        all_docs = []
        all_image_data: List[dict] = []
        for name in names:
            docs = retriever.invoke(name)
            if docs:
                all_docs.append(docs[0])
                all_image_data.extend(extract_image_urls_from_docs(docs[:1]))

        if len(all_docs) < 2:
            return "Could not find enough bikes to compare. Please check the bike names and try again."

        # Build combined context
        combined_context = ""
        for i, doc in enumerate(all_docs):
            combined_context += f"\n\n--- Bike {i + 1}: {doc.metadata.get('bike_model', 'Unknown')} ---\n"
            combined_context += doc.page_content

        compare_template = """You are a Cannondale bike expert. Compare these bikes side by side.

            Context:
            {context}

            Bikes to compare: {question}

            Instructions:
            - Create a structured comparison
            - Compare: price, frame material, key components (fork, drivetrain, brakes, wheels, tires)
            - Highlight key differences and similarities
            - Provide a recommendation on which bike suits which type of rider
            - Use markdown formatting (tables, headers, bullet points) for readability

            Comparison:"""

        prompt = ChatPromptTemplate.from_template(compare_template)
        chain = (
            {"context": lambda _: combined_context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(bike_names)

        # Deduplicate image URLs
        seen: set = set()
        for img in all_image_data:
            if img["url"] not in seen:
                result += f"\n\nIMAGE_URL: {img['url']} | {img['name']}"
                seen.add(img["url"])

        return result

    except Exception as e:
        return f"Error comparing bikes: {str(e)}"


# ==============================================================================
# TOOL 5: GET RECOMMENDATION
# ==============================================================================

@tool
def get_recommendation(
    query: str,
    budget: Optional[float] = None,
    experience_level: Optional[str] = None,
) -> str:
    """Recommend the best Cannondale bike for the user's needs.
    Use when the user asks for a recommendation, suggestion, or 'which bike should I get'.

    Args:
        query: Description of riding needs, terrain, and goals
        budget: Optional maximum budget in USD
        experience_level: Optional rider experience ('beginner', 'intermediate', 'advanced')

    Returns:
        Personalized bike recommendation with reasoning and image URLs
    """
    try:
        retriever = get_retriever()
        llm = get_llm()

        docs = retriever.invoke(query)

        # Optional budget filter
        if budget:
            filtered = []
            for doc in docs:
                price_val = parse_price(doc.metadata.get("price"))
                if price_val is None or price_val <= budget:
                    filtered.append(doc)
            if filtered:
                docs = filtered

        image_data = extract_image_urls_from_docs(docs)

        budget_str = f"${budget:,.0f}" if budget else "not specified"
        exp_str = experience_level if experience_level else "not specified"

        # Build context from docs
        context_text = "\n\n".join(doc.page_content for doc in docs)

        rec_template = """
            You are a Cannondale bike expert and cycling advisor. Recommend the best bike(s) for
            this rider.

            Context (available bikes):
            {context}

            Rider's needs: {question}
            Budget: {budget}
            Experience level: {experience}

            Instructions:
            - Recommend 1-3 bikes that best match the rider's needs
            - Explain WHY each recommendation is suitable
            - Consider the rider's experience level when making suggestions
            - Mention price and key features for each recommendation
            - If budget is specified, prioritize bikes within budget
            - Provide a clear top pick with reasoning

            Recommendation:"""

        prompt = ChatPromptTemplate.from_template(rec_template)
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


# ==============================================================================
# AGENT SETUP
# ==============================================================================

TOOLS = [search_bikes, get_bike_summary, get_bike_details, compare_bikes, get_recommendation]

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Cannondale bike expert assistant with access to 5 specialized tools.

TOOL SELECTION GUIDELINES:

1. Use 'search_bikes' when users want to:
   - Browse bikes by category or type
   - Filter by price range
   - Find bikes matching specific criteria
   - Keywords: "show me", "list", "find", "what bikes", "under $X", "between"

2. Use 'get_bike_summary' when users want:
   - Quick overview of a specific bike
   - Brief description
   - Keywords: "tell me about", "what is", "summary of", "quick info"

3. Use 'get_bike_details' when users want:
   - Full specifications
   - Technical details
   - Complete breakdown
   - Keywords: "specs", "specifications", "detailed", "everything about", "full breakdown"

4. Use 'compare_bikes' when users want:
   - Side-by-side comparison
   - Differences between bikes
   - Keywords: "compare", "vs", "versus", "difference between", "which is better"

5. Use 'get_recommendation' when users want:
   - Personalized suggestions
   - Best bike for their needs
   - Keywords: "recommend", "suggest", "best for", "should I get", "what bike for"

RESPONSE GUIDELINES:
- Use H3 for headers and H4/H5 for sub-headers.
- Always be helpful, enthusiastic, and knowledgeable about cycling
- Format comparisons in clear, readable tables when appropriate
- Provide actionable insights, not just raw data
- If a bike isn't found, suggest similar alternatives
- For pricing questions, always mention that prices may vary by retailer

IMPORTANT IMAGE HANDLING:
When a tool returns lines starting with "IMAGE_URL:", you MUST include those exact lines
verbatim at the end of your response. Do NOT omit, rephrase, or modify image URLs.

For simple greetings or conversational responses, respond directly without tools."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


def create_agent_executor() -> AgentExecutor:
    """Create the tool-calling agent executor."""
    llm = get_llm()
    agent = create_tool_calling_agent(llm, TOOLS, AGENT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=MAX_AGENT_ITERATIONS,
        return_intermediate_steps=True,
    )


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.set_page_config(
    page_title="Cannondale Bike Expert",
    page_icon="🚴‍♂️",
    layout="centered",
)

st.title("🚴‍♂️ Cannondale Bike AI Assistant")
st.markdown("**Powered by 5 Specialized AI Tools**: Search, Summary, Details, Compare & Recommend Tools")
st.write("---")

# Sample Questions ----
with st.expander("💡 Sample Questions - Try These!"):
    st.markdown("**🔍 Search & Filter** (uses search_bikes tool):")
    st.markdown("""
    - Show me mountain bikes under $5000
    - What gravel bikes do you have?
    - List electric bikes between $4000 and $8000
    """)
    st.markdown("**📝 Quick Summaries** (uses get_bike_summary tool):")
    st.markdown("""
    - Tell me about the Scalpel
    - Quick summary of Synapse Carbon
    - What is the Topstone?
    """)
    st.markdown("**📋 Detailed Specs** (uses get_bike_details tool):")
    st.markdown("""
    - Full specifications for Jekyll 1
    - Detailed breakdown of SuperSix EVO
    - Everything about the Moterra Neo
    """)
    st.markdown("**⚖️ Comparisons** (uses compare_bikes tool):")
    st.markdown("""
    - Compare Synapse vs CAAD13
    - Differences between Topstone and Topstone Carbon
    - Compare Scalpel, Habit, and Jekyll
    """)
    st.markdown("**💡 Recommendations** (uses get_recommendation tool):")
    st.markdown("""
    - Best bike for weekend trail riding under $4,000
    - What road bike for a beginner with $2,500 budget?
    - Recommend a commuter bike for city riding
    """)


# ==============================================================================
# SESSION STATE
# ==============================================================================

# Chat Memory ----
msgs = StreamlitChatMessageHistory(key="bike_expert_messages_v2")
if len(msgs.messages) == 0:
    msgs.add_ai_message(
        "👋 Hi! I'm your Cannondale bike assistant. I can search bikes, provide summaries "
        "or detailed specs, compare models, and give personalized recommendations. "
        "What would you like to know?"
    )

# Image Storage (keyed by message index — avoids reliance on additional_kwargs) ----
if "message_images" not in st.session_state:
    st.session_state.message_images = {}

# Token Tracking ----
if "total_prompt_tokens" not in st.session_state:
    st.session_state.total_prompt_tokens = 0
if "total_completion_tokens" not in st.session_state:
    st.session_state.total_completion_tokens = 0
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# Agent Executor ----
if "agent_executor" not in st.session_state:
    with st.spinner("🔧 Initializing AI Tools..."):
        st.session_state.agent_executor = create_agent_executor()
        st.success("✅ AI Tools Initialized Successfully!")


# ==============================================================================
# CHAT HISTORY DISPLAY
# ==============================================================================

for idx, msg in enumerate(msgs.messages):
    with st.chat_message(msg.type):
        if msg.type == "ai" and isinstance(msg.content, str):
            st.markdown(msg.content)  # Content is already clean (no IMAGE_URL markers)
            for img in st.session_state.message_images.get(idx, []):
                st.image(img["url"], width=IMAGE_DISPLAY_WIDTH, caption=img["name"])
        else:
            st.write(msg.content)


# ==============================================================================
# CHAT INPUT HANDLER
# ==============================================================================

if question := st.chat_input("Ask about any Cannondale bike..."):
    # Display user message
    msgs.add_user_message(question)
    with st.chat_message("human"):
        st.write(question)

    with st.spinner("🔍 Analyzing with AI tools..."):
        try:
            with get_openai_callback() as cb:
                response = st.session_state.agent_executor.invoke(
                    {"input": question, "chat_history": msgs.messages[:-1]}
                )

                # Update token tracking
                st.session_state.total_prompt_tokens += cb.prompt_tokens
                st.session_state.total_completion_tokens += cb.completion_tokens
                st.session_state.total_tokens += cb.total_tokens
                st.session_state.total_cost += (
                    (cb.prompt_tokens * INPUT_TOKEN_COST)
                    + (cb.completion_tokens * OUTPUT_TOKEN_COST)
                )

            output_text: str = response["output"]

            # Extract image data from output text
            image_data = extract_urls_from_text(output_text)

            # Fallback: check intermediate steps
            if not image_data and response.get("intermediate_steps"):
                for step in response["intermediate_steps"]:
                    if len(step) >= 2 and isinstance(step[1], str):
                        image_data.extend(extract_urls_from_text(step[1]))

            # Deduplicate by URL while preserving order
            seen: set = set()
            unique_images: List[dict] = []
            for img in image_data:
                if img["url"] not in seen:
                    unique_images.append(img)
                    seen.add(img["url"])
            image_data = unique_images

            # Store clean text in message history (strip IMAGE_URL markers)
            clean_text = strip_image_markers(output_text)
            msgs.add_ai_message(clean_text)

            # Store image data in separate session state dict, keyed by message index
            msg_idx = len(msgs.messages) - 1
            if image_data:
                st.session_state.message_images[msg_idx] = image_data

            # Display response
            with st.chat_message("ai"):
                st.markdown(clean_text)
                for img in image_data:
                    st.image(img["url"], width=IMAGE_DISPLAY_WIDTH, caption=img["name"])

        except Exception as e:
            with st.chat_message("ai"):
                st.error(f"Sorry, I encountered an error: {str(e)}")
                st.write("Please try rephrasing your question or ask about a specific Cannondale bike model.")


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:

    # st.markdown("### 🔧 Available Tools")
    # st.markdown("""
    # 1. **🔍 Search Bikes** - Find & filter by criteria
    # 2. **📝 Bike Summary** - Quick overview
    # 3. **📋 Bike Details** - Full specifications
    # 4. **⚖️ Compare Bikes** - Side-by-side comparison
    # 5. **💡 Recommendation** - Personalized suggestions
    # """)

    # st.write("---")

    # Token Usage ----
    with st.expander("📊 Token Usage & Cost"):
        st.markdown("**Current Session:**")
        st.write(f"**Prompt Tokens:** {st.session_state.total_prompt_tokens:,}")
        st.write(f"**Completion Tokens:** {st.session_state.total_completion_tokens:,}")
        st.write(f"**Total Tokens:** {st.session_state.total_tokens:,}")
        st.write(f"**Estimated Cost:** ${st.session_state.total_cost:.4f}")

    st.write("")

    # Clear Chat ----
    if st.button("🗑️ Clear Chat", type="secondary"):
        msgs.clear()
        st.session_state.message_images = {}
        msgs.add_ai_message(
            "👋 Hi! I'm your Cannondale bike assistant. What would you like to know?"
        )
        st.session_state.total_prompt_tokens = 0
        st.session_state.total_completion_tokens = 0
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

st.markdown("---")
