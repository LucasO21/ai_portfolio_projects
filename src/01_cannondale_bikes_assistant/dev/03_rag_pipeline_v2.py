# # Auto-Reload ----
%load_ext autoreload
%autoreload 2

# # Libraries ---
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool, StructuredTool


from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.callbacks import get_openai_callback

from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch

import os
import sys
from pathlib import Path
from pprint import pprint
from IPython.display import Markdown
import re
import requests
from typing import Optional, List
from dotenv import load_dotenv
load_dotenv()

# Path Setup - adjust these paths to match your setup
# project_root = Path(__file__).resolve().parents[2]  # Adjust if needed
# sys.path.append(str(project_root))

from src.global_utilities.general.paths import CANNONDALE_BIKES_ASSISTANT_DIR
from src.global_utilities.general.api_keys import get_env_key


# Variables
OPENAI_API_KEY = get_env_key("openai").strip()
EMBEDDING_MODEL = "text-embedding-ada-002"
VECTORSTORE_PATH = CANNONDALE_BIKES_ASSISTANT_DIR / "database" / "bikes_vectorstore"
MONGO_DB_URI = os.getenv("MONGO_DB_URI")

# MongoDB Connection ----
client = MongoClient(MONGO_DB_URI)
db = "cannondale_bikes_db"
collection_name = "bikes_collection"
collection = client[db][collection_name]

# Create embedding function
embedding_function = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY
)

# Load vectorstore
vectorstore = MongoDBAtlasVectorSearch(
    embedding=embedding_function,
    collection=collection,
    index_name="vector_index",
    text_key="text",
    embedding_key="embedding"
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Model
model = ChatOpenAI(
    model = 'gpt-4o-mini',
    temperature = 0.1,
    api_key = OPENAI_API_KEY
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
        # vectorstore = get_vectorstore()
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
                or bt in str(d.metadata.get("bike_name") or "").lower()
                or bt in str(d.metadata.get("description_1") or "").lower()
                or bt in str(d.metadata.get("description_2") or "").lower()
                or bt in str(d.metadata.get("highlights") or "").lower()
                or bt in str(d.metadata.get("bike_image_url") or "").lower()
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
        # retriever = get_retriever()
        llm = model

        docs = retriever.invoke(query)
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

            Output Format
            - 1 - 2 sentences followed by
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
        # retriever = get_retriever()
        llm = model

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

            Output Format
            - 3 - 4 sentences followed by
            - Follow with 8-10 bullet points of the most important features and specs

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
        # retriever = get_retriever()
        llm = model

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
        # retriever = get_retriever()
        llm = model

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
# ==============================================================================
# AGENT SETUP
# ==============================================================================

TOOLS = [
    search_bikes,
    get_bike_summary,
    get_bike_details,
    # compare_bikes,
    # get_recommendation
]

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

        # 4. Use 'compare_bikes' when users want:
        #    - Side-by-side comparison
        #    - Differences between bikes
        #    - Keywords: "compare", "vs", "versus", "difference between", "which is better"

        # 5. Use 'get_recommendation' when users want:
        #    - Personalized suggestions
        #    - Best bike for their needs
        #    - Keywords: "recommend", "suggest", "best for", "should I get", "what bike for"

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

# Create agent executor with memory
llm = model
agent = create_openai_functions_agent(llm, TOOLS, AGENT_PROMPT)

agent_executor =  AgentExecutor(
    agent=agent,
    tools=TOOLS,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=5,
    return_intermediate_steps=True,
)

# Wrap with message history
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ==============================================================================
# TEST
# ==============================================================================
session_id = "123"
msgs = ChatMessageHistory(session_id=session_id)

query = "Show me mountain bikes under $5000"

result = agent_with_history.invoke(
    {"input": query},
    config={"configurable": {"session_id": session_id}},
)

pprint(result["output"])

# image_url = extract_url_from_text(result["output"])


query = "what gravel bikes do you have?"

result = agent_with_history.invoke(
    {"input": query},
    config={"configurable": {"session_id": session_id}},
)

pprint(result["output"])


# Summary Tool Test ----
session_id = "000"
msgs = ChatMessageHistory(session_id=session_id)

query = "Give me a quick summary of Synapse Carbon"

result = agent_with_history.invoke(
    {"input": query},
    config={"configurable": {"session_id": session_id}},
)

pprint(result["output"])
Markdown(result["output"])

# Detail Tool Test ----
session_id = "000"
msgs = ChatMessageHistory(session_id=session_id)

query = "Give me a detailed description of Habit Carbon LT"

result = agent_with_history.invoke(
    {"input": query},
    config={"configurable": {"session_id": session_id}},
)

pprint(result["output"])
Markdown(result["output"])
