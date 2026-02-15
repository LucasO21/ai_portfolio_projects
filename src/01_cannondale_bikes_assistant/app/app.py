# CHALLENGE: BUILD A CANNONDALE BIKE EXPERT WITH SPECIALIZED TOOLS
# WEBSITE: https://www.cannondale.com/en-us

# Run Streamlit ----
# poetry run streamlit run src/01_cannondale_bikes_assistant/app/app.py

# Imports ----
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import MongoDBAtlasVectorSearch

import streamlit as st
import requests
import re
import yaml
import uuid
import os
import sys
from pathlib import Path

# Path Setup ----
def find_project_root(start: Path) -> Path:
    for parent in start.resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (missing pyproject.toml)")

project_root = find_project_root(Path(__file__))
sys.path.append(str(project_root / "src"))  # Add src/ to Python's import path

# Custom Imports ----
from global_utilities.general.paths import CANNONDALE_BIKES_ASSISTANT_DIR
from global_utilities.general.api_keys import get_env_key
from global_utilities.general.mongo import get_mongo_client


# Variables ----

# - Load API Key ----
OPENAI_API_KEY = get_env_key("openai")

# - Vectorstore Path ----
# persist_directory = CANNONDALE_BIKES_ASSISTANT_DIR / "database" / "bikes_vectorstore"
MONGO_CLIENT, MONGO_DB_NAME, MONGO_COLLECTION_NAME, MONGO_COLLECTION = get_mongo_client()


# Initialize the Streamlit app
st.set_page_config(
    page_title="🚴‍♂️ Cannondale Bike Expert with AI Tools",
    page_icon="🚴‍♂️",
    layout="centered"
)

# Set OpenAI API Key in environment
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["MONGO_DB_URI"] = st.secrets["MONGO_DB_URI"]
OPENAI_API_KEY = get_env_key("openai")
MONGO_DB_URI = os.getenv("MONGO_DB_URI")

st.title("🚴‍♂️ Cannondale Bike AI Assistant")
st.markdown("*Powered by Specialized AI Tools*")

# Set up Chat Memory
msgs = StreamlitChatMessageHistory(key="bike_expert_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message(" 👋 Hi! I'm your Cannondale bike assistant. What would you like to know?")

st.write("---")

# Sample Questions Expander
sample_summary_questions = """
    - Give me a quick summary of the Moterra SL LAB71
    - Tell me about the Scalpel mountain bike
    - What's a good road bike for racing?
    - Compare the Synapse and CAAD13 models
    - List 3 road bikes under $10,000 suitable for an adult
"""

sample_detailed_questions = """
    - Show me details about gravel bikes under $3000
    - What are the key features of electric mountain bikes?
    - Describe the Topstone Carbon 1 RLE in detail
    - Detailed specs for the Jekyll 1 bike
    - What hybrid bikes are good for commuting?
"""


with st.expander("💡 Sample Questions - Try These!"):
    st.markdown("**Quick Summaries** (uses summary tool):")
    st.write(sample_summary_questions)

    st.markdown("**Detailed Analysis** (uses detailed tool):")
    st.write(sample_detailed_questions)

# Initialize token tracking in session state
if 'total_prompt_tokens' not in st.session_state:
    st.session_state.total_prompt_tokens = 0
if 'total_completion_tokens' not in st.session_state:
    st.session_state.total_completion_tokens = 0
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0
# Initialize session state for agent
if 'agent_executor' not in st.session_state:
    with st.spinner("🔧 Initializing AI Tools..."):

        # Create embedding function
        embedding_function = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            api_key=OPENAI_API_KEY
        )

        # Load vectorstore
        # vectorstore = Chroma(
        #     persist_directory=str(persist_directory),
        #     embedding_function=embedding_function
        # )
        # vectorstore = MongoDBAtlasVectorSearch(
        #     # documents = documents,
        #     embedding = embedding_function,
        #     collection = MONGO_COLLECTION
        # )

        vectorstore = MongoDBAtlasVectorSearch(
            embedding=embedding_function,
            collection=MONGO_COLLECTION,
            index_name="vector_index",
            text_key="text",
            embedding_key="embedding"
        )

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # Create LLM
        model = ChatOpenAI(
            model='gpt-4.1-mini',
            temperature=0.1,
            api_key=OPENAI_API_KEY
        )

        # Tool 1: Bike Summary Tool
        @tool
        def summarize_bike_description(bike_query: str) -> str:
            """
            Provides a concise summary of a Cannondale bike based on the query.
            Use this tool when the user wants a brief overview, summary, or quick description of a bike.

            Args:
                bike_query: The bike name or query to summarize (e.g., "Moterra SL LAB71", "Scalpel mountain bike")

            Returns:
                A concise summary of the bike's key features and characteristics with image URL
            """
            # Get relevant documents to extract image URL
            relevant_docs = retriever.invoke(bike_query)

            # Extract bike image URL from the first relevant document
            bike_image_url = None
            if relevant_docs:
                metadata = relevant_docs[0].metadata
                bike_image_url = (
                    metadata.get('bike_image_url') or
                    metadata.get('main_image') or
                    metadata.get('alternate_image')
                )

            # Create summary template
            summary_template = """
                You are a Cannondale bike expert. Provide a CONCISE SUMMARY (3-4 sentences max) of the bike based on
                the context.

                Context:
                {context}

                Query: {question}

                Instructions:
                - Keep it brief and focused on the most important features
                - Mention bike type, key technology, and target use
                - Include price if available
                - Maximum 4 sentences
                - After the brief summary, include 4 - 5 bullet points of the most important features and specs of the bike.

                Summary:
                """

            summary_prompt = ChatPromptTemplate.from_template(summary_template)

            # Create summary chain
            summary_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | summary_prompt
                | model
                | StrOutputParser()
            )

            summary_result = summary_chain.invoke(bike_query)

            # DIRECTLY append the URL to the result
            if bike_image_url:
                summary_result += f"\n\nBike Image URL: {bike_image_url}"
            else:
                summary_result += f"\n\nBike Image URL: No image available"

            return summary_result

        # Tool 2: Detailed Bike Description Tool
        @tool
        def describe_bike_in_detail(bike_query: str) -> str:
            """
            Provides a comprehensive, detailed description of a Cannondale bike.
            Use this tool when the user wants in-depth information, full specifications, or detailed analysis.

            Args:
                bike_query: The bike name or query to describe in detail (e.g., "Moterra SL LAB71", "SuperSix EVO specs")

            Returns:
                A detailed description including specifications, features, components, and technical details with metadata
            """
            # Get relevant documents to extract metadata
            relevant_docs = retriever.invoke(bike_query)

            # Extract metadata
            bike_image_url = None
            bike_color = None
            model_code = None

            if relevant_docs:
                metadata = relevant_docs[0].metadata
                bike_image_url = (
                    metadata.get('bike_image_url') or
                    metadata.get('main_image') or
                    metadata.get('alternate_image')
                )
                bike_color = metadata.get('color')
                model_code = metadata.get('model_code')

            # Create detailed template
            detail_template = """
            You are a Cannondale bike expert. Provide a COMPREHENSIVE, DETAILED description of the bike based on the context.

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

            Detailed Description:
            """

            detail_prompt = ChatPromptTemplate.from_template(detail_template)

            # Create detailed chain
            detail_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | detail_prompt
                | model
                | StrOutputParser()
            )

            detail_result = detail_chain.invoke(bike_query)

            # Add metadata
            additional_info = "\n\n📋 **Additional Information:**"
            if model_code:
                additional_info += f"\n• **Model Code:** {model_code}"
            if bike_color:
                additional_info += f"\n• **Color:** {bike_color}"

            # DIRECTLY append the URL
            if bike_image_url:
                additional_info += f"\n\nBike Image URL: {bike_image_url}"
            else:
                additional_info += f"\n\nBike Image URL: No image available"

            return f"{detail_result}{additional_info}"

        # Create tools list
        tools = [summarize_bike_description, describe_bike_in_detail]

        # Create agent prompt
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Cannondale bike expert assistant with access to specialized analysis tools.

            TOOL SELECTION GUIDELINES:
            - Use 'summarize_bike_description' when users want:
            * Quick overview, brief description, summary
            * "Tell me about...", "What is...", "Quick info on..."
            * Short, concise information
            * "Give me a summary", "Quick overview"

            - Use 'describe_bike_in_detail' when users want:
            * Full specifications, detailed analysis, comprehensive info
            * "Describe in detail", "Full specs", "Everything about..."
            * "Technical specifications", "Complete details"
            * "In-depth analysis", "Comprehensive description"

            Choose the appropriate tool based on the user's request tone and keywords. Always be helpful and informative.

            IMPORTANT: When a tool returns a "Bike Image URL:", you MUST include that exact line verbatim at the end of
            your response. Do NOT omit, rephrase, or summarize the image URL. Always preserve the format:
            Bike Image URL: <url>

            For simple conversational responses, respond directly without using any tools.
            """),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create agent
        agent = create_openai_functions_agent(model, tools, agent_prompt)

        # Create agent executor with memory
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3,
            return_intermediate_steps=True
        )

        st.session_state.agent_executor = agent_executor
        st.success("✅ AI Tools Initialized Successfully!")

# Helper functions for image display
def is_valid_image_url(url):
    """Check if URL is reachable and points to an image."""
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200 and 'image' in response.headers.get('content-type', '').lower()
    except requests.RequestException:
        return False

def strip_image_url_line(text: str) -> str:
    """Remove the 'Bike Image URL: ...' line from display text."""
    return re.sub(r'\n*Bike Image URL:.*', '', text).strip()

def extract_url_from_text(text: str):
    """Extract URL from text with multiple patterns."""
    patterns = [
        r'Bike Image URL:\s*(https?://\S+)',          # Direct pattern
        r'bike_image_url[\'"]?\s*:\s*[\'"]?(https?://[^\s\'"]+)',  # From metadata format
        r'(https://embed\.widencdn\.net/[^\s]+)',     # Direct Cannondale URLs
        r'(https?://[^\s]+\.(?:jpg|jpeg|png|webp))',  # Any image URL
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            url = match.group(1).rstrip('.,!?')
            return url

    return None

# Display chat messages from history
for msg in msgs.messages:
    with st.chat_message(msg.type):
        if msg.type == "ai" and isinstance(msg.content, str):
            st.markdown(strip_image_url_line(msg.content))
            # Primary: image_url stored on the message; Fallback: extract from text
            url = msg.additional_kwargs.get("image_url") or extract_url_from_text(msg.content)
            if url:
                st.image(url, width=200, caption="Bike Image")
        else:
            st.write(msg.content)

# Chat input
if question := st.chat_input("Ask about any Cannondale bike..."):
    # Add user message to history and display it
    msgs.add_user_message(question)
    with st.chat_message("human"):
        st.write(question)

    with st.spinner("🔍 Analyzing with AI tools..."):
        try:
            # Pass chat history manually to the agent
            with get_openai_callback() as cb:
                response = st.session_state.agent_executor.invoke(
                    {"input": question, "chat_history": msgs.messages[:-1]}
                )

                st.session_state.total_prompt_tokens += cb.prompt_tokens
                st.session_state.total_completion_tokens += cb.completion_tokens
                st.session_state.total_tokens += cb.total_tokens
                st.session_state.total_cost = (cb.prompt_tokens * (5/1000000)) + (cb.completion_tokens * (15/1000000))

            output_text = response['output']

            # Extract image URL from agent output
            image_url = extract_url_from_text(output_text)

            # Fallback: extract from intermediate tool steps
            if not image_url and response.get('intermediate_steps'):
                for step in response['intermediate_steps']:
                    if len(step) >= 2 and isinstance(step[1], str):
                        image_url = extract_url_from_text(step[1])
                        if image_url:
                            break

            # Store AI message WITH image_url baked into additional_kwargs
            ai_kwargs = {"image_url": image_url} if image_url else {}
            msgs.add_message(AIMessage(content=output_text, additional_kwargs=ai_kwargs))

            # Display the response
            with st.chat_message("ai"):
                st.markdown(strip_image_url_line(output_text))
                if image_url:
                    st.image(image_url, width=200, caption="Bike Image")

        except Exception as e:
            with st.chat_message("ai"):
                st.error(f"Sorry, I encountered an error: {str(e)}")
                st.write("Please try rephrasing your question or ask about a specific Cannondale bike model.")

# Sidebar with tool info
with st.sidebar:

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    # Token usage tracking expander
    with st.expander("📊 Token Usage & Cost"):
        st.markdown("**Current Session:**")
        st.write(f"**Prompt Tokens:** {st.session_state.total_prompt_tokens:,}")
        st.write(f"**Completion Tokens:** {st.session_state.total_completion_tokens:,}")
        st.write(f"**Total Tokens:** {st.session_state.total_tokens:,}")
        st.write(f"**Total Cost:** ${st.session_state.total_cost:.4f}")

    st.write("")

    # Clear chat history button
    if st.button("🗑️ Clear Chat", type="secondary"):
        msgs.clear()
        msgs.add_ai_message(" 👋 Hi! I'm your Cannondale bike assistant. What would you like to know?")
        # Reset token counters
        st.session_state.total_prompt_tokens = 0
        st.session_state.total_completion_tokens = 0
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

st.markdown("---")

