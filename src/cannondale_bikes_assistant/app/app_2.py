# CHALLENGE: BUILD A CANNONDALE BIKE EXPERT WITH SPECIALIZED TOOLS
# WEBSITE: https://www.cannondale.com/en-us

# streamlit run PROJECT_00_CHALLENGES/cannondale_project/05_cannondale_tools_app.py

from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.callbacks import get_openai_callback
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import streamlit as st
import requests
import re
import yaml
import uuid
import os
import sys
from pathlib import Path

# Path Setup ----
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from global_utilities.paths import CANNONDALE_BIKES_ASSISTANT_DIR
from global_utilities.keys import get_env_key


# Variables ----

# - Load API Key ----
OPENAI_API_KEY = get_env_key("openai")

# - Vectorstore Path ----
VECTORSTORE_PATH = CANNONDALE_BIKES_ASSISTANT_DIR / "database" / "bikes_vectorstore_2"

# - Embedding Model ----
EMBEDDING_MODEL = "text-embedding-ada-002"


# Initialize the Streamlit app
st.set_page_config(
    page_title="🚴‍♂️ Cannondale Bike Expert with AI Tools",
    page_icon="🚴‍♂️",
    layout="centered"
)

st.title("🚴‍♂️ Cannondale Bike AI Assistant")
st.markdown("*Powered by AI Tools - Your intelligent Cannondale bike expert*")

# Set up Chat Memory
msgs = StreamlitChatMessageHistory(key="bike_expert_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("🚴‍♂️ Hi! I'm your Cannondale bike expert with specialized analysis tools. I can provide quick summaries or detailed technical specifications. What would you like to know?")

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

# Initialize Token Tracking in Session State ----
if 'total_prompt_tokens' not in st.session_state:
    st.session_state.total_prompt_tokens = 0
if 'total_completion_tokens' not in st.session_state:
    st.session_state.total_completion_tokens = 0
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0


# Helper Functions for Image Display ----
def is_valid_image_url(url):
    """Check if URL is reachable and points to an image."""
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200 and 'image' in response.headers.get('content-type', '').lower()
    except requests.RequestException:
        return False

def extract_url_from_text(text: str):
    """Extract the first http/https URL from text if present."""
    match = re.search(r'(https?://\S+)', text)
    return match.group(1) if match else None


# - Create Rag Chain ----
def create_rag_chain():

    # - Embedding Function ----
    embedding_function = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )

    # - Vectorstore ----
    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_PATH),
        embedding_function=embedding_function
    )

    # - Retriever ----
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # - LLM ----
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0.1,
        api_key=OPENAI_API_KEY
    )

    # - Contextualize Question ----
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # - History Aware Retrieval ----
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # - QA System Prompt ----
    qa_system_prompt = """You are an assistant for question-answering tasks about bike models. \
    Use the following pieces of retrieved context to answer the question concisely. \
    If a bike_image_url is available, include it in the answer by stating 'Main Image URL: [URL]'. \
    If you don't know the answer, say so. Keep the answer to three sentences maximum.\

    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # - Create QA Chain ----
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # - Combine RAG + History Aware Retriever ----
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # - Return RAG Chain ----
    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ), retriever

# Create RAG chain and retriever ----
rag_chain, retriever = create_rag_chain()


# Display Chat Messages ----
for msg in msgs.messages:
    with st.chat_message(msg.type):
        st.write(msg.content)

# Chat Input ----
if question := st.chat_input("Ask about any Cannondale bike..."):
    # Display user message first
    with st.chat_message("human"):
        st.write(question)

    # Then show spinner while processing AI response
    with st.spinner("🔍 Analyzing with AI tools..."):

        # Get response from agent with token tracking
        try:
            with get_openai_callback() as cb:
                response = rag_chain.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": "cannondale_session"}}
                )

                # - Get The Answer ----
                answer = response["answer"]

                # - Display The Answer ----
                st.chat_message("ai").write(answer)

                # - Extract Image URL ----
                image_url = extract_url_from_text(answer)

                # - Display Image ----
                if image_url and is_valid_image_url(image_url):
                    st.image(image_url, width=300, caption="Bike Image")
                else:
                    st.write("No image available for this bike model.")

                # Update token counters
                st.session_state.total_prompt_tokens += cb.prompt_tokens
                st.session_state.total_completion_tokens += cb.completion_tokens
                st.session_state.total_tokens += cb.total_tokens
                st.session_state.total_cost += cb.total_cost

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

        # st.markdown("---")
        # st.markdown("**Model:** GPT-4o-mini")
        # st.markdown("**Pricing:**")
        # st.write("• Input: $0.00015 / 1K tokens")
        # st.write("• Output: $0.0006 / 1K tokens")
    # st.write("---")



    # # Message history expander
    # with st.expander("💬 Message History"):
    #     st.write("Current conversation:")
    #     for i, msg in enumerate(msgs.messages):
    #         st.write(f"**{msg.type.title()}:** {msg.content[:100]}...")

    # Clear chat history button
    if st.button("🗑️ Clear Chat", type="secondary"):
        msgs.clear()
        msgs.add_ai_message("🚴‍♂️ Hi! I'm your Cannondale bike expert with specialized analysis tools. I can provide quick summaries or detailed technical specifications. What would you like to know?")
        # Reset token counters
        st.session_state.total_prompt_tokens = 0
        st.session_state.total_completion_tokens = 0
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

st.markdown("---")

