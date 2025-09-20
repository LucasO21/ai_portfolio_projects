

# Auto-Reload ----
%load_ext autoreload
%autoreload 2

# Libraries ---
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader, SeleniumURLLoader

import pandas as pd
import yaml
from pathlib import Path

from src.global_utilities.keys import get_env_key
from src.global_utilities.paths import CANNONDALE_BIKES_ASSISTANT_DIR


# ------------------------------------------------------------------------------
# CONSTANTS ----
# ------------------------------------------------------------------------------
OPENAI_API_KEY = get_env_key("openai")
EMBEDDING_MODEL = "text-embedding-ada-002"
VECTORSTORE_PATH = CANNONDALE_BIKES_ASSISTANT_DIR / "database" / "bikes_vectorstore"

# ------------------------------------------------------------------------------
# VECTORSTORE ----
# ------------------------------------------------------------------------------
# Note: Vectorstore was created in another project and then copied to this project.

embedding_function = OpenAIEmbeddings(model = EMBEDDING_MODEL, api_key = OPENAI_API_KEY)

vectorstore = Chroma(
    persist_directory = str(VECTORSTORE_PATH),
    embedding_function = embedding_function,
    # collection_name = "bikes"
)


# ------------------------------------------------------------------------------
# RETRIEVER ----
# ------------------------------------------------------------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5  # Return top 5 most relevant chunks
    }
)

results = retriever.invoke("SuperSix EVO road bike")
print(f"Retrieved {len(results)} documents")

# ------------------------------------------------------------------------------
# RAG PIPELINE ----
# ------------------------------------------------------------------------------

# - Enhanced Template for Bike Queries ----
template = """You are a Cannondale bike expert. Answer the question based on the provided context about Cannondale bikes.

Context:
{context}

Question: {question}

Instructions:
- Focus on specific bike models, features, and specifications
- If pricing information is not available in the context, mention that pricing may vary by retailer
- Provide detailed technical information when available
- Compare different models when relevant
- If the question cannot be answered from the context, say so clearly

Answer:"""

# - Prompt ----
prompt = ChatPromptTemplate.from_template(template)

# - LLM Model ----
model = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0.1,  # Lower temperature for more consistent, factual responses
    api_key=OPENAI_API_KEY
)

# - RAG Chain ----
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


# - Test RAG Chain ----

# Test 1: Mountain bike query
result1 = rag_chain.invoke("What is a good bike under $1000?")
print(result1)