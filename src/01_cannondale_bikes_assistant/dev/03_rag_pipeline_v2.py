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
import re
import requests
from dotenv import load_dotenv
load_dotenv()

# Path Setup - adjust these paths to match your setup
# project_root = Path(__file__).resolve().parents[2]  # Adjust if needed
# sys.path.append(str(project_root))

from src.global_utilities.general.paths import CANNONDALE_BIKES_ASSISTANT_DIR
from src.global_utilities.general.api_keys import get_env_key


# Variables
OPENAI_API_KEY = get_env_key("openai")
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
    model = 'gpt-4o',
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