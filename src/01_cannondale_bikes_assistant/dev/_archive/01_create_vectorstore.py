
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
from langchain_community.document_loaders import WebBaseLoader
# import chromadb

from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch

import pandas as pd
import yaml
from pathlib import Path
import os
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

from src.global_utilities.general.api_keys import get_env_key
from src.global_utilities.general.paths import CANNONDALE_BIKES_ASSISTANT_DIR


# Constants ----
OPENAI_API_KEY = get_env_key("openai")
EMBEDDING_MODEL = "text-embedding-ada-002"
VECTORSTORE_PATH = CANNONDALE_BIKES_ASSISTANT_DIR / "database" / "bikes_vectorstore"
CSV_PATH = CANNONDALE_BIKES_ASSISTANT_DIR / "database" / "bikes_csv" / "bikes_version_2.csv"
MONGO_DB_URI = os.getenv("MONGO_DB_URI")

# os.listdir(VECTORSTORE_PATH)


# Load Scraped Data ----
bikes_df = pd.read_csv(CSV_PATH).drop(columns = ["bike_image_url", "alternate_image"])

bikes_df.head()

bikes_df = bikes_df.rename(columns = {"platform": "bike_name", "model_name": "bike_model", "main_image": "bike_image_url"})

bikes_dict = bikes_df.to_dict(orient='records')

documents = []

for item in bikes_dict:

    content = f"""
    bike_name: {item.get("bike_name")}
    bike_model: {item.get("bike_model")}
    price: {item.get("price")}
    sale_price: {item.get("sale_price")}
    color: {item.get("color")}
    description_1: {item.get("description_1")}
    description_2: {item.get("description_2")}
    description_3: {item.get("description_3")}
    highlights: {item.get("highlights")}
    model_code: {item.get("model_code")}
    frame: {item.get("frame")}
    fork: {item.get("fork")}
    headset: {item.get("headset")}
    rear_derailleur: {item.get("rear_derailleur")}
    front_derailleur: {item.get("front_derailleur")}
    shifters: {item.get("shifters")}
    chain: {item.get("chain")}
    crank: {item.get("crank")}
    rear_cogs: {item.get("rear_cogs")}
    bottom_bracket: {item.get("bottom_bracket")}
    brakes: {item.get("brakes")}
    brake_levers: {item.get("brake_levers")}
    front_hub: {item.get("front_hub")}
    rear_hub: {item.get("rear_hub")}
    rims: {item.get("rims")}
    spokes: {item.get("spokes")}
    tire_size: {item.get("tire_size")}
    wheel_size: {item.get("wheel_size")}
    tires: {item.get("tires")}
    front_tire: {item.get("front_tire")}
    rear_tire: {item.get("rear_tire")}
    handlebar: {item.get("handlebar")}
    stem: {item.get("stem")}
    grips: {item.get("grips")}
    saddle: {item.get("saddle")}
    seatpost: {item.get("seatpost")}
    wheel_sensor: {item.get("wheel_sensor")}
    extra_1: {item.get("extra_1")}
    hubs: {item.get("hubs")}
    ingestion_hazard: {item.get("ingestion_hazard")}
    rear_shock: {item.get("rear_shock")}
    drive_unit: {item.get("drive_unit")}
    battery: {item.get("battery")}
    charger: {item.get("charger")}
    display: {item.get("display")}
    certifications: {item.get("certifications")}
    brake_type: {item.get("brake_type")}
    bike_image_url: {item.get("bike_image_url")}
    """
    # print(content)

    doc = Document(page_content=content, metadata=item)

    documents.append(doc)

documents

len(documents)

print(documents[10].metadata)

print(documents[10].page_content)

documents[0].metadata.get('bike_name')

[doc for doc in documents if doc.metadata.get('bike_model') == 'Moterra SL LAB71']


# ==============================================================================
# MONGO DB ATLAS VECTOR SEARCH ---
# ==============================================================================

# MongoDB Connection ----
client = MongoClient(MONGO_DB_URI)
db = "cannondale_bikes_db"
collection_name = "bikes_collection"
collection = client[db][collection_name]

# Embedding Function ----
embedding_function = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    api_key=OPENAI_API_KEY
)

# Vector Database ----
vectorstore = MongoDBAtlasVectorSearch.from_documents(
    documents = documents,
    embedding = embedding_function,
    collection = collection
)

# Test Retrieval ----
vector_store = MongoDBAtlasVectorSearch(
    embedding = embedding_function,
    collection = collection
)

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 5}
)

results = retriever.invoke("SuperSix EVO")
print(f"Retrieved {len(results)} documents")

# Show one document to inspect structure
# print(collection.find_one())





# ------------------------------------------------------------------------------
# CHROMA VECTOR DATABASE ---
# ------------------------------------------------------------------------------

# # Embedding Function ----
# embedding_function = OpenAIEmbeddings(
#     model='text-embedding-ada-002',
#     api_key=OPENAI_API_KEY
# )

# # Vector Database ----
# vectorstore = Chroma.from_documents(
#     documents = documents,
#     embedding = embedding_function,
#     persist_directory = str(CANNONDALE_BIKES_ASSISTANT_DIR / "vectorstore"),
#     collection_name = "bikes"
# )

# vectorstore.persist()

# chroma_client = chromadb.PersistentClient(path=str(CANNONDALE_BIKES_ASSISTANT_DIR / "database_vs"))

# collection = chroma_client.get_or_create_collection(name="bikes")

# collection.add(
#     documents=documents,
#     embeddings=embedding_function.embed_documents([doc.page_content for doc in documents]),
#     ids=[str(i) for i in range(len(documents))]
# )

# ------------------------------------------------------------------------------
# RETRIEVER ----
# ------------------------------------------------------------------------------

# Connect to Vectorstore ----
# vectorstore = Chroma(
#     persist_directory =  str(CANNONDALE_BIKES_ASSISTANT_DIR / "database" / "bikes_vectorstore"),
#     embedding_function = embedding_function,
#     collection_name = "bikes"
# )

# # Retriever ----
# retriever = vectorstore.as_retriever(
#     search_type = "similarity",
#     search_kwargs = {"k": 5}
# )

# results = retriever.invoke("SuperSix EVO LAB71")
# print(f"Retrieved {len(results)} documents")


# ------------------------------------------------------------------------------
# RAG LLM MODEL ----
# ------------------------------------------------------------------------------

# # - Enhanced Template for Bike Queries ----
# template = """You are a Cannondale bike expert. Answer the question based on the provided context about Cannondale bikes.

# Context:
# {context}

# Question: {question}

# Instructions:
# - Focus on specific bike models, features, and specifications
# - If pricing information is not available in the context, mention that pricing may vary by retailer
# - Provide detailed technical information when available
# - Compare different models when relevant
# - If the question cannot be answered from the context, say so clearly

# Answer:"""

# # - Prompt ----
# prompt = ChatPromptTemplate.from_template(template)

# # - LLM Model ----
# model = ChatOpenAI(
#     model='gpt-4o-mini',
#     temperature=0.1,  # Lower temperature for more consistent, factual responses
#     api_key=OPENAI_API_KEY
# )

# # - RAG Chain ----
# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | model
#     | StrOutputParser()
# )

# # - Test RAG Chain ----
# result1 = rag_chain.invoke("Describe the Moterra SL LAB71")
# pprint(result1)

# result2 = rag_chain.invoke("Give me the bike_image_url of the Moterra SL LAB71")
# pprint(result2)

# result3 = rag_chain.invoke("List 3 road bikes under $10,000 suitable for an adult")
# pprint(result3)

