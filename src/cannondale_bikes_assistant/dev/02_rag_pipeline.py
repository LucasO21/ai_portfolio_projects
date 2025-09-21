

# # Auto-Reload ----
# %load_ext autoreload
# %autoreload 2

# # Libraries ---
# from langchain.docstore.document import Document
# from langchain_community.vectorstores import Chroma
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import WebBaseLoader, SeleniumURLLoader

# import pandas as pd
# import yaml
# from pathlib import Path

# from src.global_utilities.keys import get_env_key
# from src.global_utilities.paths import CANNONDALE_BIKES_ASSISTANT_DIR


# # ------------------------------------------------------------------------------
# # CONSTANTS ----
# # ------------------------------------------------------------------------------
# OPENAI_API_KEY = get_env_key("openai")
# EMBEDDING_MODEL = "text-embedding-ada-002"
# VECTORSTORE_PATH = CANNONDALE_BIKES_ASSISTANT_DIR / "database" / "bikes_vectorstore"

# # ------------------------------------------------------------------------------
# # VECTORSTORE ----
# # ------------------------------------------------------------------------------
# # Note: Vectorstore was created in another project and then copied to this project.

# embedding_function = OpenAIEmbeddings(model = EMBEDDING_MODEL, api_key = OPENAI_API_KEY)

# vectorstore = Chroma(
#     persist_directory = str(VECTORSTORE_PATH),
#     embedding_function = embedding_function,
#     # collection_name = "bikes"
# )


# # ------------------------------------------------------------------------------
# # RETRIEVER ----
# # ------------------------------------------------------------------------------
# retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={
#         "k": 5  # Return top 5 most relevant chunks
#     }
# )

# results = retriever.invoke("SuperSix EVO road bike")
# print(f"Retrieved {len(results)} documents")

# # ------------------------------------------------------------------------------
# # RAG PIPELINE ----
# # ------------------------------------------------------------------------------

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

# # Test 1: Mountain bike query
# result1 = rag_chain.invoke("What is a good bike under $1000?")
# print(result1)

# test_rag_pipeline.py
# Run this to test the RAG pipeline and see what's actually being returned



from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

import os
import sys
from pathlib import Path
from pprint import pprint
import re

# Path Setup - adjust these paths to match your setup
# project_root = Path(__file__).resolve().parents[2]  # Adjust if needed
# sys.path.append(str(project_root))

from src.global_utilities.paths import CANNONDALE_BIKES_ASSISTANT_DIR
from src.global_utilities.keys import get_env_key


# Variables
OPENAI_API_KEY = get_env_key("openai")
VECTORSTORE_PATH = CANNONDALE_BIKES_ASSISTANT_DIR / "database" / "bikes_vectorstore"
EMBEDDING_MODEL = "text-embedding-ada-002"

print("=== TESTING RAG PIPELINE ===")
print(f"Vectorstore path: {VECTORSTORE_PATH}")
print(f"Path exists: {VECTORSTORE_PATH.exists()}")
print()

# Create embedding function
embedding_function = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY
)
print("✓ Created embedding function")

# Load vectorstore
vectorstore = Chroma(
    persist_directory=str(VECTORSTORE_PATH),
    embedding_function=embedding_function,
    collection_name = "bikes"
)
print("✓ Loaded vectorstore")

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
print("✓ Created retriever")

# Test direct retrieval first
print("\n=== TESTING DIRECT RETRIEVAL ===")
test_query = "Moterra SL LAB71"
docs = retriever.invoke(test_query)

print(f"Query: '{test_query}'")
print(f"Number of documents retrieved: {len(docs)}")

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---")
    print(f"Content preview (first 200 chars):")
    print(doc.page_content[:200] + "...")
    print(f"\nAll metadata keys: {list(doc.metadata.keys())}")

    # Check for image-related fields
    image_fields = ['bike_image_url', 'main_image', 'image_url', 'image', 'img_url', 'photo_url']
    print("Image-related metadata:")
    for field in image_fields:
        if field in doc.metadata:
            print(f"  {field}: {doc.metadata[field]}")

    # Print all metadata for debugging
    print(f"\nFull metadata:")
    for key, value in doc.metadata.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")

# Create LLM
llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0.1,
    api_key=OPENAI_API_KEY
)
print("\n✓ Created LLM")

# Create contextualize question prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create history aware retriever
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
print("✓ Created history aware retriever")

# Create QA system prompt - this is the key part for image URLs
qa_system_prompt = """You are an assistant for question-answering tasks about bike models. \
Use the following pieces of retrieved context to answer the question concisely. \
If you find a bike_image_url in the context metadata, include the actual URL in your answer by stating 'Main Image URL: ' followed by the complete URL. \
Look for the bike_image_url field in the provided context and use its exact value. \
If you don't know the answer, say so. Keep the answer to three sentences maximum.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create document prompt
document_prompt = PromptTemplate.from_template("Content:\n{page_content}\n\nMetadata:\n{metadata}")

# Create QA chain
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt=document_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
print("✓ Created QA chain")

# Create RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
print("✓ Created RAG chain")

# Create message history for testing
test_msgs = ChatMessageHistory()

# Create RAG chain with message history
rag_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: test_msgs,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
print("✓ Created RAG chain with history")

# Test the full RAG pipeline
print("\n=== TESTING FULL RAG PIPELINE ===")
test_question = "Tell me about the Moterra SL LAB71"
print(f"Test question: '{test_question}'")

try:
    result = rag_with_history.invoke(
        {"input": test_question},
        config={"configurable": {"session_id": "test_session"}}
    )

    print("\n=== RAG PIPELINE RESULTS ===")
    print(f"Answer: {result['answer']}")
    print(f"\nAnswer length: {len(result['answer'])} characters")

    # Check if answer contains image URL
    url_pattern = r'(https?://[^\s)>\]]+)'
    found_urls = re.findall(url_pattern, result['answer'])
    print(f"\nURLs found in answer: {found_urls}")

    # Check context documents
    if 'context' in result:
        print(f"\nNumber of context documents used: {len(result['context'])}")

        print("\n=== CONTEXT ANALYSIS ===")
        for i, doc in enumerate(result["context"]):
            print(f"\nContext Document {i+1}:")
            print(f"Content preview: {doc.page_content[:150]}...")
            print(f"Metadata keys: {list(doc.metadata.keys())}")

            # Look for image URLs in metadata
            for key, value in doc.metadata.items():
                if 'image' in key.lower() or 'photo' in key.lower() or 'url' in key.lower():
                    print(f"  IMAGE FIELD - {key}: {value}")

    # Test URL extraction function from the original code
    def extract_url_from_text(text: str):
        m = re.search(r'(https?://[^\s)>\]]+)', text)
        return m.group(1) if m else None

    extracted_url = extract_url_from_text(result['answer'])
    print(f"\nExtracted URL using original function: {extracted_url}")

except Exception as e:
    print(f"ERROR in RAG pipeline: {e}")
    import traceback
    traceback.print_exc()

print("\n=== TEST COMPLETE ===")

# Test with different questions
print("\n=== TESTING WITH DIFFERENT QUERIES ===")
test_questions = [
    "What is the Scalpel mountain bike?",
    "Show me details about the Topstone Carbon",
    "Tell me about road bikes"
]

for question in test_questions:
    print(f"\n--- Testing: '{question}' ---")
    try:
        result = rag_with_history.invoke(
            {"input": question},
            config={"configurable": {"session_id": "test_session"}}
        )
        print(f"Answer: {result['answer'][:200]}...")

        # Check for URLs
        found_urls = re.findall(r'(https?://[^\s)>\]]+)', result['answer'])
        if found_urls:
            print(f"Found URLs: {found_urls}")
        else:
            print("No URLs found in answer")

    except Exception as e:
        print(f"Error: {e}")