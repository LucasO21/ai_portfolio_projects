import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from src.global_utilities.keys import get_env_key
from src.global_utilities.paths import LANGCHAIN_BEGINNER_MASTERCLASS_DIR

# OpenAI API Key
OPENAI_API_KEY = get_env_key("openai")

# Define the persistent directory
persistent_directory = os.path.join(LANGCHAIN_BEGINNER_MASTERCLASS_DIR, "4_rag", "database", "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Define the user's question
query = "How did Juliet die?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")
