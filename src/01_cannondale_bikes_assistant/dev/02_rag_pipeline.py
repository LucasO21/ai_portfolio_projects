

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
    # documents = documents,
    embedding = embedding_function,
    collection = collection
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Model
model = ChatOpenAI(
    model = 'gpt-4.1-mini',
    temperature = 0.1,
    api_key = OPENAI_API_KEY
)


# Summarize tool
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
        You are a Cannondale bike expert. Provide a CONCISE SUMMARY (3-4 sentences max) of the bike based on the context.

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


# Detailed tool
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
        relevant_docs = retriever.get_relevant_documents(bike_query)

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

# Tool list
tools = [summarize_bike_description, describe_bike_in_detail]

# Create agent prompt
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Cannondale bike expert assistant with access to specialized analysis tools.

    TOOL SELECTION GUIDELINES:
    - Use 'summarize_bike_description' when users want:
    * Quick overview, brief description, summary of bike features
    * "Tell me about...", "What is...", "Quick info on..."
    * Short, concise information
    * "Give me a summary", "Quick overview"

    - Use 'describe_bike_in_detail' when users want:
    * Full specifications, detailed analysis, comprehensive info
    * "Describe in detail", "Full specs", "Everything about..."
    * "Technical specifications", "Complete details"
    * "In-depth analysis", "Comprehensive description"

    Choose the appropriate tool based on the user's request tone and keywords. Always be helpful and informative."""),
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
    max_iterations=3
)

# Wrap with message history
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Validate urls
def is_valid_image_url(url):
    """Check if URL is reachable and points to an image."""
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200 and 'image' in response.headers.get('content-type', '').lower()
    except requests.RequestException:
        return False

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

# Test
session_id = "123"
msgs = ChatMessageHistory(session_id=session_id)

question = "Give me a brief summary of the Adventure Neo,Adventure Neo 2 EQ."

result = agent_with_history.invoke(
    {"input": question},
    config={"configurable": {"session_id": session_id}},
)

pprint(result["output"])

image_url = extract_url_from_text(result["output"])





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


from IPython.display import Image, display
url = "https://embed.widencdn.net/img/dorelrl/3hhycofy1t/1700px@1x/C24_C65114U_LAB71_Moterra_Neo_SL_BPT_PD.webp?color=F6F6F5&q=99"
display(Image(url=url))


from PIL import Image
import requests
from io import BytesIO

url = "https://example.com/moterra-sl-lab71.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img.show()  # opens the image using your system viewer
