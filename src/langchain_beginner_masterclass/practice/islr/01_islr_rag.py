import os
import re
from pprint import pprint

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document

from src.global_utilities.keys import get_env_key
from src.global_utilities.llms import get_llm
from src.global_utilities.paths import LANGCHAIN_BEGINNER_MASTERCLASS_DIR

# OpenAI API Key
OPENAI_API_KEY = get_env_key("openai")
LLM = get_llm("openai", "gpt-4o", OPENAI_API_KEY)

# Define the persistent directory
persistent_directory = os.path.join(LANGCHAIN_BEGINNER_MASTERCLASS_DIR, "practice", "islr", "database", "chroma_db_with_metadata")
if not os.path.exists(persistent_directory):
    os.makedirs(persistent_directory)

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# Load the pdf
pdf_path = os.path.join(LANGCHAIN_BEGINNER_MASTERCLASS_DIR, "practice", "islr", "pdf", "ISLP_website.pdf")
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()
len(documents)

# Combine documents into a single document
combined_document = "\n".join([doc.page_content for doc in documents])
len(combined_document)

# Recursive Character Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # ~800–1200 tokens
    chunk_overlap=150,
    length_function=len,
)

chunks = text_splitter.split_text(combined_document)
len(chunks)

# Extract section info
def extract_section_info(text):

    pattern1 = re.match(r"(?m)^\s*(\d+(?:\.\d+)*)\s+(.+?)\s+(\d+)\s*$", text.strip())
    pattern2 = re.match(r"(?m)^\s*(\d+)\s+(\d+)\.\s+(.+?)\s*$", text.strip())

    if pattern1:
        section_number = pattern1.group(1)
        section_title = pattern1.group(2)
        page_number = pattern1.group(3)
    elif pattern2:
        section_number = pattern2.group(2)
        section_title = pattern2.group(3)
        page_number = pattern2.group(1)

    if pattern1 or pattern2:
        return section_number, section_title, page_number
    else:
        return None, None, None

# Get metadata for each chunk
section_number_list = []
section_title_list = []
page_number_list = []

for i, chunk in enumerate(chunks):
    section_number, section_title, page_number = extract_section_info(chunk)
    section_number_list.append(section_number)
    section_title_list.append(section_title)
    page_number_list.append(page_number)

# Assign metadata to each chunk
docs_with_metadata = []
for i, chunk in enumerate(chunks):
    docs_with_metadata.append(Document(
        page_content=chunk,
        metadata={
            "section_number": section_number_list[i],
            "section_title": section_title_list[i],
            "page_number": page_number_list[i],
        }
    ))

# Display content and metadata
pprint(docs_with_metadata[251].metadata)
pprint(docs_with_metadata[251].page_content)

# Count chunks with metadata
pprint(f"Number of chunks with metadata: {len([chunk for chunk in docs_with_metadata if chunk.metadata['section_number'] is not None])}")

# Create the vector store and persist it automatically
vectorstore = Chroma.from_documents(
    docs_with_metadata,
    embeddings,
    persist_directory=persistent_directory,
    collection_name="islr_chunks"
)

# Retrieve the vector store
vectorstore = Chroma(persist_directory=persistent_directory, embedding_function=embeddings, collection_name="islr_chunks")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Test the retriever
query = "What is linear regression?"
docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(docs, 1):
    print(f"Document {i}:\n{doc.page_content[:100]}...\n")
    print(f"Source: {doc.metadata['source']}\n")


# ------------------------------------------------------------------------------
# Conversational Retrieval ----
# ------------------------------------------------------------------------------

# Contextualize question prompt ----
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions ----
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever ----
history_aware_retriever = create_history_aware_retriever(
    LLM, retriever, contextualize_q_prompt
)

# QA System Prompt ----
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer concise."
    "Site your source."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(LLM, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Test the full RAG pipeline

def test_rag_pipeline(question):
    chat_history = []
    test_question = question
    print(f"Test question: '{test_question}'")
    result = rag_chain.invoke({"input": test_question, "chat_history": chat_history})
    pprint(result["answer"])
    chat_history.append(HumanMessage(content=test_question))
    chat_history.append(SystemMessage(content=rag_chain.invoke({"input": test_question, "chat_history": chat_history})["answer"]))

    for history in chat_history:
        if isinstance(history, HumanMessage):
            print(f"Human: {history.content}")
        elif isinstance(history, SystemMessage):
            print(f"System: {history.content}")


test_rag_pipeline("What is linear regression?. Site your source.")


# ------------------------------------------------------------------------------
# Agent  & Tools ----
# ------------------------------------------------------------------------------
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent

# Tool 1: Summarize output
def summarize_output(*args, **kwargs):
    return "Summarizes AI output into 3 bullet points."

# Tool List
tools = [
    Tool(
        name="Summarize Output",
        func=summarize_output,
        description="Summarizes AI output into 3 sentences.",
    )
]

# Prompt Template
prompt = hub.pull("hwchase17/react")

# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=LLM,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)


# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Test the agent with sample queries
response = agent_executor.invoke({"input": "What is linear regression?"})
pprint(response)