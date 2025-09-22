# Libraries ----
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pprint import pprint
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
import os

# Load Environment Variables ----
load_dotenv()

# Variables ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Create Model ----
openai_model = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0.1,
    api_key=OPENAI_API_KEY
)

google_model = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    temperature=0.1,
    api_key=GOOGLE_API_KEY
)

anthropic_model = ChatAnthropic(
    model='claude-3-5-sonnet-20240620',
    temperature=0.1,
    api_key=ANTHROPIC_API_KEY
)

# Example 1 ----
result = model.invoke("What is the capital of France?")
print(result.content)

# Example 2 ----
messages = [
    SystemMessage(content="You are a helpful assistant that can answer questions countries and their capitals."),
    HumanMessage(content="What is the capital of France?")
]

result = model.invoke(messages)
pprint(result.content)

# Example 3 ----
for model in [openai_model, google_model, anthropic_model]:
    result = model.invoke("Tell me a joke")
    pprint("Model: ", model.model_name, "Result: ", result.content)

# Example 4 ----
chat_history = []

system_message = SystemMessage(content="You are a helpful assistant that can answer questions about the AI.")

# chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    pprint(result.content)

    print(chat_history)

# Example 5 ----

PROJECT_ID = "langchain-beginner-masterclass"
SESSION_ID = "session-1"
COLLECTION_NAME = "chat_history"

# - Initialize Firestore Client ----
client = firestore.Client(project=PROJECT_ID)


# Initialize Firestore Chat Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = openai_model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")

