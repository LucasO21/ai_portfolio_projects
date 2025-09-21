# Libraries ----
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pprint import pprint

# Load Environment Variables ----
load_dotenv()

# Variables ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create Model ----
model = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0.1,
    api_key=OPENAI_API_KEY
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

]