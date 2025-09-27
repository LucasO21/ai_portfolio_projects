
from dotenv import load_dotenv
import logging
import os

from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI?
from langchain_anthropic import ChatAnthropic

load_dotenv()

from src.global_utilities.keys import get_env_key

OPENAI_API_KEY = get_env_key("openai")
GOOGLE_API_KEY = get_env_key("gemini")
ANTHROPIC_API_KEY = get_env_key("anthropic")


logger = logging.getLogger(__name__)

def get_llm(model_provider, model_name, api_key, temperature=0.1):
    if model_provider == "openai":
        # logger.info(f"Provider: OpenAI | Model: {model_name} | Temp: {temperature}")
        print(f"Provider: OpenAI | Model: {model_name} | Temp: {temperature}")
        return ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=temperature)

    # elif model_provider == "gemini":
    #     # logger.info(f"Provider: Gemini | Model: {model_name} | Temp: {temperature}")
    #     print(f"Provider: Gemini | Model: {model_name} | Temp: {temperature}")
    #     return ChatGoogleGenerativeAI(model=model_name, api_key=GOOGLE_API_KEY, temperature=temperature)

    elif model_provider == "anthropic":
        # logger.info(f"Provider: Anthropic | Model: {model_name} | Temp: {temperature}")
        print(f"Provider: Anthropic | Model: {model_name} | Temp: {temperature}")
        return ChatAnthropic(model=model_name, api_key=ANTHROPIC_API_KEY, temperature=temperature)

    else:
        raise ValueError(f"Model provider {model_provider} not supported.")
