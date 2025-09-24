import os
from pathlib import Path
import sys
import yaml
from dotenv import load_dotenv



# Get Environment Key ----
def get_env_key(key = "openai"):

    load_dotenv()

    if key == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        print("OpenAI API key retrieved successfully.")
        return api_key

    elif key == "gemini":
        api_key = os.getenv('GEMINI_API_KEY')
        print("Google API key retrieved successfully.")
        return api_key

    elif key == "anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        print("Anthropic API key retrieved successfully.")
        return api_key

    elif key == "deepseek":
        api_key = os.getenv('DEEPSEEK_API_KEY')
        print("DeepSeek API key retrieved successfully.")
        return api_key

    elif key == "firecrawl":
        api_key = os.getenv('FIRECRAWL_API_KEY')
        print("Firecrawl API key retrieved successfully.")
        return api_key

    elif key == "tavily":
        api_key = os.getenv('TAVILY_API_KEY')
        print("Tavily API key retrieved successfully.")
        return api_key
    else:
        raise ValueError(f"Key {key} not found in environment variables.")

