import os
from pathlib import Path
import sys
import yaml
from dotenv import load_dotenv


# Get Environment Key ----
def get_env_key(key = "openai"):

    load_dotenv()

    if key == "openai":
        return os.getenv('OPENAI_API_KEY')
        print("OpenAI API key retrieved successfully.")

    elif key == "gemini":
        return os.getenv('GEMINI_API_KEY')
        print("Google API key retrieved successfully.")

    elif key == "anthropic":
        return os.getenv('ANTHROPIC_API_KEY')
        print("Anthropic API key retrieved successfully.")
    elif key == "deepseek":
        return os.getenv('DEEPSEEK_API_KEY')
        print("DeepSeek API key retrieved successfully.")
    elif key == "firecrawl":
        return os.getenv('FIRECRAWL_API_KEY')
        print("Firecrawl API key retrieved successfully.")

    else:
        raise ValueError(f"Key {key} not found in environment variables.")

