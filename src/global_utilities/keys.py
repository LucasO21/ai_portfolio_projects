import os
from pathlib import Path
import sys
import yaml
from dotenv import load_dotenv




OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# LLM API Key as Environment Variable ----
# def set_env_variables(key = "openai"):

#     if key == "openai":
#         try:
#             os.environ['OPENAI_API_KEY'] = yaml.safe_load(open(CREDENTIALS_DIR))['openai']
#             print("OpenAI API key set successfully.")

#         except Exception as e:
#             print(f"Error setting environment variables: {e}")


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

    else:
        raise ValueError(f"Key {key} not found in environment variables.")

