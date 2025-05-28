# deepseek_wrapper.py

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Initialize DeepSeek-compatible OpenAI client
client = OpenAI(api_key="sk-1a87e1e386e4478395b227b17a8ebdc9", base_url="https://api.deepseek.com")

def ask_deepseek(messages: list[dict], model="deepseek-chat", stream=False) -> str:
    """Send a chat request to DeepSeek API."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream
    )
    return response.choices[0].message.content
