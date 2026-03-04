"""
Credentials module — loads API keys from environment variables.

Setup:
    1. Copy .env.example to .env in the project root
    2. Fill in your Azure OpenAI credentials
    3. The dotenv library loads them automatically
"""
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
