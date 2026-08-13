from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os


load_dotenv()

LLM = ChatOpenAI (
    api_key = os.getenv("OPENROUTER_API_KEY"),
    base_url = "https://openrouter.ai/api/v1",
    model = "openrouter/free",
    temperature = 0.8
)

