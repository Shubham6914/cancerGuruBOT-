# Configs like API keys, DB connection strings
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()   # load environment variables from .env file

class Settings(BaseSettings):
    # LLM API settings
    llm_api_url: str = os.getenv("LLM_API_URL", "http://localhost:8000/api/generate")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "llama3.2-vision")

    # Qdrant vector DB settings
    qdrant_host: str = os.getenv("QDRANT_HOST", "http://localhost:6333")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "medical_articles")

    class Config:
        env_file = ".env"

# singleton instance to access settings
settings = Settings()