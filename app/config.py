import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    HF_API_KEY: str = os.getenv("HF_API_KEY", "")
    
    # Embedding model (Hugging Face)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM settings
    LLM_MODEL: str = "llama-3.1-8b-instant"
    
    # Query settings
    TOP_K: int = 5

settings = Settings()
