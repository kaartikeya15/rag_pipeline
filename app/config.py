import os
from pydantic import BaseModel

class Settings(BaseModel):
    # API key from environment
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")

    # Models
    EMBED_MODEL: str = "mistral-embed"
    CHAT_MODEL: str = "mistral-large-latest"

    # Chunking parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", 6))
    ALPHA: float = float(os.getenv("ALPHA", 0.6))   # weight for semantic similarity
    BETA: float = float(os.getenv("BETA", 0.4))     # weight for keyword similarity
    COSINE_THRESHOLD: float = float(os.getenv("COSINE_THRESHOLD", 0.25))

settings = Settings()