import os
from pydantic import BaseModel

class Settings(BaseModel):
    # Custom API key
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "Lrs4xsjb1CUPX3B5bZx24XmH992ChYjE")

    # Models
    EMBED_MODEL: str = "mistral-embed"
    CHAT_MODEL: str = "mistral-large-latest"

    # Chunking parameters
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200

    # Retrieval
    TOP_K: int = 6
    ALPHA: float = 0.6   # weight for semantic similarity
    BETA: float = 0.4    # weight for keyword similarity
    COSINE_THRESHOLD: float = 0.25  # if too low, return "insufficient evidence"

settings = Settings()