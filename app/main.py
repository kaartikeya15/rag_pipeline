from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.store import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init DB
    init_db()
    yield
    # Shutdown: (optional cleanup here)
    print("Shutting down...")

app = FastAPI(title="StackAI RAG", version="0.1", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "RAG backend is running with DB initialized!"}