from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import shutil, os
from app.store import init_db
from app.pdf_ingest import process_pdf
from app.retrieval import hybrid_search
from mistralai import Mistral
from app.config import settings
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.database import conn

client = Mistral(api_key=settings.MISTRAL_API_KEY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init DB
    init_db()
    yield

app = FastAPI(title="StackAI RAG", version="0.1", lifespan=lifespan)

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        doc_info = process_pdf(temp_path, file.filename)
        results.append(doc_info)
        os.remove(temp_path)

    return {"ingested": results}

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_rag(req: QueryRequest):
    query = req.query
    try:
        # --- Intent check ---
        if query.lower().strip() in [
            "hi", "hello", "help", "how are you",
            "what can you do", "who are you", "what is your name"
        ]:
            return {"answer": "Hello! I'm your personal assistant. Ask me about your documents."}

        # --- Retrieve top-k chunks ---
        results = hybrid_search(query, top_k=settings.TOP_K)
        if not results:
            return {"answer": "I couldn’t find any relevant content in your documents."}

        # --- Threshold check ---
        avg_sem = sum(r[1] for r in results) / len(results)
        if avg_sem < settings.COSINE_THRESHOLD:
            return {"answer": "Insufficient evidence to answer confidently."}

        # --- Build context for LLM ---
        context = "\n\n".join([f"[{r[3]}:p{r[4]}:{r[6]}] {r[5]}" for r in results])

        system_prompt = (
            "You are a helpful assistant. "
            "Answer only using the provided context. "
            "Cite sources in brackets [doc:page:chunk]. "
        )
        user_prompt = f"Query: {query}\n\nContext:\n{context}"

        # --- Call Mistral ---
        resp = client.chat.complete(
            model=settings.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        answer = resp.choices[0].message.content if resp.choices else None

        if not answer:
            return {"answer": "I couldn’t generate an answer from the context."}

        return {"answer": answer, "sources": [r[3] for r in results]}

    except Exception as e:
        # Catch-all fallback
        return {"answer": None, "error": str(e)}

@app.post("/reset")
async def reset_db():
    """
    Clear all ingested documents, chunks, embeddings, and stats.
    Effectively resets the knowledge base.
    """
    with conn() as c:
        c.execute("DELETE FROM documents")
        c.execute("DELETE FROM chunks")
        c.execute("DELETE FROM embeddings")
        c.execute("DELETE FROM terms")
        c.execute("DELETE FROM df")
    return {"status": "Knowledge base cleared."}

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join("app/static", "index.html"))