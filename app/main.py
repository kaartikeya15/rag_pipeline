from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import shutil, os
from app.store import init_db
from app.pdf_ingest import process_pdf
from app.retrieval import hybrid_search
from mistralai import Mistral
from app.config import settings

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

@app.post("/query")
async def query_rag(query: str):
    # Intent check
    if query.lower().strip() in ["hi", "hello", "help", "how are you", "what can you do", "who are you", "what is your name"]:
        return {"answer": "Hello! I'm your personal assistant. Ask me about your documents."}

    # Retrieve top-k chunks
    results = hybrid_search(query, top_k=settings.TOP_K)

    # Threshold check
    avg_sem = sum(r[1] for r in results) / len(results)
    if avg_sem < settings.COSINE_THRESHOLD:
        return {"answer": "Insufficient evidence to answer confidently."}

    # Build context for LLM
    context = "\n\n".join([f"[{r[3]}:p{r[4]}:{r[6]}] {r[5]}" for r in results])

    system_prompt = (
        "You are a helpful assistant. "
        "Answer only using the provided context. "
        "Cite sources in brackets [doc:page:chunk]."
    )

    user_prompt = f"Query: {query}\n\nContext:\n{context}"

    resp = client.chat.complete(
        model=settings.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    answer = resp.choices[0].message.content
    return {"answer": answer, "sources": [r[3] for r in results]}

@app.get("/")
def root():
    return {"message": "RAG backend is running!"}