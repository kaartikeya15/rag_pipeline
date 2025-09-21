from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import shutil, os
from app.store import init_db
from app.pdf_ingest import process_pdf

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

@app.get("/")
def root():
    return {"message": "RAG backend is running with DB initialized!"}