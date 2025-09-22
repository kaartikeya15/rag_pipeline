import re
from PyPDF2 import PdfReader
from app.config import settings
from app.store import save_document, save_chunk, save_embedding, save_terms, bump_df
import numpy as np
from mistralai import Mistral

client = Mistral(api_key=settings.MISTRAL_API_KEY)

def clean_text(text: str) -> str:
    """Basic cleaning of extracted text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap
    return chunks

def embed_text(text: str) -> np.ndarray:
    """Get embedding from Mistral API."""
    resp = client.embeddings.create(model=settings.EMBED_MODEL, inputs=[text])
    return np.array(resp.data[0].embedding, dtype="float32")

def tokenize(text: str):
    """Simple whitespace tokenizer."""
    tokens = re.findall(r'\w+', text.lower())
    return tokens

def term_freq(tokens):
    """Compute term frequency map."""
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf

def process_pdf(file_path: str, file_name: str):
    """Extract text, chunk, embed, and save to DB."""
    doc_id = save_document(file_name)
    reader = PdfReader(file_path)

    for page_num, page in enumerate(reader.pages):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue

        # Split into chunks
        for chunk in chunk_text(text):
            tokens = tokenize(chunk)
            tf_map = term_freq(tokens)

            # Save chunk
            chunk_id = save_chunk(doc_id, page_num + 1, chunk, len(tokens))

            # Save embedding
            vec = embed_text(chunk)
            save_embedding(chunk_id, vec)

            # Save keyword stats
            save_terms(chunk_id, tf_map)
            bump_df(set(tf_map.keys()))

    return {"document_id": doc_id, "name": file_name}