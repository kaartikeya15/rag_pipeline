import re
from PyPDF2 import PdfReader
from app.config import settings
from app.store import save_document, save_chunk, save_embedding, save_terms, bump_df
import numpy as np
from mistralai import Mistral

# Initialize Mistral API client using API key from config
client = Mistral(api_key=settings.MISTRAL_API_KEY)


def embed_text(text: str) -> np.ndarray:
    """
    Generate an embedding vector for a single text snippet.
    Used in retrieval (query embeddings).
    """
    resp = client.embeddings.create(
        model=settings.EMBED_MODEL,  # Embedding model name (e.g. mistral-embed)
        inputs=[text]                 # Input: a list (batch of texts); here just one
    )
    # Convert to NumPy array for math operations later (cosine similarity)
    return np.array(resp.data[0].embedding, dtype="float32")


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Collapse multiple spaces/newlines into one
    - Strip leading/trailing whitespace
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text: str, size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP):
    """
    Split long text into overlapping chunks.
    Example: size=500, overlap=50 → chunk1 = 0–500, chunk2 = 450–950, etc.
    Ensures continuity of context across chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]       # Slice text from start to end
        chunks.append(chunk)
        start += size - overlap       # Slide forward by (size - overlap)
    return chunks


def tokenize(text: str):
    """
    Tokenize text into lowercase words.
    Uses regex to capture alphanumeric words.
    Example: "Hello World!" → ["hello", "world"]
    """
    return re.findall(r'\w+', text.lower())


def term_freq(tokens):
    """
    Compute term frequency (TF) for a chunk.
    Example: ["apple", "apple", "banana"] → {"apple": 2, "banana": 1}
    """
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


def process_pdf(file_path: str, file_name: str):
    """
    Full pipeline for processing a single PDF:
    1. Register doc in DB
    2. Read pages and clean text
    3. Chunk page text
    4. Batch embed chunks
    5. Save chunks, embeddings, and TF stats in DB
    """
    # Save doc metadata in DB and get its unique ID
    doc_id = save_document(file_name)

    # Initialize PDF reader
    reader = PdfReader(file_path)

    # Loop through all pages in the PDF
    for page_num, page in enumerate(reader.pages):
        # Extract text from the page and clean it
        text = clean_text(page.extract_text() or "")
        if not text:
            continue  # Skip empty pages

        # Break the page into overlapping chunks
        chunks = chunk_text(text)

        # Process chunks in batches (to reduce API calls)
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Request embeddings for all chunks in this batch
            resp = client.embeddings.create(
                model=settings.EMBED_MODEL,
                inputs=batch
            )

            # Iterate over batch results
            for j, chunk in enumerate(batch):
                # Tokenize and compute term frequency for keyword search
                tokens = tokenize(chunk)
                tf_map = term_freq(tokens)

                # Save chunk text in DB (with metadata)
                chunk_id = save_chunk(doc_id, page_num + 1, chunk, len(tokens))

                # Save embedding vector in DB
                vec = np.array(resp.data[j].embedding, dtype="float32")
                save_embedding(chunk_id, vec)

                # Save term frequency stats in DB
                save_terms(chunk_id, tf_map)

                # Update document frequency counts (global keyword stats)
                bump_df(set(tf_map.keys()))

    # Return basic info to API caller
    return {"document_id": doc_id, "name": file_name}