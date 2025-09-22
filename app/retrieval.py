import numpy as np
import math
from app.config import settings
from app.store import load_all_embeddings, get_term_stats, get_chunk_terms
from app.pdf_ingest import tokenize, embed_text

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def bm25_score(query_tokens, chunk_id, avg_len=200, k1=1.2, b=0.75):
    tf_map = get_chunk_terms(chunk_id)
    N, df_map = get_term_stats(query_tokens)
    score = 0.0
    len_doc = sum(tf_map.values())

    for t in query_tokens:
        if t not in tf_map or df_map[t] == 0:
            continue
        tf = tf_map[t]
        df = df_map[t]
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * len_doc / avg_len)
        score += idf * numerator / denominator
    return score

def hybrid_search(query: str, top_k=6):
    # Get query embedding
    q_vec = embed_text(query)
    q_tokens = tokenize(query)

    # Load all stored embeddings + chunks
    chunks = load_all_embeddings()
    results = []

    for chunk_id, vec, doc_name, page, text in chunks:
        sem = cosine_similarity(q_vec, vec)
        kw = bm25_score(q_tokens, chunk_id)
        final = settings.ALPHA * sem + settings.BETA * (kw / (kw + 1))  # normalize BM25
        results.append((final, sem, kw, doc_name, page, text, chunk_id))

    # Sort by score
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]