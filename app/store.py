import sqlite3
import numpy as np
from contextlib import contextmanager

DB_PATH = "rag.db"

@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    yield c
    c.commit()
    c.close()

def init_db():
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS documents(
            id INTEGER PRIMARY KEY,
            name TEXT
        )""")

        c.execute("""CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY,
            doc_id INTEGER,
            page INTEGER,
            text TEXT,
            tokens INTEGER,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        )""")

        c.execute("""CREATE TABLE IF NOT EXISTS embeddings(
            chunk_id INTEGER PRIMARY KEY,
            vector BLOB,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id)
        )""")

        c.execute("""CREATE TABLE IF NOT EXISTS terms(
            chunk_id INTEGER,
            term TEXT,
            tf REAL,
            PRIMARY KEY(chunk_id, term)
        )""")

        c.execute("""CREATE TABLE IF NOT EXISTS df(
            term TEXT PRIMARY KEY,
            df INTEGER
        )""")

def save_document(name: str) -> int:
    with conn() as c:
        cur = c.execute("INSERT INTO documents(name) VALUES (?)", (name,))
        return cur.lastrowid

def save_chunk(doc_id: int, page: int, text: str, tokens: int) -> int:
    with conn() as c:
        cur = c.execute("INSERT INTO chunks(doc_id, page, text, tokens) VALUES (?,?,?,?)",
                        (doc_id, page, text, tokens))
        return cur.lastrowid

def save_embedding(chunk_id: int, vec: np.ndarray):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO embeddings(chunk_id, vector) VALUES (?,?)",
                  (chunk_id, vec.astype("float32").tobytes()))

def save_terms(chunk_id: int, tf_map: dict):
    with conn() as c:
        c.executemany("INSERT OR REPLACE INTO terms(chunk_id, term, tf) VALUES (?,?,?)",
                      [(chunk_id, t, float(tf)) for t, tf in tf_map.items()])

def bump_df(terms: set):
    with conn() as c:
        for t in terms:
            row = c.execute("SELECT df FROM df WHERE term=?", (t,)).fetchone()
            if row:
                c.execute("UPDATE df SET df=? WHERE term=?", (row[0] + 1, t))
            else:
                c.execute("INSERT INTO df(term, df) VALUES (?,?)", (t, 1))

def load_all_embeddings():
    with conn() as c:
        rows = c.execute("""
            SELECT e.chunk_id, e.vector, d.name, c.page, c.text
            FROM embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN documents d ON d.id = c.doc_id
        """).fetchall()

    out = []
    for chunk_id, blob, doc_name, page, text in rows:
        vec = np.frombuffer(blob, dtype="float32")
        out.append((chunk_id, vec, doc_name, page, text))
    return out

def get_term_stats(terms: list):
    """
    Return total number of chunks (N) and df (document frequency) for each query term.
    """
    with conn() as c:
        df_rows = {
            t: (c.execute("SELECT df FROM df WHERE term=?", (t,)).fetchone() or (0,))[0]
            for t in terms
        }
        N = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return N, df_rows

def get_chunk_terms(chunk_id: int):
    """
    Return term frequency map for a specific chunk.
    """
    with conn() as c:
        rows = c.execute("SELECT term, tf FROM terms WHERE chunk_id=?", (chunk_id,)).fetchall()
    return dict(rows)