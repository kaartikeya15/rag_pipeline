import sqlite3

DB_PATH = "rag.db"

def conn():
    return sqlite3.connect(DB_PATH)