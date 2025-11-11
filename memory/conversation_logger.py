import sqlite3
from datetime import datetime

DB_PATH = "nova_conversation_logs.db"

def initialise_db():
    conn = sqlite3.connect(DB_PATH)
    # Create cursor object, acts as a pointer to the database
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT CHECK(role IN ('user','assistant')) NOT NULL,
            message TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def log_message(session_id: str, role: str, message: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO conversations (session_id, role, message, timestamp)
        VALUES (?, ?, ?, ?)
    """, (session_id, role, message, datetime.now().isoformat()))
    # ? parameterizes the data, prevents SQL injection
    conn.commit()
    conn.close()

# Get all messages for a session
# SELECT session_id, role, message, timestamp
# FROM conversations
# ORDER BY id DESC
# LIMIT 10;

