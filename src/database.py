"""
SQLite-backed storage for chat conversations and messages.
Supports multiple named conversations, similar to ChatGPT's sidebar.
"""

import sqlite3
import uuid
import json
from contextlib import contextmanager

DB_PATH = "chat_history.db"


@contextmanager
def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    """Create the conversations and messages tables if they don't exist."""
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                from_cache INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        """)
        conn.commit()


def create_conversation(title="New chat", db_path=DB_PATH):
    """Create a new conversation and return its id."""
    conv_id = str(uuid.uuid4())
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            (conv_id, title),
        )
        conn.commit()
    return conv_id


def get_conversations(db_path=DB_PATH):
    """Return all conversations, most recently created first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def rename_conversation(conversation_id, new_title, db_path=DB_PATH):
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (new_title, conversation_id),
        )
        conn.commit()


def delete_conversation(conversation_id, db_path=DB_PATH):
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()


def add_message(conversation_id, role, content, sources=None, from_cache=False, db_path=DB_PATH):
    """Add a message to a conversation. `sources` is a list of dicts, stored as JSON."""
    sources_json = json.dumps(sources) if sources else None
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, from_cache) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, sources_json, int(from_cache)),
        )
        conn.commit()


def get_messages(conversation_id, db_path=DB_PATH):
    """Return all messages for a conversation in chronological order."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT role, content, sources, from_cache, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()

    messages = []
    for row in rows:
        msg = dict(row)
        msg["sources"] = json.loads(msg["sources"]) if msg["sources"] else []
        msg["from_cache"] = bool(msg["from_cache"])
        messages.append(msg)
    return messages
