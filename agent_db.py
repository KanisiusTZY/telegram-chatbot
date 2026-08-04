"""
SQLite persistence layer for the Telegram AI agent.

Tables:
  messages  — conversation history per user
  notes     — personal notes per user
  reminders — scheduled reminders per user
"""

import sqlite3
import threading
from datetime import datetime
from typing import Optional

DB_PATH = "agent.db"
MAX_HISTORY = 20  # rows stored; trimmed on insert

_local = threading.local()


def _conn() -> sqlite3.Connection:
    """Return a thread-local connection (creates one if needed)."""
    if not getattr(_local, "conn", None):
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


def init_db() -> None:
    """Create tables if they don't exist."""
    c = _conn()
    init_ai_prefs(c)
    c.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            role        TEXT    NOT NULL,  -- 'user' | 'assistant'
            content     TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);

        CREATE TABLE IF NOT EXISTS notes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            content     TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_notes_user ON notes(user_id);

        CREATE TABLE IF NOT EXISTS reminders (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            message     TEXT    NOT NULL,
            remind_at   TEXT    NOT NULL,  -- ISO-8601 UTC
            sent        INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_reminders_pending
            ON reminders(sent, remind_at);
    """)
    c.commit()


# ─── Messages ────────────────────────────────────────────────────────────────

def save_message(user_id: int, role: str, content: str) -> None:
    c = _conn()
    c.execute(
        "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
        (user_id, role, content),
    )
    # Keep only the last MAX_HISTORY rows per user
    c.execute("""
        DELETE FROM messages
        WHERE user_id = ?
          AND id NOT IN (
              SELECT id FROM messages
              WHERE user_id = ?
              ORDER BY id DESC
              LIMIT ?
          )
    """, (user_id, user_id, MAX_HISTORY))
    c.commit()


def load_history(user_id: int, limit: int = 10) -> list[dict]:
    """Return the last `limit` messages as [{role, content}, ...]."""
    c = _conn()
    rows = c.execute("""
        SELECT role, content FROM (
            SELECT id, role, content
            FROM messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        ) ORDER BY id ASC
    """, (user_id, limit)).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def clear_history(user_id: int) -> None:
    c = _conn()
    c.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    c.commit()


# ─── Notes ───────────────────────────────────────────────────────────────────

def save_note(user_id: int, content: str) -> int:
    """Insert a note and return its id."""
    c = _conn()
    cur = c.execute(
        "INSERT INTO notes (user_id, content) VALUES (?, ?)",
        (user_id, content),
    )
    c.commit()
    return cur.lastrowid


def get_notes(user_id: int) -> list[dict]:
    """Return all notes for a user as [{id, content, created_at}, ...]."""
    c = _conn()
    rows = c.execute(
        "SELECT id, content, created_at FROM notes WHERE user_id = ? ORDER BY id",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_note(user_id: int, note_id: int) -> bool:
    """Delete a note owned by user_id. Returns True if a row was deleted."""
    c = _conn()
    cur = c.execute(
        "DELETE FROM notes WHERE id = ? AND user_id = ?",
        (note_id, user_id),
    )
    c.commit()
    return cur.rowcount > 0


# ─── Reminders ───────────────────────────────────────────────────────────────

def save_reminder(user_id: int, message: str, remind_at: datetime) -> int:
    """Insert a reminder and return its id."""
    c = _conn()
    cur = c.execute(
        "INSERT INTO reminders (user_id, message, remind_at) VALUES (?, ?, ?)",
        (user_id, message, remind_at.strftime("%Y-%m-%d %H:%M:%S")),
    )
    c.commit()
    return cur.lastrowid


def get_due_reminders(now: Optional[datetime] = None) -> list[dict]:
    """Return unsent reminders whose remind_at <= now."""
    if now is None:
        now = datetime.utcnow()
    c = _conn()
    rows = c.execute("""
        SELECT id, user_id, message, remind_at
        FROM reminders
        WHERE sent = 0 AND remind_at <= ?
        ORDER BY remind_at
    """, (now.strftime("%Y-%m-%d %H:%M:%S"),)).fetchall()
    return [dict(r) for r in rows]


def mark_reminder_sent(reminder_id: int) -> None:
    c = _conn()
    c.execute("UPDATE reminders SET sent = 1 WHERE id = ?", (reminder_id,))
    c.commit()


# ─── AI Enable/Disable ────────────────────────────────────────────────────────

def init_ai_prefs(c_conn) -> None:
    """Called from init_db — creates tables for AI on/off state."""
    c_conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_ai_prefs (
            user_id    INTEGER PRIMARY KEY,
            ai_enabled INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS global_settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        INSERT OR IGNORE INTO global_settings (key, value)
        VALUES ('ai_enabled', '1');
    """)
    c_conn.commit()


def is_ai_enabled_global() -> bool:
    c = _conn()
    row = c.execute(
        "SELECT value FROM global_settings WHERE key = 'ai_enabled'"
    ).fetchone()
    return row is None or row["value"] == "1"


def set_ai_enabled_global(enabled: bool) -> None:
    c = _conn()
    c.execute(
        "INSERT OR REPLACE INTO global_settings (key, value) VALUES ('ai_enabled', ?)",
        ("1" if enabled else "0",),
    )
    c.commit()


def is_ai_enabled_for_user(user_id: int) -> bool:
    """Returns False only if the user has been explicitly disabled."""
    c = _conn()
    row = c.execute(
        "SELECT ai_enabled FROM user_ai_prefs WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    # No row = default on
    return row is None or bool(row["ai_enabled"])


def set_ai_enabled_for_user(user_id: int, enabled: bool) -> None:
    c = _conn()
    c.execute(
        "INSERT OR REPLACE INTO user_ai_prefs (user_id, ai_enabled) VALUES (?, ?)",
        (user_id, 1 if enabled else 0),
    )
    c.commit()
