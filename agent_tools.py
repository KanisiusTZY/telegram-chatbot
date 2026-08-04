"""
Tool definitions and implementations for the Telegram AI agent.

Each tool has:
  - A JSON schema (TOOLS list) — passed to Groq's tools= parameter
  - An implementation function — called when the model requests it
"""

import json
import logging
from datetime import datetime, timezone

from simpleeval import simple_eval, EvalWithCompoundTypes, InvalidExpression
from duckduckgo_search import DDGS

import agent_db

log = logging.getLogger(__name__)

# ─── Tool Schemas (OpenAI / Groq format) ─────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Cari informasi terkini di internet pakai DuckDuckGo. "
                "Gunakan kalau user nanya soal berita, fakta terbaru, atau hal yang "
                "lo gak yakin jawabannya."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query pencarian dalam bahasa Indonesia atau Inggris.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": (
                "Simpan pengingat untuk dikirim ke user pada waktu tertentu. "
                "remind_at harus dalam format ISO-8601 UTC, contoh: '2026-08-05T09:00:00'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Isi pesan pengingat.",
                    },
                    "remind_at": {
                        "type": "string",
                        "description": "Waktu pengingat dalam ISO-8601 UTC (YYYY-MM-DDTHH:MM:SS).",
                    },
                },
                "required": ["message", "remind_at"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Simpan catatan personal untuk user. Cocok buat nyimpen info penting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Isi catatan yang mau disimpan.",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_notes",
            "description": "Ambil semua catatan personal milik user.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Hitung ekspresi matematika dengan aman. "
                "Contoh: '2 ** 10', '(3.14 * 5**2)', 'sqrt(144)' (butuh 'math.' prefix: 'math.sqrt(144)'). "
                "Jangan pakai buat hal di luar matematika."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Ekspresi matematika yang mau dihitung.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


# ─── Tool Implementations ─────────────────────────────────────────────────────

def _tool_web_search(user_id: int, query: str, max_results: int = 5) -> str:
    max_results = min(int(max_results), 10)
    log.info(f"[tool:web_search] user={user_id} query={query!r} n={max_results}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return json.dumps({"error": "Gak ada hasil ditemukan."})
        formatted = [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in results
        ]
        return json.dumps(formatted, ensure_ascii=False)
    except Exception as e:
        log.error(f"[tool:web_search] error: {e}")
        return json.dumps({"error": f"Search gagal: {e}"})


import re
from datetime import datetime, timedelta, timezone


def parse_flexible_time(time_str: str) -> datetime | None:
    """Parse relative time strings ('2 detik', '5m', '+30s') or ISO-8601 timestamps."""
    time_str = time_str.strip()

    m = re.search(r'(\d+)\s*(detik|sec|second|s|menit|min|minute|m|jam|hour|h|hari|day|d)', time_str, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        unit = m.group(2).lower()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if unit in ['detik', 'sec', 'second', 'seconds', 's']:
            return now + timedelta(seconds=val)
        if unit in ['menit', 'min', 'minute', 'minutes', 'm']:
            return now + timedelta(minutes=val)
        if unit in ['jam', 'hour', 'hours', 'h']:
            return now + timedelta(hours=val)
        if unit in ['hari', 'day', 'days', 'd']:
            return now + timedelta(days=val)

    remind_at_clean = time_str.replace("Z", "").strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(remind_at_clean, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(remind_at_clean)
    except ValueError:
        return None


def _tool_set_reminder(user_id: int, message: str, remind_at: str) -> str:
    log.info(f"[tool:set_reminder] user={user_id} remind_at={remind_at!r}")
    try:
        dt = parse_flexible_time(remind_at)
        if dt is None:
            return json.dumps({"error": f"Format waktu '{remind_at}' tidak dikenali. Gunakan relatif seperti '5 menit', '30 detik', atau ISO format."})
        rid = agent_db.save_reminder(user_id, message, dt)
        return json.dumps({
            "ok": True,
            "reminder_id": rid,
            "remind_at": dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "message": message,
        })
    except Exception as e:
        log.error(f"[tool:set_reminder] error: {e}")
        return json.dumps({"error": str(e)})


def _tool_save_note(user_id: int, content: str) -> str:
    log.info(f"[tool:save_note] user={user_id}")
    try:
        note_id = agent_db.save_note(user_id, content)
        return json.dumps({"ok": True, "note_id": note_id, "content": content})
    except Exception as e:
        log.error(f"[tool:save_note] error: {e}")
        return json.dumps({"error": str(e)})


def _tool_get_notes(user_id: int) -> str:
    log.info(f"[tool:get_notes] user={user_id}")
    try:
        notes = agent_db.get_notes(user_id)
        if not notes:
            return json.dumps({"notes": [], "message": "Belum ada catatan tersimpan."})
        return json.dumps({"notes": notes}, ensure_ascii=False, default=str)
    except Exception as e:
        log.error(f"[tool:get_notes] error: {e}")
        return json.dumps({"error": str(e)})


def _tool_calculate(user_id: int, expression: str) -> str:
    log.info(f"[tool:calculate] user={user_id} expr={expression!r}")
    try:
        import math
        names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        names["math"] = math
        result = simple_eval(expression, names=names)
        return json.dumps({"expression": expression, "result": result})
    except InvalidExpression as e:
        return json.dumps({"error": f"Ekspresi tidak valid: {e}"})
    except ZeroDivisionError:
        return json.dumps({"error": "Pembagian dengan nol."})
    except Exception as e:
        return json.dumps({"error": f"Gagal hitung: {e}"})


# ─── Dispatcher ───────────────────────────────────────────────────────────────

_TOOL_MAP = {
    "web_search": _tool_web_search,
    "set_reminder": _tool_set_reminder,
    "save_note": _tool_save_note,
    "get_notes": _tool_get_notes,
    "calculate": _tool_calculate,
}


def execute_tool(user_id: int, tool_name: str, tool_args: dict) -> str:
    """Run a tool by name and return its JSON string result."""
    fn = _TOOL_MAP.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Tool '{tool_name}' tidak dikenal."})
    return fn(user_id=user_id, **tool_args)
