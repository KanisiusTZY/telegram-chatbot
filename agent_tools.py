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
                "Cari informasi real-time dan fakta di internet pakai DuckDuckGo. "
                "WAJIB digunakan saat user menanyakan pertanyaan tentang istilah, singkatan, tim esport/olahraga (misal BTR, RRQ, EVOS), "
                "tokoh, berita, fakta, harga, cuaca, atau topik umum apa pun agar jawaban selalu akurat dan terbaru."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query pencarian dalam bahasa Indonesia atau Inggris (misal 'apa itu BTR esport', 'harga btc hari ini').",
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
                "WAJIB panggil tool ini setiap kali user minta pengingat/reminder dalam durasi atau waktu apapun "
                "(misal '2 detik', '5 menit', '1 jam', 'besok jam 9'). "
                "bisa menerima string waktu relatif seperti '2 detik', '5 menit', '+10s' atau ISO format."
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
                        "description": "Waktu pengingat (contoh: '2 detik', '5 menit', '1 jam', '+30s', atau ISO format).",
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
    {
        "type": "function",
        "function": {
            "name": "file_convert",
            "description": (
                "Konversi file yang baru saja dikirim user ke format lain. "
                "Gunakan setelah user mengirim file dan meminta konversi (misal 'ubah ke PDF', 'jadiin docx', 'jadiin png', 'ekstrak ke txt')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_format": {
                        "type": "string",
                        "enum": ["pdf", "docx", "jpg", "png", "txt"],
                        "description": "Format tujuan konversi (pdf, docx, jpg, png, txt)",
                    },
                },
                "required": ["target_format"],
            },
        },
    },
]


# ─── Tool Implementations ─────────────────────────────────────────────────────

USER_LAST_FILES: dict[int, dict] = {}
PENDING_CONVERTED_FILES: dict[int, dict] = {}


def set_user_last_file(user_id: int, file_path: str, filename: str, ext: str) -> None:
    import time
    USER_LAST_FILES[user_id] = {
        "path": file_path,
        "filename": filename,
        "ext": ext.lower().lstrip("."),
        "time": time.time(),
    }


def pop_pending_converted_file(user_id: int) -> dict | None:
    return PENDING_CONVERTED_FILES.pop(user_id, None)


def _tool_file_convert(user_id: int, target_format: str) -> str:
    import os
    import subprocess
    import time

    target_format = target_format.lower().lstrip(".")
    log.info(f"[tool:file_convert] user={user_id} target_format={target_format}")

    file_info = USER_LAST_FILES.get(user_id)
    if not file_info or not os.path.exists(file_info["path"]):
        return json.dumps({
            "error": "Belum ada file yang kamu kirim. Silakan kirim file dulu (DOCX, PDF, JPG, PNG, TXT), baru minta konversi."
        })

    src_path = file_info["path"]
    src_filename = file_info["filename"]
    src_ext = file_info["ext"]

    if src_ext == target_format:
        return json.dumps({
            "error": f"File '{src_filename}' sudah dalam format {target_format.upper()}."
        })

    base_name = os.path.splitext(src_filename)[0]
    out_filename = f"{base_name}.{target_format}"
    temp_dir = os.path.dirname(src_path) or "temp_files"
    out_path = os.path.join(temp_dir, f"out_{user_id}_{int(time.time())}_{out_filename}")

    try:
        # 1. Image -> PDF
        if src_ext in ["jpg", "jpeg", "png", "webp", "bmp"] and target_format == "pdf":
            from PIL import Image
            img = Image.open(src_path)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(out_path, "PDF")

        # 2. Image -> Image (JPG / PNG)
        elif src_ext in ["jpg", "jpeg", "png", "webp", "bmp"] and target_format in ["jpg", "jpeg", "png"]:
            from PIL import Image
            img = Image.open(src_path)
            fmt = "JPEG" if target_format in ["jpg", "jpeg"] else "PNG"
            if fmt == "JPEG" and img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(out_path, fmt)

        # 3. PDF -> Image (JPG / PNG)
        elif src_ext == "pdf" and target_format in ["jpg", "jpeg", "png"]:
            from pdf2image import convert_from_path
            images = convert_from_path(src_path, first_page=1, last_page=1)
            if not images:
                return json.dumps({"error": "Gagal membaca halaman dari file PDF."})
            fmt = "JPEG" if target_format in ["jpg", "jpeg"] else "PNG"
            images[0].save(out_path, fmt)

        # 4. PDF -> DOCX
        elif src_ext == "pdf" and target_format == "docx":
            from pdf2docx import Converter
            cv = Converter(src_path)
            cv.convert(out_path, start=0, end=None)
            cv.close()

        # 5. DOCX -> PDF
        elif src_ext in ["docx", "doc"] and target_format == "pdf":
            converted = False
            try:
                cmd = ["soffice", "--headless", "--convert-to", "pdf", "--outdir", temp_dir, src_path]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25)
                lo_out = os.path.join(temp_dir, os.path.splitext(os.path.basename(src_path))[0] + ".pdf")
                if os.path.exists(lo_out):
                    os.rename(lo_out, out_path)
                    converted = True
            except Exception as e:
                log.warning(f"[file_convert] LibreOffice conversion warning: {e}")

            if not converted:
                import docx
                doc = docx.Document(src_path)
                full_text = "\n".join([p.text for p in doc.paragraphs if p.text])
                if not full_text:
                    return json.dumps({"error": "Dokumen DOCX kosong atau tidak bisa dibaca."})
                out_txt = os.path.splitext(out_path)[0] + ".txt"
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(full_text)
                out_path = out_txt
                out_filename = base_name + ".txt"

        # 6. PDF / DOCX / Image -> TXT
        elif target_format == "txt":
            full_text = ""
            if src_ext == "pdf":
                from pypdf import PdfReader
                reader = PdfReader(src_path)
                full_text = "\n\n".join([p.extract_text() or "" for p in reader.pages])
            elif src_ext in ["docx", "doc"]:
                import docx
                doc = docx.Document(src_path)
                full_text = "\n".join([p.text for p in doc.paragraphs])
            else:
                full_text = f"Teks dari file {src_filename}"

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(full_text if full_text.strip() else "Tidak ada teks yang dapat diekstrak.")

        else:
            return json.dumps({
                "error": f"Konversi dari format .{src_ext} ke .{target_format} belum didukung."
            })

        if not os.path.exists(out_path):
            return json.dumps({"error": f"Gagal membuat file hasil konversi {target_format.upper()}."})

        PENDING_CONVERTED_FILES[user_id] = {
            "out_path": out_path,
            "out_filename": out_filename,
            "src_path": src_path,
        }

        return json.dumps({
            "ok": True,
            "out_filename": out_filename,
            "message": f"Konversi '{src_filename}' ke '{out_filename}' berhasil!",
        })

    except Exception as e:
        log.error(f"[file_convert] error: {e}", exc_info=True)
        return json.dumps({"error": f"Gagal mengonversi file: {e}"})


def _search_ddg_html(query: str, max_results: int = 5) -> list[dict]:
    import urllib.request
    import urllib.parse
    import re
    import ssl

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers=headers)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        results = []
        snippets = re.findall(r'result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
        titles = re.findall(r'result__a[^>]*>(.*?)</a>', html, re.DOTALL)

        for t, s in zip(titles, snippets):
            t_clean = re.sub(r'<[^>]+>', '', t).replace('\n', ' ').strip()
            s_clean = re.sub(r'<[^>]+>', '', s).replace('\n', ' ').strip()
            if t_clean and s_clean:
                results.append({"title": t_clean, "snippet": s_clean})
                if len(results) >= max_results:
                    break
        return results
    except Exception as e:
        log.error(f"[html_search] error: {e}")
        return []


def _tool_web_search(user_id: int, query: str, max_results: int = 5) -> str:
    max_results = min(int(max_results), 10)
    log.info(f"[tool:web_search] user={user_id} query={query!r} n={max_results}")

    results = []
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        if raw:
            results = [
                {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
                for r in raw
            ]
    except Exception as e:
        log.warning(f"[tool:web_search] DDGS library failed ({e}), trying HTML fallback...")

    if not results:
        results = _search_ddg_html(query, max_results=max_results)

    if not results:
        return json.dumps({"error": f"Tidak ada hasil ditemukan untuk '{query}'."})

    return json.dumps(results, ensure_ascii=False)


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
    "file_convert": _tool_file_convert,
}


def execute_tool(user_id: int, tool_name: str, tool_args: dict) -> str:
    """Run a tool by name and return its JSON string result."""
    fn = _TOOL_MAP.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Tool '{tool_name}' tidak dikenal."})
    return fn(user_id=user_id, **tool_args)
