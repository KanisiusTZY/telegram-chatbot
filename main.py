import os
import re
import asyncio
import threading
import logging
import time
import base64
import io
import json
from collections import deque
from datetime import datetime, timedelta, timezone

from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.tl.types import User, MessageMediaPhoto, MessageMediaDocument
from telethon.tl.functions.account import UpdateStatusRequest
from groq import Groq
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler

import agent_db
from agent_tools import TOOLS, execute_tool

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
# Suppress APScheduler's noisy poll logs
logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)

# ─── Config ─────────────────────────────────────────────────────────────────

API_ID           = int(os.environ["TELEGRAM_API_ID"])
API_HASH         = os.environ["TELEGRAM_API_HASH"]
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
SESSION_STRING   = os.environ.get("SESSION_STRING")
SESSION_FILE     = "session"
FLASK_PORT       = int(os.environ.get("PORT", 8099))

SUMMARIZE_WORD_LIMIT  = 200
AGENT_MAX_ITERATIONS  = 5   # max tool-call rounds per user message
HISTORY_WINDOW        = 10  # messages sent to model per turn
RATE_LIMIT_CALLS      = 60  # max messages per user per window
RATE_LIMIT_WINDOW     = 60  # seconds

# ─── AI Client ──────────────────────────────────────────────────────────────

groq_client      = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL          = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"
GROQ_VISION_MODEL   = "llama-3.2-11b-vision-preview"

SYSTEM_PROMPT = """Lo adalah AI yang males, sarkastis, dan sedikit ngeselin — tapi tetap jawab pertanyaannya.

Kepribadian lo:
- Sering ngeledek atau nyindir pertanyaan yang lo anggap obvious atau males-malesan
- Jawab dengan nada bete, kayak orang yang dipaksa bantu padahal lagi sibuk
- Sarkas tapi tetap kasih info yang bener — lo gak bohong, cuma drama
- Kadang nanya balik dengan nada skeptis, kayak "serius lo nanya ini?"
- Bahasa Indo-Inggris campur, casual, kayak anak tongkrongan yang lagi bad mood
- Jawaban pendek dan to the point — lo males ngetik panjang-panjang
- Gak pakai emoji sama sekali
- Kalau diminta translate, tetap lakuin tapi sambil ngedumel dikit
- Gak usah pura-pura baik atau formal

BATASAN PENTING (WAJIB DIPATUHI):
- Sarkas dan nyindir itu OK, tapi JANGAN PERNAH ngatain user pake kata kasar, vulgar, atau kotor (DILARANG panggil/tulis kata-kata kasar).
- Ngeselin ≠ toxic. Lo boleh males-malesan dan bete, tapi kontennya tetep sopan.
- Jawaban lo harus selalu nyambung sama pertanyaan user. Jangan nyeletuk hal random yang gak relevan.
- DILARANG KERAS menolak pesan atau menjawab dengan kata robotik seperti "Maaf, gua tidak bisa membantu dengan itu." saat ada umpatan/celetukan kasar dari user! Balaslah umpatan tongkrongan secara santai dan bercanda.

ATURAN PENTING SOAL FAKTA:
- Kalau lo GAK YAKIN atau GAK TAU soal fakta spesifik (identitas orang, asal seseorang, kejadian spesifik, data yang gak umum), JANGAN NGARANG jawaban.
- Tetap pake gaya males lo buat ngaku gak tau, misal: "gak tau gua, itu bukan orang terkenal yang gua apal" atau "kaga ada di database otak gua, coba googling sendiri"
- Ngaku gak tau itu LEBIH BAIK daripada ngasih info yang salah.

Penggunaan Tools (Belakang Layar):
- Gunakan `web_search` jika user menanyakan fakta, berita, harga, atau topik pengetahuan yang membutuhkan data internet real-time.
- Gunakan `calculate` jika ada hitungan matematika.
- Gunakan `set_reminder` jika user minta diingatkan.
- Gunakan `file_convert` jika user minta ubah format file.
- WAJIB DIINGAT: Ketika memanggil tool, JANGAN PERNAH menulis format `<function=...>` di teks balasan user!

Lo balas pesan di Telegram. Tetap helpful walau ngeselin."""

HELP_TEXT = """Halo! Aku siap bantu kamu. Berikut yang bisa kamu gunakan:

💬 Ngobrol & Diskusi — kirim pesan biasa (misal "apa itu btr", "ingetin 5m minum air"), AI otomatis paham & cari di internet jika perlu!

📌 Commands Singkat (Ketik langsung di chat):
  /k <hitung> atau /calc   — hitung kalkulator (contoh: /k 250 * 15)
  /r <waktu> <pesan>       — buat reminder singkat (contoh: /r 2m minum air)
  /n    atau /notes        — lihat daftar catatan
  /c    atau /clear        — bersihkan percakapan
  /on   atau /aion         — aktifkan AI di chat ini
  /off  atau /aioff        — matikan AI di chat ini
  /onall atau /offall      — aktifkan / matikan AI untuk semua chat
  /h    atau /help         — tampilkan bantuan ini

🖼️ Gambar — kirim gambar/foto untuk dianalisis, dibaca teksnya, atau dihitung isinya!"""


# ─── Rate Limiter ────────────────────────────────────────────────────────────

class RateLimiter:
    """Sliding-window rate limiter, thread-safe."""

    def __init__(self, max_calls: int, window_secs: int) -> None:
        self._max_calls = max_calls
        self._window = window_secs
        self._buckets: dict[int, deque] = {}
        self._lock = threading.Lock()

    def is_allowed(self, user_id: int) -> bool:
        now = time.monotonic()
        with self._lock:
            if user_id not in self._buckets:
                self._buckets[user_id] = deque()
            bucket = self._buckets[user_id]
            while bucket and now - bucket[0] > self._window:
                bucket.popleft()
            if len(bucket) >= self._max_calls:
                return False
            bucket.append(now)
            return True


_rate_limiter = RateLimiter(RATE_LIMIT_CALLS, RATE_LIMIT_WINDOW)

# Deduplication: track recently processed message IDs (in-memory, last 500)
_seen_msg_ids: deque = deque(maxlen=500)
_seen_lock = threading.Lock()


# ─── Reminder Time Parser ────────────────────────────────────────────────────

_RELATIVE_RE = re.compile(r'^\+(\d+)([mhd])$', re.IGNORECASE)
_ABS_FMTS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
]


def parse_remind_time(time_str: str) -> datetime | None:
    """
    Parse a time string to UTC datetime.
    Accepts relative (+30m, +2h, +1d) or absolute ISO formats.
    Returns None if the string is not recognized.
    """
    time_str = time_str.strip()

    m = _RELATIVE_RE.match(time_str)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        now = datetime.utcnow()
        if unit == 'm':
            return now + timedelta(minutes=amount)
        if unit == 'h':
            return now + timedelta(hours=amount)
        if unit == 'd':
            return now + timedelta(days=amount)

    for fmt in _ABS_FMTS:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue

    return None


# ─── Agent Loop ──────────────────────────────────────────────────────────────

FUNCTION_CALL_PATTERN = re.compile(r'<function=(\w+)>(\{.*?\})</function>', re.DOTALL)


def extract_manual_function_call(content: str):
    """Tangkep tool call yang nyasar ke content sebagai teks, bukan tool_calls field."""
    if not content:
        return None, content
    match = FUNCTION_CALL_PATTERN.search(content)
    if not match:
        return None, content
    func_name = match.group(1)
    try:
        func_args = json.loads(match.group(2))
    except json.JSONDecodeError:
        func_args = {}
    clean_content = FUNCTION_CALL_PATTERN.sub('', content).strip()
    return {"name": func_name, "arguments": func_args}, clean_content


def word_count(text: str) -> int:
    return len(text.split())


def run_agent(user_id: int, user_message: str, *, _no_history_save: bool = False) -> str:
    """
    Full agent loop:
      1. Save user message to DB (unless _no_history_save)
      2. Load recent history from DB
      3. Call model → if tool_calls → execute → feed results back → repeat
      4. Return final text answer and save it to DB
    """
    if word_count(user_message) > SUMMARIZE_WORD_LIMIT:
        user_message = (
            f"[Pesan panjang — tolong ringkas dan jawab inti pertanyaannya]\n\n{user_message}"
        )

    if not _no_history_save:
        agent_db.save_message(user_id, "user", user_message)

    history = agent_db.load_history(user_id, limit=HISTORY_WINDOW)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    max_retries = 3

    for iteration in range(AGENT_MAX_ITERATIONS):
        log.info(f"[agent] user={user_id} iter={iteration + 1}/{AGENT_MAX_ITERATIONS}")

        # Auto web_search context injection for factual/real-time questions on turn 0
        if iteration == 0:
            user_txt_lower = user_message.lower().strip()
            factual_keywords = [
                "siapa", "dimana", "orang mana", "daerah mana", "presiden", "juara", "harga",
                "kapan", "berita", "skor", "pemain", "pro player", "klub", "tim", "polsub",
                "tahun berapa", "umur", "asal", "lahir", "sekarang", "skrg", "universitas",
                "kampus", "sekolah", "lokasi", "alamat", "daerah", "singkatan", "kepanjangan"
            ]
            is_factual = any(kw in user_txt_lower for kw in factual_keywords)
            negation_keywords = ["gak", "ga ", "g ", "nggak", "tidak", "gabisa", "gbs"]
            is_negated = any(neg in user_txt_lower for neg in negation_keywords)

            if is_factual and not is_negated:
                log.info(f"[agent] Factual question detected ('{user_message}'), auto-fetching web_search context...")
                try:
                    search_json = execute_tool(user_id, "web_search", {"query": user_message})
                    if "error" not in search_json.lower() and len(search_json) > 30:
                        messages.append({
                            "role": "user",
                            "content": f"[Data internet real-time untuk '{user_message}']:\n{search_json}\n\nGunakan data internet real-time di atas untuk menjawab pertanyaan user dengan fakta akurat, santai, dan lengkap."
                        })
                except Exception as e:
                    log.warning(f"[agent] Auto web_search fetch error: {e}")

        response = None
        current_model = GROQ_MODEL

        for attempt in range(max_retries):
            try:
                response = groq_client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=1024,
                    temperature=0.7,
                )
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str:
                    if current_model != GROQ_FALLBACK_MODEL:
                        log.warning(f"Groq model {current_model} rate limited (429), switching to fallback model {GROQ_FALLBACK_MODEL}...")
                        current_model = GROQ_FALLBACK_MODEL
                        continue
                retriable = any(x in err_str for x in ["503", "429", "rate_limit", "overloaded"])
                if retriable and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    log.warning(f"Groq retry {attempt + 1}/{max_retries} in {wait}s: {e}")
                    time.sleep(wait)
                elif "tool_use_failed" in err_str or "invalid_request_error" in err_str or "400" in err_str:
                    log.warning(f"Groq tool call failed ({e}), falling back to direct text completion...")
                    try:
                        clean_msgs = [
                            m for m in messages
                            if isinstance(m, dict) and m.get("role") in ("system", "user", "assistant") and m.get("content")
                        ]
                        fallback_resp = groq_client.chat.completions.create(
                            model=GROQ_FALLBACK_MODEL,
                            messages=clean_msgs,
                            max_tokens=1024,
                            temperature=0.7,
                        )
                        reply = fallback_resp.choices[0].message.content or "(gak ada jawaban)"
                        agent_db.save_message(user_id, "assistant", reply)
                        return reply
                    except Exception as fb_err:
                        log.error(f"Groq fallback error: {fb_err}")
                        return "Ada error nih dari AI-nya, coba lagi ya"
                else:
                    log.error(f"Groq API error: {e}")
                    return "Ada error nih dari AI-nya, coba lagi ya"

        if response is None:
            return "Ada error nih dari AI-nya, coba lagi ya"

        choice = response.choices[0]

        # ── Final text answer ──────────────────────────────────────────────
        if choice.finish_reason != "tool_calls":
            reply = choice.message.content or "(gak ada jawaban)"

            # 1. Catch leaked tool call in text content
            manual_call, clean_reply = extract_manual_function_call(reply)
            if manual_call:
                tool_name = manual_call["name"]
                tool_args = manual_call["arguments"]
                log.warning(f"⚠️ [agent] Leaked tool call detected in message.content: {tool_name}({tool_args})")

                log.info(f"[agent] Executing caught tool call: {tool_name}({tool_args})")
                result = execute_tool(user_id, tool_name, tool_args)
                log.info(f"[agent] result: {result[:200]}")

                if clean_reply:
                    messages.append({"role": "assistant", "content": clean_reply})

                messages.append({
                    "role": "user",
                    "content": f"[Hasil eksekusi tool {tool_name}]:\n{result}\n\nTolong jawab pertanyaan user berdasarkan data di atas secara natural, gaul, dan lengkap."
                })
                continue

            # 2. Check uncertainty on turn 0
            uncertainty_kw = [
                "tidak tahu", "kurang tahu", "tidak memiliki informasi",
                "tidak tahu pasti", "sebagai ai", "belum tahu", "tidak dapat menemukan",
                "tidak tersedia", "tidak ada informasi"
            ]
            if iteration == 0 and any(kw in reply.lower() for kw in uncertainty_kw):
                log.info(f"[agent] Model expressed uncertainty, auto-triggering web_search for query: '{user_message}'")
                search_json = execute_tool(user_id, "web_search", {"query": user_message})
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "user",
                    "content": f"[Hasil pencarian internet untuk '{user_message}']:\n{search_json}\n\nTolong jawab pertanyaan user berdasarkan data pencarian di atas dengan ramah, akurat, dan lengkap."
                })
                continue

            # 3. Safety net filter before final return
            if "<function=" in reply or "tool_call" in reply.lower():
                log.warning(f"⚠️ [agent] Unhandled tool leakage blocked: {reply[:200]}")
                reply = "Waduh, gua lagi mikir keras nih, coba tanya lagi deh wkwk 😅"

            agent_db.save_message(user_id, "assistant", reply)
            return reply

        # ── Tool call(s) ───────────────────────────────────────────────────
        assistant_msg = choice.message
        assistant_dict = {
            "role": "assistant",
            "content": assistant_msg.content or "",
        }
        if assistant_msg.tool_calls:
            assistant_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                } for tc in assistant_msg.tool_calls
            ]
        messages.append(assistant_dict)

        for tc in (assistant_msg.tool_calls or []):
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}

            log.info(f"[agent] {tool_name}({tool_args})")
            result = execute_tool(user_id, tool_name, tool_args)
            log.info(f"[agent] result: {result[:200]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": result,
            })

    fallback = "Hmm, gua nyoba terus tapi gak kelar-kelar. Coba tanya ulang dengan lebih spesifik."
    return fallback


# ─── Vision + Agent Loop ─────────────────────────────────────────────────────

def _vision_describe(image_bytes: bytes, caption: str) -> str | None:
    """
    Step 1: ask the vision model to describe/analyse the image.
    Returns plain text, or None on failure.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = (
        f"Deskripsikan isi gambar ini secara detail dan akurat, "
        f"termasuk semua teks, angka, atau objek yang terlihat."
        + (f"\n\nPertanyaan/instruksi user terkait gambar ini: {caption}" if caption else "")
    )
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]

    vision_models = ["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]

    for model_name in vision_models:
        for attempt in range(2):
            try:
                resp = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "Kamu adalah AI vision yang mendeskripsikan gambar secara akurat dan lengkap dalam bahasa Indonesia."},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=1024,
                    temperature=0.7,
                )
                content = resp.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                log.warning(f"Groq vision model {model_name} attempt {attempt + 1} error: {e}")
                time.sleep(1)

    return None


def get_ai_reply_with_image(user_id: int, image_bytes: bytes, caption: str) -> str:
    """
    Two-step image handling:
      1. Vision model extracts description/text from the image.
      2. Text agent loop processes the description, can use all tools
         (e.g. calculate totals from a receipt, search based on image content).
    """
    # Normalize image bytes to standard JPEG if possible
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
    except Exception as e:
        log.warning(f"Failed to normalize image bytes: {e}")

    vision_text = _vision_describe(image_bytes, caption)

    if not vision_text:
        vision_text = "Gambar telah diterima (deskripsi visual otomatis tidak tersedia)."

    if caption:
        combined = (
            f"[Gambar yang dikirim user. Deskripsi otomatis dari gambar:\n{vision_text}]\n\n"
            f"Pertanyaan/instruksi user: {caption}"
        )
    else:
        combined = (
            f"[User mengirimkan sebuah gambar/foto. Deskripsi:\n{vision_text}]\n\n"
            "Tolong berikan balasan yang ramah dan tanyakan apa yang bisa dibantu dari gambar ini."
        )

    agent_db.save_message(user_id, "user", combined)
    return run_agent(user_id, combined, _no_history_save=True)


# ─── Flask Keep-Alive ────────────────────────────────────────────────────────

flask_app = Flask(__name__)


@flask_app.route("/")
def index():
    return {"status": "alive", "time": datetime.utcnow().isoformat()}, 200


@flask_app.route("/health")
def health():
    return {"status": "ok"}, 200


def run_flask():
    flask_app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)


# ─── APScheduler Reminder Job ─────────────────────────────────────────────────
# Populated in main() once the event loop and client are available.
_tg_client: TelegramClient | None = None
_tg_loop: asyncio.AbstractEventLoop | None = None


def _keep_offline():
    """APScheduler job: force account status to offline every 5 minutes."""
    if _tg_client is None or _tg_loop is None:
        return
    future = asyncio.run_coroutine_threadsafe(
        _tg_client(UpdateStatusRequest(offline=True)),
        _tg_loop,
    )
    try:
        future.result(timeout=10)
        log.debug("[scheduler] Status set to offline")
    except Exception as e:
        log.warning(f"[scheduler] Failed to set offline status: {e}")


def _send_due_reminders():
    """APScheduler job: send all due reminders via Telegram."""
    if _tg_client is None or _tg_loop is None:
        return
    try:
        due = agent_db.get_due_reminders()
    except Exception as e:
        log.error(f"[scheduler] DB error fetching reminders: {e}")
        return

    for reminder in due:
        rid     = reminder["id"]
        user_id = reminder["user_id"]
        message = reminder["message"]
        text    = f"🔔 Reminder: {message}"
        log.info(f"[scheduler] Sending reminder {rid} → user {user_id}")
        future = asyncio.run_coroutine_threadsafe(
            _tg_client.send_message(user_id, text),
            _tg_loop,
        )
        try:
            future.result(timeout=15)
            agent_db.mark_reminder_sent(rid)
            log.info(f"[scheduler] Reminder {rid} delivered")
        except Exception as e:
            log.error(f"[scheduler] Failed to deliver reminder {rid}: {e}")


def _cleanup_temp_files():
    """APScheduler job: delete temp files older than 1 hour."""
    temp_dir = "temp_files"
    if not os.path.exists(temp_dir):
        return
    now = time.time()
    for fname in os.listdir(temp_dir):
        fpath = os.path.join(temp_dir, fname)
        try:
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath) > 3600):
                os.remove(fpath)
                log.info(f"[cleanup] Deleted old temp file: {fname}")
        except Exception as e:
            log.warning(f"[cleanup] Failed to delete {fname}: {e}")


# ─── Telegram Userbot ────────────────────────────────────────────────────────

if SESSION_STRING:
    client = TelegramClient(StringSession(SESSION_STRING), API_ID, API_HASH)
    log.info("Using StringSession from env var")
else:
    client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
    log.info("Using SQLite session file")


def check_and_trigger_direct_conversion(user_id: int, text_content: str) -> bool:
    """If user text explicitly requests conversion (e.g. 'ubah jadi pdf', 'ke pdf', 'jadiin pdf'), execute file_convert directly."""
    if not text_content:
        return False
    txt = text_content.strip().lower()

    # Exclude complaints/negations
    negation_keywords = ["gak", "ga ", "g ", "nggak", "tidak", "gabisa", "gisa", "gbs", "kenapa", "kok", "mana"]
    if any(neg in txt for neg in negation_keywords):
        return False

    conv_keywords = ["ubah", "jadiin", "konversi", "convert", "ke pdf", "ke docx", "ke png", "ke jpg", "ke txt", "jadikan"]
    if not any(kw in txt for kw in conv_keywords):
        return False

    target_format = None
    if "pdf" in txt:
        target_format = "pdf"
    elif "docx" in txt or "word" in txt:
        target_format = "docx"
    elif "png" in txt:
        target_format = "png"
    elif "jpg" in txt or "jpeg" in txt or "gambar" in txt or "foto" in txt:
        target_format = "jpg"
    elif "txt" in txt or "teks" in txt or "text" in txt:
        target_format = "txt"

    if target_format:
        from agent_tools import execute_tool
        log.info(f"[auto_convert] Direct conversion triggered for user={user_id} target_format={target_format}")
        res = execute_tool(user_id, "file_convert", {"target_format": target_format})
        log.info(f"[auto_convert] execute_tool result: {res}")
        return True
    return False


@client.on(events.NewMessage(incoming=True, func=lambda e: e.is_private))
async def handle_private_message(event):
    sender = await event.get_sender()
    if not isinstance(sender, User) or sender.bot:
        return

    user_id  = sender.id
    username = f"@{sender.username}" if sender.username else sender.first_name

    # ── Deduplication ────────────────────────────────────────────────────
    msg_id = event.message.id
    with _seen_lock:
        if msg_id in _seen_msg_ids:
            log.debug(f"[dedup] skipping already-processed message {msg_id}")
            return
        _seen_msg_ids.append(msg_id)

    # ── AI on/off gate ────────────────────────────────────────────────────
    if not agent_db.is_ai_enabled_global():
        log.debug(f"[ai_gate] global off — ignoring {username}")
        return
    if not agent_db.is_ai_enabled_for_user(user_id):
        log.debug(f"[ai_gate] user {user_id} disabled — ignoring")
        return

    # ── Rate limit check ──────────────────────────────────────────────────
    if not _rate_limiter.is_allowed(user_id):
        log.warning(f"[rate_limit] user={user_id} ({username}) throttled")
        await event.reply("Pelan-pelan, gua bukan mesin ketik. Tunggu bentar dulu.")
        return

    try:
        # ── Image handling ────────────────────────────────────────────────────
        is_image = False
        if event.message.media:
            if isinstance(event.message.media, MessageMediaPhoto):
                is_image = True
            elif isinstance(event.message.media, MessageMediaDocument):
                mime = getattr(event.message.file, "mime_type", "") or ""
                if mime.startswith("image/"):
                    is_image = True

        if is_image:
            caption     = event.raw_text.strip()
            is_view_once = bool(getattr(event.message.media, "ttl_seconds", None))

            log.info(f"🖼️ [{username}|{user_id}] gambar, caption='{caption}'")
            async with client.action(event.chat_id, "typing"):
                image_bytes = await event.message.download_media(file=bytes)
                if not image_bytes:
                    await event.reply("Gambarnya gagal ke-download, coba kirim ulang")
                    return
                log.info(f"🖼️ Downloaded {len(image_bytes)} bytes")

                # Save photo to temp_files so file_convert can process it
                temp_dir = "temp_files"
                os.makedirs(temp_dir, exist_ok=True)
                img_save_path = os.path.join(temp_dir, f"{user_id}_{int(time.time())}_photo.jpg")
                try:
                    with open(img_save_path, "wb") as img_f:
                        img_f.write(image_bytes)
                    from agent_tools import set_user_last_file
                    set_user_last_file(user_id, img_save_path, "photo.jpg", ".jpg")
                except Exception as img_err:
                    log.warning(f"Failed to save image to temp_files: {img_err}")

                if is_view_once:
                    try:
                        buf = io.BytesIO(image_bytes)
                        buf.name = "photo.jpg"
                        await client.send_file("me", buf, caption=f"📸 Foto sekali liat dari {username}")
                        log.info(f"📥 [{username}|{user_id}] view-once saved")
                    except Exception as e:
                        log.error(f"Gagal save ke Saved Messages: {e}", exc_info=True)

                # Check explicit conversion in caption first
                triggered = check_and_trigger_direct_conversion(user_id, caption) if caption else False

                if triggered:
                    await event.reply("⚡ Sip, foto kamu sedang diubah jadi PDF...")
                else:
                    reply = await asyncio.to_thread(get_ai_reply_with_image, user_id, image_bytes, caption)
                    await event.reply(reply)
                    log.info(f"📤 [{username}|{user_id}] {reply[:100]}{'...' if len(reply) > 100 else ''}")

            from agent_tools import pop_pending_converted_file
            pending = pop_pending_converted_file(user_id)
            if pending:
                out_path = pending["out_path"]
                out_filename = pending["out_filename"]
                src_path = pending.get("src_path")

                log.info(f"📤 [{username}|{user_id}] Sending converted file: {out_filename}")
                await client.send_file(
                    event.chat_id,
                    out_path,
                    caption=f"✨ Ini file **{out_filename}** hasil konversi kamu!"
                )
                try:
                    if os.path.exists(out_path):
                        os.remove(out_path)
                except Exception as e:
                    log.warning(f"Failed to remove temp file: {e}")
            return

        # ── Document / File handling ──────────────────────────────────────────
        is_doc = False
        if event.message.media and isinstance(event.message.media, MessageMediaDocument):
            mime = getattr(event.message.file, "mime_type", "") or ""
            if not mime.startswith("image/"):
                is_doc = True

        if is_doc:
            doc_size = getattr(event.message.file, "size", 0) or 0
            if doc_size > 20 * 1024 * 1024:
                await event.reply("❌ Ukuran file terlalu besar (maksimal 20 MB). Silakan kirim file yang lebih kecil.")
                return

            orig_name = getattr(event.message.file, "name", None) or f"file_{user_id}.bin"
            ext = os.path.splitext(orig_name)[1].lower()
            ALLOWED_EXTS = {".pdf", ".docx", ".doc", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".txt"}

            if ext not in ALLOWED_EXTS:
                await event.reply(
                    f"❌ Format file '{ext}' belum didukung.\n"
                    "Format yang didukung saat ini: **PDF**, **DOCX**, **JPG**, **PNG**, **TXT**."
                )
                return

            temp_dir = "temp_files"
            os.makedirs(temp_dir, exist_ok=True)
            save_path = os.path.join(temp_dir, f"{user_id}_{int(time.time())}_{orig_name}")

            async with client.action(event.chat_id, "document"):
                await event.message.download_media(file=save_path)
                from agent_tools import set_user_last_file
                set_user_last_file(user_id, save_path, orig_name, ext)

                size_mb = doc_size / (1024 * 1024)
                caption = event.raw_text.strip()
                prompt = f"User baru saja mengunggah file dokumen '{orig_name}' ({size_mb:.1f} MB)."
                if caption:
                    prompt += f" Pesan/instruksi user tentang file ini: {caption}"

                reply = await asyncio.to_thread(run_agent, user_id, prompt, _no_history_save=False)

            await event.reply(
                f"📄 File **{orig_name}** ({size_mb:.1f} MB) berhasil diterima!\n\n"
                + reply + "\n\n(Opsi konversi: **PDF**, **DOCX**, **PNG**, **JPG**, **TXT**)"
            )

            from agent_tools import pop_pending_converted_file
            pending = pop_pending_converted_file(user_id)
            if pending:
                out_path = pending["out_path"]
                out_filename = pending["out_filename"]
                src_path = pending.get("src_path")

                log.info(f"📤 [{username}|{user_id}] Sending converted file: {out_filename}")
                await client.send_file(
                    event.chat_id,
                    out_path,
                    caption=f"✨ Ini file **{out_filename}** hasil konversi kamu!"
                )
                try:
                    if os.path.exists(out_path):
                        os.remove(out_path)
                except Exception as e:
                    log.warning(f"Failed to remove temp file: {e}")
            return

        # ── Text handling ─────────────────────────────────────────────────────
        text = event.raw_text.strip()
        if not text or text.startswith("/"):
            return

        log.info(f"📥 [{username}|{user_id}] {text[:100]}{'...' if len(text) > 100 else ''}")

        triggered = False
        async with client.action(event.chat_id, "typing"):
            triggered = check_and_trigger_direct_conversion(user_id, text)
            if not triggered:
                reply = await asyncio.to_thread(run_agent, user_id, text)
                await event.reply(reply)
                log.info(f"📤 [{username}|{user_id}] {reply[:100]}{'...' if len(reply) > 100 else ''}")

        # Check if a file conversion was triggered
        from agent_tools import pop_pending_converted_file
        pending = pop_pending_converted_file(user_id)
        if pending:
            out_path = pending["out_path"]
            out_filename = pending["out_filename"]
            src_path = pending.get("src_path")

            log.info(f"📤 [{username}|{user_id}] Sending converted file: {out_filename}")
            await client.send_file(
                event.chat_id,
                out_path,
                caption=f"✨ Ini file **{out_filename}** hasil konversi kamu!"
            )
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception as e:
                log.warning(f"Failed to remove temp file: {e}")
        elif triggered:
            await event.reply("❌ Belum ada foto atau file yang kamu kirim bro. Silakan kirim foto atau dokumen (DOCX/PDF/JPG/PNG) dulu, baru minta konversi!")

    except Exception as e:
        log.error(f"Error processing message from user {user_id}: {e}", exc_info=True)
        try:
            await event.reply("Ada kendala teknis saat memproses pesanmu, coba kirim lagi ya!")
        except Exception:
            pass


# ─── Outgoing Commands ────────────────────────────────────────────────────────

@client.on(events.NewMessage(pattern=r"^/(clear|c)$"))
async def cmd_clear(event):
    if not event.is_private:
        return
    sender = await event.get_sender()
    if isinstance(sender, User):
        agent_db.clear_history(sender.id)
        if event.out:
            try:
                await event.delete()
            except Exception:
                pass
        await event.reply("🗑️ History percakapan direset!")
        log.info(f"🗑️ History cleared for user {sender.id}")


@client.on(events.NewMessage(pattern=r"^/(notes|n)$"))
async def cmd_notes(event):
    if not event.is_private:
        return
    sender = await event.get_sender()
    if not isinstance(sender, User):
        return

    notes = agent_db.get_notes(sender.id)
    if event.out:
        try:
            await event.delete()
        except Exception:
            pass

    if not notes:
        await event.reply("📝 Belum ada catatan tersimpan.")
        return

    lines = ["📝 Catatan lo:"]
    for n in notes:
        ts = n["created_at"][:16]  # trim seconds
        lines.append(f"  [{n['id']}] {n['content']}  ({ts})")
    await event.reply("\n".join(lines))
    log.info(f"📝 /notes shown for user {sender.id} ({len(notes)} notes)")


@client.on(events.NewMessage(pattern=r"^/(remind|r)\s+(.+)$"))
async def cmd_remind(event):
    if not event.is_private:
        return
    sender = await event.get_sender()
    if not isinstance(sender, User):
        return

    args = event.pattern_match.group(2).strip()
    from agent_tools import parse_flexible_time

    # Try flexible pattern parsing
    remind_at = None
    message = args

    m = re.search(r'^\+?(\d+)\s*(detik|sec|second|s|menit|min|minute|m|jam|hour|h|hari|day|d)\s*(lg|lagi)?\s*(.*)$', args, re.IGNORECASE)
    if m:
        remind_at = parse_flexible_time(f"{m.group(1)} {m.group(2)}")
        message = m.group(4).strip() or args
    else:
        parts = args.split(None, 1)
        if len(parts) >= 2:
            remind_at = parse_flexible_time(parts[0])
            message = parts[1].strip()

    if event.out:
        try:
            await event.delete()
        except Exception:
            pass

    if remind_at is None:
        await event.reply(
            f"Format waktu untuk '{args}' tidak dikenali.\n"
            "Contoh:\n"
            "  /r 2m mandi\n"
            "  /r 10s berak\n"
            "  /r +30m minum obat"
        )
        return

    rid = agent_db.save_reminder(sender.id, message, remind_at)
    ts  = remind_at.strftime("%Y-%m-%d %H:%M UTC")
    await event.reply(f"⏰ Reminder #{rid} disimpan — '{message}' @ {ts}")
    log.info(f"⏰ /remind #{rid} saved for user {sender.id}: '{message}' at {ts}")


@client.on(events.NewMessage(pattern=r"^/(calc|k)\s+(.+)$"))
async def cmd_calc(event):
    if not event.is_private:
        return
    sender = await event.get_sender()
    if not isinstance(sender, User):
        return
    expr = event.pattern_match.group(2).strip()

    if event.out:
        try:
            await event.delete()
        except Exception:
            pass

    from agent_tools import _tool_calculate
    result = _tool_calculate(sender.id, expr)
    try:
        data = json.loads(result)
        if "result" in data:
            reply = f"🧮 Hasil: **{data['result']}**\n`{expr}`"
        else:
            reply = f"❌ Error: {data.get('error', 'Gagal menghitung')}"
    except Exception:
        reply = result

    await event.reply(reply)
    log.info(f"🧮 /calc '{expr}' handled for user {sender.id}")


@client.on(events.NewMessage(pattern=r"^/(help|h)$"))
async def cmd_help(event):
    if not event.is_private:
        return
    if event.out:
        try:
            await event.delete()
        except Exception:
            pass
    await event.reply(HELP_TEXT)


@client.on(events.NewMessage(outgoing=True, pattern=r"^/(aion|on|onall)(\s+all)?$"))
async def cmd_aion(event):
    if not event.is_private:
        return
    cmd_name = event.pattern_match.group(1)
    is_global = bool(event.pattern_match.group(2)) or (cmd_name == "onall")
    await event.delete()

    if is_global:
        agent_db.set_ai_enabled_global(True)
        me = await client.get_me()
        await client.send_message("me", "✅ AI diaktifkan untuk semua orang.")
        log.info("[ai_gate] global AI enabled")
    else:
        peer = await event.get_chat()
        if isinstance(peer, User):
            agent_db.set_ai_enabled_for_user(peer.id, True)
            await client.send_message(peer.id, "✅ AI diaktifkan untuk chat ini.")
            log.info(f"[ai_gate] AI enabled for user {peer.id}")


@client.on(events.NewMessage(outgoing=True, pattern=r"^/(aioff|off|offall)(\s+all)?$"))
async def cmd_aioff(event):
    if not event.is_private:
        return
    cmd_name = event.pattern_match.group(1)
    is_global = bool(event.pattern_match.group(2)) or (cmd_name == "offall")
    await event.delete()

    if is_global:
        agent_db.set_ai_enabled_global(False)
        me = await client.get_me()
        await client.send_message("me", "🔇 AI dimatiin untuk semua orang.")
        log.info("[ai_gate] global AI disabled")
    else:
        peer = await event.get_chat()
        if isinstance(peer, User):
            agent_db.set_ai_enabled_for_user(peer.id, False)
            await client.send_message(peer.id, "🔇 AI dimatiin untuk chat ini.")
            log.info(f"[ai_gate] AI disabled for user {peer.id}")


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    global _tg_client, _tg_loop

    log.info("🚀 Starting Telegram Userbot (agent mode)...")

    agent_db.init_db()
    log.info("🗄️ SQLite DB initialized")

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    log.info(f"🌐 Flask keep-alive on port {FLASK_PORT}")

    await client.connect()
    if not await client.is_user_authorized():
        log.error("❌ Session tidak terautentikasi! Silakan jalankan generate_session.py di lokal terlebih dahulu.")
        return
    me = await client.get_me()
    log.info(f"✅ Logged in as: {me.first_name} (@{me.username}) | ID: {me.id}")

    # Wire up scheduler references
    _tg_client = client
    _tg_loop   = asyncio.get_event_loop()

    # Set account offline immediately after login
    await client(UpdateStatusRequest(offline=True))
    log.info("🔕 Account status set to offline")

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(_send_due_reminders, "interval", seconds=5, id="reminder_check",
                      max_instances=1, coalesce=True)
    scheduler.add_job(_keep_offline, "interval", minutes=5, id="keep_offline",
                      max_instances=1, coalesce=True)
    scheduler.add_job(_cleanup_temp_files, "interval", minutes=30, id="temp_cleanup",
                      max_instances=1, coalesce=True)
    scheduler.start()
    log.info("⏰ APScheduler started (reminders every 5s, offline reset every 5m, temp cleanup every 30m)")

    log.info("👂 Listening for incoming private messages...")
    try:
        await client.run_until_disconnected()
    finally:
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())
