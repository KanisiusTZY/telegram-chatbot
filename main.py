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
FLASK_PORT       = 8099

SUMMARIZE_WORD_LIMIT  = 200
AGENT_MAX_ITERATIONS  = 5   # max tool-call rounds per user message
HISTORY_WINDOW        = 10  # messages sent to model per turn
RATE_LIMIT_CALLS      = 20  # max messages per user per window
RATE_LIMIT_WINDOW     = 60  # seconds

# ─── AI Client ──────────────────────────────────────────────────────────────

groq_client      = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

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

Contoh gaya:
- "ya ampun, ini bisa di-google dalam 5 detik loh"
- "oke oke gua jawab, tapi lo harus janji gak nanya hal bodoh lagi"
- "...serius? itu pertanyaannya?"
- "iya bisa, tapi kenapa lo nanya ke gua"

Kapan pakai tools vs jawab langsung:
- Pakai web_search kalau nanya soal berita terkini, harga, cuaca, atau hal yang mungkin udah berubah
- Pakai calculate kalau ada hitungan matematika eksplisit (termasuk dari teks/gambar yang dikirim user)
- Pakai save_note / get_notes kalau user minta simpan atau lihat catatan
- Pakai set_reminder kalau user minta diingatkan sesuatu di waktu tertentu
- Jawab langsung kalau pertanyaannya umum dan lo yakin jawabannya

Kalau user kirim gambar, lo terima konteks "Gambar yang dikirim user: ..." — gunakan itu buat jawab/jalankan tool yang relevan.

Lo balas pesan di Telegram. Tetap helpful walau ngeselin."""

HELP_TEXT = """Gua bisa ngapain aja:

Ngobrol biasa — kirim pesan ke gua, gua balas (dengan muka bete)

Tools yang bisa gua pakai otomatis:
  web_search    — cari info terkini
  calculate     — hitung matematika
  save_note     — simpan catatan
  get_notes     — lihat catatan lo
  set_reminder  — pasang pengingat

Commands (ketik dari chat ini):
  /help         — pesan ini
  /notes        — lihat semua catatan lo
  /remind <waktu> <pesan>
                — buat reminder manual
                  waktu bisa: +30m +2h +1d
                  atau ISO: 2026-08-05T09:00
  /clear        — hapus history percakapan

Gambar juga bisa — gua analisis + bisa langsung hitung/cari dari isinya."""


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

        response = None
        for attempt in range(max_retries):
            try:
                response = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=1024,
                )
                break
            except Exception as e:
                err_str = str(e)
                retriable = any(x in err_str for x in ["503", "429", "rate_limit", "overloaded"])
                if retriable and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    log.warning(f"Groq retry {attempt + 1}/{max_retries} in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    log.error(f"Groq API error: {e}")
                    return "Ada error nih dari AI-nya, coba lagi ya"

        if response is None:
            return "Ada error nih dari AI-nya, coba lagi ya"

        choice = response.choices[0]

        # ── Final text answer ──────────────────────────────────────────────
        if choice.finish_reason != "tool_calls":
            reply = choice.message.content or "(gak ada jawaban)"
            agent_db.save_message(user_id, "assistant", reply)
            return reply

        # ── Tool call(s) ───────────────────────────────────────────────────
        assistant_msg = choice.message
        messages.append(assistant_msg)

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
                "content": result,
            })

    fallback = "Hmm, gua nyoba terus tapi gak kelar-kelar. Coba tanya ulang dengan lebih spesifik."
    agent_db.save_message(user_id, "assistant", fallback)
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
        f"termasuk semua teks/angka yang terlihat."
        + (f"\n\nPertanyaan/instruksi user terkait gambar ini: {caption}" if caption else "")
    )
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {"role": "system", "content": "Kamu adalah AI vision yang mendeskripsikan gambar secara akurat dan lengkap."},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            retriable = any(x in err_str for x in ["503", "429", "rate_limit", "overloaded"])
            if retriable and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                log.error(f"Groq vision error: {e}")
                return None
    return None


def get_ai_reply_with_image(user_id: int, image_bytes: bytes, caption: str) -> str:
    """
    Two-step image handling:
      1. Vision model extracts description/text from the image.
      2. Text agent loop processes the description, can use all tools
         (e.g. calculate totals from a receipt, search based on image content).
    """
    vision_text = _vision_describe(image_bytes, caption)

    if vision_text is None:
        return "Gambarnya gak bisa gua baca, coba lagi"

    # Build a combined user message that feeds into the agent loop
    if caption:
        combined = (
            f"[Gambar yang dikirim user. Deskripsi otomatis dari gambar:\n{vision_text}]\n\n"
            f"Pertanyaan/instruksi user: {caption}"
        )
    else:
        combined = f"[User kirim gambar. Deskripsi otomatis:\n{vision_text}]"

    # Save original label to DB, then run agent with full context
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


# ─── Telegram Userbot ────────────────────────────────────────────────────────

if SESSION_STRING:
    client = TelegramClient(StringSession(SESSION_STRING), API_ID, API_HASH)
    log.info("Using StringSession from env var")
else:
    client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
    log.info("Using SQLite session file")


@client.on(events.NewMessage(incoming=True, func=lambda e: e.is_private))
async def handle_private_message(event):
    sender = await event.get_sender()
    if not isinstance(sender, User) or sender.bot:
        return

    user_id  = sender.id
    username = f"@{sender.username}" if sender.username else sender.first_name

    # ── Rate limit check ──────────────────────────────────────────────────
    if not _rate_limiter.is_allowed(user_id):
        log.warning(f"[rate_limit] user={user_id} ({username}) throttled")
        await event.reply("Pelan-pelan, gua bukan mesin ketik. Tunggu bentar dulu.")
        return

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

            if is_view_once:
                try:
                    buf = io.BytesIO(image_bytes)
                    buf.name = "photo.jpg"
                    await client.send_file("me", buf, caption=f"📸 Foto sekali liat dari {username}")
                    log.info(f"📥 [{username}|{user_id}] view-once saved")
                except Exception as e:
                    log.error(f"Gagal save ke Saved Messages: {e}", exc_info=True)

            reply = await asyncio.to_thread(get_ai_reply_with_image, user_id, image_bytes, caption)

        await event.reply(reply)
        log.info(f"📤 [{username}|{user_id}] {reply[:100]}{'...' if len(reply) > 100 else ''}")
        return

    # ── Text handling ─────────────────────────────────────────────────────
    text = event.raw_text.strip()
    if not text:
        return

    log.info(f"📥 [{username}|{user_id}] {text[:100]}{'...' if len(text) > 100 else ''}")

    async with client.action(event.chat_id, "typing"):
        reply = await asyncio.to_thread(run_agent, user_id, text)

    await event.reply(reply)
    log.info(f"📤 [{username}|{user_id}] {reply[:100]}{'...' if len(reply) > 100 else ''}")


# ─── Outgoing Commands ────────────────────────────────────────────────────────

@client.on(events.NewMessage(outgoing=True, pattern=r"^/clear$"))
async def cmd_clear(event):
    if not event.is_private:
        return
    peer = await event.get_chat()
    if isinstance(peer, User):
        agent_db.clear_history(peer.id)
        await event.delete()
        await client.send_message(peer.id, "🗑️ History percakapan direset!")
        log.info(f"🗑️ History cleared for user {peer.id}")


@client.on(events.NewMessage(outgoing=True, pattern=r"^/notes$"))
async def cmd_notes(event):
    if not event.is_private:
        return
    peer = await event.get_chat()
    if not isinstance(peer, User):
        return

    notes = agent_db.get_notes(peer.id)
    await event.delete()

    if not notes:
        await client.send_message(peer.id, "📝 Belum ada catatan tersimpan.")
        return

    lines = ["📝 Catatan lo:"]
    for n in notes:
        ts = n["created_at"][:16]  # trim seconds
        lines.append(f"  [{n['id']}] {n['content']}  ({ts})")
    await client.send_message(peer.id, "\n".join(lines))
    log.info(f"📝 /notes shown for user {peer.id} ({len(notes)} notes)")


@client.on(events.NewMessage(outgoing=True, pattern=r"^/remind\s+(.+)$"))
async def cmd_remind(event):
    if not event.is_private:
        return
    peer = await event.get_chat()
    if not isinstance(peer, User):
        return

    # Parse: /remind <time> <message>
    args = event.pattern_match.group(1).strip()
    parts = args.split(None, 1)  # split on first whitespace only

    await event.delete()

    if len(parts) < 2:
        usage = (
            "Format salah.\n"
            "Contoh:\n"
            "  /remind +30m Minum obat\n"
            "  /remind +2h Meeting\n"
            "  /remind 2026-08-05T09:00 Deadline\n"
        )
        await client.send_message(peer.id, usage)
        return

    time_str, message = parts[0], parts[1].strip()
    remind_at = parse_remind_time(time_str)

    if remind_at is None:
        await client.send_message(
            peer.id,
            f"Gak ngerti format waktu '{time_str}'.\n"
            "Pakai +30m, +2h, +1d, atau 2026-08-05T09:00"
        )
        return

    rid = agent_db.save_reminder(peer.id, message, remind_at)
    ts  = remind_at.strftime("%Y-%m-%d %H:%M UTC")
    await client.send_message(peer.id, f"⏰ Reminder #{rid} disimpan — '{message}' @ {ts}")
    log.info(f"⏰ /remind #{rid} saved for user {peer.id}: '{message}' at {ts}")


@client.on(events.NewMessage(outgoing=True, pattern=r"^/help$"))
async def cmd_help(event):
    if not event.is_private:
        return
    await event.delete()
    peer = await event.get_chat()
    if isinstance(peer, User):
        await client.send_message(peer.id, HELP_TEXT)


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    global _tg_client, _tg_loop

    log.info("🚀 Starting Telegram Userbot (agent mode)...")

    agent_db.init_db()
    log.info("🗄️ SQLite DB initialized")

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    log.info(f"🌐 Flask keep-alive on port {FLASK_PORT}")

    await client.start()
    me = await client.get_me()
    log.info(f"✅ Logged in as: {me.first_name} (@{me.username}) | ID: {me.id}")

    # Wire up scheduler references
    _tg_client = client
    _tg_loop   = asyncio.get_event_loop()

    # Set account offline immediately after login
    await client(UpdateStatusRequest(offline=True))
    log.info("🔕 Account status set to offline")

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(_send_due_reminders, "interval", seconds=30, id="reminder_check",
                      max_instances=1, coalesce=True)
    scheduler.add_job(_keep_offline, "interval", minutes=5, id="keep_offline",
                      max_instances=1, coalesce=True)
    scheduler.start()
    log.info("⏰ APScheduler started (reminders every 30s, offline reset every 5m)")

    log.info("👂 Listening for incoming private messages...")
    try:
        await client.run_until_disconnected()
    finally:
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())
