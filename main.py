import os
import asyncio
import threading
import logging
import time
import base64
import io
import json
from datetime import datetime

from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.tl.types import User, MessageMediaPhoto, MessageMediaDocument
from groq import Groq
from flask import Flask

import agent_db
from agent_tools import TOOLS, execute_tool

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ─────────────────────────────────────────────────────────────────

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
SESSION_STRING = os.environ.get("SESSION_STRING")
SESSION_FILE = "session"
SUMMARIZE_WORD_LIMIT = 200
FLASK_PORT = 8099
AGENT_MAX_ITERATIONS = 5  # max tool-call rounds per message
HISTORY_WINDOW = 10       # messages sent to model per turn

# ─── AI Client ──────────────────────────────────────────────────────────────

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"
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
- Pakai calculate kalau ada hitungan matematika eksplisit
- Pakai save_note / get_notes kalau user minta simpan atau lihat catatan
- Pakai set_reminder kalau user minta diingatkan sesuatu di waktu tertentu
- Jawab langsung kalau pertanyaannya umum dan lo yakin jawabannya

Lo balas pesan di Telegram. Tetap helpful walau ngeselin."""


# ─── Agent Loop ──────────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def run_agent(user_id: int, user_message: str) -> str:
    """
    Full agent loop:
      1. Save user message to DB
      2. Load recent history from DB
      3. Call model → if tool_calls → execute → feed results back → repeat
      4. Return final text answer
    """
    if word_count(user_message) > SUMMARIZE_WORD_LIMIT:
        user_message = (
            f"[Pesan panjang — tolong ringkas dan jawab inti pertanyaannya]\n\n{user_message}"
        )

    agent_db.save_message(user_id, "user", user_message)
    history = agent_db.load_history(user_id, limit=HISTORY_WINDOW)

    # Build initial messages list for this turn
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    max_retries = 3

    for iteration in range(AGENT_MAX_ITERATIONS):
        log.info(f"[agent] user={user_id} iteration={iteration + 1}/{AGENT_MAX_ITERATIONS}")

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
                is_retriable = any(x in err_str for x in ["503", "429", "rate_limit", "overloaded"])
                if is_retriable and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    log.warning(f"Groq error, retry {attempt + 1}/{max_retries} in {wait}s...")
                    time.sleep(wait)
                else:
                    log.error(f"Groq API error: {e}")
                    # Roll back the user message we already saved
                    return "Ada error nih dari AI-nya, coba lagi ya"

        if response is None:
            return "Ada error nih dari AI-nya, coba lagi ya"

        choice = response.choices[0]
        finish_reason = choice.finish_reason

        # ── Final text answer ──────────────────────────────────────────────
        if finish_reason != "tool_calls":
            reply = choice.message.content or "(gak ada jawaban)"
            agent_db.save_message(user_id, "assistant", reply)
            return reply

        # ── Tool call(s) requested ─────────────────────────────────────────
        assistant_msg = choice.message
        # Append assistant message (with tool_calls) to the running context
        messages.append(assistant_msg)

        tool_calls = assistant_msg.tool_calls or []
        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}

            log.info(f"[agent] tool_call: {tool_name}({tool_args})")
            result = execute_tool(user_id, tool_name, tool_args)
            log.info(f"[agent] tool_result: {result[:200]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Exhausted max iterations without a final answer
    fallback = "Hmm, gua nyoba terus tapi gak kelar-kelar. Coba tanya ulang dengan lebih spesifik."
    agent_db.save_message(user_id, "assistant", fallback)
    return fallback


def get_ai_reply_with_image(user_id: int, image_bytes: bytes, caption: str) -> str:
    """Vision call — stays as a single-shot call (no tools for images)."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt_text = caption if caption else "Gambar ini isinya apa?"

    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
        {"type": "text", "text": prompt_text},
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=1024,
            )
            reply = response.choices[0].message.content
            agent_db.save_message(user_id, "user", f"[Kirim gambar] {prompt_text}")
            agent_db.save_message(user_id, "assistant", reply)
            return reply
        except Exception as e:
            err_str = str(e)
            is_retriable = any(x in err_str for x in ["503", "429", "rate_limit", "overloaded"])
            if is_retriable and attempt < max_retries - 1:
                wait = 2 ** attempt
                log.warning(f"Groq vision error, retry {attempt + 1}/{max_retries} in {wait}s...")
                time.sleep(wait)
                continue
            log.error(f"Groq Vision API error: {e}")
            return "Gambarnya gak bisa gua baca, coba lagi"


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


# ─── Reminder Daemon ─────────────────────────────────────────────────────────

def run_reminder_daemon(tg_client: TelegramClient, loop: asyncio.AbstractEventLoop):
    """
    Background thread: polls for due reminders every 30 seconds and
    sends them via Telegram using the running event loop.
    """
    log.info("⏰ Reminder daemon started")
    while True:
        time.sleep(30)
        try:
            due = agent_db.get_due_reminders()
            for reminder in due:
                rid = reminder["id"]
                user_id = reminder["user_id"]
                message = reminder["message"]
                log.info(f"⏰ Sending reminder {rid} to user {user_id}")
                text = f"🔔 Reminder: {message}"
                future = asyncio.run_coroutine_threadsafe(
                    tg_client.send_message(user_id, text),
                    loop,
                )
                try:
                    future.result(timeout=15)
                    agent_db.mark_reminder_sent(rid)
                    log.info(f"⏰ Reminder {rid} sent and marked as done")
                except Exception as send_err:
                    log.error(f"⏰ Failed to send reminder {rid}: {send_err}")
        except Exception as e:
            log.error(f"⏰ Reminder daemon error: {e}", exc_info=True)


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

    user_id = sender.id
    username = f"@{sender.username}" if sender.username else sender.first_name

    is_image = False
    if event.message.media:
        if isinstance(event.message.media, MessageMediaPhoto):
            is_image = True
        elif isinstance(event.message.media, MessageMediaDocument):
            mime = getattr(event.message.file, "mime_type", "") or ""
            if mime.startswith("image/"):
                is_image = True

    if is_image:
        caption = event.raw_text.strip()
        is_view_once = bool(getattr(event.message.media, "ttl_seconds", None))

        log.info(f"🖼️ [{username} | {user_id}] gambar diterima, caption: '{caption}'")
        async with client.action(event.chat_id, "typing"):
            image_bytes = await event.message.download_media(file=bytes)
            if not image_bytes:
                await event.reply("Gambarnya gagal ke-download, coba kirim ulang")
                return
            log.info(f"🖼️ Image downloaded: {len(image_bytes)} bytes")

            if is_view_once:
                try:
                    buf = io.BytesIO(image_bytes)
                    buf.name = "photo.jpg"
                    await client.send_file(
                        "me",
                        buf,
                        caption=f"📸 Foto sekali liat dari {username}",
                    )
                    log.info(f"📥 [{username} | {user_id}] foto sekali liat disimpen ke Saved Messages")
                except Exception as e:
                    log.error(f"Gagal simpen ke Saved Messages: {e}", exc_info=True)

            reply = await asyncio.to_thread(get_ai_reply_with_image, user_id, image_bytes, caption)
        await event.reply(reply)
        log.info(f"📤 [{username} | {user_id}] {reply[:100]}{'...' if len(reply) > 100 else ''}")
        return

    text = event.raw_text.strip()
    if not text:
        return

    log.info(f"📥 [{username} | {user_id}] {text[:100]}{'...' if len(text) > 100 else ''}")

    async with client.action(event.chat_id, "typing"):
        reply = await asyncio.to_thread(run_agent, user_id, text)

    await event.reply(reply)
    log.info(f"📤 [{username} | {user_id}] {reply[:100]}{'...' if len(reply) > 100 else ''}")


@client.on(events.NewMessage(outgoing=True, pattern=r"^/clear$"))
async def handle_clear_command(event):
    if not event.is_private:
        return

    peer = await event.get_chat()
    if isinstance(peer, User):
        target_user_id = peer.id
        agent_db.clear_history(target_user_id)
        await event.delete()
        await client.send_message(peer.id, "🗑️ History percakapan direset!")
        log.info(f"🗑️ History cleared for user {target_user_id}")


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    log.info("🚀 Starting Telegram Userbot (agent mode)...")

    agent_db.init_db()
    log.info("🗄️ SQLite DB initialized")

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    log.info(f"🌐 Flask keep-alive running on port {FLASK_PORT}")

    await client.start()
    me = await client.get_me()
    log.info(f"✅ Logged in as: {me.first_name} (@{me.username}) | ID: {me.id}")

    loop = asyncio.get_event_loop()
    reminder_thread = threading.Thread(
        target=run_reminder_daemon, args=(client, loop), daemon=True
    )
    reminder_thread.start()
    log.info("⏰ Reminder daemon started")

    log.info("👂 Listening for incoming private messages...")
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
