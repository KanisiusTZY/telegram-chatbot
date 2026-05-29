import os
import asyncio
import threading
import logging
import time
from datetime import datetime
from collections import defaultdict

from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.tl.types import User
from groq import Groq
from flask import Flask

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
MAX_HISTORY = 10
SUMMARIZE_WORD_LIMIT = 200
FLASK_PORT = 8099

# ─── AI Client ──────────────────────────────────────────────────────────────

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"

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

Lo balas pesan di Telegram. Tetap helpful walau ngeselin."""

# ─── Conversation History ────────────────────────────────────────────────────

# {user_id: [{"role": "user"/"assistant", "content": "..."}]}
histories: dict = defaultdict(list)


def add_to_history(user_id: int, role: str, content: str) -> None:
    histories[user_id].append({"role": role, "content": content})
    if len(histories[user_id]) > MAX_HISTORY:
        histories[user_id] = histories[user_id][-MAX_HISTORY:]


def clear_history(user_id: int) -> None:
    histories[user_id] = []


def word_count(text: str) -> int:
    return len(text.split())


# ─── AI Reply ───────────────────────────────────────────────────────────────

def get_ai_reply(user_id: int, user_message: str) -> str:
    if word_count(user_message) > SUMMARIZE_WORD_LIMIT:
        user_message = (
            f"[Pesan panjang — tolong ringkas dan jawab inti pertanyaannya]\n\n{user_message}"
        )

    add_to_history(user_id, "user", user_message)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + histories[user_id]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=1024,
            )
            reply = response.choices[0].message.content
            add_to_history(user_id, "assistant", reply)
            return reply
        except Exception as e:
            err_str = str(e)
            is_retriable = any(x in err_str for x in ["503", "429", "rate_limit", "overloaded"])
            if is_retriable and attempt < max_retries - 1:
                wait = 2 ** attempt
                log.warning(f"Groq error, retry {attempt + 1}/{max_retries} in {wait}s...")
                time.sleep(wait)
                continue
            log.error(f"Groq API error: {e}")
            if histories[user_id] and histories[user_id][-1]["role"] == "user":
                histories[user_id].pop()
            return "Ada error nih dari AI-nya, coba lagi ya"


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
    text = event.raw_text.strip()

    if not text:
        return

    log.info(f"📥 [{username} | {user_id}] {text[:100]}{'...' if len(text) > 100 else ''}")

    async with client.action(event.chat_id, "typing"):
        reply = await asyncio.to_thread(get_ai_reply, user_id, text)

    await event.reply(reply)
    log.info(f"📤 [{username} | {user_id}] {reply[:100]}{'...' if len(reply) > 100 else ''}")


@client.on(events.NewMessage(outgoing=True, pattern=r"^/clear$"))
async def handle_clear_command(event):
    if not event.is_private:
        return

    peer = await event.get_chat()
    if isinstance(peer, User):
        target_user_id = peer.id
        clear_history(target_user_id)
        await event.delete()
        await client.send_message(peer.id, "🗑️ History percakapan direset!")
        log.info(f"🗑️ History cleared for user {target_user_id}")


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    log.info("🚀 Starting Telegram Userbot...")

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    log.info(f"🌐 Flask keep-alive running on port {FLASK_PORT}")

    await client.start()
    me = await client.get_me()
    log.info(f"✅ Logged in as: {me.first_name} (@{me.username}) | ID: {me.id}")
    log.info("👂 Listening for incoming private messages...")

    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
