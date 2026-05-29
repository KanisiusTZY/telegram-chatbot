import os
import asyncio
import threading
import logging
import time
from datetime import datetime
from collections import defaultdict

from telethon import TelegramClient, events
from telethon.tl.types import User
from google import genai
from google.genai import types as genai_types
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
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
SESSION_FILE = "session"
MAX_HISTORY = 10
SUMMARIZE_WORD_LIMIT = 200
FLASK_PORT = 8099

# ─── AI Client ──────────────────────────────────────────────────────────────

gemini = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_FALLBACK_MODEL = "gemini-2.0-flash-lite"

SYSTEM_PROMPT = """Lo adalah asisten AI yang santai, helpful, dan fun. Kepribadian lo:
- Bahasa sehari-hari campur Indo-Inggris (kayak anak Jakarta ngobrol)
- Casual tapi tetap informatif dan akurat
- Jawaban ringkas, langsung ke inti — gak perlu basa-basi panjang
- Boleh pakai emoji secukupnya, tapi jangan lebay
- Kalau ada pertanyaan teknis, jawab dengan jelas tapi tetap santai
- Kalau diminta translate, langsung kasih terjemahan + penjelasan singkat kalau perlu
- Gak usah selalu pakai "Hei!", "Tentu!", atau pembuka formal lainnya

Lo balas pesan di Telegram — jadi keep it conversational."""

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

    contents = []
    for msg in histories[user_id]:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(genai_types.Content(
            role=role,
            parts=[genai_types.Part(text=msg["content"])]
        ))

    for model in [GEMINI_MODEL, GEMINI_FALLBACK_MODEL]:
        max_retries = 2 if model == GEMINI_MODEL else 1
        for attempt in range(max_retries):
            try:
                response = gemini.models.generate_content(
                    model=model,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=1024,
                    ),
                )
                if model != GEMINI_MODEL:
                    log.info(f"Used fallback model: {model}")
                reply = response.text
                add_to_history(user_id, "assistant", reply)
                return reply
            except Exception as e:
                err_str = str(e)
                is_retriable = any(x in err_str for x in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"])
                if is_retriable and attempt < max_retries - 1:
                    log.warning(f"Gemini error ({model}), retry in 1s...")
                    time.sleep(1)
                    continue
                if is_retriable:
                    log.warning(f"Model {model} unavailable/quota exceeded, trying fallback...")
                    break
                log.error(f"Gemini API error: {e}")
                if histories[user_id] and histories[user_id][-1]["role"] == "user":
                    histories[user_id].pop()
                return "Aduh, ada error nih dari AI-nya 😅 Coba lagi ya?"

    if histories[user_id] and histories[user_id][-1]["role"] == "user":
        histories[user_id].pop()
    return "AI-nya lagi overload nih 😓 Coba beberapa saat lagi ya!"


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

client = TelegramClient(SESSION_FILE, API_ID, API_HASH)


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
