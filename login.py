import os
import asyncio
from telethon import TelegramClient

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]

async def main():
    print("=== Telegram Userbot Login ===")
    print("Proses ini akan menyimpan session ke file 'session.session'")
    print("Setelah login berhasil, jalankan workflow 'Telegram Userbot'\n")

    client = TelegramClient("session", API_ID, API_HASH)
    await client.start()
    me = await client.get_me()
    print(f"\n✅ Login berhasil sebagai: {me.first_name} (@{me.username})")
    print("Session tersimpan. Sekarang kamu bisa jalankan workflow utama!")
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
