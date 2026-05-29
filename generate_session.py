import os
import asyncio
from telethon import TelegramClient
from telethon.sessions import StringSession

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]

async def main():
    print("=== Generate SESSION_STRING ===")
    print("Login dulu, nanti dikasih string yang bisa dipaste ke Railway.\n")

    async with TelegramClient(StringSession(), API_ID, API_HASH) as client:
        session_string = client.session.save()
        me = await client.get_me()
        print(f"\n✅ Login berhasil sebagai: {me.first_name} (@{me.username})")
        print("\n" + "="*60)
        print("SESSION_STRING lo (copy semua ini, paste ke Railway):")
        print("="*60)
        print(session_string)
        print("="*60)
        print("\nPaste string di atas ke Railway → Variables → SESSION_STRING")

if __name__ == "__main__":
    asyncio.run(main())
