import os
import threading
import asyncio
import aiohttp
from http.server import BaseHTTPRequestHandler, HTTPServer
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import requests
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

# Load configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

if not BOT_TOKEN or not GROQ_API_KEY:
    raise ValueError("BOT_TOKEN and GROQ_API_KEY must be set in environment variables")

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Самая щедрая бесплатная модель на Groq (проверено 14.09.2026):
# openai/gpt-oss-120b — 120B, 131k контекст, бесплатно, без карты
# Запасной вариант с большими лимитами: openai/gpt-oss-20b
MODEL = "openai/gpt-oss-120b"


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я бот с бесплатным ИИ от Groq.\n"
        "Модель: openai/gpt-oss-120b (120B, бесплатно, быстро).\n"
        "Просто задавай вопросы!",
        reply_markup=types.ReplyKeyboardRemove(),
    )


@dp.message(F.text)
async def handle_message(message: types.Message):
    user_text = message.text
    print(f"INCOMING user_id={message.from_user.id} username=@{message.from_user.username} text={user_text!r}", flush=True)

    try:
        response = requests.post(
            GROQ_ENDPOINT,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": user_text}],
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            timeout=60,
        )

        if response.status_code != 200:
            await message.answer(
                f"❌ Ошибка от Groq: {response.status_code}\n"
                f"Ответ: {response.text[:1000]}"
            )
            return

        data = response.json()
        ai_response = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not ai_response:
            await message.answer("❌ Пустой ответ от ИИ, попробуй ещё раз")
            return

        # Без parse_mode, чтобы < > в ответе не ломали HTML-разметку
        await message.answer(f"🤖 {ai_response}")

    except Exception as e:
        await message.answer(f"❌ Произошла ошибка: {str(e)}")


async def main():
    _start_health_server()
    asyncio.create_task(_self_ping())
    print("Бот запускается...", flush=True)
    await dp.start_polling(bot)


# --- Self-ping для Render Web Service (бесплатный тариф) ---
# Render засыпает через ~15 мин без HTTP-трафика.
# Этот фоновый таск каждые 4 минуты стучится в свой URL, чтобы не усыпили.
RENDER_URL = os.getenv("RENDER_URL", "https://agent-ju76.onrender.com")
PING_INTERVAL = 240  # секунды (4 минуты)

async def _self_ping():
    await asyncio.sleep(10)  # дать боту запуститься
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(RENDER_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    print(f"Self-ping {resp.status}", flush=True)
            except Exception as e:
                print(f"Self-ping error: {e}", flush=True)
            await asyncio.sleep(PING_INTERVAL)
class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


def _start_health_server():
    port = int(os.getenv("PORT", "10000"))
    try:
        server = HTTPServer(("0.0.0.0", port), _HealthHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"Healthcheck слушает порт {port}", flush=True)
    except Exception as e:
        print(f"Health-сервер не стартовал: {e}", flush=True)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())