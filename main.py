import os
import json
import html
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
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))

if not BOT_TOKEN or not GROQ_API_KEY:
    raise ValueError("BOT_TOKEN and GROQ_API_KEY must be set in environment variables")

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

MODEL = "openai/gpt-oss-120b"

# --- User storage ---
USERS_FILE = "users.json"
_user_cache = {}
_user_lock = asyncio.Lock()

def _load_users():
    global _user_cache
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            _user_cache = json.load(f)
    except FileNotFoundError:
        _user_cache = {}

def _save_users():
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(_user_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Save users error: {e}", flush=True)

def _get_user(uid: int) -> dict:
    key = str(uid)
    if key not in _user_cache:
        _user_cache[key] = {
            "chat_id": uid,
            "name": "",
            "username": "",
            "preferences": "",
            "history": [],
            "first_seen": "",
            "last_seen": ""
        }
    return _user_cache[key]

def _update_user(message: types.Message):
    from datetime import datetime
    u = _get_user(message.from_user.id)
    u["name"] = message.from_user.first_name or ""
    u["username"] = message.from_user.username or ""
    now = datetime.utcnow().isoformat()
    if not u["first_seen"]:
        u["first_seen"] = now
    u["last_seen"] = now
    _save_users()

def _add_history(uid: int, role: str, content: str):
    u = _get_user(uid)
    u["history"].append({"role": role, "content": content})
    if len(u["history"]) > 20:
        u["history"] = u["history"][-20:]
    _save_users()

def _build_system_prompt(uid: int) -> str:
    u = _get_user(uid)
    name = u.get("name") or "друг"
    prefs = u.get("preferences") or ""
    base = (
        f"Ты — дружелюбный ИИ-ассистент. Обращайся к пользователю по имени: {name}. "
        "Говори естественно, кратко и по делу. Избегай шаблонных фраз."
    )
    if prefs:
        base += f" П предпочтения пользователя: {prefs}."
    return base

# --- Keyboards ---
def _admin_keyboard() -> types.InlineKeyboardMarkup:
    return types.InlineKeyboardMarkup(inline_keyboard=[[
        types.InlineKeyboardButton(text="📢 Рассылка", callback_data="admin_broadcast"),
        types.InlineKeyboardButton(text="📊 Статистика", callback_data="admin_stats"),
    ], [
        types.InlineKeyboardButton(text="⬅️ Назад", callback_data="admin_back"),
    ]])

# --- Handlers ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    _update_user(message)
    u = _get_user(message.from_user.id)
    name = u.get("name") or "друг"
    kb = _admin_keyboard() if message.from_user.id == ADMIN_ID else types.ReplyKeyboardRemove()
    await message.answer(
        f"Привет, {name}! Я бот с бесплатным ИИ от Groq.\n"
        f"Модель: {MODEL} (120B, бесплатно, быстро).\n"
        "Просто задавай вопросы — я запомню твои предпочтения.",
        reply_markup=kb,
    )

@dp.message(Command("broadcast"))
async def cmd_broadcast(message: types.Message):
    if message.from_user.id != ADMIN_ID:
        return
    text = message.text.split(maxsplit=1)
    if len(text) < 2:
        await message.answer("Использование: /broadcast <текст>")
        return
    await _do_broadcast(text[1], message)

@dp.callback_query(F.data == "admin_broadcast")
async def cb_broadcast(callback: types.CallbackQuery):
    if callback.from_user.id != ADMIN_ID:
        await callback.answer("Только для админа", show_alert=True)
        return
    await callback.message.answer("Введи текст рассылки (ответь на это сообщение):")
    await callback.answer()

@dp.callback_query(F.data == "admin_stats")
async def cb_stats(callback: types.CallbackQuery):
    if callback.from_user.id != ADMIN_ID:
        await callback.answer("Только для админа", show_alert=True)
        return
    total = len(_user_cache)
    active = sum(1 for u in _user_cache.values() if u.get("history"))
    await callback.message.answer(f"👥 Пользователей: {total}\n💬 С историей: {active}")
    await callback.answer()

@dp.callback_query(F.data == "admin_back")
async def cb_back(callback: types.CallbackQuery):
    if callback.from_user.id != ADMIN_ID:
        await callback.answer("Только для админа", show_alert=True)
        return
    await callback.message.edit_reply_markup(reply_markup=_admin_keyboard())
    await callback.answer()

@dp.message(F.text)
async def handle_message(message: types.Message):
    _update_user(message)
    _add_history(message.from_user.id, "user", message.text)

    # Если админ ответил на просьбу ввести текст рассылки
    if message.from_user.id == ADMIN_ID and message.reply_to_message:
        if "Введи текст рассылки" in (message.reply_to_message.text or ""):
            await _do_broadcast(message.text, message)
            return

    try:
        u = _get_user(message.from_user.id)
        system_prompt = _build_system_prompt(message.from_user.id)
        messages = [{"role": "system", "content": system_prompt}] + u["history"][-10:]

        response = requests.post(
            GROQ_ENDPOINT,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            timeout=60,
        )

        if response.status_code != 200:
            await message.answer(f"❌ Ошибка от Groq: {response.status_code}")
            return

        data = response.json()
        ai_response = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not ai_response:
            await message.answer("❌ Пустой ответ от ИИ, попробуй ещё раз")
            return

        _add_history(message.from_user.id, "assistant", ai_response)
        # Экранируем HTML, чтобы не было ошибок парсинга
        safe = html.escape(ai_response)
        await message.answer(f"🤖 {safe}")

    except Exception as e:
        await message.answer(f"❌ Произошла ошибка: {str(e)}")

async def _do_broadcast(text: str, message: types.Message):
    sent = 0
    failed = 0
    safe = html.escape(text)
    for uid_str in _user_cache.keys():
        try:
            await bot.send_message(int(uid_str), f"📢 <b>Рассылка от админа:</b>\n\n{safe}", parse_mode=ParseMode.HTML)
            sent += 1
            await asyncio.sleep(0.05)
        except Exception:
            failed += 1
    await message.answer(f"✅ Рассылка завершена: отправлено {sent}, ошибок {failed}")

# --- Self-ping ---
RENDER_URL = os.getenv("RENDER_URL", "https://agent-ju76.onrender.com")
PING_INTERVAL = 240

async def _self_ping():
    await asyncio.sleep(10)
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(RENDER_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    print(f"Self-ping {resp.status}", flush=True)
            except Exception as e:
                print(f"Self-ping error: {e}", flush=True)
            await asyncio.sleep(PING_INTERVAL)

# --- Healthcheck ---
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

async def main():
    _load_users()
    _start_health_server()
    asyncio.create_task(_self_ping())
    print("Бот запускается...", flush=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())