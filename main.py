import os
import json
import html
import threading
import asyncio
import random
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
            "last_seen": "",
            "awaiting_broadcast": False
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

# --- Search state ---
_SEARCH_TASKS: dict = {}

def _get_search_state(uid: int) -> dict:
    key = str(uid)
    if key not in _SEARCH_TASKS:
        _SEARCH_TASKS[key] = {
            "query": "",
            "enabled": False,
            "task": None,
        }
    return _SEARCH_TASKS[key]

# --- Keyboards ---
def _main_keyboard() -> types.ReplyKeyboardMarkup:
    return types.ReplyKeyboardMarkup(
        keyboard=[[types.KeyboardButton(text="🔍 Поиск вакансий"),
                  types.KeyboardButton(text="📢 Рассылка")],
               [types.KeyboardButton(text="📊 Статистика"),
                  types.KeyboardButton(text="⬅️ Назад")]],
        resize_keyboard=True,
        selective=True,
    )

def _admin_keyboard() -> types.InlineKeyboardMarkup:
    return types.InlineKeyboardMarkup(inline_keyboard=[[
        types.InlineKeyboardButton(text="📢 Рассылка", callback_data="admin_broadcast"),
        types.InlineKeyboardButton(text="📊 Статистика", callback_data="admin_stats"),
    ], [
        types.InlineKeyboardButton(text="⬅️ Назад", callback_data="admin_back"),
    ]])

def _search_keyboard() -> types.ReplyKeyboardMarkup:
    return types.ReplyKeyboardMarkup(
        keyboard=[[
            types.KeyboardButton(text="🔍 Начать поиск"),
            types.KeyboardButton(text="⏹️ Стоп поиск"),
        ]],
        resize_keyboard=True,
        selective=True,
    )

# --- Handlers ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    _update_user(message)
    u = _get_user(message.from_user.id)
    name = u.get("name") or "друг"
    kb = _main_keyboard() if message.from_user.id != ADMIN_ID else _admin_keyboard()
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
    u = _get_user(callback.from_user.id)
    u["awaiting_broadcast"] = True
    _save_users()
    await callback.message.answer("✍️ Пришли текст рассылки следующим сообщением:")
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
    u = _get_user(callback.from_user.id)
    u["awaiting_broadcast"] = False
    _save_users()
    try:
        await callback.message.edit_reply_markup(reply_markup=_admin_keyboard())
    except Exception:
        await callback.message.answer("Главное меню:", reply_markup=_admin_keyboard())
    await callback.answer()

@dp.message(F.text == "🔍 Начать поиск")
async def btn_start_search(message: types.Message):
    _update_user(message)
    u = _get_user(message.from_user.id)
    state = _get_search_state(message.from_user.id)
    if state["enabled"]:
        await message.answer("Поиск уже запущен", reply_markup=_search_keyboard())
        return
    # Просим пользователя ввести запрос
    await message.answer("Введите запрос для поиска вакансий:", reply_markup=_search_keyboard())
    u["awaiting_search_query"] = True
    _save_users()

@dp.message(F.text == "⏹️ Стоп поиск")
async def btn_stop_search(message: types.Message):
    _update_user(message)
    state = _get_search_state(message.from_user.id)
    state["enabled"] = False
    if state["task"]:
        state["task"].cancel()
        state["task"] = None
    await message.answer("Поиск остановлен✅", reply_markup=types.ReplyKeyboardRemove())
    u = _get_user(message.from_user.id)
    u["awaiting_search_query"] = False
    _save_users()

@dp.message(Command("search"))
async def cmd_search(message: types.Message):
    _update_user(message)
    u = _get_user(message.from_user.id)
    state = _get_search_state(message.from_user.id)
    if state["enabled"]:
        await message.answer("Поиск уже запущен", reply_markup=_search_keyboard())
        return
    await message.answer("Введите запрос для поиска вакансий:", reply_markup=_search_keyboard())
    u["awaiting_search_query"] = True
    _save_users()

# Хэндлер для всех текстовых сообщений (кроме команд)
@dp.message()
async def handle_message(message: types.Message):
    _update_user(message)

    # Если пользователь ожидает ввод запроса для поиска
    u = _get_user(message.from_user.id)
    if u.get("awaiting_search_query"):
        u["awaiting_search_query"] = False
        query = message.text
        state = _get_search_state(message.from_user.id)
        state["query"] = query
        state["enabled"] = True
        
        # Выполняем поиск с задержками
        await message.answer(f"🔍 Начинаю поиск вакансий по запросу: {query}", reply_markup=types.ReplyKeyboardRemove())
        await _do_search_job(message, query)
        return

    # Если админ в режиме ожидания текста рассылки
    if message.from_user.id == ADMIN_ID:
        if u.get("awaiting_broadcast"):
            u["awaiting_broadcast"] = False
            _save_users()
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


async def _random_delay(min_seconds: int = 5, max_seconds: int = 15):
    """Random delay to avoid rate limits"""
    delay = random.uniform(min_seconds, max_seconds)
    await asyncio.sleep(delay)


async def _do_search_job(message: types.Message, query: str):
    """Search HH.RU for vacancies and send 3 results"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    
    vacancies_found = 0
    page = 0
    
    async with aiohttp.ClientSession() as session:
        while vacancies_found < 3:
            try:
                url = f"https://hh.ru/search/vacancy?text={query}&page={page}&per_page=50"
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        break
                    # Try to parse as JSON, fallback to text
                    try:
                        data = await resp.json()
                    except Exception:
                        text = await resp.text()
                        # If JSON fails, try to extract from HTML (basic approach)
                        data = {"items": []}
                    
                    vacancies = data.get("items", [])
                    
                    for vacancy in vacancies:
                        if vacancies_found >= 3:
                            break
                        title = vacancy.get("name", "Без названия")
                        company = vacancy.get("employer", {}).get("name", "Компания не указана")
                        salary = vacancy.get("salary", {})
                        salary_text = f"{salary.get('from', 0)} - {salary.get('to', 0)} {salary.get('currency', 'RUB')}" if salary else "По договоренности"
                        url_link = vacancy.get("alternate_url", "")
                        
                        msg = f"📋 {title}\n🏢 {company}\n💰 {salary_text}\n🔗 {url_link}"
                        try:
                            await message.answer(msg, parse_mode=ParseMode.HTML)
                            vacancies_found += 1
                        except Exception:
                            pass
                        
                        # Random delay between requests to avoid hitting limits
                        await asyncio.sleep(random.uniform(2, 7))
                
                page += 1
                if page > 10:  # Safety limit
                    break
                    
            except Exception as e:
                print(f"Search error: {e}", flush=True)
                break
    
    if vacancies_found == 0:
        await message.answer("Вакансии не найдены по этому запросу")
    else:
        await message.answer(f"✅ Найдено {vacancies_found} вакансий (выводим первые 3)")
    
    # Continue search in 3 hours if still enabled
    state = _get_search_state(message.from_user.id)
    if state["enabled"]:
        await asyncio.sleep(10800)  # 3 hours
        # Re-check state since another task might have canceled it
        state = _get_search_state(message.from_user.id)
        if state["enabled"] and state["query"]:
            await _do_search_job(message, state["query"])

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