"""Embedded Telegram bot: replies to @mentions and replies-to-bot in group chats.

Runs as an asyncio task *inside* the FastAPI service (started/stopped in the lifespan),
talking to the in-process :class:`ModelManager` directly — no HTTP round-trip, sharing the
single model slot + lock with the OpenAI API. Disabled unless ``MLX_LAZYSERVE_TG_BOT_TOKEN``
is set, and the extra deps (``httpx`` + ``telegramify-markdown``, the ``telegram`` extra) are
imported lazily so the core install stays lean.

Behavior:
- **Groups + gated DMs.** In groups it triggers on @mentions or replies to it. Private chats
  are open but gated: a stranger's first DM asks an owner (``MLX_LAZYSERVE_TG_OWNER_IDS``) to
  approve via inline buttons; approved user ids are remembered in SQLite. Other bots and the
  bot's own messages are ignored.
- **Per-(chat, user) memory + prefs.** A short bounded conversation history per user, plus each
  user's own model (``/model``) and thinking toggle (``/think``) — all persisted to SQLite.
- **Interrupt + merge.** If a user sends a new message while the bot is still generating
  *that user's* reply, the in-flight generation is aborted and a fresh one runs over the
  **merged** messages — so the user gets one coherent answer, not two.
- **No streaming.** The whole reply is generated, then sent (auto-split, rich-formatted).

Rendering uses recent Bot API formatting features: telegramify-markdown -> message
**entities** (so there's no MarkdownV2-escaping pitfall), expandable blockquotes, message
reactions (👀 ack), the ``typing`` chat-action, reply threading, and disabled link previews.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, field

from .config import Settings
from .manager import ModelManager

logger = logging.getLogger("mlx_lazyserve.telegram")

API_BASE = "https://api.telegram.org"
LONG_POLL_TIMEOUT = 50  # seconds the getUpdates call parks server-side
TYPING_INTERVAL = 4.0  # re-send "typing" before Telegram clears it (~5 s)
MAX_MESSAGE_LEN = 4000  # < Telegram's 4096 hard cap (UTF-16 units), with headroom
ACK_REACTION = "👀"  # reaction set on the triggering message while we think
PROMPT_MARGIN = 256  # tokens held back from the context window (template/counting slack)


@dataclass
class Incoming:
    chat_id: int
    user_id: int
    message_id: int
    text: str


@dataclass
class Channel:
    """Per-(chat, user) serialized work queue with interrupt+merge state.

    All access happens on the single asyncio loop thread, so plain lists/fields are safe;
    the only cross-thread object is ``abort`` (set on the loop, read by the worker thread).
    """

    pending: list[Incoming] = field(default_factory=list)  # received, not yet answered
    history: list[dict] = field(default_factory=list)  # [{role, content}, ...] (cache of DB)
    abort: threading.Event | None = None  # set => a generation is in flight
    worker: asyncio.Task | None = None  # the loop draining `pending`
    loaded: bool = False  # whether history + prefs were hydrated from the store yet
    load_lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # serialize hydration
    model: str | None = None  # per-user model override (None = bot default)
    thinking: bool | None = None  # per-user thinking override (None = bot default)


class HistoryStore:
    """Tiny SQLite store for per-(chat, user) conversation history.

    All methods are synchronous and serialized by a lock; call them via ``asyncio.to_thread``
    from the event loop so a disk write never blocks generation or polling.
    """

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS messages("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL, "
            "user_id INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_user ON messages(chat_id, user_id, id)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS prefs("
            "chat_id INTEGER NOT NULL, user_id INTEGER NOT NULL, "
            "model TEXT, thinking INTEGER, PRIMARY KEY(chat_id, user_id))"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS dm_allowed(user_id INTEGER PRIMARY KEY)"
        )
        self._conn.commit()

    def load(self, chat_id: int, user_id: int, limit: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE chat_id=? AND user_id=? "
                "ORDER BY id DESC LIMIT ?",
                (chat_id, user_id, limit),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def append(self, chat_id: int, user_id: int, msgs: list[dict], cap: int) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO messages(chat_id, user_id, role, content) VALUES(?,?,?,?)",
                [(chat_id, user_id, m["role"], m["content"]) for m in msgs],
            )
            # Keep only the most recent `cap` rows for this (chat, user).
            self._conn.execute(
                "DELETE FROM messages WHERE chat_id=? AND user_id=? AND id NOT IN "
                "(SELECT id FROM messages WHERE chat_id=? AND user_id=? ORDER BY id DESC LIMIT ?)",
                (chat_id, user_id, chat_id, user_id, cap),
            )
            self._conn.commit()

    def clear(self, chat_id: int, user_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM messages WHERE chat_id=? AND user_id=?", (chat_id, user_id)
            )
            self._conn.commit()

    def get_prefs(self, chat_id: int, user_id: int) -> tuple[str | None, bool | None]:
        with self._lock:
            row = self._conn.execute(
                "SELECT model, thinking FROM prefs WHERE chat_id=? AND user_id=?",
                (chat_id, user_id),
            ).fetchone()
        if not row:
            return None, None
        model, thinking = row
        return model, (None if thinking is None else bool(thinking))

    def set_model(self, chat_id: int, user_id: int, model: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO prefs(chat_id, user_id, model) VALUES(?,?,?) "
                "ON CONFLICT(chat_id, user_id) DO UPDATE SET model=excluded.model",
                (chat_id, user_id, model),
            )
            self._conn.commit()

    def set_thinking(self, chat_id: int, user_id: int, thinking: bool) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO prefs(chat_id, user_id, thinking) VALUES(?,?,?) "
                "ON CONFLICT(chat_id, user_id) DO UPDATE SET thinking=excluded.thinking",
                (chat_id, user_id, int(thinking)),
            )
            self._conn.commit()

    def all_dm_allowed(self) -> list[int]:
        with self._lock:
            return [r[0] for r in self._conn.execute("SELECT user_id FROM dm_allowed").fetchall()]

    def allow_dm(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute("INSERT OR IGNORE INTO dm_allowed(user_id) VALUES(?)", (user_id,))
            self._conn.commit()


class TelegramBot:
    def __init__(self, settings: Settings, manager: ModelManager) -> None:
        self.settings = settings
        self.manager = manager
        self.token = settings.tg_bot_token
        self.default_model = settings.tg_model or settings.default_model  # per-user overridable
        self.bot_id = 0
        self.username = ""
        self.channels: dict[tuple[int, int], Channel] = {}
        self._client = None  # httpx.AsyncClient, set in run()
        self._closing = False  # set on shutdown: abort generations + stop workers
        self._bg: set[asyncio.Task] = set()  # strong refs so fire-and-forget tasks aren't GC'd
        self._store: HistoryStore | None = None  # SQLite history, opened in run()
        self._dm_allowed: set[int] = set()  # user ids approved for private chat (mirrors DB)
        self._dm_pending: set[int] = set()  # DM auth requests awaiting an owner decision
        self._dm_denied: set[int] = set()  # DM users an owner rejected (silently ignored)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    # ------------------------------------------------------------------ lifecycle

    async def run(self, stop: asyncio.Event) -> None:
        if not self.token:
            return
        if not self.default_model or self.default_model not in self.settings.models:
            logger.error(
                "Telegram bot: model %r is not configured; bot disabled", self.default_model
            )
            return
        try:
            import httpx
        except ImportError:
            logger.error(
                "Telegram bot enabled but deps missing — install the 'telegram' extra "
                "(uv sync --extra telegram)"
            )
            return

        # httpx logs one INFO line per request — noisy, and it prints the bot token in the URL.
        # Quiet it to WARNING so the service log stays useful (and tokenless) for debugging.
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self._setup_markdown()
        if self.settings.tg_db_path:
            try:
                self._store = HistoryStore(str(self.settings.tg_db_path))
                self._dm_allowed = set(self._store.all_dm_allowed())
                logger.info(
                    "Telegram history persisted to %s (%d DM-approved users)",
                    self.settings.tg_db_path, len(self._dm_allowed),
                )
            except Exception as exc:
                logger.warning("could not open history DB (%s); using in-memory history", exc)
                self._store = None
        timeout = httpx.Timeout(LONG_POLL_TIMEOUT + 15.0, connect=10.0)
        async with httpx.AsyncClient(base_url=API_BASE, timeout=timeout) as client:
            self._client = client
            try:
                me = await self._api("getMe")
            except Exception as exc:
                logger.error("Telegram getMe failed (%s); bot disabled", exc)
                return
            self.bot_id = me["id"]
            self.username = me.get("username") or ""
            logger.info(
                "Telegram bot @%s (id=%s) online; default model=%s",
                self.username, self.bot_id, self.default_model,
            )
            # Drop any backlog so a restart doesn't replay stale @mentions.
            await self._api_quiet("deleteWebhook", drop_pending_updates=True)
            await self._api_quiet(
                "setMyCommands",
                commands=[
                    {"command": "model", "description": "选择模型 / choose model"},
                    {"command": "think", "description": "开关思考 / toggle reasoning"},
                    {"command": "reset", "description": "清空上下文 / clear context"},
                ],
            )
            try:
                await self._poll_loop(stop)
            finally:
                # On stop/cancel, abort in-flight generations (sync, safe under cancellation)
                # so they release the model lock promptly and manager.shutdown() doesn't block.
                self._abort_all()
        logger.info("Telegram bot stopped")

    def _abort_all(self) -> None:
        self._closing = True
        for chan in self.channels.values():
            if chan.abort is not None:
                chan.abort.set()

    def _setup_markdown(self) -> None:
        """Make long quotes expandable and drop telegramify's emoji heading prefixes."""
        try:
            from telegramify_markdown.config import get_runtime_config

            cfg = get_runtime_config()
            cfg.cite_expandable = True
            for lvl in range(1, 7):
                if hasattr(cfg.markdown_symbol, f"heading_level_{lvl}"):
                    setattr(cfg.markdown_symbol, f"heading_level_{lvl}", "")
        except Exception as exc:
            logger.warning("telegramify-markdown config unavailable: %s", exc)

    async def _poll_loop(self, stop: asyncio.Event) -> None:
        offset: int | None = None
        backoff = 1.0
        while not stop.is_set():
            try:
                updates = await self._api(
                    "getUpdates",
                    offset=offset,
                    timeout=LONG_POLL_TIMEOUT,
                    allowed_updates=["message", "callback_query", "my_chat_member"],
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("getUpdates failed (%s); retrying in %.0fs", exc, backoff)
                await self._sleep_or_stop(stop, backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            backoff = 1.0
            for update in updates or []:
                offset = update["update_id"] + 1
                try:
                    if "message" in update:
                        self._handle_update(update)
                    elif "callback_query" in update:
                        self._spawn(self._handle_callback(update["callback_query"]))
                    elif "my_chat_member" in update:
                        self._spawn(self._handle_my_chat_member(update["my_chat_member"]))
                except Exception:
                    logger.exception("error handling update %s", update.get("update_id"))

    @staticmethod
    async def _sleep_or_stop(stop: asyncio.Event, secs: float) -> None:
        try:
            await asyncio.wait_for(stop.wait(), timeout=secs)
        except asyncio.TimeoutError:
            pass

    # ------------------------------------------------------------------ dispatch

    def _handle_update(self, update: dict) -> None:
        message = update.get("message")
        if not message:
            return
        chat = message.get("chat") or {}
        ctype = chat.get("type")
        frm = message.get("from") or {}
        if frm.get("is_bot") or frm.get("id") is None:
            return
        if ctype == "private":
            self._handle_private(message, chat.get("id"), frm)
            return
        if ctype not in ("group", "supergroup"):
            return
        chat_id = chat.get("id")

        text = (message.get("text") or message.get("caption") or "").strip()
        cmd, arg = self._parse_command(text)
        if cmd is not None:
            uid, mid = frm["id"], message.get("message_id")
            if cmd == "reset":
                self._reset(chat_id, uid)
                self._spawn(self._send_plain(chat_id, "🧹 已清空我们的对话上下文。", mid))
            elif cmd == "model":
                self._spawn(self._cmd_model(chat_id, uid, mid, arg))
            elif cmd in ("think", "thinking"):
                self._spawn(self._cmd_think(chat_id, uid, mid, arg))
            return

        req = self._extract_request(message, chat_id, frm["id"], text)
        if req is not None:
            self.submit(req)

    def _handle_private(self, message: dict, chat_id: int, frm: dict) -> None:
        """Private chat: gated by owner approval; approved users (and owners) chat normally."""
        user_id = frm["id"]
        mid = message.get("message_id")
        text = (message.get("text") or message.get("caption") or "").strip()
        if not self._dm_authorized(user_id):
            if user_id not in self._dm_denied:
                self._spawn(self._request_dm_auth(user_id, chat_id, frm))
            return
        cmd, arg = self._parse_command(text)
        if cmd is not None:
            if cmd == "reset":
                self._reset(chat_id, user_id)
                self._spawn(self._send_plain(chat_id, "🧹 已清空我们的对话上下文。", mid))
            elif cmd == "model":
                self._spawn(self._cmd_model(chat_id, user_id, mid, arg))
            elif cmd in ("think", "thinking"):
                self._spawn(self._cmd_think(chat_id, user_id, mid, arg))
            elif cmd in ("start", "help"):
                self._spawn(self._send_plain(
                    chat_id,
                    "你好,直接发消息就行。命令:/model 选模型 · /think 开关思考 · /reset 清空上下文",
                    mid,
                ))
            return
        if text:
            self.submit(Incoming(chat_id, user_id, mid, text))

    def _dm_authorized(self, user_id: int) -> bool:
        if not self.settings.tg_owner_ids:
            return True  # no owner configured -> private chat is open to everyone
        return user_id in self.settings.tg_owner_ids or user_id in self._dm_allowed

    async def _request_dm_auth(self, user_id: int, chat_id: int, frm: dict) -> None:
        """Tell the stranger we're asking, and DM each owner an approve/deny prompt."""
        if user_id in self._dm_pending or not self.settings.tg_owner_ids:
            return  # already outstanding, or nobody to ask
        self._dm_pending.add(user_id)
        await self._send_plain(
            chat_id, "👋 你好。我需要管理员授权后才能回复——已发送授权请求,通过后即可对话。"
        )
        name = ((frm.get("first_name") or "") + " " + (frm.get("last_name") or "")).strip()
        uname = frm.get("username")
        who = f"{name} (@{uname})" if uname else (name or "(无用户名)")
        kb = {"inline_keyboard": [[
            {"text": "✅ 允许", "callback_data": f"auth:ok:{user_id}"},
            {"text": "❌ 拒绝", "callback_data": f"auth:no:{user_id}"},
        ]]}
        notice = f"🔔 私聊授权请求\n用户:{who}\nID: {user_id}\n是否允许 ta 私聊机器人?"
        sent = False
        for owner in self.settings.tg_owner_ids:
            if await self._api_quiet(
                "sendMessage", chat_id=owner, text=notice, reply_markup=kb,
                link_preview_options={"is_disabled": True},
            ) is not None:
                sent = True
        if not sent:
            # owner unreachable (hasn't started the bot?) — drop pending so a later retry works
            self._dm_pending.discard(user_id)
            logger.warning("could not deliver DM-auth request for %s (owner not reachable)", user_id)

    async def _handle_my_chat_member(self, upd: dict) -> None:
        """Owner gate: leave any group we're added to by a non-owner (if an allowlist is set)."""
        chat = upd.get("chat") or {}
        if chat.get("type") not in ("group", "supergroup"):
            return  # ignore private start/block transitions and channels
        old = (upd.get("old_chat_member") or {}).get("status")
        new = (upd.get("new_chat_member") or {}).get("status")
        added = old in ("left", "kicked") and new in ("member", "administrator", "restricted")
        if not added:
            return  # promotion/demotion/leaving — not a fresh add
        adder_id = (upd.get("from") or {}).get("id")
        if self.settings.tg_owner_ids and adder_id not in self.settings.tg_owner_ids:
            chat_id = chat.get("id")
            logger.info("unauthorized add by user %s to chat %s; leaving", adder_id, chat_id)
            await self._api_quiet(
                "sendMessage",
                chat_id=chat_id,
                text="抱歉，我只接受授权用户的邀请，正在退出本群。",
                link_preview_options={"is_disabled": True},
            )
            await self._api_quiet("leaveChat", chat_id=chat_id)

    def _extract_request(self, message: dict, chat_id, user_id, text: str) -> Incoming | None:
        if not text:
            return None
        entities = message.get("entities") or message.get("caption_entities") or []
        triggered = False

        reply_from = (message.get("reply_to_message") or {}).get("from") or {}
        if reply_from.get("id") == self.bot_id:
            triggered = True
        if not triggered and self.username:
            # A 'mention' entity + the @handle appearing in text (substring check avoids the
            # UTF-16-vs-codepoint offset mismatch that slicing by entity offset would hit).
            tag = f"@{self.username}".lower()
            if any(e.get("type") == "mention" for e in entities) and tag in text.lower():
                triggered = True
        if not triggered:
            for e in entities:
                if e.get("type") == "text_mention" and (e.get("user") or {}).get("id") == self.bot_id:
                    triggered = True
                    break
        if not triggered:
            return None

        clean = self._strip_mention(text) or text
        return Incoming(chat_id=chat_id, user_id=user_id, message_id=message["message_id"], text=clean)

    def _strip_mention(self, text: str) -> str:
        if not self.username:
            return text.strip()
        return re.sub(rf"@{re.escape(self.username)}\b", "", text, flags=re.IGNORECASE).strip()

    def _parse_command(self, text: str) -> tuple[str | None, str]:
        """If text is a slash command addressed to us (or unqualified), return (name, arg)."""
        if not text.startswith("/"):
            return None, ""
        head, _, arg = text.partition(" ")
        base, _, at = head[1:].partition("@")
        if at and at.lower() != self.username.lower():
            return None, ""  # addressed to a different bot
        return base.lower(), arg.strip()

    def _get_channel(self, chat_id: int, user_id: int) -> Channel:
        key = (chat_id, user_id)
        chan = self.channels.get(key)
        if chan is None:
            chan = self.channels[key] = Channel()
        return chan

    def _reset(self, chat_id, user_id) -> None:
        chan = self._get_channel(chat_id, user_id)
        chan.history.clear()
        chan.pending.clear()
        chan.loaded = True  # authoritative empty cache; don't re-hydrate the DB we just cleared
        if self._store is not None:
            try:
                self._store.clear(chat_id, user_id)  # tiny delete; fine to run inline
            except Exception as exc:
                logger.warning("history clear failed: %s", exc)

    # ------------------------------------------------------------------ commands / prefs

    async def _cmd_model(self, chat_id, user_id, reply_to, arg: str) -> None:
        """/model — pick the model this user chats with (inline keyboard, or /model <name>)."""
        chan = self._get_channel(chat_id, user_id)
        await self._ensure_loaded(chan, chat_id, user_id)
        if arg:
            if arg in self.settings.models:
                chan.model = arg
                await self._save_pref(chat_id, user_id, model=arg)
                await self._send_plain(chat_id, f"✅ 你的模型已切到 {arg}", reply_to)
            else:
                opts = "、".join(self.settings.models)
                await self._send_plain(chat_id, f"未知模型：{arg}\n可选：{opts}", reply_to)
            return
        current = chan.model or self.default_model
        await self._api_quiet(
            "sendMessage",
            chat_id=chat_id,
            text=f"你当前的模型：{current}\n点下面切换（每人独立）：",
            reply_parameters={"message_id": reply_to, "allow_sending_without_reply": True},
            reply_markup=self._model_keyboard(current),
        )

    async def _cmd_think(self, chat_id, user_id, reply_to, arg: str) -> None:
        """/think — toggle this user's reasoning (inline keyboard, or /think on|off)."""
        chan = self._get_channel(chat_id, user_id)
        await self._ensure_loaded(chan, chat_id, user_id)
        a = arg.lower()
        if a in ("on", "1", "true", "开", "开启"):
            chan.thinking = True
            await self._save_pref(chat_id, user_id, thinking=True)
            await self._send_plain(chat_id, "✅ 思考已开启", reply_to)
            return
        if a in ("off", "0", "false", "关", "关闭"):
            chan.thinking = False
            await self._save_pref(chat_id, user_id, thinking=False)
            await self._send_plain(chat_id, "✅ 思考已关闭", reply_to)
            return
        current = chan.thinking if chan.thinking is not None else self.settings.tg_enable_thinking
        await self._api_quiet(
            "sendMessage",
            chat_id=chat_id,
            text=f"思考模式当前：{'开' if current else '关'}（每人独立）",
            reply_parameters={"message_id": reply_to, "allow_sending_without_reply": True},
            reply_markup=self._think_keyboard(current),
        )

    async def _handle_callback(self, cb: dict) -> None:
        """Inline-keyboard tap: prefs go to the tapper; auth taps to owners only."""
        cb_id = cb.get("id")
        data = cb.get("data") or ""
        msg = cb.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        user_id = (cb.get("from") or {}).get("id")
        if chat_id is None or user_id is None:
            await self._api_quiet("answerCallbackQuery", callback_query_id=cb_id)
            return
        if data.startswith("auth:"):
            await self._handle_auth_callback(cb_id, data, user_id, msg)
            return
        chan = self._get_channel(chat_id, user_id)
        await self._ensure_loaded(chan, chat_id, user_id)
        if data.startswith("m:") and data[2:] in self.settings.models:
            chan.model = data[2:]
            await self._save_pref(chat_id, user_id, model=chan.model)
            toast = f"✅ 模型已切到 {chan.model}"
        elif data in ("t:1", "t:0"):
            chan.thinking = data == "t:1"
            await self._save_pref(chat_id, user_id, thinking=chan.thinking)
            toast = "✅ 思考已开启" if chan.thinking else "✅ 思考已关闭"
        else:
            toast = None
        await self._api_quiet("answerCallbackQuery", callback_query_id=cb_id, text=toast)

    def _model_keyboard(self, current: str) -> dict:
        rows = [
            [{"text": ("✅ " if name == current else "") + name, "callback_data": f"m:{name}"}]
            for name in self.settings.models
        ]
        return {"inline_keyboard": rows}

    def _think_keyboard(self, current: bool) -> dict:
        return {
            "inline_keyboard": [[
                {"text": ("✅ " if current else "") + "开启思考", "callback_data": "t:1"},
                {"text": ("✅ " if not current else "") + "关闭思考", "callback_data": "t:0"},
            ]]
        }

    async def _save_pref(
        self, chat_id, user_id, *, model: str | None = None, thinking: bool | None = None
    ) -> None:
        if self._store is None:
            return
        try:
            if model is not None:
                await asyncio.to_thread(self._store.set_model, chat_id, user_id, model)
            if thinking is not None:
                await asyncio.to_thread(self._store.set_thinking, chat_id, user_id, thinking)
        except Exception as exc:
            logger.warning("save pref failed for (%s,%s): %s", chat_id, user_id, exc)

    async def _handle_auth_callback(self, cb_id, data: str, actor_id: int, msg: dict) -> None:
        """An owner approving/denying a stranger's private-chat request."""
        if actor_id not in self.settings.tg_owner_ids:
            await self._api_quiet("answerCallbackQuery", callback_query_id=cb_id, text="无权操作")
            return
        try:
            _, action, target = data.split(":", 2)
            target_id = int(target)
        except (ValueError, TypeError):
            await self._api_quiet("answerCallbackQuery", callback_query_id=cb_id)
            return
        self._dm_pending.discard(target_id)
        if action == "ok":
            self._dm_allowed.add(target_id)
            self._dm_denied.discard(target_id)
            await self._save_dm_allowed(target_id)
            await self._api_quiet("answerCallbackQuery", callback_query_id=cb_id, text=f"✅ 已允许 {target_id}")
            await self._api_quiet(
                "sendMessage", chat_id=target_id,
                text="✅ 管理员已通过,现在可以直接发消息和我聊天了。",
                link_preview_options={"is_disabled": True},
            )
            result = f"✅ 已允许 {target_id} 私聊"
        else:
            self._dm_denied.add(target_id)
            await self._api_quiet("answerCallbackQuery", callback_query_id=cb_id, text=f"已拒绝 {target_id}")
            result = f"❌ 已拒绝 {target_id}"
        if msg.get("message_id") and (msg.get("chat") or {}).get("id") is not None:
            await self._api_quiet(
                "editMessageText", chat_id=msg["chat"]["id"], message_id=msg["message_id"], text=result
            )

    async def _save_dm_allowed(self, user_id: int) -> None:
        if self._store is None:
            return
        try:
            await asyncio.to_thread(self._store.allow_dm, user_id)
        except Exception as exc:
            logger.warning("persist dm-allow failed for %s: %s", user_id, exc)

    # ------------------------------------------------------------------ per-user queue

    def submit(self, msg: Incoming) -> None:
        key = (msg.chat_id, msg.user_id)
        chan = self._get_channel(msg.chat_id, msg.user_id)
        chan.pending.append(msg)
        if chan.abort is not None:
            chan.abort.set()  # interrupt the in-flight generation -> it re-batches + merges
        if chan.worker is None or chan.worker.done():
            chan.worker = asyncio.create_task(self._worker(key))

    async def _worker(self, key: tuple[int, int]) -> None:
        chan = self.channels[key]
        chat_id, user_id = key
        reacted: int | None = None
        while True:
            # Drain synchronously (no await) so submit() can't interleave and orphan a
            # message between the empty-check and the worker exiting (single-thread asyncio).
            batch, chan.pending = chan.pending, []
            if not batch or self._closing:
                chan.worker = None
                if reacted is not None and not self._closing:
                    await self._react(chat_id, reacted, None)
                return

            # Each Telegram message stays its own native user turn (no string concatenation);
            # a merged batch is simply several consecutive user messages in the request.
            batch_msgs = [{"role": "user", "content": m.text} for m in batch if m.text]
            reply_to = batch[-1].message_id
            if not batch_msgs:
                continue

            await self._ensure_loaded(chan, chat_id, user_id)
            model = chan.model or self.default_model
            thinking = chan.thinking if chan.thinking is not None else self.settings.tg_enable_thinking

            # Arm the interrupt BEFORE any await, so a message arriving during the
            # acknowledgement round-trips still aborts this generation (no lost merge).
            # The window where chan.abort is None is now pure synchronous code.
            abort = threading.Event()
            chan.abort = abort

            if reacted is not None and reacted != reply_to:
                await self._react(chat_id, reacted, None)
            await self._react(chat_id, reply_to, ACK_REACTION)
            reacted = reply_to

            messages = self._build_messages(chan.history, batch_msgs)
            typing_stop = asyncio.Event()
            typing = asyncio.create_task(self._typing_loop(chat_id, typing_stop))
            try:
                text, reasoning, interrupted, error = await self._generate(
                    messages, abort, model, thinking
                )
            finally:
                typing_stop.set()
                await asyncio.gather(typing, return_exceptions=True)
                chan.abort = None

            if interrupted:
                if self._closing:
                    chan.worker = None
                    return
                # A newer message arrived mid-generation: put this batch back in FRONT so the
                # next drain merges old+new into a single regenerated answer.
                chan.pending[0:0] = batch
                continue

            await self._react(chat_id, reply_to, None)
            reacted = None
            if error:
                await self._send_plain(chat_id, f"⚠️ 生成失败：{error}", reply_to)
                continue
            if not text.strip():
                await self._send_plain(chat_id, "（模型返回了空回复）", reply_to)
                continue

            turn = batch_msgs + [{"role": "assistant", "content": text}]
            chan.history.extend(turn)
            self._trim_history(chan)
            await self._persist(chat_id, user_id, turn)
            await self._send_reply(chat_id, text, reasoning, reply_to)

    def _build_messages(self, history: list[dict], new_msgs: list[dict]) -> list[dict]:
        messages: list[dict] = []
        if self.settings.tg_system_prompt:
            messages.append({"role": "system", "content": self.settings.tg_system_prompt})
        messages.extend(history)
        messages.extend(new_msgs)
        return messages

    def _trim_history(self, chan: Channel) -> None:
        cap = max(0, self.settings.tg_history_turns) * 2
        if len(chan.history) > cap:
            del chan.history[: len(chan.history) - cap]

    async def _ensure_loaded(self, chan: Channel, chat_id: int, user_id: int) -> None:
        """Hydrate a channel's history + prefs from SQLite once, on first use after (re)start.

        Guarded by a per-channel lock so a command handler and the worker can't both load
        concurrently (which could skip with stale-empty data or clobber an appended turn).
        """
        if self._store is None or chan.loaded:
            return
        async with chan.load_lock:
            if chan.loaded:
                return
            try:
                cap = max(0, self.settings.tg_history_turns) * 2
                chan.history = await asyncio.to_thread(self._store.load, chat_id, user_id, cap)
                model, thinking = await asyncio.to_thread(self._store.get_prefs, chat_id, user_id)
                if model and model in self.settings.models:
                    chan.model = model
                if thinking is not None:
                    chan.thinking = thinking
                if chan.history:
                    logger.info(
                        "restored %d history msgs for (%s,%s)", len(chan.history), chat_id, user_id
                    )
            except Exception as exc:
                logger.warning("history load failed for (%s,%s): %s", chat_id, user_id, exc)
            finally:
                chan.loaded = True

    async def _persist(self, chat_id: int, user_id: int, turn: list[dict]) -> None:
        if self._store is None:
            return
        try:
            cap = max(0, self.settings.tg_history_turns) * 2
            await asyncio.to_thread(self._store.append, chat_id, user_id, turn, cap)
        except Exception as exc:
            logger.warning("history persist failed for (%s,%s): %s", chat_id, user_id, exc)

    # ------------------------------------------------------------------ generation

    def _prompt_budget(self, model: str) -> int:
        """Max prompt tokens = model context window − reserved output − a small margin."""
        spec = self.settings.models.get(model)
        context = spec.context if spec else 8192
        reserve = min(self.settings.tg_max_tokens, max(256, context // 2))
        return max(1024, context - reserve - PROMPT_MARGIN)

    async def _generate(self, messages: list[dict], abort: threading.Event, model: str, enable_thinking: bool):
        """Run the blocking model stream in a thread, bridging events over a queue.

        Mirrors the server's `_stream_completion`. ``abort`` is NOT passed to the manager;
        the producer checks it itself and reports ``interrupted`` at the moment it breaks —
        so a message arriving just after a *natural* finish never discards a completed reply.
        Returns ``(content, reasoning, interrupted, error)``.
        """
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        params = {
            "max_tokens": self.settings.tg_max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "enable_thinking": enable_thinking,
            # Bot KV is quantized harder than the API (default 4-bit vs 8): chats are frequent
            # and short, so the smaller cache trims memory/bandwidth at a tiny quality cost.
            "kv_bits": self.settings.tg_kv_bits,
            # Trim oldest history so prompt + reserved output fit the model's context window.
            "max_prompt_tokens": self._prompt_budget(model),
        }

        def produce() -> None:
            interrupted = False
            try:
                gen = self.manager.generate_stream(model, messages, **params)
                for event in gen:
                    if abort.is_set():
                        interrupted = True
                        gen.close()  # its finally closes the mlx stream + releases the lock
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", interrupted))

        producer = asyncio.create_task(asyncio.to_thread(produce))
        content: list[str] = []
        reasoning: list[str] = []
        error: str | None = None
        interrupted = False
        try:
            while True:
                kind, value = await queue.get()
                if kind == "done":
                    interrupted = bool(value)
                    break
                if kind == "error":
                    error = value
                    continue
                if value.get("content"):
                    content.append(value["content"])
                elif value.get("reasoning"):
                    reasoning.append(value["reasoning"])
                # tool_calls / usage are not surfaced in the chat bot
        finally:
            abort.set()
            await asyncio.gather(producer, return_exceptions=True)
        return "".join(content), "".join(reasoning), interrupted, error

    # ------------------------------------------------------------------ sending

    def _compose_markdown(self, text: str, reasoning: str) -> str:
        text = (text or "").strip()
        # Show reasoning whenever the model produced it (i.e. the user has thinking on);
        # telegramify collapses long quotes into an expandable blockquote.
        if reasoning and reasoning.strip():
            quoted = "\n".join("> " + ln for ln in reasoning.strip().splitlines())
            return f"{quoted}\n\n{text}" if text else quoted
        return text

    async def _send_reply(self, chat_id, text: str, reasoning: str, reply_to: int) -> None:
        md = self._compose_markdown(text, reasoning)
        rp = {"message_id": reply_to, "allow_sending_without_reply": True}
        try:
            from telegramify_markdown import telegramify
            from telegramify_markdown.content import ContentType

            items = await telegramify(
                md,
                max_message_length=MAX_MESSAGE_LEN,
                render_mermaid=False,  # no Playwright dep; mermaid stays a code block
                min_file_lines=40,  # keep short/medium code inline; only big dumps -> files
            )
        except Exception as exc:
            logger.warning("telegramify failed (%s); sending plain text", exc)
            await self._send_plain(chat_id, text, reply_to)
            return
        if not items:
            await self._send_plain(chat_id, text, reply_to)
            return
        for i, item in enumerate(items):
            await self._send_item(chat_id, item, ContentType, rp if i == 0 else None)

    async def _send_item(self, chat_id, item, content_type, reply_parameters) -> None:
        ct = item.content_type
        if ct == content_type.TEXT:
            entities = [e.to_dict() for e in (item.entities or [])]
            try:
                await self._api(
                    "sendMessage",
                    chat_id=chat_id,
                    text=item.text,
                    entities=entities or None,
                    reply_parameters=reply_parameters,
                    link_preview_options={"is_disabled": True},
                )
            except Exception as exc:
                logger.warning("sendMessage(entities) failed (%s); retrying as plain text", exc)
                await self._api_quiet(
                    "sendMessage",
                    chat_id=chat_id,
                    text=item.text,
                    reply_parameters=reply_parameters,
                    link_preview_options={"is_disabled": True},
                )
        elif ct == content_type.PHOTO:
            await self._send_media("sendPhoto", "photo", chat_id, item, reply_parameters)
        elif ct == content_type.FILE:
            await self._send_media("sendDocument", "document", chat_id, item, reply_parameters)
        else:  # RICH or any future type: degrade to plain text rather than dropping it
            txt = getattr(item, "text", None)
            if txt:
                await self._api_quiet(
                    "sendMessage",
                    chat_id=chat_id,
                    text=txt,
                    reply_parameters=reply_parameters,
                    link_preview_options={"is_disabled": True},
                )

    async def _send_media(self, method, field_name, chat_id, item, reply_parameters) -> None:
        file_data = getattr(item, "file_data", None)
        if file_data is None:
            return
        data = {"chat_id": str(chat_id)}
        caption = getattr(item, "caption_text", None)
        if caption:
            data["caption"] = caption
        cap_entities = getattr(item, "caption_entities", None)
        if cap_entities:
            data["caption_entities"] = json.dumps([e.to_dict() for e in cap_entities])
        if reply_parameters:
            data["reply_parameters"] = json.dumps(reply_parameters)
        files = {field_name: (getattr(item, "file_name", "file"), file_data)}
        try:
            resp = await self._client.post(f"/bot{self.token}/{method}", data=data, files=files)
            payload = resp.json()
            if not payload.get("ok"):
                raise RuntimeError(payload.get("description"))
        except Exception as exc:
            logger.warning("%s failed (%s)", method, exc)

    async def _send_plain(self, chat_id, text: str, reply_to: int | None = None) -> None:
        for i, chunk in enumerate(self._split_plain(text, MAX_MESSAGE_LEN)):
            rp = (
                {"message_id": reply_to, "allow_sending_without_reply": True}
                if reply_to and i == 0
                else None
            )
            await self._api_quiet(
                "sendMessage",
                chat_id=chat_id,
                text=chunk,
                reply_parameters=rp,
                link_preview_options={"is_disabled": True},
            )

    @staticmethod
    def _split_plain(text: str, limit: int) -> list[str]:
        text = text or ""
        if len(text) <= limit:
            return [text or " "]
        parts: list[str] = []
        buf = ""
        for line in text.splitlines(keepends=True):
            while len(line) > limit:  # a single over-long line: hard-split it
                if buf:
                    parts.append(buf)
                    buf = ""
                parts.append(line[:limit])
                line = line[limit:]
            if len(buf) + len(line) > limit:
                parts.append(buf)
                buf = line
            else:
                buf += line
        if buf:
            parts.append(buf)
        return parts

    # ------------------------------------------------------------------ reactions / typing

    async def _react(self, chat_id, message_id, emoji: str | None) -> None:
        reaction = [{"type": "emoji", "emoji": emoji}] if emoji else []
        await self._api_quiet(
            "setMessageReaction", chat_id=chat_id, message_id=message_id, reaction=reaction
        )

    async def _typing_loop(self, chat_id, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self._api_quiet("sendChatAction", chat_id=chat_id, action="typing")
            await self._sleep_or_stop(stop, TYPING_INTERVAL)

    # ------------------------------------------------------------------ HTTP

    async def _api(self, method: str, **params):
        payload = {k: v for k, v in params.items() if v is not None}
        resp = await self._client.post(f"/bot{self.token}/{method}", json=payload)
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"{method}: HTTP {resp.status_code}")
        if not data.get("ok"):
            raise RuntimeError(f"{method}: {data.get('description')} (code {data.get('error_code')})")
        return data.get("result")

    async def _api_quiet(self, method: str, **params):
        try:
            return await self._api(method, **params)
        except Exception as exc:
            logger.debug("%s failed (ignored): %s", method, exc)
            return None


async def run_bot(settings: Settings, manager: ModelManager, stop: asyncio.Event) -> None:
    """Entry point used by the server lifespan; returns immediately if the bot is disabled."""
    await TelegramBot(settings, manager).run(stop)
