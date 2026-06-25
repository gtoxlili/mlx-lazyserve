"""Embedded Telegram bot: replies to @mentions and replies-to-bot in group chats.

Runs as an asyncio task *inside* the FastAPI service (started/stopped in the lifespan),
talking to the in-process :class:`ModelManager` directly — no HTTP round-trip, sharing the
single model slot + lock with the OpenAI API. Disabled unless ``MLX_LAZYSERVE_TG_BOT_TOKEN``
is set, and the extra deps (``httpx`` + ``telegramify-markdown``, the ``telegram`` extra) are
imported lazily so the core install stays lean.

Behavior:
- **Groups only.** Triggers when a message @mentions the bot or replies to one of the bot's
  messages. DMs, other bots, and the bot's own messages are ignored.
- **Per-(chat, user) memory.** A short bounded conversation history per user.
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
    history: list[dict] = field(default_factory=list)  # [{role, content}, ...]
    abort: threading.Event | None = None  # set => a generation is in flight
    worker: asyncio.Task | None = None  # the loop draining `pending`


class TelegramBot:
    def __init__(self, settings: Settings, manager: ModelManager) -> None:
        self.settings = settings
        self.manager = manager
        self.token = settings.tg_bot_token
        self.model = settings.tg_model or settings.default_model
        self.bot_id = 0
        self.username = ""
        self.channels: dict[tuple[int, int], Channel] = {}
        self._client = None  # httpx.AsyncClient, set in run()
        self._closing = False  # set on shutdown: abort generations + stop workers
        self._bg: set[asyncio.Task] = set()  # strong refs so fire-and-forget tasks aren't GC'd

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    # ------------------------------------------------------------------ lifecycle

    async def run(self, stop: asyncio.Event) -> None:
        if not self.token:
            return
        if not self.model or self.model not in self.settings.models:
            logger.error("Telegram bot: model %r is not configured; bot disabled", self.model)
            return
        try:
            import httpx
        except ImportError:
            logger.error(
                "Telegram bot enabled but deps missing — install the 'telegram' extra "
                "(uv sync --extra telegram)"
            )
            return

        self._setup_markdown()
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
                "Telegram bot @%s (id=%s) online; model=%s", self.username, self.bot_id, self.model
            )
            # Drop any backlog so a restart doesn't replay stale @mentions.
            await self._api_quiet("deleteWebhook", drop_pending_updates=True)
            await self._api_quiet(
                "setMyCommands",
                commands=[{"command": "reset", "description": "清空与你的对话上下文 / clear our context"}],
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
                    allowed_updates=["message"],
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
                    self._handle_update(update)
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
        if chat.get("type") not in ("group", "supergroup"):
            return  # groups only
        chat_id = chat.get("id")
        if self.settings.tg_allowed_chats and chat_id not in self.settings.tg_allowed_chats:
            return
        frm = message.get("from") or {}
        if frm.get("is_bot") or frm.get("id") is None:
            return

        text = (message.get("text") or message.get("caption") or "").strip()
        if self._is_reset_command(text):
            self._reset(chat_id, frm["id"])
            self._spawn(
                self._send_plain(chat_id, "🧹 已清空我们的对话上下文。", message.get("message_id"))
            )
            return

        req = self._extract_request(message, chat_id, frm["id"], text)
        if req is not None:
            self.submit(req)

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

    def _is_reset_command(self, text: str) -> bool:
        if not text:
            return False
        first = text.split(maxsplit=1)[0].lower()
        base, _, at = first.partition("@")
        return base == "/reset" and (at == "" or at == self.username.lower())

    def _reset(self, chat_id, user_id) -> None:
        chan = self.channels.get((chat_id, user_id))
        if chan is not None:
            chan.history.clear()
            chan.pending.clear()

    # ------------------------------------------------------------------ per-user queue

    def submit(self, msg: Incoming) -> None:
        key = (msg.chat_id, msg.user_id)
        chan = self.channels.get(key)
        if chan is None:
            chan = self.channels[key] = Channel()
        chan.pending.append(msg)
        if chan.abort is not None:
            chan.abort.set()  # interrupt the in-flight generation -> it re-batches + merges
        if chan.worker is None or chan.worker.done():
            chan.worker = asyncio.create_task(self._worker(key))

    async def _worker(self, key: tuple[int, int]) -> None:
        chan = self.channels[key]
        chat_id, _ = key
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

            user_text = "\n\n".join(m.text for m in batch if m.text).strip()
            reply_to = batch[-1].message_id
            if not user_text:
                continue

            # Arm the interrupt BEFORE any await, so a message arriving during the
            # acknowledgement round-trips still aborts this generation (no lost merge).
            # The window where chan.abort is None is now pure synchronous code.
            abort = threading.Event()
            chan.abort = abort

            if reacted is not None and reacted != reply_to:
                await self._react(chat_id, reacted, None)
            await self._react(chat_id, reply_to, ACK_REACTION)
            reacted = reply_to

            messages = self._build_messages(chan.history, user_text)
            typing_stop = asyncio.Event()
            typing = asyncio.create_task(self._typing_loop(chat_id, typing_stop))
            try:
                text, reasoning, interrupted, error = await self._generate(messages, abort)
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

            chan.history.append({"role": "user", "content": user_text})
            chan.history.append({"role": "assistant", "content": text})
            self._trim_history(chan)
            await self._send_reply(chat_id, text, reasoning, reply_to)

    def _build_messages(self, history: list[dict], user_text: str) -> list[dict]:
        messages: list[dict] = []
        if self.settings.tg_system_prompt:
            messages.append({"role": "system", "content": self.settings.tg_system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def _trim_history(self, chan: Channel) -> None:
        cap = max(0, self.settings.tg_history_turns) * 2
        if len(chan.history) > cap:
            del chan.history[: len(chan.history) - cap]

    # ------------------------------------------------------------------ generation

    async def _generate(self, messages: list[dict], abort: threading.Event):
        """Run the blocking model stream in a thread, bridging events over a queue.

        Mirrors the server's `_stream_completion`. ``abort`` is NOT passed to the manager;
        the producer checks it itself and reports ``interrupted`` at the moment it breaks —
        so a message arriving just after a *natural* finish never discards a completed reply.
        Returns ``(content, reasoning, interrupted, error)``.
        """
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        model = self.model
        params = {
            "max_tokens": self.settings.tg_max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "enable_thinking": self.settings.tg_enable_thinking,
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
        if self.settings.tg_enable_thinking and reasoning and reasoning.strip():
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
