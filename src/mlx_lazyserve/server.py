"""OpenAI-compatible HTTP API (chat completions) backed by the model manager."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import load_settings
from .manager import ModelManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

settings = load_settings()
manager = ModelManager(settings)


def _set_wired_limit(mb: int) -> None:
    """Best-effort: set the Metal GPU wired-memory limit via a NOPASSWD sudo rule.

    Needs a scoped /etc/sudoers.d rule (see README). Without it this logs a
    warning and the service keeps running on the default ~75% cap.
    """
    log = logging.getLogger("mlx_lazyserve")
    try:
        subprocess.run(
            ["sudo", "-n", "/usr/sbin/sysctl", f"iogpu.wired_limit_mb={mb}"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        log.info("set iogpu.wired_limit_mb=%d", mb)
    except Exception as exc:
        log.warning("could not set iogpu.wired_limit_mb=%d (%s)", mb, exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if settings.wired_limit_mb > 0:
        _set_wired_limit(settings.wired_limit_mb)
    bot_stop: asyncio.Event | None = None
    bot_task: asyncio.Task | None = None
    if settings.tg_bot_token:
        from .telegram import run_bot

        bot_stop = asyncio.Event()
        bot_task = asyncio.create_task(run_bot(settings, manager, bot_stop))
    try:
        yield
    finally:
        if bot_task is not None:
            bot_stop.set()
            bot_task.cancel()
            try:
                await bot_task
            except (asyncio.CancelledError, Exception):
                pass
        manager.shutdown()
        if settings.wired_limit_mb > 0:
            _set_wired_limit(0)  # restore the default cap on graceful shutdown


app = FastAPI(title="mlx-lazyserve", version="0.1.0", lifespan=lifespan)


def _require_auth(request: Request) -> None:
    if not settings.api_keys:
        return
    token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if token not in settings.api_keys:
        raise HTTPException(status_code=401, detail="invalid api key")


def _resolve_model(body: dict) -> str:
    name = body.get("model") or settings.default_model
    if name is None or name not in settings.models:
        raise HTTPException(status_code=404, detail=f"unknown model: {name!r}")
    return name


def _params(body: dict) -> dict:
    """OpenAI sampling/control fields → engine stream kwargs (optional ones only when sent)."""
    p: dict = {
        "max_tokens": int(
            body.get("max_tokens")
            or body.get("max_completion_tokens")
            or settings.default_max_tokens
        ),
        "temperature": float(body.get("temperature", 0.7)),
        "top_p": float(body.get("top_p", 0.95)),
        "kv_bits": int(body.get("kv_bits", settings.default_kv_bits) or 0),
        # Anti-repetition defaults (Ollama-style). Clients may override both — send
        # repetition_penalty 1.0 to disable the penalty, loop_guard false to opt out.
        "repetition_penalty": float(
            body.get("repetition_penalty", settings.default_repetition_penalty)
        ),
        "repetition_context_size": settings.repetition_context_size,
        "min_p": float(body.get("min_p", settings.default_min_p)),
        "loop_guard": bool(body.get("loop_guard", settings.loop_guard)),
    }
    if body.get("top_k") is not None:
        p["top_k"] = int(body["top_k"])
    if body.get("seed") is not None:
        p["seed"] = int(body["seed"])
    if body.get("presence_penalty") is not None:
        p["presence_penalty"] = float(body["presence_penalty"])
    if body.get("frequency_penalty") is not None:
        p["frequency_penalty"] = float(body["frequency_penalty"])
    if isinstance(body.get("logit_bias"), dict) and body["logit_bias"]:
        p["logit_bias"] = {int(k): float(v) for k, v in body["logit_bias"].items()}
    stop = body.get("stop")
    if stop:
        p["stop"] = [stop] if isinstance(stop, str) else [s for s in stop if s]
    return p


def _resolve_tools(body: dict):
    """OpenAI tools/tool_choice -> the tools list fed to the chat template (or None)."""
    if body.get("tool_choice") == "none":
        return None
    return body.get("tools") or None


def _enable_thinking(body: dict) -> bool:
    """Per-request thinking switch: body.enable_thinking or chat_template_kwargs, else default."""
    if "enable_thinking" in body:
        return bool(body["enable_thinking"])
    cck = body.get("chat_template_kwargs")
    if isinstance(cck, dict) and "enable_thinking" in cck:
        return bool(cck["enable_thinking"])
    return settings.default_enable_thinking


def _split_multimodal(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """Flatten OpenAI message content into text + a list of image references.

    Image parts (``{"type": "image_url", "image_url": {"url": ...}}``) are pulled
    out so the vision engine can consume them separately; text parts are joined.
    Plain string content passes through unchanged.
    """
    images: list[str] = []
    clean: list[dict] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    texts.append(str(part))
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    texts.append(part.get("text", ""))
                elif ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url")
                    if url:
                        images.append(url)
                elif ptype == "image":
                    url = part.get("image") or part.get("url")
                    if url:
                        images.append(url)
            clean.append({**message, "content": "\n".join(t for t in texts if t)})
        else:
            clean.append(message)
    return clean, images


def _chunk(cid: str, created: int, model: str, delta: dict, finish=None) -> str:
    obj = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _maintenance_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": "3600"},
        content={
            "error": {
                "message": "The service is paused for scheduled maintenance and is not "
                "accepting requests right now. Please try again later.",
                "type": "service_unavailable",
                "code": "maintenance",
            }
        },
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "loaded": manager.current_name(),
        "maintenance": manager.is_paused(),
    }


@app.get("/v1/models")
def list_models(request: Request) -> dict:
    _require_auth(request)
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": "mlx-lazyserve"}
            for name in settings.models
        ],
    }


@app.get("/admin/maintenance")
def get_maintenance(request: Request) -> dict:
    _require_auth(request)
    return {"maintenance": manager.is_paused(), "loaded": manager.current_name()}


@app.post("/admin/maintenance")
async def set_maintenance(request: Request) -> dict:
    """Toggle maintenance mode. Body: {"enabled": true|false}.

    Enabling frees the loaded model from memory and makes inference endpoints
    return 503; the state survives restarts via the pause-marker file.
    """
    _require_auth(request)
    body = await request.json()
    if bool(body.get("enabled")):
        manager.pause()
    else:
        manager.resume()
    return {"maintenance": manager.is_paused(), "loaded": manager.current_name()}


async def _watch_disconnect(request: Request, abort: threading.Event) -> None:
    """Set `abort` as soon as the client disconnects, so generation can stop early."""
    try:
        while not abort.is_set():
            if await request.is_disconnected():
                abort.set()
                return
            await asyncio.sleep(0.3)
    except asyncio.CancelledError:
        pass


async def _stream_completion(cid, created, model, messages, params, abort, include_usage=False):
    """SSE generator. Runs the blocking model stream in a worker thread and bridges it
    through a queue. On client disconnect Starlette closes this generator, the `finally`
    sets `abort`, and the worker stops + releases the model lock — so an aborted caller
    never leaves a generation hogging the GPU and the single model slot.

    With ``include_usage`` (OpenAI ``stream_options.include_usage``) a final usage-only
    chunk (``choices: []`` + ``usage``) is emitted before ``[DONE]``."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def produce():
        try:
            for event in manager.generate_stream(model, messages, abort=abort, **params):
                loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
        except Exception as exc:  # surface load/inference errors into the stream
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    producer = asyncio.create_task(asyncio.to_thread(produce))
    finish = "stop"
    usage = None
    try:
        yield _chunk(cid, created, model, {"role": "assistant"})
        while True:
            kind, value = await queue.get()
            if kind == "done":
                break
            if kind == "error":
                logging.error("generation failed: %s", value)
                yield _chunk(cid, created, model, {"content": f"\n[error: {value}]"}, finish="stop")
                yield "data: [DONE]\n\n"
                return
            if "usage" in value:
                usage = value["usage"]
            elif "reasoning" in value:
                yield _chunk(cid, created, model, {"reasoning_content": value["reasoning"]})
            elif "tool_calls" in value:
                finish = "tool_calls"
                yield _chunk(cid, created, model, {"tool_calls": value["tool_calls"]})
            elif value.get("content"):
                yield _chunk(cid, created, model, {"content": value["content"]})
        yield _chunk(cid, created, model, {}, finish=finish)
        if include_usage and usage:
            usage_chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage,
            }
            yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        abort.set()
        try:
            await producer
        except Exception:
            pass


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _require_auth(request)
    if manager.is_paused():
        return _maintenance_response()
    body = await request.json()
    model = _resolve_model(body)
    messages, images = _split_multimodal(body.get("messages") or [])
    tools = _resolve_tools(body)
    rf = body.get("response_format")
    structured = isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema")
    if structured and tools:
        raise HTTPException(
            status_code=400, detail="response_format and tools cannot be used together"
        )
    params = _params(body)
    params["images"] = images
    params["tools"] = tools
    # structured output emits JSON directly — incompatible with free-form <think> reasoning
    params["enable_thinking"] = False if structured else _enable_thinking(body)
    params["response_format"] = rf if structured else None
    if structured:
        params["loop_guard"] = False  # never truncate grammar-constrained JSON mid-stream
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    abort = threading.Event()

    if body.get("stream"):
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return StreamingResponse(
            _stream_completion(cid, created, model, messages, params, abort, include_usage),
            media_type="text/event-stream",
        )

    # Non-streaming: generate in a worker thread; a watcher aborts it if the client leaves.
    watcher = asyncio.create_task(_watch_disconnect(request, abort))
    try:
        events = await asyncio.to_thread(
            lambda: list(manager.generate_stream(model, messages, abort=abort, **params))
        )
    finally:
        abort.set()
        watcher.cancel()

    content = "".join(e["content"] for e in events if "content" in e)
    reasoning = "".join(e["reasoning"] for e in events if "reasoning" in e)
    tool_calls = [tc for e in events if "tool_calls" in e for tc in e["tool_calls"]]
    usage = next(
        (e["usage"] for e in reversed(events) if "usage" in e),
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    message: dict = {"role": "assistant", "content": content or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": usage,
    }
