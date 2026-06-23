"""OpenAI-compatible HTTP API (chat completions) backed by the model manager."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .config import load_settings
from .manager import ModelManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

settings = load_settings()
manager = ModelManager(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    manager.shutdown()


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
    return {
        "max_tokens": int(body.get("max_tokens") or settings.default_max_tokens),
        "temperature": float(body.get("temperature", 0.7)),
        "top_p": float(body.get("top_p", 0.95)),
    }


def _chunk(cid: str, created: int, model: str, delta: dict, finish=None) -> str:
    obj = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "loaded": manager.current_name()}


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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _require_auth(request)
    body = await request.json()
    model = _resolve_model(body)
    messages = body.get("messages") or []
    params = _params(body)
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if body.get("stream"):

        def sse():
            yield _chunk(cid, created, model, {"role": "assistant"})
            try:
                for text in manager.generate_stream(model, messages, **params):
                    if text:
                        yield _chunk(cid, created, model, {"content": text})
            except Exception as exc:  # surface load/inference errors into the stream
                logging.exception("generation failed")
                yield _chunk(cid, created, model, {"content": f"\n[error: {exc}]"}, finish="stop")
                yield "data: [DONE]\n\n"
                return
            yield _chunk(cid, created, model, {}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    def collect() -> str:
        return "".join(manager.generate_stream(model, messages, **params))

    text = await run_in_threadpool(collect)
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
