"""MiniMax H3-Context-IR client: expand a plain prompt into H3-Base's input format.

The full H3 system is three modules and only the middle one is open: Context-IR
and Regenerate-2K stay behind MiniMax's Open Platform API. We run H3-Base
locally and call Context-IR remotely, which is the split their own reproduction
scripts use.

This matters more than it sounds. H3-Base is trained on a heavily structured
prompt — shot-by-shot blocking with timecodes, a separate soundscape track and a
separate score track — and Context-IR is what turns one ordinary sentence into
that. Feeding H3-Base a bare sentence works, but it is not the input the weights
were tuned for.

Expansion is an async task: POST creates it, then poll until it leaves the queue.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_TERMINAL_FAIL = {"fail", "failed", "error"}


class ContextIRError(RuntimeError):
    pass


class ContextIRClient:
    def __init__(self, api_key: str, base_url: str, *, timeout: float = 180.0) -> None:
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self._key)

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(
            f"{self._base}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise ContextIRError(f"Context-IR HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ContextIRError(f"Context-IR unreachable: {exc}") from exc

    def expand(self, prompt: str, *, duration: int, ratio: str) -> str:
        """Return the expanded, H3-Base-shaped prompt. Raises on failure."""
        if not self.enabled:
            raise ContextIRError("no MiniMax API key configured")
        created = self._call(
            "POST",
            "/v2/h3_context_ir",
            {
                "model": "MiniMax-H3",
                "content": [{"type": "text", "text": prompt}],
                "duration": duration,
                "ratio": ratio,
            },
        )
        task_id = created.get("task_id")
        if not task_id:
            raise ContextIRError(f"no task_id in Context-IR response: {created}")

        deadline = time.monotonic() + self._timeout
        delay = 1.0
        while time.monotonic() < deadline:
            time.sleep(delay)
            delay = min(delay * 1.5, 8.0)  # back off; expansion takes tens of seconds
            res = self._call("GET", f"/v2/query/video_generation/{task_id}")
            status = str(res.get("status") or res.get("task", {}).get("status") or "").lower()
            expanded = (res.get("task") or {}).get("content", {}).get("prompt")
            if expanded:
                logger.info("Context-IR expanded prompt to %d chars", len(expanded))
                return expanded
            if status in _TERMINAL_FAIL:
                raise ContextIRError(f"Context-IR task {task_id} failed: {res}")
        raise ContextIRError(f"Context-IR task {task_id} timed out after {self._timeout:.0f}s")


def expand_or_passthrough(
    client: ContextIRClient | None, prompt: str, *, duration: int, ratio: str, mode: str
) -> tuple[str, str | None]:
    """Best-effort expansion. Returns (prompt_to_use, expanded_or_None).

    Degrading to the raw prompt is the right call here: expansion is a quality
    lever, and losing a queued multi-hour job because a remote API blipped would
    be a worse outcome than a slightly weaker prompt.
    """
    if mode == "raw" or client is None or not client.enabled:
        return prompt, None
    try:
        expanded = client.expand(prompt, duration=duration, ratio=ratio)
        return expanded, expanded
    except ContextIRError as exc:
        logger.warning("Context-IR expansion failed, using the raw prompt: %s", exc)
        return prompt, None
