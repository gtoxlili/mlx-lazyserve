"""Firecrawl web tools for the embedded Telegram bot: keyless web search + page/PDF scrape.

Firecrawl's free "keyless" tier answers plain REST calls with **no API key** (rate-limited per
source IP); set ``MLX_LAZYSERVE_FIRECRAWL_API_KEY`` to raise the limits. This module exposes two
OpenAI-style tool schemas (:data:`TOOL_SCHEMAS` — ``web_search`` / ``web_scrape``) that the bot
advertises to the model, plus an async :class:`FirecrawlClient` whose :meth:`~FirecrawlClient.
dispatch` runs one tool call and returns a plain string to feed back as the ``tool`` message.

``dispatch`` never raises: a timeout, a 429, or a malformed response degrades into a short note
the model can reason about ("rate-limited — answer from memory") instead of crashing the reply.
Results are capped to a character budget so a long article can't blow the model's context.

``httpx`` is imported lazily (it ships in the ``telegram`` extra), so importing this module
never pulls a hard dependency into the core install.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("mlx_lazyserve.firecrawl")

# OpenAI-style function schemas advertised to the model. Kept intentionally to two verbs —
# search (discover) and scrape (read one URL / PDF) — which the Firecrawl keyless tier serves
# without a key; crawl/map/extract/agent need a key and are out of scope for the chat bot.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for up-to-date information and get back a ranked list of "
                "results (title, URL, short snippet). Use this when the user asks about current "
                "events, recent facts, prices, versions, or anything you are unsure of, or to "
                "find pages worth reading in full with web_scrape."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "limit": {
                        "type": "integer",
                        "description": "How many results to return (1-10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_scrape",
            "description": (
                "Fetch a single web page or PDF by its URL and return the main content as clean "
                "Markdown. Use this to read a specific link in full — including one returned by "
                "web_search — when the snippet isn't enough."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The absolute http(s) URL to fetch.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


# --- content cleaning ---------------------------------------------------------------------
# Scraped markdown carries junk a text model can't use and shouldn't pay context for: inline
# images, base64 blobs, URL tracking params, trailing whitespace, and long blank runs. Strip
# them before feeding content back (~20% smaller on a typical page, and every image + tracking
# query string gone). Kept conservative — link text/URLs and code fences survive, so nothing the
# model might need to quote is lost.
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # ![alt](url) inline images
_DATA_IMG = re.compile(r"data:image/[^)\s]+")  # any base64 image blob left inline
_TRACK = re.compile(  # tracking query params inside URLs (utm_*, gclid, fbclid, …)
    r"(?i)([?&])(utm_[a-z_]+|gclid|fbclid|mc_[a-z]+|ref|ref_src|igshid|si)=[^&#)\s]*"
)
_DANGLING_Q = re.compile(r"[?&]+([)#\s])")  # a URL left ending in '?' or '&' after the strip
_TRAILING_WS = re.compile(r"[ \t]+$", re.M)
_BLANK_RUN = re.compile(r"\n{3,}")


def _clean_markdown(md: str) -> str:
    """Strip images/base64/tracking/whitespace noise from scraped markdown (see module note)."""
    if not md:
        return ""
    md = _IMG.sub("", md)
    md = _DATA_IMG.sub("", md)
    md = _TRACK.sub(r"\1", md)
    md = _DANGLING_Q.sub(r"\1", md)
    md = _TRAILING_WS.sub("", md)
    md = _BLANK_RUN.sub("\n\n", md)
    return md.strip()


def _oneline(s) -> str:
    """Collapse a search snippet's whitespace/newlines into a single clean line."""
    return re.sub(r"\s+", " ", (s or "")).strip()


class FirecrawlClient:
    """Thin async wrapper over Firecrawl's ``/v2/search`` and ``/v2/scrape`` REST endpoints."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.firecrawl.dev",
        api_key: str = "",
        result_chars: int = 6000,
        search_limit: int = 5,
        timeout: float = 45.0,
    ) -> None:
        import httpx

        self._base = base_url.rstrip("/")
        self._result_chars = max(500, int(result_chars))
        self._search_limit = max(1, min(10, int(search_limit)))
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"  # else keyless (rate-limited per IP)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0), headers=headers
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def dispatch(self, name: str, args: dict) -> str:
        """Run one tool call and return a model-facing string. Never raises."""
        import httpx

        try:
            if name == "web_search":
                return await self._search(str(args.get("query") or "").strip(), args.get("limit"))
            if name == "web_scrape":
                return await self._scrape(str(args.get("url") or "").strip())
            return f"(error: unknown tool '{name}')"
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code == 429:
                return (
                    "(the web tool is rate-limited right now — answer from your own knowledge "
                    "and tell the user the information may be out of date)"
                )
            logger.warning("firecrawl %s -> HTTP %s", name, code)
            return f"(web tool error: HTTP {code})"
        except Exception as exc:  # timeout, connection, bad JSON, …
            logger.warning("firecrawl %s failed: %s", name, exc)
            return f"(web tool error: {exc})"

    # ------------------------------------------------------------------ endpoints

    async def _search(self, query: str, limit) -> str:
        if not query:
            return "(error: empty search query)"
        n = self._search_limit
        if limit is not None:
            try:
                n = max(1, min(10, int(limit)))
            except (ValueError, TypeError):
                pass
        data = await self._post("/v2/search", {"query": query, "limit": n})
        web = ((data.get("data") or {}).get("web")) or []
        if not web:
            return f"(no web results for {query!r})"
        lines = []
        for i, w in enumerate(web, 1):
            title = _oneline(w.get("title"))
            url = (w.get("url") or "").strip()
            desc = _oneline(w.get("description"))
            lines.append(f"{i}. {title}\n   {url}\n   {desc}".rstrip())
        return self._cap("\n".join(lines))

    async def _scrape(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return "(error: web_scrape needs an absolute http(s) URL)"
        data = await self._post("/v2/scrape", {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,     # drop nav/header/footer/sidebars server-side
            "removeBase64Images": True,  # never ship inline base64 image blobs back
        })
        md = _clean_markdown((data.get("data") or {}).get("markdown") or "")
        if not md:
            return f"(no readable content at {url})"
        return self._cap(md)

    async def _post(self, path: str, payload: dict) -> dict:
        resp = await self._client.post(f"{self._base}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def _cap(self, text: str) -> str:
        if len(text) <= self._result_chars:
            return text
        return text[: self._result_chars].rstrip() + "\n…(truncated)"
