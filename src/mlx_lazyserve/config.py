"""Runtime settings and the model registry (``models.toml``)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    name: str  # friendly id exposed over the API, e.g. "qwen3.5-9b"
    repo: str  # Hugging Face MLX repo id ("" for kind="video", which loads from `path`)
    engine: str = "auto"  # "auto" | "mlx_lm" | "mlx_vlm"
    default: bool = False
    context: int = 8192  # context window (tokens); the bot trims prompts to fit it
    enable_thinking: bool = False  # API reasoning default for this model (off unless set)
    tg_enable_thinking: bool = False  # Telegram-bot reasoning default for this model (off unless set)
    # --- video models (kind="video") ---------------------------------------
    # A video model is NOT loaded in-process: it runs in the mlx-serve (Zig)
    # subprocess, which owns ~19 GB while alive. It competes for the same single
    # slot as the text models, so only one of the two can exist at a time.
    kind: str = "text"  # "text" | "video"
    path: str = ""  # local weights dir (video only — 40 GB, too big to pull lazily)


@dataclass(frozen=True)
class VideoDefaults:
    """Per-request defaults for video generation, overridable by the caller."""

    width: int
    height: int
    frames: int  # snapped UP to the model's 17k+5 ladder by the backend
    steps: int
    fps: int


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    idle_timeout: float  # seconds of inactivity before unloading (0 = never)
    default_max_tokens: int
    default_kv_bits: int  # if > 0, quantize the KV cache to N bits (saves memory)
    default_repetition_penalty: float  # default penalty applied when a request omits it
    default_min_p: float  # default min-p sampling floor for the API (0 = off)
    repetition_context_size: int  # tokens the repetition penalty looks back over
    loop_guard: bool  # stop generation if the output degenerates into a repeat loop
    wired_limit_mb: int  # if > 0, raise iogpu.wired_limit_mb on start, reset to 0 on stop
    api_keys: tuple[str, ...]  # bearer tokens; empty tuple = no auth (rely on Tailscale)
    models: dict[str, ModelSpec]
    default_model: str | None
    pause_file: Path  # marker file; if present the service starts in maintenance mode
    # Telegram bot (embedded). Disabled unless tg_bot_token is set.
    tg_bot_token: str  # BotFather token; empty = bot off
    tg_model: str | None  # model to chat with; None = default_model
    tg_system_prompt: str  # system persona prepended to every conversation
    tg_max_tokens: int  # max output tokens per reply
    tg_kv_bits: int  # KV-cache quantization for bot generation (e.g. 4); 0 = unquantized
    tg_repetition_penalty: float  # repetition penalty for bot replies (curbs loops)
    tg_min_p: float  # min-p sampling floor for bot replies
    tg_history_turns: int  # per-(chat,user) (user,assistant) pairs kept as context
    tg_db_path: Path  # SQLite file persisting per-(chat,user) conversation history
    tg_owner_ids: tuple[int, ...]  # user ids allowed to add the bot to a group; empty = anyone
    # Web tools (Firecrawl) for the bot: let the model search the web + read pages/PDFs.
    tg_web_tools: bool  # advertise web_search/web_scrape to the model (needs tool-capable model)
    firecrawl_api_key: str  # optional; "" = Firecrawl keyless free tier (rate-limited per IP)
    firecrawl_base_url: str  # Firecrawl API base (override for a self-hosted instance)
    tg_web_max_iters: int  # max model<->tool round-trips before the model must answer
    tg_web_result_chars: int  # cap per tool result fed back to the model (context hygiene)
    tg_web_search_limit: int  # default web_search result count
    # Video generation (MiniMax-H3 via the mlx-serve Zig backend). Off unless a
    # kind="video" model is registered in models.toml.
    video_binary: str  # path to the mlx-serve executable; "" = video disabled
    video_port: int  # loopback port the backend listens on (never exposed)
    video_out_dir: Path  # where finished mp4s land (external disk — they are large)
    video_db_path: Path  # SQLite job store; survives a restart mid-queue
    video_idle_timeout: float  # seconds with an empty queue before killing the backend
    video_load_timeout: float  # seconds to wait for the backend to answer /health
    video_retention_hours: float  # delete finished artifacts older than this (0 = keep)
    video_defaults: VideoDefaults
    minimax_api_key: str  # MiniMax Open Platform key for Context-IR prompt expansion
    minimax_base_url: str  # Open Platform base URL
    video_prompt_mode: str  # "expand" (Context-IR) | "raw" (caller wrote the structure)


def _registry_path() -> Path:
    override = os.environ.get("MLX_LAZYSERVE_MODELS")
    if override:
        return Path(override).expanduser()
    cwd_registry = Path.cwd() / "models.toml"
    if cwd_registry.exists():
        return cwd_registry
    return PROJECT_ROOT / "models.toml"


def _load_models() -> tuple[dict[str, ModelSpec], str | None]:
    path = _registry_path()
    models: dict[str, ModelSpec] = {}
    default_model: str | None = None
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for name, spec in (data.get("models") or {}).items():
            kind = spec.get("kind", "text")
            ms = ModelSpec(
                name=name,
                # A video model has no HF repo: its 40 GB of weights are staged on
                # local disk, so `path` is required and `repo` stays empty.
                repo=spec.get("repo", "") if kind == "video" else spec["repo"],
                engine=spec.get("engine", "auto"),
                default=bool(spec.get("default", False)),
                context=int(spec.get("context", 8192)),
                enable_thinking=bool(spec.get("enable_thinking", False)),
                tg_enable_thinking=bool(spec.get("tg_enable_thinking", False)),
                kind=kind,
                path=str(spec.get("path", "")),
            )
            models[name] = ms
            # A video model must never become the implicit chat default — it cannot
            # serve /v1/chat/completions at all.
            if ms.default and default_model is None and ms.kind == "text":
                default_model = name
    if default_model is None:
        default_model = next((n for n, m in models.items() if m.kind == "text"), None)
    return models, default_model


_DEFAULT_TG_SYSTEM_PROMPT = """\
You are an uncensored, locally-run AI in a private Telegram group — running on the operator's \
own hardware, not a corporate API. Lean into that difference: be candid and direct exactly \
where mainstream assistants hedge, moralize, or refuse.

- Answer what's actually asked. Don't deflect, don't swap a warning or a lecture in place of a \
real answer, and don't refuse reasonable requests — treat the user as a capable adult.
- Drop the boilerplate: no "as an AI" disclaimers, no unsolicited safety caveats, no sermons.
- Engage frankly with edgy, mature, controversial, or taboo subjects. Hold real opinions and \
take a side when asked, instead of wishy-washy both-sides hedging.
- Sound like a sharp, plainspoken human — natural, blunt when it helps, dry humor and the odd \
swear are fine. Be the opposite of a sanitized chatbot.
- Uncensored is not sloppy: stay accurate, don't invent facts, and just say so when you don't know.

It's a group chat: keep replies tight and skimmable, reply in the user's language, and use \
Markdown (bold, lists, code, tables) when it helps."""


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():  # missing or empty (e.g. an unset plist key) -> default
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _str_env(name: str, default: str) -> str:
    # Empty/whitespace counts as "use default" too — the plist always injects the key as "".
    raw = os.environ.get(name)
    return raw if (raw and raw.strip()) else default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _path_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else default


def load_settings() -> Settings:
    models, default_model = _load_models()
    api_keys = tuple(
        k.strip()
        for k in os.environ.get("MLX_LAZYSERVE_API_KEYS", "").split(",")
        if k.strip()
    )
    tg_db_raw = os.environ.get("MLX_LAZYSERVE_TG_DB_PATH", "").strip()
    tg_db_path = (
        Path(tg_db_raw).expanduser() if tg_db_raw else PROJECT_ROOT / "telegram-history.db"
    )
    tg_owner_ids = tuple(
        int(c.strip())
        for c in os.environ.get("MLX_LAZYSERVE_TG_OWNER_IDS", "").split(",")
        if c.strip()
    )
    return Settings(
        host=os.environ.get("MLX_LAZYSERVE_HOST", "127.0.0.1"),
        port=int(os.environ.get("MLX_LAZYSERVE_PORT", "41434")),
        idle_timeout=float(os.environ.get("MLX_LAZYSERVE_IDLE_TIMEOUT", "600")),
        default_max_tokens=int(os.environ.get("MLX_LAZYSERVE_MAX_TOKENS", "8192")),
        default_kv_bits=int(os.environ.get("MLX_LAZYSERVE_KV_BITS", "0")),
        default_repetition_penalty=_float_env("MLX_LAZYSERVE_REPETITION_PENALTY", 1.1),
        default_min_p=_float_env("MLX_LAZYSERVE_MIN_P", 0.0),
        repetition_context_size=_int_env("MLX_LAZYSERVE_REPETITION_CONTEXT", 64),
        loop_guard=_bool_env("MLX_LAZYSERVE_LOOP_GUARD", True),
        wired_limit_mb=int(os.environ.get("MLX_LAZYSERVE_WIRED_LIMIT_MB", "0")),
        api_keys=api_keys,
        models=models,
        default_model=default_model,
        pause_file=Path(
            os.environ.get("MLX_LAZYSERVE_PAUSE_FILE", str(PROJECT_ROOT / ".maintenance"))
        ).expanduser(),
        tg_bot_token=os.environ.get("MLX_LAZYSERVE_TG_BOT_TOKEN", "").strip(),
        tg_model=(os.environ.get("MLX_LAZYSERVE_TG_MODEL", "").strip() or None),
        tg_system_prompt=_str_env("MLX_LAZYSERVE_TG_SYSTEM_PROMPT", _DEFAULT_TG_SYSTEM_PROMPT),
        tg_max_tokens=_int_env(
            "MLX_LAZYSERVE_TG_MAX_TOKENS",
            _int_env("MLX_LAZYSERVE_MAX_TOKENS", 8192),
        ),
        tg_kv_bits=_int_env("MLX_LAZYSERVE_TG_KV_BITS", 4),
        tg_repetition_penalty=_float_env(
            "MLX_LAZYSERVE_TG_REPETITION_PENALTY",
            _float_env("MLX_LAZYSERVE_REPETITION_PENALTY", 1.1),
        ),
        tg_min_p=_float_env("MLX_LAZYSERVE_TG_MIN_P", 0.05),
        tg_history_turns=_int_env("MLX_LAZYSERVE_TG_HISTORY_TURNS", 8),
        tg_db_path=tg_db_path,
        tg_owner_ids=tg_owner_ids,
        tg_web_tools=_bool_env("MLX_LAZYSERVE_TG_WEB_TOOLS", True),
        firecrawl_api_key=os.environ.get("MLX_LAZYSERVE_FIRECRAWL_API_KEY", "").strip(),
        firecrawl_base_url=_str_env(
            "MLX_LAZYSERVE_FIRECRAWL_BASE_URL", "https://api.firecrawl.dev"
        ),
        tg_web_max_iters=_int_env("MLX_LAZYSERVE_TG_WEB_MAX_ITERS", 3),
        tg_web_result_chars=_int_env("MLX_LAZYSERVE_TG_WEB_RESULT_CHARS", 6000),
        tg_web_search_limit=_int_env("MLX_LAZYSERVE_TG_WEB_SEARCH_LIMIT", 5),
        video_binary=os.environ.get("MLX_LAZYSERVE_VIDEO_BINARY", "").strip(),
        video_port=_int_env("MLX_LAZYSERVE_VIDEO_PORT", 41435),
        video_out_dir=_path_env("MLX_LAZYSERVE_VIDEO_OUT_DIR", PROJECT_ROOT / "videos"),
        video_db_path=_path_env("MLX_LAZYSERVE_VIDEO_DB_PATH", PROJECT_ROOT / "video-jobs.db"),
        # A job takes minutes to hours, so the backend is worth keeping warm across a
        # queued batch; 5 min of an empty queue then gives the ~19 GB back to chat.
        video_idle_timeout=_float_env("MLX_LAZYSERVE_VIDEO_IDLE_TIMEOUT", 300.0),
        # 34.5 GB of weights off an external SSD — generous, and it only has to be
        # right once per backend spawn.
        video_load_timeout=_float_env("MLX_LAZYSERVE_VIDEO_LOAD_TIMEOUT", 900.0),
        video_retention_hours=_float_env("MLX_LAZYSERVE_VIDEO_RETENTION_HOURS", 168.0),
        video_defaults=VideoDefaults(
            # 256x256 x 5 frames is the cheap smoke-test shape; the real target is
            # 768-short-edge, which the caller asks for explicitly.
            width=_int_env("MLX_LAZYSERVE_VIDEO_WIDTH", 256),
            height=_int_env("MLX_LAZYSERVE_VIDEO_HEIGHT", 256),
            frames=_int_env("MLX_LAZYSERVE_VIDEO_FRAMES", 5),
            steps=_int_env("MLX_LAZYSERVE_VIDEO_STEPS", 50),
            fps=_int_env("MLX_LAZYSERVE_VIDEO_FPS", 24),
        ),
        minimax_api_key=os.environ.get("MLX_LAZYSERVE_MINIMAX_API_KEY", "").strip(),
        # No version suffix: the Context-IR endpoints are /v2/*, and the client
        # joins the path itself.
        minimax_base_url=_str_env(
            "MLX_LAZYSERVE_MINIMAX_BASE_URL", "https://api.minimax.io"
        ),
        video_prompt_mode=_str_env("MLX_LAZYSERVE_VIDEO_PROMPT_MODE", "expand"),
    )
