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
    repo: str  # Hugging Face MLX repo id
    engine: str = "auto"  # "auto" | "mlx_lm" | "mlx_vlm"
    default: bool = False


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    idle_timeout: float  # seconds of inactivity before unloading (0 = never)
    default_max_tokens: int
    wired_limit_mb: int  # if > 0, raise iogpu.wired_limit_mb on start, reset to 0 on stop
    api_keys: tuple[str, ...]  # bearer tokens; empty tuple = no auth (rely on Tailscale)
    models: dict[str, ModelSpec]
    default_model: str | None
    pause_file: Path  # marker file; if present the service starts in maintenance mode


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
            ms = ModelSpec(
                name=name,
                repo=spec["repo"],
                engine=spec.get("engine", "auto"),
                default=bool(spec.get("default", False)),
            )
            models[name] = ms
            if ms.default and default_model is None:
                default_model = name
    if default_model is None and models:
        default_model = next(iter(models))
    return models, default_model


def load_settings() -> Settings:
    models, default_model = _load_models()
    api_keys = tuple(
        k.strip()
        for k in os.environ.get("MLX_LAZYSERVE_API_KEYS", "").split(",")
        if k.strip()
    )
    return Settings(
        host=os.environ.get("MLX_LAZYSERVE_HOST", "127.0.0.1"),
        port=int(os.environ.get("MLX_LAZYSERVE_PORT", "41434")),
        idle_timeout=float(os.environ.get("MLX_LAZYSERVE_IDLE_TIMEOUT", "600")),
        default_max_tokens=int(os.environ.get("MLX_LAZYSERVE_MAX_TOKENS", "8192")),
        wired_limit_mb=int(os.environ.get("MLX_LAZYSERVE_WIRED_LIMIT_MB", "0")),
        api_keys=api_keys,
        models=models,
        default_model=default_model,
        pause_file=Path(
            os.environ.get("MLX_LAZYSERVE_PAUSE_FILE", str(PROJECT_ROOT / ".maintenance"))
        ).expanduser(),
    )
