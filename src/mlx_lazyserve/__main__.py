"""Entry point: ``mlx-lazyserve`` (console script) or ``python -m mlx_lazyserve``."""

from __future__ import annotations


def main() -> None:
    import uvicorn

    from .config import load_settings

    settings = load_settings()
    uvicorn.run(
        "mlx_lazyserve.server:app",
        host=settings.host,
        port=settings.port,
        workers=1,  # single process: keep exactly one model in unified memory
        log_level="info",
    )


if __name__ == "__main__":
    main()
