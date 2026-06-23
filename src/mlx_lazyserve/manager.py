"""Single-slot model manager: lazy load, serialized generation, idle unload."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator

from .config import Settings
from .engine import load_model

logger = logging.getLogger(__name__)


class ModelManager:
    """Holds at most one model in unified memory at a time.

    - lazy: a model is loaded on the first request that needs it
    - single-slot: requesting a different model evicts the current one (24 GB)
    - idle unload: a background reaper frees the model after ``idle_timeout``
    - serialized: generation holds a lock, so one GPU stream runs at a time
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Plain Lock (not RLock): the streaming generator holds this across yields that
        # Starlette drives on DIFFERENT threadpool threads, so it must be releasable from
        # a thread other than the one that acquired it. RLock forbids that and would leave
        # the lock orphaned (held forever) → every later request then deadlocks on it.
        self._lock = threading.Lock()
        self._model = None
        self._model_name: str | None = None
        self._last_used = time.monotonic()
        self._paused = settings.pause_file.exists()
        self._stop = threading.Event()
        if settings.idle_timeout > 0:
            threading.Thread(
                target=self._idle_reaper, name="idle-reaper", daemon=True
            ).start()

    def current_name(self) -> str | None:
        return self._model_name  # atomic read, lock-free so /health never blocks on a generation

    def is_paused(self) -> bool:
        return self._paused  # atomic read, lock-free so the async event loop never blocks here

    def pause(self) -> None:
        """Enter maintenance mode: free the loaded model and refuse new work."""
        with self._lock:
            self._paused = True
            self._unload_locked()
        try:
            self._settings.pause_file.touch()
        except OSError as exc:
            logger.warning("could not write pause marker: %s", exc)

    def resume(self) -> None:
        """Leave maintenance mode and serve again."""
        with self._lock:
            self._paused = False
        try:
            self._settings.pause_file.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("could not remove pause marker: %s", exc)

    def _idle_reaper(self) -> None:
        timeout = self._settings.idle_timeout
        interval = min(30.0, max(1.0, timeout / 2))
        while not self._stop.wait(interval):
            with self._lock:
                idle = time.monotonic() - self._last_used
                if self._model is not None and idle > timeout:
                    logger.info(
                        "idle %.0fs > %.0fs; unloading %s", idle, timeout, self._model_name
                    )
                    self._unload_locked()

    def _unload_locked(self) -> None:
        if self._model is not None:
            self._model.close()
            self._model = None
            self._model_name = None

    def _ensure_locked(self, name: str) -> None:
        if self._paused:
            raise RuntimeError("service is paused (maintenance mode)")
        if self._model_name == name:
            return
        spec = self._settings.models.get(name)
        if spec is None:
            raise KeyError(name)
        self._unload_locked()  # evict the previous model first
        t0 = time.monotonic()
        logger.info("loading model %s (%s)...", name, spec.repo)
        self._model = load_model(spec)
        self._model_name = name
        logger.info("loaded %s in %.1fs", name, time.monotonic() - t0)

    def generate_stream(self, name: str, messages: list[dict], **params) -> Iterator[str]:
        with self._lock:
            self._ensure_locked(name)
            model = self._model
            self._last_used = time.monotonic()
            try:
                for chunk in model.stream(messages, **params):
                    yield chunk
                    self._last_used = time.monotonic()
            finally:
                self._last_used = time.monotonic()

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._unload_locked()
