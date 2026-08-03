"""Single-slot model manager: lazy load, serialized generation, idle unload."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator

from .config import Settings
from .engine import load_model

logger = logging.getLogger(__name__)


class VideoBusy(RuntimeError):
    """A video job owns the GPU slot, so no text model can be loaded right now.

    Carries the running job's id and ETA so the HTTP layer can answer with a
    useful Retry-After instead of a bare 503.
    """

    def __init__(self, job_id: str | None, eta_seconds: float | None) -> None:
        self.job_id = job_id
        self.eta_seconds = eta_seconds
        eta = f", ~{eta_seconds:.0f}s remaining" if eta_seconds else ""
        super().__init__(
            f"a video job is using the GPU (job {job_id or 'unknown'}{eta}). "
            "Video and text models cannot be resident at the same time on 24 GB."
        )


class ModelManager:
    """Holds at most one model in unified memory at a time.

    - lazy: a model is loaded on the first request that needs it
    - single-slot: requesting a different model evicts the current one (24 GB)
    - idle unload: a background reaper frees the model after ``idle_timeout``
    - serialized: generation holds a lock, so one GPU stream runs at a time

    The slot is also what the video backend competes for. A text model and the
    MiniMax-H3 backend cannot coexist (5-19 GB plus ~11 GB against 24 GB total),
    so exactly one of them may hold the slot. The video side is an out-of-process
    child: taking the slot means evicting the text model and spawning it, and
    releasing means killing it, which is the only way the memory really returns.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Set while the video backend owns the slot. Guarded by _lock for writes,
        # read lock-free (a plain reference read is atomic) so /health never blocks.
        self._video = None
        self._video_model: str | None = None
        # Filled in by the server once the job store exists; used only to make the
        # VideoBusy error carry a real ETA.
        self.video_jobs = None
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

    # ── video slot ────────────────────────────────────────────────────────

    def video_active(self) -> bool:
        return self._video is not None  # atomic read; see __init__

    def _video_busy_error(self) -> VideoBusy:
        job_id, eta = None, None
        store = self.video_jobs
        if store is not None:
            job = store.active()
            if job is not None:
                job_id = job.id
                api = job.to_api().get("progress") or {}
                eta = api.get("eta_seconds")
        return VideoBusy(job_id, eta)

    def acquire_video_slot(self, model_name: str):
        """Evict the text model and start the video backend. Returns the backend.

        Called only by the single video worker. The lock is held just long enough
        to swap slot ownership — never across the generation itself, which runs
        for hours and would otherwise block /health and every chat request.
        """
        from .video_backend import VideoBackend, VideoBackendError

        with self._lock:
            if self._paused:
                raise RuntimeError("service is paused (maintenance mode)")
            if self._video is not None and self._video_model == model_name:
                return self._video
            if self._video is not None:  # different video model — restart the child
                self._video.stop()
                self._video = None
                self._video_model = None
            spec = self._settings.models.get(model_name)
            if spec is None or spec.kind != "video":
                raise KeyError(model_name)
            if not self._settings.video_binary:
                raise VideoBackendError(
                    "video backend is not configured (set MLX_LAZYSERVE_VIDEO_BINARY)"
                )
            self._unload_locked()  # the text model goes first; its RAM is the budget
            logger.info("video job taking the GPU slot (%s)", model_name)
            backend = VideoBackend(
                self._settings.video_binary,
                spec.path,
                self._settings.video_port,
                load_timeout=self._settings.video_load_timeout,
                log_path=str(self._settings.video_out_dir / "backend.log"),
            )
            backend.start()
            self._video = backend
            self._video_model = model_name
            self._last_used = time.monotonic()
            return backend

    def release_video_slot(self) -> None:
        with self._lock:
            if self._video is not None:
                self._video.stop()
                self._video = None
                self._video_model = None
                logger.info("video slot released")

    def _ensure_locked(self, name: str) -> None:
        if self._paused:
            raise RuntimeError("service is paused (maintenance mode)")
        if self._video is not None:
            # Default policy is exclusive: a video job keeps the GPU until it
            # finishes. Silently queueing a chat request behind a multi-hour job
            # is worse than refusing it with an ETA the caller can act on.
            raise self._video_busy_error()
        if self._model_name == name:
            return
        spec = self._settings.models.get(name)
        if spec is None:
            raise KeyError(name)
        if spec.kind != "text":
            raise KeyError(f"{name} is a {spec.kind} model; use /v1/videos")
        self._unload_locked()  # evict the previous model first
        t0 = time.monotonic()
        logger.info("loading model %s (%s)...", name, spec.repo)
        self._model = load_model(spec)
        self._model_name = name
        logger.info("loaded %s in %.1fs", name, time.monotonic() - t0)

    def generate_stream(
        self,
        name: str,
        messages: list[dict],
        *,
        abort: threading.Event | None = None,
        **params,
    ) -> Iterator[dict]:
        with self._lock:
            self._ensure_locked(name)
            model = self._model
            self._last_used = time.monotonic()
            stream = model.stream(messages, **params)
            try:
                for chunk in stream:
                    if abort is not None and abort.is_set():
                        break  # caller went away — stop generating, release the lock
                    yield chunk
                    self._last_used = time.monotonic()
            finally:
                # stop the underlying mlx generator promptly (frees the GPU on abort)
                closer = getattr(stream, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        pass
                self._last_used = time.monotonic()

    def shutdown(self) -> None:
        self._stop.set()
        self.release_video_slot()
        with self._lock:
            self._unload_locked()
