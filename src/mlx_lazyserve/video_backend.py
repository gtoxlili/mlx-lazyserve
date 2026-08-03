"""Lifecycle + HTTP client for the mlx-serve (Zig) video backend.

Why a subprocess and not an in-process binding: a single H3 job is minutes to
hours of compute, so IPC cost is noise, while the two things that actually
matter both favour a separate process.

  1. Killing the process is the only way to be *sure* the ~11 GB of DiT weights
     go back to the OS. On a 24 GB box that reclamation is the whole game — the
     text models need 5-19 GB of the same pool.
  2. An MLX over-commit inside the backend hard-crashes it. As a subprocess that
     is a failed job; in-process it would take the chat API down with it, and
     this server is exposed through a tunnel.

The backend loads its weights lazily on the first generation request, so spawn
is cheap (~1 s) and the real cost lands on the first job.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# The backend logs one line per denoise step; the job worker tails stderr to turn
# those into progress. Kept here so the parser and the producer stay together.
_STEP_LINE = "[minimax-h3] step "


class VideoBackendError(RuntimeError):
    """Backend could not be started, or returned an error for a request."""


class VideoBackend:
    """Owns one mlx-serve child process bound to loopback.

    Not thread-safe by itself: callers go through ModelManager, which serializes
    start/stop, and only the single job worker calls ``generate``.
    """

    def __init__(
        self,
        binary: str,
        model_dir: str,
        port: int,
        *,
        load_timeout: float,
        log_path: str | None = None,
    ) -> None:
        self._binary = binary
        self._model_dir = model_dir
        self._port = port
        self._load_timeout = load_timeout
        self._log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._log_fh = None
        # Last step line seen, as (step, total). Read by the job worker for progress.
        self._progress: tuple[int, int] | None = None
        self._progress_lock = threading.Lock()
        self._tail: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def progress(self) -> tuple[int, int] | None:
        with self._progress_lock:
            return self._progress

    def start(self) -> None:
        if self.is_running():
            return
        cmd = [
            self._binary,
            "--model", self._model_dir,
            "--serve",
            "--host", "127.0.0.1",
            "--port", str(self._port),
            # The pre-flight counts only *free* pages, but macOS reclaims file
            # cache as MLX allocates, so on a 24 GB box it refuses loads that do
            # fit. We gate on our own single-slot rule instead: nothing else of
            # size is resident when the backend runs.
            "--skip-mem-preflight",
        ]
        logger.info("starting video backend on :%d", self._port)
        self._log_fh = open(self._log_path, "ab", buffering=0) if self._log_path else subprocess.DEVNULL
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Own process group, so stop() can signal the whole tree and never
            # the parent server.
            start_new_session=True,
        )
        self._progress = None
        self._tail = threading.Thread(target=self._tail_output, name="video-tail", daemon=True)
        self._tail.start()
        self._await_health()

    def _tail_output(self) -> None:
        """Mirror backend output to the log file and scrape step progress from it."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for raw in proc.stdout:
            if self._log_fh not in (None, subprocess.DEVNULL):
                try:
                    self._log_fh.write(raw)
                except OSError:
                    pass
            line = raw.decode("utf-8", "replace")
            idx = line.find(_STEP_LINE)
            if idx == -1:
                continue
            # "[minimax-h3] step 12/50 sigma 0.83 (1379 ms)"
            try:
                cur, _, total = line[idx + len(_STEP_LINE):].partition("/")
                total = total.split()[0]
                with self._progress_lock:
                    self._progress = (int(cur), int(total))
            except (ValueError, IndexError):
                continue

    def _await_health(self) -> None:
        deadline = time.monotonic() + 60.0  # spawn only; weights load on first request
        last: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise VideoBackendError(
                    f"video backend exited during startup (code {self._proc.returncode}); "
                    f"see {self._log_path}"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=3) as r:
                    if r.status == 200:
                        logger.info("video backend healthy on :%d", self._port)
                        return
            except (urllib.error.URLError, OSError) as exc:
                last = exc
            time.sleep(0.5)
        self.stop()
        raise VideoBackendError(f"video backend did not become healthy in 60s ({last})")

    def generate(self, payload: dict, *, timeout: float | None) -> dict:
        """POST one generation. Blocks for the whole job — callers run this off-loop.

        ``timeout=None`` means block indefinitely, which is deliberate: at 768p a
        legitimate job runs for hours. Interrupting one is done by stopping the
        backend (see ``stop``), which drops the socket and raises here.
        """
        if not self.is_running():
            raise VideoBackendError("video backend is not running")
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v1/video/generations",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise VideoBackendError(f"backend HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            # A hard MLX over-commit kills the child mid-request; surface that as
            # the real cause instead of a bare connection-reset.
            if self._proc is not None and self._proc.poll() is not None:
                raise VideoBackendError(
                    f"video backend died during generation (code {self._proc.returncode}); "
                    "likely an MLX over-commit — lower resolution or frame count"
                ) from exc
            raise VideoBackendError(f"backend unreachable: {exc}") from exc

    def stop(self) -> None:
        """Terminate the child and wait for it, so the memory is actually back."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            logger.info("stopping video backend (pid %d)", proc.pid)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                logger.warning("video backend ignored SIGTERM; killing")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait(timeout=10)
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass
        if self._log_fh not in (None, subprocess.DEVNULL):
            try:
                self._log_fh.close()
            except OSError:
                pass
        self._log_fh = None
        with self._progress_lock:
            self._progress = None
