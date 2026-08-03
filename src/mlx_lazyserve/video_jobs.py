"""Job store + serial worker for video generation.

Video generation cannot be a blocking HTTP request. One job is minutes to hours,
and this server is reached through a tunnel where every proxy in the path has an
idle timeout far shorter than that. So the API is job-shaped — submit, poll,
fetch — which is also the shape MiniMax's own Open Platform API uses.

Jobs live in SQLite (same precedent as the Telegram history), so a restart mid-
queue loses nothing but the in-flight job, which is re-queued on the next boot.

Exactly one job runs at a time. That is not a simplification: the backend needs
~11 GB resident for the DiT and the box has 24 GB, so a second concurrent job
would over-commit and hard-crash both.
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .video_backend import VideoBackendError
from .video_mux import MuxError, mux

logger = logging.getLogger(__name__)

QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED = (
    "queued", "running", "completed", "failed", "cancelled",
)
TERMINAL = frozenset({COMPLETED, FAILED, CANCELLED})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id             TEXT PRIMARY KEY,
    status         TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt         TEXT NOT NULL,
    expanded_prompt TEXT,
    params         TEXT NOT NULL,
    artifact       TEXT,
    error          TEXT,
    progress_step  INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    created_at     REAL NOT NULL,
    started_at     REAL,
    finished_at    REAL
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at);
"""


@dataclass(frozen=True)
class Job:
    id: str
    status: str
    model: str
    prompt: str
    expanded_prompt: str | None
    params: dict
    artifact: str | None
    error: str | None
    progress_step: int
    progress_total: int
    created_at: float
    started_at: float | None
    finished_at: float | None

    def to_api(self) -> dict:
        out: dict = {
            "id": self.id,
            "object": "video.generation",
            "status": self.status,
            "model": self.model,
            "created_at": self.created_at,
            "params": self.params,
        }
        if self.status == RUNNING and self.progress_total:
            progress: dict = {"step": self.progress_step, "steps": self.progress_total}
            if self.progress_step > 0:
                # Extrapolate from steps actually observed. Deliberately excludes
                # the fixed VAE-decode tail, so it reads slightly optimistic near
                # the end rather than stalling at "1 second left".
                elapsed = time.time() - (self.started_at or time.time())
                per_step = elapsed / self.progress_step
                progress["eta_seconds"] = round(
                    per_step * (self.progress_total - self.progress_step), 1
                )
            else:
                # Still in the ~45 s load phase (text encode + DiT). Dividing the
                # elapsed time by a step count of zero would produce an ETA that
                # grows the longer you wait, which is worse than none.
                progress["phase"] = "loading"
            out["progress"] = progress
        if self.expanded_prompt:
            out["expanded_prompt"] = self.expanded_prompt
        if self.artifact:
            out["content_url"] = f"/v1/videos/{self.id}/content"
        if self.error:
            out["error"] = self.error
        return out


def _row_to_job(r: sqlite3.Row) -> Job:
    return Job(
        id=r["id"], status=r["status"], model=r["model"], prompt=r["prompt"],
        expanded_prompt=r["expanded_prompt"], params=json.loads(r["params"]),
        artifact=r["artifact"], error=r["error"],
        progress_step=r["progress_step"], progress_total=r["progress_total"],
        created_at=r["created_at"], started_at=r["started_at"],
        finished_at=r["finished_at"],
    )


class VideoJobStore:
    """SQLite-backed job queue with one worker thread."""

    def __init__(self, settings: Settings, manager) -> None:
        self._settings = settings
        self._manager = manager
        self._db_path = settings.video_db_path
        self._out_dir = settings.video_out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + one lock: the worker and the request threads
        # share this connection, and every write goes through _write().
        self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._cancelled: set[str] = set()
        self._cancel_lock = threading.Lock()
        self._requeue_orphans()
        self._worker = threading.Thread(target=self._run, name="video-worker", daemon=True)
        self._worker.start()

    # ── storage ───────────────────────────────────────────────────────────

    def _write(self, sql: str, args: tuple = ()) -> None:
        with self._db_lock:
            self._db.execute(sql, args)
            self._db.commit()

    def _query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._db_lock:
            return self._db.execute(sql, args).fetchall()

    def _requeue_orphans(self) -> None:
        """A job marked running at boot means we died mid-generation. Re-queue it."""
        rows = self._query("SELECT id FROM jobs WHERE status = ?", (RUNNING,))
        if rows:
            logger.info("re-queueing %d job(s) orphaned by a restart", len(rows))
            self._write(
                "UPDATE jobs SET status = ?, started_at = NULL WHERE status = ?",
                (QUEUED, RUNNING),
            )

    # ── public API ────────────────────────────────────────────────────────

    def submit(self, model: str, prompt: str, params: dict) -> Job:
        job_id = f"vid_{uuid.uuid4().hex[:20]}"
        now = time.time()
        self._write(
            "INSERT INTO jobs (id, status, model, prompt, params, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (job_id, QUEUED, model, prompt, json.dumps(params), now),
        )
        self._wake.set()
        logger.info("queued video job %s (%s)", job_id, model)
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> Job | None:
        rows = self._query("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return _row_to_job(rows[0]) if rows else None

    def set_expanded(self, job_id: str, expanded: str) -> None:
        """Record the Context-IR output alongside the caller's original prompt."""
        self._write(
            "UPDATE jobs SET expanded_prompt = ? WHERE id = ?", (expanded, job_id)
        )

    def list(self, limit: int = 50) -> list[Job]:
        rows = self._query(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [_row_to_job(r) for r in rows]

    def queue_depth(self) -> int:
        return len(self._query("SELECT id FROM jobs WHERE status = ?", (QUEUED,)))

    def active(self) -> Job | None:
        rows = self._query("SELECT * FROM jobs WHERE status = ? LIMIT 1", (RUNNING,))
        return _row_to_job(rows[0]) if rows else None

    def cancel(self, job_id: str) -> bool:
        """Cancel a job. Queued ones drop out; a running one has its backend killed.

        There is no polite way to interrupt a running generation: the worker is
        blocked in a socket read on a request that may have hours left, and the
        backend has no cancel endpoint. Killing the child is the cancel — it drops
        the connection, ``generate`` raises, and _execute marks the job cancelled.
        """
        job = self.get(job_id)
        if job is None or job.status in TERMINAL:
            return False
        with self._cancel_lock:
            self._cancelled.add(job_id)
        if job.status == QUEUED:
            self._finish(job_id, CANCELLED, error="cancelled before start")
        else:
            logger.info("cancelling running job %s by stopping the backend", job_id)
            self._manager.release_video_slot()
        return True

    def _is_cancelled(self, job_id: str) -> bool:
        with self._cancel_lock:
            return job_id in self._cancelled

    def _finish(self, job_id: str, status: str, *, artifact: str | None = None,
                error: str | None = None) -> None:
        self._write(
            "UPDATE jobs SET status=?, artifact=?, error=?, finished_at=? WHERE id=?",
            (status, artifact, error, time.time(), job_id),
        )

    # ── worker ────────────────────────────────────────────────────────────

    def _next_queued(self) -> Job | None:
        rows = self._query(
            "SELECT * FROM jobs WHERE status = ? ORDER BY created_at LIMIT 1", (QUEUED,)
        )
        return _row_to_job(rows[0]) if rows else None

    def _run(self) -> None:
        idle_since: float | None = None
        while not self._stop.is_set():
            job = self._next_queued()
            if job is None:
                # Queue is empty. Give the backend's ~11 GB back once the idle
                # window passes, so the text models can use the slot again.
                if self._manager.video_active():
                    if idle_since is None:
                        idle_since = time.monotonic()
                    elif time.monotonic() - idle_since > self._settings.video_idle_timeout:
                        logger.info("video queue idle; releasing the slot")
                        self._manager.release_video_slot()
                        idle_since = None
                self._wake.wait(timeout=5.0)
                self._wake.clear()
                continue
            idle_since = None
            self._execute(job)

    def _execute(self, job: Job) -> None:
        if self._is_cancelled(job.id):
            self._finish(job.id, CANCELLED, error="cancelled before start")
            return
        self._write(
            "UPDATE jobs SET status=?, started_at=?, progress_total=? WHERE id=?",
            (RUNNING, time.time(), int(job.params.get("steps", 0)), job.id),
        )
        t0 = time.monotonic()
        progress_stop = threading.Event()
        try:
            backend = self._manager.acquire_video_slot(job.model)
            threading.Thread(
                target=self._pump_progress,
                args=(job.id, backend, progress_stop),
                name=f"video-progress-{job.id[:8]}",
                daemon=True,
            ).start()
            result = backend.generate(
                job.params,
                # No per-job wall-clock cap: at 768p a legitimate job runs for
                # hours, and a timeout here would kill work that is progressing
                # fine. Cancellation is the caller's lever instead.
                timeout=None,
            )
            out = self._out_dir / f"{job.id}.mp4"
            mux(result, out)
            self._finish(job.id, COMPLETED, artifact=str(out))
            logger.info("job %s completed in %.1f min", job.id, (time.monotonic() - t0) / 60)
        except (VideoBackendError, MuxError) as exc:
            # A cancel kills the backend mid-request, which surfaces here as a
            # backend error. Report what actually happened, not a spurious failure.
            if self._is_cancelled(job.id):
                logger.info("job %s cancelled", job.id)
                self._finish(job.id, CANCELLED, error="cancelled while running")
            else:
                logger.error("job %s failed: %s", job.id, exc)
                self._finish(job.id, FAILED, error=str(exc))
        except Exception as exc:  # noqa: BLE001 — a worker thread must never die
            logger.exception("job %s crashed", job.id)
            self._finish(job.id, FAILED, error=f"{type(exc).__name__}: {exc}")
        finally:
            progress_stop.set()

    def _pump_progress(self, job_id: str, backend, stop: threading.Event) -> None:
        """Mirror the backend's step counter into the job row for pollers."""
        while not stop.wait(2.0):
            p = backend.progress()
            if p is None:
                continue
            self._write(
                "UPDATE jobs SET progress_step=?, progress_total=? WHERE id=?",
                (p[0], p[1], job_id),
            )

    def prune(self) -> int:
        """Delete artifacts (and rows) past the retention window. Returns count."""
        hours = self._settings.video_retention_hours
        if hours <= 0:
            return 0
        cutoff = time.time() - hours * 3600
        rows = self._query(
            "SELECT id, artifact FROM jobs WHERE finished_at IS NOT NULL AND finished_at < ?",
            (cutoff,),
        )
        for r in rows:
            if r["artifact"]:
                Path(r["artifact"]).unlink(missing_ok=True)
            self._write("DELETE FROM jobs WHERE id = ?", (r["id"],))
        if rows:
            logger.info("pruned %d expired video job(s)", len(rows))
        return len(rows)

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        self._worker.join(timeout=5)
        with self._db_lock:
            self._db.close()
