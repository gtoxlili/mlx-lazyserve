"""Mux the backend's raw frames + PCM into an mp4.

The Zig backend returns pixels, not a container: base64 RGB8 frames plus base64
PCM s16le stereo. Building the mp4 is this layer's job.

Video is encoded with ``h264_videotoolbox`` — Apple's hardware encoder. On this
box software x264 would compete with nothing (the GPU is idle by the time we
mux) but VideoToolbox still wins on wall-clock and leaves the CPU free for the
next job's text-encode stage.
"""

from __future__ import annotations

import base64
import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

logger = logging.getLogger(__name__)


class MuxError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _write_wav(pcm: bytes, path: Path, *, sample_rate: int, channels: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # pcm_s16le
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def mux(result: dict, out_path: Path, *, crf_quality: int = 65) -> Path:
    """Turn one backend response into an mp4 at ``out_path``.

    ``result`` is the backend's JSON: frames/width/height/fps/format/data plus the
    optional audio_* fields. Raises MuxError on anything unexpected — a silently
    truncated video is worse than a failed job.
    """
    if not ffmpeg_available():
        raise MuxError("ffmpeg not found on PATH")
    fmt = result.get("format")
    if fmt != "rgb8":
        raise MuxError(f"unexpected frame format {fmt!r} (expected 'rgb8')")

    width = int(result["width"])
    height = int(result["height"])
    frames = int(result["frames"])
    fps = int(result.get("fps", 24))
    rgb = base64.b64decode(result["data"])

    expected = width * height * frames * 3
    if len(rgb) != expected:
        raise MuxError(
            f"frame buffer is {len(rgb)} bytes, expected {expected} "
            f"({frames}f {width}x{height} rgb8)"
        )

    audio_b64 = result.get("audio_data")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="h3mux-") as tmp:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
        ]
        if audio_b64:
            wav_path = Path(tmp) / "audio.wav"
            _write_wav(
                base64.b64decode(audio_b64),
                wav_path,
                sample_rate=int(result.get("audio_sample_rate", 32000)),
                channels=int(result.get("audio_channels", 2)),
            )
            cmd += ["-i", str(wav_path), "-c:a", "aac", "-b:a", "192k"]
        cmd += [
            "-c:v", "h264_videotoolbox",
            "-q:v", str(crf_quality),
            "-pix_fmt", "yuv420p",
            # No -shortest. The two tracks never land on exactly the same duration
            # (5 frames @24fps is 208 ms, its audio is 200 ms), and -shortest
            # resolves that by dropping the trailing video frame — 20% of a short
            # clip. mp4 carries tracks of unequal length fine, so keep every frame
            # the model actually generated and let the audio end a few ms early.
            "-movflags", "+faststart",
            str(out_path),
        ]
        proc = subprocess.run(cmd, input=rgb, capture_output=True)
        if proc.returncode != 0:
            raise MuxError(
                f"ffmpeg failed ({proc.returncode}): "
                f"{proc.stderr.decode('utf-8', 'replace')[:400]}"
            )

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise MuxError(f"ffmpeg produced no output at {out_path}")
    logger.info(
        "muxed %s (%df %dx%d @%dfps, %.1f MB)",
        out_path.name, frames, width, height, fps, out_path.stat().st_size / 1e6,
    )
    return out_path
