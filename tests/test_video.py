import base64
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlx_lazyserve.config import ModelSpec, _load_models
from mlx_lazyserve.contextir import ContextIRClient, expand_or_passthrough
from mlx_lazyserve.video_mux import MuxError, mux


class VideoModelSpecTests(unittest.TestCase):
    def test_video_spec_needs_no_repo(self):
        spec = ModelSpec(name="h3", repo="", kind="video", path="/tmp/weights")
        self.assertEqual(spec.kind, "video")
        self.assertEqual(spec.repo, "")

    def test_video_model_never_becomes_the_chat_default(self):
        """A video model can't serve /v1/chat/completions, so it must not be the default."""
        toml = (
            '[models."h3"]\nkind = "video"\npath = "/tmp/w"\ndefault = true\n'
            '[models."small"]\nrepo = "org/small"\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.toml"
            path.write_text(toml)
            with mock.patch("mlx_lazyserve.config._registry_path", return_value=path):
                models, default = _load_models()
        self.assertEqual(models["h3"].kind, "video")
        self.assertEqual(default, "small")


class MuxTests(unittest.TestCase):
    def _payload(self, frames=2, w=32, h=32, data=None):
        return {
            "format": "rgb8",
            "frames": frames,
            "width": w,
            "height": h,
            "fps": 24,
            "data": base64.b64encode(data if data is not None else b"\x00" * (frames * w * h * 3)).decode(),
        }

    def test_rejects_unexpected_frame_format(self):
        payload = self._payload()
        payload["format"] = "yuv420p"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MuxError) as cm:
                mux(payload, Path(tmp) / "out.mp4")
        self.assertIn("yuv420p", str(cm.exception))

    def test_rejects_truncated_frame_buffer(self):
        """A short buffer means a silently truncated video — must fail, not pad."""
        payload = self._payload(frames=4, data=b"\x00" * 100)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MuxError) as cm:
                mux(payload, Path(tmp) / "out.mp4")
        self.assertIn("expected", str(cm.exception))


class ContextIRTests(unittest.TestCase):
    def test_passthrough_when_disabled(self):
        prompt, expanded = expand_or_passthrough(
            None, "a cat", duration=5, ratio="16:9", mode="expand"
        )
        self.assertEqual(prompt, "a cat")
        self.assertIsNone(expanded)

    def test_raw_mode_skips_a_configured_client(self):
        client = ContextIRClient("key", "https://example.invalid")
        with mock.patch.object(client, "expand", side_effect=AssertionError("must not call")):
            prompt, expanded = expand_or_passthrough(
                client, "a cat", duration=5, ratio="16:9", mode="raw"
            )
        self.assertEqual(prompt, "a cat")
        self.assertIsNone(expanded)

    def test_expansion_failure_degrades_to_the_raw_prompt(self):
        """Losing a queued multi-hour job to a remote blip is worse than a weaker prompt."""
        from mlx_lazyserve.contextir import ContextIRError

        client = ContextIRClient("key", "https://example.invalid")
        with mock.patch.object(client, "expand", side_effect=ContextIRError("boom")):
            prompt, expanded = expand_or_passthrough(
                client, "a cat", duration=5, ratio="16:9", mode="expand"
            )
        self.assertEqual(prompt, "a cat")
        self.assertIsNone(expanded)


if __name__ == "__main__":
    unittest.main()
