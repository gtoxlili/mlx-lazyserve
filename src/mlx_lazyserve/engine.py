"""Inference engines wrapping mlx-lm and mlx-vlm behind one tiny interface.

A loaded model exposes ``.stream(messages, *, max_tokens, temperature, top_p,
images=None)`` yielding text deltas, and ``.close()`` to release unified memory.
``images`` is a list of image references (http(s) URL, local path, or a
``data:`` base64 URI); only the vision engine consumes them.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator

logger = logging.getLogger(__name__)


def clear_mlx_cache() -> None:
    """Return cached MLX/Metal buffers to the system after unloading a model."""
    try:
        import mlx.core as mx
    except Exception:
        return
    for owner in (mx, getattr(mx, "metal", None)):
        fn = getattr(owner, "clear_cache", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            return


class MlxLmModel:
    """Text LLMs via mlx-lm."""

    def __init__(self, repo: str) -> None:
        from mlx_lm import load

        self.repo = repo
        self.model, self.tokenizer = load(repo)

    def stream(
        self, messages, *, max_tokens, temperature, top_p, images=None
    ) -> Iterator[str]:
        if images:
            raise ValueError(
                "this model is text-only (mlx-lm); image input requires a vision model"
            )

        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        sampler = make_sampler(temp=temperature, top_p=top_p)
        for response in stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
        ):
            text = getattr(response, "text", None)
            yield text if text is not None else str(response)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        clear_mlx_cache()


class MlxVlmModel:
    """Vision-language models via mlx-vlm (also handles text-only generation)."""

    def __init__(self, repo: str) -> None:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.repo = repo
        self.model, self.processor = load(repo)
        try:
            self.config = load_config(repo)
        except Exception:
            self.config = getattr(self.model, "config", None)

    @staticmethod
    def _resolve_images(images) -> list:
        """Turn API image refs into things mlx-vlm accepts (URL/path/PIL.Image)."""
        resolved = []
        for ref in images or []:
            if isinstance(ref, str) and ref.startswith("data:"):
                import base64
                import io

                from PIL import Image

                _, b64 = ref.split(",", 1)
                resolved.append(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
            else:
                resolved.append(ref)
        return resolved

    def stream(
        self, messages, *, max_tokens, temperature, top_p, images=None
    ) -> Iterator[str]:
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        image_refs = self._resolve_images(images)
        prompt = apply_chat_template(
            self.processor, self.config, messages, num_images=len(image_refs)
        )
        for chunk in stream_generate(
            self.model,
            self.processor,
            prompt,
            image=image_refs or None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            text = getattr(chunk, "text", None)
            yield text if text is not None else str(chunk)

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.config = None
        gc.collect()
        clear_mlx_cache()


def load_model(spec) -> MlxLmModel | MlxVlmModel:
    """Instantiate the right engine for a ModelSpec, with auto-fallback."""
    if spec.engine in ("mlx_lm", "auto"):
        try:
            logger.info("loading %s via mlx-lm", spec.repo)
            return MlxLmModel(spec.repo)
        except Exception as exc:
            if spec.engine == "mlx_lm":
                raise
            logger.warning("mlx-lm could not load %s (%s); trying mlx-vlm", spec.repo, exc)
    logger.info("loading %s via mlx-vlm", spec.repo)
    return MlxVlmModel(spec.repo)
