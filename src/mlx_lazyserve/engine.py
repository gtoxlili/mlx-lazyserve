"""Inference engines wrapping mlx-lm and mlx-vlm behind one tiny interface.

A loaded model exposes ``.stream(messages, ...)`` yielding text deltas and
``.close()`` to release unified memory.
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

    def stream(self, messages, *, max_tokens, temperature, top_p) -> Iterator[str]:
        from mlx_lm import stream_generate

        try:
            from mlx_lm.sample_utils import make_sampler
        except Exception:
            make_sampler = None

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        kwargs: dict = {"max_tokens": max_tokens}
        if make_sampler is not None:
            kwargs["sampler"] = make_sampler(temp=temperature, top_p=top_p)
        else:  # older mlx-lm accepted sampling params directly
            kwargs["temp"] = temperature

        for response in stream_generate(self.model, self.tokenizer, prompt, **kwargs):
            text = getattr(response, "text", None)
            yield text if text is not None else str(response)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        clear_mlx_cache()


class MlxVlmModel:
    """Vision-language models via mlx-vlm (also handles text-only generation).

    mlx-vlm's generate/template API has shifted across releases; this targets
    mlx-vlm >= 0.4.x and is exercised on the first smoke test.
    """

    def __init__(self, repo: str) -> None:
        from mlx_vlm import load

        self.repo = repo
        self.model, self.processor = load(repo)
        try:
            from mlx_vlm.utils import load_config

            self.config = load_config(repo)
        except Exception:
            self.config = getattr(self.model, "config", None)

    def stream(self, messages, *, max_tokens, temperature, top_p) -> Iterator[str]:
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(self.processor, self.config, messages, num_images=0)
        # kwarg name for temperature has differed between versions; try both.
        attempts = (
            {"max_tokens": max_tokens, "temperature": temperature},
            {"max_tokens": max_tokens, "temp": temperature},
        )
        last_err: Exception | None = None
        for kwargs in attempts:
            try:
                gen = stream_generate(self.model, self.processor, prompt, [], **kwargs)
            except TypeError as exc:  # signature mismatch for this mlx-vlm version
                last_err = exc
                continue
            for chunk in gen:
                text = getattr(chunk, "text", None)
                yield text if text is not None else str(chunk)
            return
        if last_err is not None:
            raise last_err

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
