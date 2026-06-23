"""Inference engines wrapping mlx-lm and mlx-vlm behind one tiny interface.

A loaded model exposes
``.stream(messages, *, max_tokens, temperature, top_p, images=None, tools=None,
enable_thinking=False)`` which yields **typed events**:

    {"reasoning": str}    incremental thinking text (only when enable_thinking)
    {"content": str}      incremental answer text
    {"tool_calls": [...]}  a final OpenAI-shaped tool_calls list (only when the
                          request supplied ``tools`` and the model emitted calls)

and ``.close()`` to release unified memory. ``images`` is a list of image refs
(http(s) URL, local path, or ``data:`` base64 URI); only the vision engine uses them.

Reasoning splitting and tool-call parsing reuse mlx-vlm's pure-Python helpers plus
mlx-lm/mlx-vlm's per-model ``tool_parsers`` — no model-specific parsing is hand-rolled.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from types import SimpleNamespace

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


def _parse_events(raw_iter, enable_thinking, tool_module, tools) -> Iterator[dict]:
    """Turn a raw text-delta iterator into typed events.

    Splits ``<think>…</think>`` reasoning from content (mlx-vlm's ThinkingStreamState),
    suppresses tool-call markup from the content stream, and at the end parses tool
    calls from the full output into OpenAI items. Degrades to plain content streaming
    if mlx-vlm's helpers aren't importable.
    """
    try:
        from mlx_vlm.server.responses_state import (
            ThinkingStreamState,
            process_tool_calls,
            suppress_tool_call_content,
        )
    except Exception:  # mlx-vlm not installed (core-only) — stream plain content
        for text in raw_iter:
            if text:
                yield {"content": text}
        return

    state = ThinkingStreamState(enable_thinking)
    tc_start = getattr(tool_module, "tool_call_start", None) if tool_module else None
    in_tool = False
    full = ""
    for text in raw_iter:
        if not text:
            continue
        full += text
        delta = state.feed(text)
        if delta.reasoning:
            yield {"reasoning": delta.reasoning}
        content = delta.content
        if content is not None:
            in_tool, content = suppress_tool_call_content(full, in_tool, tc_start, content)
            if content:
                yield {"content": content}
    if tool_module is not None and tools:
        try:
            result = process_tool_calls(full, tool_module, tools)
        except Exception as exc:
            logger.warning("tool-call parse failed: %s", exc)
            result = None
        if result and result.get("calls"):
            yield {"tool_calls": result["calls"]}


class MlxLmModel:
    """Text LLMs via mlx-lm."""

    def __init__(self, repo: str) -> None:
        from mlx_lm import load

        self.repo = repo
        self.model, self.tokenizer = load(repo)

    def _tool_module(self):
        """Build a tool-parser shim from the tokenizer's native tool-call attributes."""
        tok = self.tokenizer
        if not getattr(tok, "has_tool_calling", False):
            return None
        start = getattr(tok, "tool_call_start", None)
        parser = getattr(tok, "tool_parser", None)
        if not start or parser is None:
            return None
        return SimpleNamespace(
            tool_call_start=start,
            tool_call_end=getattr(tok, "tool_call_end", "") or "",
            parse_tool_call=parser,
        )

    def _build_prompt(self, messages, tools, enable_thinking):
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tools=tools or None,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            logger.warning("apply_chat_template(tools/enable_thinking) failed (%s); plain", exc)
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def stream(
        self,
        messages,
        *,
        max_tokens,
        temperature,
        top_p,
        images=None,
        tools=None,
        enable_thinking=False,
    ) -> Iterator[dict]:
        if images:
            raise ValueError(
                "this model is text-only (mlx-lm); image input requires a vision model"
            )

        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self._build_prompt(messages, tools, enable_thinking)
        sampler = make_sampler(temp=temperature, top_p=top_p)
        tool_module = self._tool_module() if tools else None

        def raw():
            for response in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
            ):
                t = getattr(response, "text", None)
                yield t if t is not None else str(response)

        yield from _parse_events(raw(), enable_thinking, tool_module, tools)

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

    def _tool_module(self):
        try:
            from mlx_vlm.tool_parsers import (
                _infer_tool_parser_from_processor,
                load_tool_module,
            )

            tp = _infer_tool_parser_from_processor(self.processor)
            return load_tool_module(tp) if tp else None
        except Exception as exc:
            logger.warning("vlm tool-parser discovery failed: %s", exc)
            return None

    def _build_prompt(self, messages, num_images, tools, enable_thinking):
        from mlx_vlm.prompt_utils import apply_chat_template

        try:
            return apply_chat_template(
                self.processor,
                self.config,
                messages,
                num_images=num_images,
                tools=tools or None,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            logger.warning("vlm apply_chat_template(tools/enable_thinking) failed (%s); plain", exc)
            return apply_chat_template(
                self.processor, self.config, messages, num_images=num_images
            )

    def stream(
        self,
        messages,
        *,
        max_tokens,
        temperature,
        top_p,
        images=None,
        tools=None,
        enable_thinking=False,
    ) -> Iterator[dict]:
        from mlx_vlm import stream_generate

        image_refs = self._resolve_images(images)
        prompt = self._build_prompt(messages, len(image_refs), tools, enable_thinking)
        tool_module = self._tool_module() if tools else None

        def raw():
            for chunk in stream_generate(
                self.model,
                self.processor,
                prompt,
                image=image_refs or None,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                t = getattr(chunk, "text", None)
                yield t if t is not None else str(chunk)

        yield from _parse_events(raw(), enable_thinking, tool_module, tools)

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
