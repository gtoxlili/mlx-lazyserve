"""Inference engines wrapping mlx-lm and mlx-vlm behind one tiny interface.

A loaded model exposes ``.stream(messages, *, max_tokens, temperature, top_p,
images=None, tools=None, enable_thinking=False, top_k, min_p, seed,
repetition_penalty, presence_penalty, frequency_penalty, logit_bias, stop,
response_format, kv_bits)`` which yields **typed events**:

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


def _find_stop(text: str, stops: list[str]) -> int:
    """Earliest index in ``text`` of any stop string, or -1."""
    best = -1
    for s in stops:
        i = text.find(s)
        if i != -1 and (best == -1 or i < best):
            best = i
    return best


def _parse_events(raw_iter, enable_thinking, tool_module, tools, stop=None) -> Iterator[dict]:
    """Turn a raw text-delta iterator into typed events.

    Splits ``<think>…</think>`` reasoning from content (mlx-vlm's ThinkingStreamState),
    suppresses tool-call markup from the content stream, honors ``stop`` sequences
    (truncating + halting the underlying generator), and at the end parses tool calls
    from the full output into OpenAI items. Degrades to plain content streaming if
    mlx-vlm's helpers aren't importable.
    """
    stops = [s for s in (stop or []) if s]

    try:
        from mlx_vlm.server.responses_state import (
            ThinkingStreamState,
            process_tool_calls,
            suppress_tool_call_content,
        )
    except Exception:  # mlx-vlm not installed (core-only) — stream plain content
        acc = ""
        for text in raw_iter:
            if not text:
                continue
            if stops:
                acc += text
                idx = _find_stop(acc, stops)
                if idx != -1:
                    head = acc[: idx]
                    emitted = acc[: len(acc) - len(text)]
                    if len(head) > len(emitted):
                        yield {"content": head[len(emitted):]}
                    _close(raw_iter)
                    return
            yield {"content": text}
        return

    state = ThinkingStreamState(enable_thinking)
    tc_start = getattr(tool_module, "tool_call_start", None) if tool_module else None
    in_tool = False
    full = ""
    keep = max((len(s) for s in stops), default=1) - 1  # hold-back so a split stop isn't missed
    pending = ""
    stopped = False
    for text in raw_iter:
        if not text:
            continue
        full += text
        delta = state.feed(text)
        if delta.reasoning:
            yield {"reasoning": delta.reasoning}
        content = delta.content
        if content is None:
            continue
        in_tool, content = suppress_tool_call_content(full, in_tool, tc_start, content)
        if not content:
            continue
        pending += content
        if stops:
            idx = _find_stop(pending, stops)
            if idx != -1:
                if idx:
                    yield {"content": pending[:idx]}
                pending = ""
                stopped = True
                _close(raw_iter)  # halt the underlying mlx generator promptly
                break
            if len(pending) > keep:
                yield {"content": pending[: len(pending) - keep]}
                pending = pending[len(pending) - keep:]
        else:
            yield {"content": pending}
            pending = ""
    if not stopped and pending:
        yield {"content": pending}
    if not stopped and tool_module is not None and tools:
        try:
            result = process_tool_calls(full, tool_module, tools)
        except Exception as exc:
            logger.warning("tool-call parse failed: %s", exc)
            result = None
        if result and result.get("calls"):
            yield {"tool_calls": result["calls"]}


def _close(it) -> None:
    closer = getattr(it, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


def _build_sampler(temperature, top_p, top_k, min_p):
    from mlx_lm.sample_utils import make_sampler

    return make_sampler(
        temp=float(temperature),
        top_p=float(top_p or 0.0),
        min_p=float(min_p or 0.0),
        top_k=int(top_k or 0),
    )


def _build_logits_processors(
    logit_bias, repetition_penalty, presence_penalty, frequency_penalty, structured
):
    """mlx-lm logit-bias/penalty processors + an optional structured-output processor."""
    procs: list = []
    if any(
        v is not None
        for v in (logit_bias, repetition_penalty, presence_penalty, frequency_penalty)
    ):
        from mlx_lm.sample_utils import make_logits_processors

        procs = list(
            make_logits_processors(
                logit_bias=logit_bias,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
        )
    if structured is not None:
        procs.append(structured)
    return procs or None


def _build_structured(hf_tokenizer, response_format):
    """An llguidance logits processor enforcing OpenAI ``response_format``, or None.

    Reuses mlx-vlm's ``build_json_schema_logits_processor`` (backed by the already
    installed ``llguidance``) so the model can only sample schema-valid JSON tokens —
    a *guarantee*, not best-effort prompting. Degrades to None (unconstrained) on any
    failure so a bad schema never 500s the request.
    """
    if not isinstance(response_format, dict):
        return None
    rtype = response_format.get("type")
    if rtype not in ("json_object", "json_schema"):
        return None  # "text" / unknown -> unconstrained
    try:
        from mlx_vlm.structured import build_json_schema_logits_processor
    except Exception as exc:
        logger.warning("response_format needs mlx-vlm+llguidance (%s); ignoring", exc)
        return None
    if rtype == "json_object":
        schema: object = {"type": "object"}
    else:
        js = response_format.get("json_schema") or {}
        schema = js.get("schema") or js
    try:
        return build_json_schema_logits_processor(hf_tokenizer, schema)
    except Exception as exc:
        logger.warning("could not compile JSON grammar (%s); ignoring response_format", exc)
        return None


def _apply_seed(seed) -> None:
    if seed is None:
        return
    try:
        import mlx.core as mx

        mx.random.seed(int(seed))
    except Exception as exc:
        logger.warning("could not set seed (%s)", exc)


def _capture_usage(chunk, usage: dict) -> None:
    """Record cumulative token counts off a stream_generate result object.

    Both mlx-lm's GenerationResponse and mlx-vlm's GenerationResult expose
    ``.prompt_tokens`` and ``.generation_tokens`` (cumulative); the last object wins.
    """
    p = getattr(chunk, "prompt_tokens", None)
    g = getattr(chunk, "generation_tokens", None)
    if p is not None:
        usage["prompt_tokens"] = int(p)
    if g is not None:
        usage["completion_tokens"] = int(g)


def _usage_event(usage: dict):
    """An OpenAI usage object wrapped as a typed event, or None if no counts captured."""
    if not usage:
        return None
    p = int(usage.get("prompt_tokens", 0))
    c = int(usage.get("completion_tokens", 0))
    return {"usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}}


def _raw_with_kv_fallback(make_gen, gen_kw, usage) -> Iterator[str]:
    """Iterate a stream_generate, yielding ``.text`` deltas and recording token counts.

    If KV-cache quantization isn't supported for this model — e.g. Gemma's
    sliding-window ``RotatingKVCache`` raises ``NotImplementedError`` — retry once
    without ``kv_bits`` (only safe before any token was produced).
    """
    produced = False
    try:
        for chunk in make_gen(gen_kw):
            produced = True
            _capture_usage(chunk, usage)
            t = getattr(chunk, "text", None)
            yield t if t is not None else str(chunk)
    except NotImplementedError as exc:
        if produced or not gen_kw.get("kv_bits"):
            raise
        logger.warning("kv_bits unsupported for this model (%s); retrying unquantized", exc)
        kw = {k: v for k, v in gen_kw.items() if k != "kv_bits"}
        for chunk in make_gen(kw):
            _capture_usage(chunk, usage)
            t = getattr(chunk, "text", None)
            yield t if t is not None else str(chunk)


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
        top_k=None,
        min_p=None,
        seed=None,
        repetition_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        logit_bias=None,
        stop=None,
        response_format=None,
        kv_bits=0,
    ) -> Iterator[dict]:
        if images:
            raise ValueError(
                "this model is text-only (mlx-lm); image input requires a vision model"
            )

        from mlx_lm import stream_generate

        prompt = self._build_prompt(messages, tools, enable_thinking)
        tool_module = self._tool_module() if tools else None
        hf_tok = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        structured = _build_structured(hf_tok, response_format)
        processors = _build_logits_processors(
            logit_bias, repetition_penalty, presence_penalty, frequency_penalty, structured
        )
        _apply_seed(seed)
        gen_kw = {
            "max_tokens": max_tokens,
            "sampler": _build_sampler(temperature, top_p, top_k, min_p),
        }
        if processors:
            gen_kw["logits_processors"] = processors
        if kv_bits:
            gen_kw["kv_bits"] = int(kv_bits)

        usage: dict = {}

        def raw():
            yield from _raw_with_kv_fallback(
                lambda kw: stream_generate(self.model, self.tokenizer, prompt, **kw),
                gen_kw,
                usage,
            )

        yield from _parse_events(raw(), enable_thinking, tool_module, tools, stop)
        ev = _usage_event(usage)
        if ev:
            yield ev

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
        top_k=None,
        min_p=None,
        seed=None,
        repetition_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        logit_bias=None,
        stop=None,
        response_format=None,
        kv_bits=0,
    ) -> Iterator[dict]:
        from mlx_vlm import stream_generate

        image_refs = self._resolve_images(images)
        prompt = self._build_prompt(messages, len(image_refs), tools, enable_thinking)
        tool_module = self._tool_module() if tools else None
        hf_tok = getattr(self.processor, "tokenizer", self.processor)
        structured = _build_structured(hf_tok, response_format)
        processors = _build_logits_processors(
            logit_bias, repetition_penalty, presence_penalty, frequency_penalty, structured
        )
        _apply_seed(seed)
        gen_kw = {
            "image": image_refs or None,
            "max_tokens": max_tokens,
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if top_k:
            gen_kw["top_k"] = int(top_k)
        if min_p:
            gen_kw["min_p"] = float(min_p)
        if processors:
            gen_kw["logits_processors"] = processors
        if kv_bits:
            gen_kw["kv_bits"] = int(kv_bits)

        usage: dict = {}

        def raw():
            yield from _raw_with_kv_fallback(
                lambda kw: stream_generate(self.model, self.processor, prompt, **kw),
                gen_kw,
                usage,
            )

        yield from _parse_events(raw(), enable_thinking, tool_module, tools, stop)
        ev = _usage_event(usage)
        if ev:
            yield ev

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
