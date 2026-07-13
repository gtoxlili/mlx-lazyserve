"""Inference engines wrapping mlx-lm and mlx-vlm behind one tiny interface.

A loaded model exposes ``.stream(messages, *, max_tokens, temperature, top_p,
images=None, tools=None, enable_thinking=False, top_k, min_p, seed,
repetition_penalty, presence_penalty, frequency_penalty, repetition_context_size,
logit_bias, stop, response_format, kv_bits, loop_guard, max_prompt_tokens)`` which
yields **typed events**:

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
import json
import logging
from collections.abc import Iterator
from types import SimpleNamespace

logger = logging.getLogger(__name__)

_FIM_MARKERS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
)


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


def _strip_fim_from_messages(messages):
    """Remove leaked FIM control markers from prior assistant turns.

    User/system text is deliberately left alone so a user can discuss the marker itself.
    Only messages that need rewriting are copied.
    """
    out = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            out.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, str):
            out.append(message)
            continue
        clean = content
        for marker in _FIM_MARKERS:
            clean = clean.replace(marker, "")
        out.append({**message, "content": clean} if clean != content else message)
    return out


def _without_fim_markers(raw_iter: Iterator[str]) -> Iterator[str]:
    """Strip FIM markers from a text stream, including markers split across chunks."""
    pending = ""
    try:
        for text in raw_iter:
            if not text:
                continue
            pending += text
            for marker in _FIM_MARKERS:
                pending = pending.replace(marker, "")

            # Retain only a suffix that could be the beginning of a split marker.
            hold = 0
            for marker in _FIM_MARKERS:
                for length in range(min(len(marker) - 1, len(pending)), 0, -1):
                    if pending.endswith(marker[:length]):
                        hold = max(hold, length)
                        break
            if len(pending) > hold:
                yield pending[:-hold] if hold else pending
                pending = pending[-hold:] if hold else ""
    except GeneratorExit:
        _close(raw_iter)
        raise
    if pending:
        # An incomplete marker is ordinary text and must not be dropped.
        yield pending
    _close(raw_iter)


def _fim_token_ids(tokenizer) -> tuple[int, ...]:
    """Return FIM markers that are represented by one exact tokenizer token."""
    found = []
    for marker in _FIM_MARKERS:
        try:
            ids = list(tokenizer.encode(marker, add_special_tokens=False))
            if len(ids) != 1:
                continue
            token_id = int(ids[0])
            if tokenizer.decode([token_id], skip_special_tokens=False) == marker:
                found.append(token_id)
        except Exception:
            continue
    return tuple(dict.fromkeys(found))


def _events_with_empty_content_retry(
    make_events, enable_thinking, repo="", allow_retry=True
) -> Iterator[dict]:
    """Retry once without thinking when a thinking pass produces no answer.

    Reasoning from the first pass can be streamed immediately. Usage is held until both
    passes finish so callers receive one cumulative usage event.
    """
    usages = []
    has_content = False
    has_tool_calls = False

    for event in make_events(enable_thinking):
        if "usage" in event:
            usages.append(event["usage"])
            continue
        if event.get("content") and event["content"].strip():
            has_content = True
        if event.get("tool_calls"):
            has_tool_calls = True
        yield event

    if allow_retry and enable_thinking and not has_content and not has_tool_calls:
        logger.warning("%s produced no answer after thinking; retrying without thinking", repo)
        for event in make_events(False):
            if "usage" in event:
                usages.append(event["usage"])
                continue
            yield event

    if usages:
        prompt_tokens = sum(int(u.get("prompt_tokens", 0)) for u in usages)
        completion_tokens = sum(int(u.get("completion_tokens", 0)) for u in usages)
        yield {
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        }


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


# --- repetition / degeneration guard ------------------------------------------------
#
# A degraded model (heavy quant, abliteration) can fall into a degenerate loop, emitting
# the same short block forever (e.g. "傲娇傲娇傲娇…"). A ``repetition_penalty`` usually
# prevents it, but as a model-agnostic backstop we also watch the decoded stream and cut
# generation when a short cycle repeats far past anything a real answer would contain.
# The same detector powers ``trim_degenerate`` so a looped reply never reaches the user
# or the bot's saved history (a stored loop would re-prime it on the next turn).
_LOOP_MAX_PERIOD = 32  # longest repeating block (chars) we scan for
_LOOP_MIN_REPEATS = 4  # need at least this many back-to-back copies, ...
_LOOP_MIN_RUN_CHARS = 60  # ...and the whole run must span at least this many chars
_LOOP_TAIL_WINDOW = 512  # only inspect this many trailing chars while streaming
_LOOP_KEEP_CYCLES = 3  # cycles kept when collapsing a run for display/history


def _trailing_run(s: str):
    """Longest run of a ``≤_LOOP_MAX_PERIOD``-char block repeated at the END of ``s``.

    Returns ``(run_chars, period, repeats)`` for the run spanning the most characters
    (ties prefer the shortest period — the tightest cycle), or ``None`` when ``s`` is empty.
    """
    n = len(s)
    best = None
    for period in range(1, min(_LOOP_MAX_PERIOD, n // 2) + 1):
        seg = s[n - period : n]
        reps, i = 1, n - period
        while i - period >= 0 and s[i - period : i] == seg:
            reps += 1
            i -= period
        run_chars = reps * period
        if best is None or run_chars > best[0]:
            best = (run_chars, period, reps)
    return best


def _is_degenerate(s: str) -> bool:
    run = _trailing_run(s)
    return bool(run and run[2] >= _LOOP_MIN_REPEATS and run[0] >= _LOOP_MIN_RUN_CHARS)


def trim_degenerate(text: str) -> str:
    """Collapse a trailing degenerate repetition run to a few cycles + an ellipsis.

    No-op on normal text. Lets the bot keep a looped reply out of both the chat and the
    persisted history (a stored loop would re-prime the model on the next turn).
    """
    run = _trailing_run(text or "")
    if not (run and run[2] >= _LOOP_MIN_REPEATS and run[0] >= _LOOP_MIN_RUN_CHARS):
        return text
    run_chars, period, _ = run
    start = len(text) - run_chars
    return text[:start] + text[start : start + period] * _LOOP_KEEP_CYCLES + "…"


def _loop_guarded(raw_iter: Iterator[str]) -> Iterator[str]:
    """Pass text deltas through, halting the generator if the output degenerates into a loop."""
    tail = ""
    try:
        for text in raw_iter:
            yield text
            if not text:
                continue
            tail = (tail + text)[-_LOOP_TAIL_WINDOW:]
            if _is_degenerate(tail):
                logger.warning("degeneration loop detected; halting generation")
                break
    except GeneratorExit:
        _close(raw_iter)
        raise
    _close(raw_iter)


def _fit_prompt(messages, build, token_len, max_prompt_tokens):
    """Drop the oldest non-system messages until the built prompt fits ``max_prompt_tokens``.

    ``build(msgs)`` returns a prompt (str or token ids); ``token_len(prompt)`` counts its
    tokens. A leading system message and the newest message are always kept. Returns the
    final prompt. No-op when ``max_prompt_tokens`` is falsy or the prompt already fits.
    """
    prompt = build(messages)
    if not max_prompt_tokens:
        return prompt
    msgs = list(messages)
    while token_len(prompt) > max_prompt_tokens and len(msgs) > 1:
        drop = next(
            (i for i, m in enumerate(msgs) if not (isinstance(m, dict) and m.get("role") == "system")),
            None,
        )
        if drop is None or drop >= len(msgs) - 1:  # only a system msg + the newest remain
            break
        del msgs[drop]
        prompt = build(msgs)
    return prompt


def _normalize_tool_call_args(messages):
    """Coerce each ``tool_calls[].function.arguments`` from a JSON *string* into a dict.

    HF chat templates iterate a tool call's ``arguments`` as a mapping (Qwen's does
    ``{% for k, v in tool_call.function.arguments | items %}``), but the OpenAI wire format —
    and our own tool-call parser (mlx-vlm's ``process_tool_calls``) — carry ``arguments`` as a
    JSON string. Feeding that back on a follow-up turn makes ``apply_chat_template`` raise
    ``TypeError: Can only get item pairs from a mapping``, which surfaces as a 500 / failed
    reply. Convert string args to dicts so a multi-turn tool exchange renders; leave dicts (and
    unparseable strings) untouched. Returns a new list, shallow-copying only the messages/calls
    it rewrites — the caller's list is never mutated.
    """
    out = []
    for m in messages:
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        if not calls:
            out.append(m)
            continue
        new_calls = []
        changed = False
        for tc in calls:
            fn = tc.get("function") if isinstance(tc, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, str):
                try:
                    parsed = json.loads(args or "{}")
                except (ValueError, TypeError):
                    new_calls.append(tc)  # leave a malformed string as-is
                    continue
                new_calls.append({**tc, "function": {**fn, "arguments": parsed}})
                changed = True
            else:
                new_calls.append(tc)
        out.append({**m, "tool_calls": new_calls} if changed else m)
    return out


def _build_sampler(temperature, top_p, top_k, min_p):
    from mlx_lm.sample_utils import make_sampler

    return make_sampler(
        temp=float(temperature),
        top_p=float(top_p or 0.0),
        min_p=float(min_p or 0.0),
        top_k=int(top_k or 0),
    )


def _build_logits_processors(
    logit_bias,
    repetition_penalty,
    presence_penalty,
    frequency_penalty,
    repetition_context_size,
    structured,
    blocked_token_ids=(),
):
    """mlx-lm logit-bias/penalty processors + an optional structured-output processor."""
    procs: list = []
    if any(
        v is not None
        for v in (logit_bias, repetition_penalty, presence_penalty, frequency_penalty)
    ):
        from mlx_lm.sample_utils import make_logits_processors

        kwargs: dict = {
            "logit_bias": logit_bias,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        if repetition_context_size:
            kwargs["repetition_context_size"] = int(repetition_context_size)
        procs = list(make_logits_processors(**kwargs))
    if structured is not None:
        procs.append(structured)
    if blocked_token_ids:
        import mlx.core as mx

        indices = mx.array(list(blocked_token_ids))

        def block_control_tokens(_, logits):
            # Apply last so neither caller logit_bias nor a grammar can re-enable FIM tokens.
            return logits.at[:, indices].add(-mx.inf)

        procs.append(block_control_tokens)
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
        hf_tok = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        self._blocked_token_ids = _fim_token_ids(hf_tok)
        if self._blocked_token_ids:
            logger.info("blocking FIM token ids for %s: %s", repo, self._blocked_token_ids)

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

    def _prompt_token_len(self, prompt) -> int:
        if isinstance(prompt, str):
            return len(self.tokenizer.encode(prompt))
        return len(prompt)  # already token ids

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
        repetition_context_size=None,
        logit_bias=None,
        stop=None,
        response_format=None,
        kv_bits=0,
        loop_guard=True,
        max_prompt_tokens=None,
    ) -> Iterator[dict]:
        if images:
            raise ValueError(
                "this model is text-only (mlx-lm); image input requires a vision model"
            )

        from mlx_lm import stream_generate

        messages = _strip_fim_from_messages(_normalize_tool_call_args(messages))
        tool_module = self._tool_module() if tools else None
        hf_tok = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        structured = _build_structured(hf_tok, response_format)
        processors = _build_logits_processors(
            logit_bias,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            repetition_context_size,
            structured,
            self._blocked_token_ids,
        )
        gen_kw = {
            "max_tokens": max_tokens,
            "sampler": _build_sampler(temperature, top_p, top_k, min_p),
        }
        if processors:
            gen_kw["logits_processors"] = processors
        if kv_bits:
            gen_kw["kv_bits"] = int(kv_bits)

        def make_events(thinking):
            prompt = _fit_prompt(
                messages,
                lambda ms: self._build_prompt(ms, tools, thinking),
                self._prompt_token_len,
                max_prompt_tokens,
            )
            _apply_seed(seed)
            usage: dict = {}

            def raw():
                yield from _raw_with_kv_fallback(
                    lambda kw: stream_generate(self.model, self.tokenizer, prompt, **kw),
                    gen_kw,
                    usage,
                )

            clean = _without_fim_markers(raw())
            guarded = _loop_guarded(clean) if loop_guard else clean
            yield from _parse_events(guarded, thinking, tool_module, tools, stop)
            ev = _usage_event(usage)
            if ev:
                yield ev

        yield from _events_with_empty_content_retry(
            make_events, enable_thinking, self.repo, allow_retry=not bool(stop)
        )

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
        hf_tok = getattr(self.processor, "tokenizer", self.processor)
        self._blocked_token_ids = _fim_token_ids(hf_tok)
        if self._blocked_token_ids:
            logger.info("blocking FIM token ids for %s: %s", repo, self._blocked_token_ids)

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

    def _prompt_token_len(self, prompt) -> int:
        if isinstance(prompt, str):
            tok = getattr(self.processor, "tokenizer", self.processor)
            try:
                return len(tok.encode(prompt))
            except Exception:
                return max(1, len(prompt) // 4)  # rough fallback if encode is unavailable
        return len(prompt)

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
        repetition_context_size=None,
        logit_bias=None,
        stop=None,
        response_format=None,
        kv_bits=0,
        loop_guard=True,
        max_prompt_tokens=None,
    ) -> Iterator[dict]:
        from mlx_vlm import stream_generate

        image_refs = self._resolve_images(images)
        messages = _strip_fim_from_messages(_normalize_tool_call_args(messages))
        tool_module = self._tool_module() if tools else None
        hf_tok = getattr(self.processor, "tokenizer", self.processor)
        structured = _build_structured(hf_tok, response_format)
        processors = _build_logits_processors(
            logit_bias,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            repetition_context_size,
            structured,
            self._blocked_token_ids,
        )
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

        def make_events(thinking):
            prompt = _fit_prompt(
                messages,
                lambda ms: self._build_prompt(ms, len(image_refs), tools, thinking),
                self._prompt_token_len,
                max_prompt_tokens,
            )
            _apply_seed(seed)
            usage: dict = {}

            def raw():
                yield from _raw_with_kv_fallback(
                    lambda kw: stream_generate(self.model, self.processor, prompt, **kw),
                    gen_kw,
                    usage,
                )

            clean = _without_fim_markers(raw())
            guarded = _loop_guarded(clean) if loop_guard else clean
            yield from _parse_events(guarded, thinking, tool_module, tools, stop)
            ev = _usage_event(usage)
            if ev:
                yield ev

        yield from _events_with_empty_content_retry(
            make_events, enable_thinking, self.repo, allow_retry=not bool(stop)
        )

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
