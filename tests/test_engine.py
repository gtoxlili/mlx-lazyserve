import math
import unittest

from unittest import mock

from mlx_lazyserve.engine import (
    ContextOverflow,
    _build_logits_processors,
    _events_with_empty_content_retry,
    _fim_token_ids,
    _fit_output,
    _fit_prompt,
    _PrefixCache,
    _strip_fim_from_messages,
    _without_fim_markers,
)


class FakeTokenizer:
    tokens = {
        "<|fim_prefix|>": 5,
        "<|fim_middle|>": 6,
        "<|fim_suffix|>": 7,
        # Simulate a missing token being encoded as multiple ordinary pieces.
        "<|fim_pad|>": [90, 91],
    }

    def encode(self, text, add_special_tokens=False):
        token = self.tokens[text]
        return token if isinstance(token, list) else [token]

    def decode(self, ids, skip_special_tokens=False):
        reverse = {value: key for key, value in self.tokens.items() if isinstance(value, int)}
        return "".join(reverse.get(token_id, "?") for token_id in ids)


class EngineSafetyTests(unittest.TestCase):
    def test_fim_token_ids_only_returns_exact_single_tokens(self):
        self.assertEqual(_fim_token_ids(FakeTokenizer()), (5, 6, 7))

    def test_fim_stream_filter_handles_split_markers(self):
        chunks = ["before <|fim_", "prefix|> after ", "<|fim_middle|>", " done"]
        self.assertEqual("".join(_without_fim_markers(iter(chunks))), "before  after  done")

    def test_fim_stream_filter_preserves_incomplete_marker(self):
        self.assertEqual("".join(_without_fim_markers(iter(["text <|fim_pre"]))), "text <|fim_pre")

    def test_only_assistant_history_is_sanitized(self):
        messages = [
            {"role": "user", "content": "Explain <|fim_prefix|>"},
            {"role": "assistant", "content": "leaked <|fim_prefix|> marker"},
        ]
        clean = _strip_fim_from_messages(messages)
        self.assertEqual(clean[0]["content"], messages[0]["content"])
        self.assertEqual(clean[1]["content"], "leaked  marker")
        self.assertIs(clean[0], messages[0])
        self.assertIsNot(clean[1], messages[1])

    def test_fim_logits_are_blocked_after_other_processors(self):
        import mlx.core as mx

        processors = _build_logits_processors(
            logit_bias={5: 1000.0},
            repetition_penalty=None,
            presence_penalty=None,
            frequency_penalty=None,
            repetition_context_size=None,
            structured=None,
            blocked_token_ids=(5, 6),
        )
        logits = mx.zeros((1, 10))
        for processor in processors:
            logits = processor([], logits)
        values = logits.tolist()[0]
        self.assertTrue(math.isinf(values[5]) and values[5] < 0)
        self.assertTrue(math.isinf(values[6]) and values[6] < 0)

    def test_empty_thinking_pass_retries_and_combines_usage(self):
        calls = []

        def make_events(thinking):
            calls.append(thinking)
            if thinking:
                yield {"reasoning": "I should answer."}
                yield {
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
                }
            else:
                yield {"content": "Recovered answer"}
                yield {
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
                }

        events = list(_events_with_empty_content_retry(make_events, True, "test-model"))
        self.assertEqual(calls, [True, False])
        self.assertEqual(events[0], {"reasoning": "I should answer."})
        self.assertEqual(events[1], {"content": "Recovered answer"})
        self.assertEqual(
            events[2],
            {"usage": {"prompt_tokens": 22, "completion_tokens": 7, "total_tokens": 29}},
        )

    def test_content_or_tool_call_does_not_retry(self):
        for event in ({"content": "answer"}, {"tool_calls": [{"id": "1"}]}):
            calls = []

            def make_events(thinking):
                calls.append(thinking)
                yield event

            self.assertEqual(
                list(_events_with_empty_content_retry(make_events, True, "test-model")),
                [event],
            )
            self.assertEqual(calls, [True])

    def test_explicit_stop_can_disable_empty_retry(self):
        calls = []

        def make_events(thinking):
            calls.append(thinking)
            yield {"reasoning": "stopped"}

        self.assertEqual(
            list(
                _events_with_empty_content_retry(
                    make_events, True, "test-model", allow_retry=False
                )
            ),
            [{"reasoning": "stopped"}],
        )
        self.assertEqual(calls, [True])


if __name__ == "__main__":
    unittest.main()


class ContextBudgetTests(unittest.TestCase):
    """prompt + output must stay inside the window, or the KV cache walks past the
    Metal wired limit and the process dies rather than degrading."""

    def test_no_context_configured_leaves_max_tokens_alone(self):
        kw = {"max_tokens": 999999}
        self.assertIs(_fit_output(kw, 10, 0), kw)

    def test_room_to_spare_is_untouched(self):
        kw = {"max_tokens": 100}
        self.assertIs(_fit_output(kw, 500, 8192), kw)

    def test_output_is_clamped_to_what_is_left(self):
        # 8192 window - 8000 prompt - 256 margin leaves nothing sane, so pick a case with room
        out = _fit_output({"max_tokens": 4000}, 5000, 8192)
        self.assertEqual(out["max_tokens"], 8192 - 5000 - 256)

    def test_clamp_does_not_mutate_the_caller_dict(self):
        kw = {"max_tokens": 4000, "sampler": object()}
        out = _fit_output(kw, 5000, 8192)
        self.assertEqual(kw["max_tokens"], 4000)
        self.assertIs(out["sampler"], kw["sampler"])  # other keys carried through

    def test_prompt_filling_the_window_is_refused(self):
        # _fit_prompt always keeps the newest message, so one oversized message reaches
        # here untrimmed. Refusing beats handing it to prefill.
        with self.assertRaises(ContextOverflow):
            _fit_output({"max_tokens": 100}, 8192, 8192)

    def test_prompt_far_past_the_window_is_refused(self):
        with self.assertRaises(ContextOverflow):
            _fit_output({"max_tokens": 100}, 500000, 8192)


class FitPromptTests(unittest.TestCase):
    def test_returns_prompt_and_the_fitted_messages(self):
        msgs = [{"role": "user", "content": "a"}]
        prompt, fitted = _fit_prompt(msgs, lambda m: list(range(len(m))), len, 0)
        self.assertEqual(prompt, [0])
        self.assertEqual(fitted, msgs)

    def test_oldest_non_system_messages_are_dropped_until_it_fits(self):
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "old"},
            {"role": "user", "content": "mid"},
            {"role": "user", "content": "new"},
        ]
        # one "token" per message keeps the arithmetic obvious
        prompt, fitted = _fit_prompt(msgs, lambda m: list(range(len(m))), len, 2)
        self.assertEqual(len(prompt), 2)
        self.assertEqual([m["content"] for m in fitted], ["s", "new"])


class PrefixCacheTests(unittest.TestCase):
    """The snapshot is the only thing that survives generation, so its bookkeeping has to
    be exact: a cache we cannot describe token-for-token would silently corrupt output."""

    SIG = (8,)

    def setUp(self):
        self.cache = _PrefixCache()
        self.saved = {}
        # Stand in for the mlx-lm cache API. The fake cache records how many tokens it has
        # seen so a wrong split shows up as a wrong count rather than passing silently.
        self.patches = [
            mock.patch("mlx_lm.models.cache.make_prompt_cache",
                       lambda model, max_kv_size=None: {"n": 0}),
            mock.patch("mlx_lm.models.cache.save_prompt_cache",
                       lambda path, c: self.saved.update(n=c["n"])),
            mock.patch("mlx_lm.models.cache.load_prompt_cache",
                       lambda path: {"n": self.saved["n"]}),
            mock.patch("mlx_lazyserve.engine._prefill",
                       lambda model, cache, ids, kv: cache.update(n=cache["n"] + len(ids))),
        ]
        for p in self.patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self.patches])

    def start(self, history, prompt, sig=None, **kw):
        return self.cache.start(object(), history, prompt, sig or self.SIG, 8, **kw)

    def test_cold_start_prefills_everything_after_the_snapshot_point(self):
        prompt = list(range(1000))
        _, feed = self.start(prompt[:990], prompt)
        self.assertEqual(len(feed), 10)          # 990 snapshotted, 10 left for generation
        self.assertEqual(self.cache.tokens, prompt[:990])

    def test_a_strict_extension_reuses_the_snapshot(self):
        p1 = list(range(1000))
        self.start(p1[:990], p1)
        p2 = p1 + list(range(5000, 5100))
        cache, feed = self.start(p2[:1090], p2)
        # 990 restored + 100 prefilled to the new history boundary, 10 left to feed
        self.assertEqual(cache["n"], 1090)
        self.assertEqual(len(feed), 10)

    def test_divergence_inside_the_cached_span_starts_over(self):
        p1 = list(range(1000))
        self.start(p1[:990], p1)
        p2 = list(range(500)) + [999999] + list(range(6000, 6100))
        cache, feed = self.start(p2[:591], p2)
        self.assertEqual(cache["n"], 591)        # rebuilt from zero, not from 990
        self.assertEqual(len(feed), len(p2) - 591)

    def test_changing_kv_bits_invalidates_the_snapshot(self):
        # generate_step quantizes the cache in place, so a cache built at one kv_bits is
        # not the same object at another.
        p = list(range(1000))
        self.start(p[:990], p)
        cache, _ = self.start(p[:990], p, sig=(4,))
        self.assertEqual(cache["n"], 990)        # re-prefilled, not restored

    def test_prefix_shorter_than_the_floor_is_not_snapshotted(self):
        p = list(range(100))
        _, feed = self.start(p[:90], p)
        self.assertEqual(len(feed), 100)         # whole prompt fed, nothing cached
        self.assertEqual(self.cache.tokens, [])

    def test_identical_prompt_still_leaves_a_token_to_feed(self):
        # Reusing everything would hand stream_generate an empty prompt.
        p = list(range(1000))
        self.cache.tokens = list(p)
        self.cache.sig = self.SIG
        _, feed = self.start(p, p)
        self.assertTrue(feed)

    def test_invalidate_forces_a_full_prefill(self):
        p = list(range(1000))
        self.start(p[:990], p)
        self.cache.invalidate()
        cache, _ = self.start(p[:990], p)
        self.assertEqual(cache["n"], 990)

    def test_retry_render_must_not_overwrite_the_snapshot(self):
        # The empty-answer retry re-renders with thinking off; that template diverges at its
        # second token, so snapshotting it would poison every following request.
        p1 = list(range(1000))
        self.start(p1[:990], p1)
        fallback = list(range(2000, 2900))
        self.start(fallback[:890], fallback, snapshot=False)
        self.assertEqual(self.cache.tokens, p1[:990])
