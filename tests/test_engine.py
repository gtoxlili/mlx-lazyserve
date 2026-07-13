import math
import unittest

from mlx_lazyserve.engine import (
    _build_logits_processors,
    _events_with_empty_content_retry,
    _fim_token_ids,
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
