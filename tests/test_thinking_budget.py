import unittest

import mlx.core as mx

from mlx_lazyserve.engine import _thinking_budget_processor


class FakeHFTokenizer:
    """Minimal stand-in: ThinkingBudgetCriteria only ever calls ``encode``."""

    IDS = {"<think>": [1], "</think>": [2], "\n": [3]}

    def encode(self, text, add_special_tokens=False):
        return self.IDS[text]


class FakeWrapper:
    """Shaped like mlx-lm's TokenizerWrapper: real tokenizer under ``_tokenizer``."""

    think_start = "<think>"
    think_end = "</think>"

    def __init__(self):
        self._tokenizer = FakeHFTokenizer()


def logits(n=8):
    return mx.arange(n, dtype=mx.float32)[None] * 0.0 + 1.0


def forced_id(out):
    """The single id a collapsed distribution leaves reachable, or None if untouched."""
    finite = [i for i, v in enumerate(out[0].tolist()) if v != float("-inf")]
    return finite[0] if len(finite) == 1 else None


class DisabledCasesTests(unittest.TestCase):
    def test_no_processor_when_thinking_is_off(self):
        self.assertIsNone(_thinking_budget_processor(FakeWrapper(), 10, False, "<think>"))

    def test_no_processor_when_budget_is_zero(self):
        self.assertIsNone(_thinking_budget_processor(FakeWrapper(), 0, True, "<think>"))

    def test_no_processor_when_budget_is_negative(self):
        self.assertIsNone(_thinking_budget_processor(FakeWrapper(), -1, True, "<think>"))


class BudgetEnforcementTests(unittest.TestCase):
    BUDGET = 4

    def _run(self, prompt, n_steps, token=9):
        """Drive the processor like generate_step does and report when it forces a token."""
        proc = _thinking_budget_processor(FakeWrapper(), self.BUDGET, True, prompt)
        self.assertIsNotNone(proc)
        tokens = mx.array([7, 7, 7])  # prompt tail, as handed to the first call
        out = []
        for _ in range(n_steps):
            out.append(forced_id(proc(tokens, logits())))
            tokens = mx.concat([tokens, mx.array([token])])
        return out

    def test_forces_think_end_once_the_budget_is_spent(self):
        # Prompt pre-opens the block, so every generated token counts against the budget.
        forced = self._run("...<think>\n", n_steps=9)
        self.assertNotIn(2, forced[: self.BUDGET])  # untouched while under budget
        self.assertIn(2, forced)  # </think> forced once over it

    def test_the_first_call_only_anchors(self):
        # It lands on the prompt tail; consuming it would let a historical </think> in the
        # conversation flip the criteria's state before generation even starts.
        self.assertIsNone(self._run("...<think>\n", n_steps=1)[0])

    def test_closed_thinking_prompt_does_not_count_generated_tokens(self):
        # A prompt whose last block is already closed means generation starts outside
        # thinking, so a plain answer must never be truncated.
        self.assertEqual(self._run("<think>x</think> answer", n_steps=9).count(2), 0)

    def test_forced_token_is_the_only_reachable_one(self):
        out = [f for f in self._run("...<think>\n", n_steps=9) if f is not None]
        self.assertTrue(out, "expected the cap to fire")
        self.assertEqual(set(out) - {2, 3}, set())  # only "\n" and "</think>" are ever forced


class TokenIdPromptTests(unittest.TestCase):
    """apply_chat_template returns ids for some models, text for others."""

    def test_preopen_detected_from_token_ids(self):
        proc = _thinking_budget_processor(FakeWrapper(), 2, True, [5, 5, 1])  # ...<think>
        tokens = mx.array([5, 5, 1])
        seen = []
        for _ in range(8):
            seen.append(forced_id(proc(tokens, logits())))
            tokens = mx.concat([tokens, mx.array([9])])
        self.assertIn(2, seen)

    def test_closed_block_in_token_ids_is_not_treated_as_open(self):
        proc = _thinking_budget_processor(FakeWrapper(), 2, True, [1, 9, 2])  # <think>x</think>
        tokens = mx.array([1, 9, 2])
        seen = []
        for _ in range(8):
            seen.append(forced_id(proc(tokens, logits())))
            tokens = mx.concat([tokens, mx.array([9])])
        self.assertNotIn(2, seen)


if __name__ == "__main__":
    unittest.main()
