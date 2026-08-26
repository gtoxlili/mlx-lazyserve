import unittest

from mlx_lazyserve.telegram import Channel, Incoming, TelegramBot

BOT_ID = 999


def bot() -> TelegramBot:
    """A bot with just the fields the quoting helpers touch — the real __init__ wants a
    live token, an HTTP client and a model manager, none of which these are about."""
    b = object.__new__(TelegramBot)
    b.bot_id = BOT_ID
    return b


def msg(**kw):
    return {"message_id": 1, **kw}


class QuotedContextTests(unittest.TestCase):
    """A group reply carries its subject in reply_to_message, delivered separately from the
    user's own text. Losing it means answering "这条什么意思" with no idea what 这条 is."""

    def test_plain_message_has_no_quote(self):
        self.assertIsNone(bot()._quoted_context(msg(text="hi")))

    def test_another_persons_message_is_quoted_with_their_name(self):
        out = bot()._quoted_context(msg(
            reply_to_message={"from": {"first_name": "张", "last_name": "三"}, "text": "原文"},
        ))
        self.assertIn("张 三 说", out)
        self.assertIn("原文", out)

    def test_username_is_used_when_there_is_no_display_name(self):
        out = bot()._quoted_context(msg(
            reply_to_message={"from": {"username": "someone"}, "text": "原文"},
        ))
        self.assertIn("someone", out)

    def test_a_hand_picked_quote_beats_the_whole_message(self):
        # Selecting part of a message narrows what is being asked about on purpose.
        out = bot()._quoted_context(msg(
            reply_to_message={"from": {"first_name": "张"}, "text": "很长的原文全部内容"},
            quote={"text": "全部内容"},
        ))
        self.assertIn("全部内容", out)
        self.assertNotIn("很长的原文", out)

    def test_caption_is_used_when_there_is_no_text(self):
        out = bot()._quoted_context(msg(
            reply_to_message={"from": {"first_name": "张"}, "caption": "图说"},
        ))
        self.assertIn("图说", out)

    def test_captionless_media_still_reports_that_something_was_pointed_at(self):
        out = bot()._quoted_context(msg(
            reply_to_message={"from": {"first_name": "张"}, "photo": [{"file_id": "x"}]},
        ))
        self.assertIn("[photo]", out)

    def test_a_reply_to_nothing_quotable_is_dropped(self):
        self.assertIsNone(bot()._quoted_context(msg(
            reply_to_message={"from": {"first_name": "张"}},
        )))

    def test_long_quotes_are_truncated(self):
        out = bot()._quoted_context(msg(
            reply_to_message={"from": {"first_name": "张"}, "text": "字" * 9000},
        ))
        self.assertLess(len(out), 4200)
        self.assertIn("已截断", out)

    def test_the_bots_own_message_is_tagged_as_such(self):
        out = bot()._quoted_context(msg(
            reply_to_message={"from": {"id": BOT_ID}, "text": "我说过的话"},
        ))
        self.assertTrue(out.startswith(TelegramBot.SELF_QUOTE))


class ComposeTurnTests(unittest.TestCase):
    def test_text_without_a_quote_is_unchanged(self):
        chan = Channel()
        self.assertEqual(bot()._compose_turn(Incoming(1, 2, 3, "问题"), chan), "问题")

    def test_a_quote_is_prefixed_to_the_users_text(self):
        chan = Channel()
        out = bot()._compose_turn(Incoming(1, 2, 3, "这啥意思", "[引用 · 张 说]\n原文"), chan)
        self.assertTrue(out.startswith("[引用 · 张 说]"))
        self.assertTrue(out.endswith("这啥意思"))

    def test_quoting_the_bots_last_line_back_at_it_is_dropped(self):
        # Replying to the bot is how you continue a thread; that text is already the previous
        # assistant turn, so re-quoting would pay for the same tokens twice.
        chan = Channel()
        chan.history = [{"role": "assistant", "content": "我说过的话"}]
        quoted = f"{TelegramBot.SELF_QUOTE}\n我说过的话"
        self.assertEqual(bot()._compose_turn(Incoming(1, 2, 3, "继续", quoted), chan), "继续")

    def test_quoting_an_older_bot_message_is_kept(self):
        # Only the newest assistant turn is guaranteed to still be in context.
        chan = Channel()
        chan.history = [{"role": "assistant", "content": "最近说的"}]
        quoted = f"{TelegramBot.SELF_QUOTE}\n很久以前说的另一段"
        out = bot()._compose_turn(Incoming(1, 2, 3, "这个", quoted), chan)
        self.assertIn("很久以前说的", out)


if __name__ == "__main__":
    unittest.main()
