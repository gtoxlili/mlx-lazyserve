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


class ChatHistoryStoreTests(unittest.TestCase):
    """Storage for messages nobody addressed to the bot — the ones that make
    "what did the group decide about X" answerable at all."""

    def setUp(self):
        import os
        import tempfile

        from mlx_lazyserve.telegram import HistoryStore
        self.store = HistoryStore(os.path.join(tempfile.mkdtemp(), "t.db"))

    def add(self, mid, uid, name, uname, text, ts=1000, cap=1000):
        self.store.log_group(1, mid, uid, name, uname, text, ts, cap)

    def test_the_same_message_is_never_logged_twice(self):
        self.add(1, 11, "张三", "zhangsan", "早")
        self.add(1, 11, "张三", "zhangsan", "早")
        self.assertEqual(len(self.store.recent(1, None, 10)), 1)

    def test_search_spans_the_whole_group(self):
        self.add(1, 11, "张三", "zhangsan", "方案要分两期")
        self.add(2, 22, "李四", "lisi", "方案预算不够")
        self.assertEqual(len(self.store.search(1, ["方案"], None, 10)), 2)

    def test_search_can_be_limited_to_one_person(self):
        self.add(1, 11, "张三", "zhangsan", "方案要分两期")
        self.add(2, 22, "李四", "lisi", "方案预算不够")
        rows = self.store.search(1, ["方案"], 11, 10)
        self.assertEqual([r[1] for r in rows], ["方案要分两期"])

    def test_more_matching_terms_ranks_higher(self):
        self.add(1, 11, "张三", "zhangsan", "只提到方案")
        self.add(2, 11, "张三", "zhangsan", "方案和排期都定了")
        rows = self.store.search(1, ["方案", "排期"], None, 1)
        self.assertEqual(rows[0][1], "方案和排期都定了")

    def test_an_old_match_beats_a_recent_non_match(self):
        # The point of searching is reaching past the recency window.
        self.add(1, 11, "张三", "zhangsan", "关于方案的老发言")
        for i in range(2, 40):
            self.add(i, 11, "张三", "zhangsan", f"闲聊{i}")
        self.assertEqual([r[1] for r in self.store.search(1, ["方案"], None, 5)],
                         ["关于方案的老发言"])

    def test_two_character_chinese_terms_match(self):
        # Why this is a scan and not FTS5: trigram needs 3+ characters and would silently
        # miss most Chinese words.
        self.add(1, 11, "张三", "zhangsan", "这个方案可以")
        self.assertTrue(self.store.search(1, ["方案"], None, 5))

    def test_like_wildcards_in_a_query_are_literals(self):
        self.add(1, 11, "张三", "zhangsan", "AxB")
        self.assertFalse(self.store.search(1, ["A_B"], None, 5))

    def test_recent_returns_oldest_first(self):
        for i in range(1, 6):
            self.add(i, 11, "张三", "zhangsan", f"m{i}")
        self.assertEqual([r[1] for r in self.store.recent(1, None, 3)], ["m3", "m4", "m5"])

    def test_the_log_is_pruned_to_the_cap(self):
        for i in range(1, 31):
            self.add(i, 11, "张三", "zhangsan", f"m{i}", cap=10)
        rows = self.store.recent(1, None, 50)
        self.assertLessEqual(len(rows), 10)
        self.assertEqual(rows[-1][1], "m30")

    def test_a_person_resolves_by_handle_or_display_name(self):
        self.add(1, 11, "张三", "zhangsan", "早")
        self.assertEqual(self.store.resolve_name(1, "zhangsan"), 11)
        self.assertEqual(self.store.resolve_name(1, "ZhangSan"), 11)
        self.assertEqual(self.store.resolve_name(1, "张三"), 11)
        self.assertIsNone(self.store.resolve_name(1, "查无此人"))


class ChatSearchToolTests(unittest.IsolatedAsyncioTestCase):
    """The tool is what the model actually reaches for, so its argument handling has to
    survive whatever the model puts in the JSON."""

    def setUp(self):
        import os
        import tempfile
        from types import SimpleNamespace

        from mlx_lazyserve.telegram import HistoryStore
        self.b = bot()
        self.b._store = HistoryStore(os.path.join(tempfile.mkdtemp(), "t.db"))
        self.b.settings = SimpleNamespace(tg_recall_chars=1200, tg_group_log_cap=50000)
        self.b._store.log_group(1, 1, 11, "张三", "zhangsan", "方案分两期做", 1000, 100)
        self.b._store.log_group(1, 2, 22, "李四", "lisi", "预算只有一半", 1001, 100)

    async def test_a_keyword_query_finds_the_message(self):
        out = await self.b._chat_history_search(1, {"query": "方案"})
        self.assertIn("方案分两期做", out)
        self.assertIn("张三", out)

    async def test_person_narrows_the_search(self):
        out = await self.b._chat_history_search(1, {"query": "", "person": "@lisi"})
        self.assertIn("预算只有一半", out)
        self.assertNotIn("方案分两期做", out)

    async def test_an_unknown_person_says_so_instead_of_guessing(self):
        out = await self.b._chat_history_search(1, {"query": "方案", "person": "查无此人"})
        self.assertIn("没有找到", out)

    async def test_an_empty_query_falls_back_to_recent_messages(self):
        out = await self.b._chat_history_search(1, {"query": ""})
        self.assertIn("预算只有一半", out)

    async def test_a_junk_limit_does_not_blow_up(self):
        out = await self.b._chat_history_search(1, {"query": "方案", "limit": "很多"})
        self.assertIn("方案分两期做", out)

    async def test_no_match_is_reported_plainly(self):
        self.assertIn("没有匹配", await self.b._chat_history_search(1, {"query": "量子计算"}))


class ToolAvailabilityTests(unittest.TestCase):
    def make(self, has_store=True, cap=50000, fc=None):
        from types import SimpleNamespace
        b = bot()
        b._store = object() if has_store else None
        b._fc = fc
        b.settings = SimpleNamespace(tg_group_log_cap=cap, tg_recall_chars=1200)
        return b

    def names(self, b, chat_id):
        return [t["function"]["name"] for t in (b._tools_for(chat_id) or [])]

    def test_groups_get_the_chat_history_tool(self):
        self.assertIn("chat_history_search", self.names(self.make(), -1001))

    def test_private_chats_do_not(self):
        # Nothing writes group_log for a private chat, so the tool could only answer
        # "nothing found" — and each wasted round costs a full generate.
        self.assertNotIn("chat_history_search", self.names(self.make(), 584544685))

    def test_it_survives_web_tools_being_off(self):
        # Searching what the group already said needs no network.
        self.assertEqual(self.names(self.make(fc=None), -1001), ["chat_history_search"])

    def test_disabling_the_group_log_removes_the_tool(self):
        self.assertIsNone(self.make(cap=0, fc=None)._tools_for(-1001))


class StreamTests(unittest.IsolatedAsyncioTestCase):
    """Telegram has no streaming API — this is one message edited on a throttle, so the
    throttle and the backoff are the whole mechanism."""

    def make(self):
        from mlx_lazyserve.telegram import TelegramBot
        b = bot()
        self.calls = []

        async def api_quiet(method, **kw):
            self.calls.append((method, kw))
            return {"message_id": 7}

        async def stream_edit(chat_id, mid, text):
            self.calls.append(("edit", {"text": text}))
            return self.edit_ok

        b._api_quiet = api_quiet
        b._stream_edit = stream_edit
        self.edit_ok = True
        return TelegramBot._Stream(b, -100, 5)

    async def test_the_first_push_sends_a_message(self):
        st = self.make()
        await st.push("你好")
        self.assertEqual(self.calls[0][0], "sendMessage")
        self.assertEqual(st.mid, 7)

    async def test_edits_inside_the_interval_are_skipped(self):
        st = self.make()
        await st.push("一")
        await st.push("一二")          # immediately after — throttled away
        self.assertEqual([c[0] for c in self.calls], ["sendMessage"])

    async def test_an_edit_lands_once_the_interval_has_passed(self):
        st = self.make()
        await st.push("一")
        st.last = 0.0                  # pretend the interval elapsed
        await st.push("一二")
        self.assertEqual(self.calls[-1], ("edit", {"text": "一二"}))

    async def test_identical_text_never_costs_a_request(self):
        st = self.make()
        await st.push("一")
        st.last = 0.0
        await st.push("一")
        self.assertEqual(len(self.calls), 1)

    async def test_a_refused_edit_widens_the_interval(self):
        st = self.make()
        await st.push("一")
        st.last = 0.0
        self.edit_ok = False
        before = st.interval
        await st.push("一二")
        self.assertGreater(st.interval, before)
        self.assertEqual(st.shown, "一")   # not recorded as shown, so it retries later

    async def test_outgrowing_one_message_stops_streaming(self):
        # Past Telegram's per-message limit the live preview gives up and the final,
        # properly split send takes over.
        st = self.make()
        await st.push("x" * 9000)
        self.assertTrue(st.overflow)
        self.assertEqual(self.calls, [])

    async def test_drop_deletes_the_live_message(self):
        st = self.make()
        await st.push("一")
        await st.drop()
        self.assertEqual(self.calls[-1][0], "deleteMessage")
        self.assertIsNone(st.mid)

    async def test_drop_without_a_message_is_a_no_op(self):
        st = self.make()
        await st.drop()
        self.assertEqual(self.calls, [])
