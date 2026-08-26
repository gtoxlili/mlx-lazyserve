import unittest

from mlx_lazyserve.firecrawl import _clean_markdown, _prune_markdown


class PruneMarkdownTests(unittest.TestCase):
    """Scraped pages arrive with site chrome intact; prefill runs ~65 tok/s, so every
    kilobyte of navigation is seconds the user waits before seeing an answer."""

    def test_a_navigation_row_is_dropped(self):
        nav = "[首页](http://a) [预报](http://b) [预警](http://c) [雷达](http://d)"
        self.assertEqual(_prune_markdown(nav), "")

    def test_a_sentence_with_links_is_kept(self):
        line = "广州今天[晴](http://a)，最高 34 度"
        self.assertIn("广州今天晴，最高 34 度", _prune_markdown(line))

    def test_link_text_survives_but_the_url_does_not(self):
        self.assertEqual(_prune_markdown("见[中央气象台](http://www.nmc.cn/x?a=1)发布"),
                         "见中央气象台发布")

    def test_repeated_lines_collapse(self):
        self.assertEqual(_prune_markdown("加载中...\n加载中...\n正文"), "加载中...\n正文")

    def test_short_labels_are_preserved(self):
        # The rejected heuristic — dropping short label-only lines — saved another ~50% and
        # would have deleted exactly these, which on a weather page are the answer.
        out = _prune_markdown("30/\n26℃\n多云\n东风\n<3级")
        for token in ("26℃", "多云", "东风"):
            self.assertIn(token, out)

    def test_pruning_runs_inside_the_normal_clean(self):
        md = "![x](http://img)\n[a](http://1) [b](http://2) [c](http://3)\n真正的正文"
        out = _clean_markdown(md)
        self.assertIn("真正的正文", out)
        self.assertNotIn("http", out)

    def test_empty_input_is_safe(self):
        self.assertEqual(_clean_markdown(""), "")
