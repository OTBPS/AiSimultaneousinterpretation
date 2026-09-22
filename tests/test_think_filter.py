"""ThinkFilter must survive tags split across arbitrary token boundaries."""
import pytest

from app.think_filter import ThinkFilter, strip_think


def feed_all(chunks):
    f = ThinkFilter()
    return "".join(f.feed(c) for c in chunks) + f.flush()


def test_empty_think_pair_removed():
    assert feed_all(["<think>", "", "</think>", "你好世界"]) == "你好世界"


def test_tag_split_across_chunks():
    # the exact failure mode a naive str.replace has
    assert feed_all(["<thi", "nk>", "</th", "ink>", "你好"]) == "你好"


def test_one_char_at_a_time():
    s = "<think></think>调度器执行上下文切换。"
    assert feed_all(list(s)) == "调度器执行上下文切换。"


def test_thinking_content_is_dropped():
    out = feed_all(["<think>", "Let me analyse this sentence", "</think>", "译文"])
    assert out == "译文"
    assert "analyse" not in out


def test_plain_text_passes_through():
    assert feed_all(["调度器", "执行", "上下文切换。"]) == "调度器执行上下文切换。"


def test_leading_label_stripped():
    assert feed_all(["翻译：", "你好"]) == "你好"
    assert feed_all(["<think></think>\n\n", "译文：", "你好"]) == "你好"


def test_angle_bracket_in_content_survives():
    assert feed_all(["a < b", " and c"]) == "a < b and c"


def test_strip_think_oneshot():
    assert strip_think("<think>\n\n</think>\n\n你好。") == "你好。"
    assert strip_think('"你好。"') == "你好。"


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7])
def test_arbitrary_chunk_sizes(n):
    s = "<think></think>这是一个测试句子。"
    chunks = [s[i:i + n] for i in range(0, len(s), n)]
    assert feed_all(chunks) == "这是一个测试句子。"
