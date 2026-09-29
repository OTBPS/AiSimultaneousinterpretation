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


# ---------------------------------------------- unclosed <think> (the big one)
#
# The model very often emits the opening tag and then the answer, never
# closing it:   "<think>\n\n请看下一张。"
# Treating <think> as "suppress until </think>" discarded the whole
# translation, and the subtitle rendered its English source with nothing
# under it. 20 of 28 sample utterances were lost this way.
def test_unclosed_think_still_yields_the_answer():
    assert feed_all(["<think>\n\n请看下一张。"]) == "请看下一张。"


def test_unclosed_think_split_across_chunks():
    assert feed_all(["<thi", "nk>", "\n\n", "以及第三，", "招聘计划。"]) \
        == "以及第三，招聘计划。"


def test_unclosed_think_streams_rather_than_waiting_for_flush():
    """The answer must appear as it arrives, not all at once at the end."""
    f = ThinkFilter()
    seen = f.feed("<think>\n\n嗯，B是")
    assert "嗯，B是" in seen, "answer should stream out of an unclosed block"


def test_closed_english_reasoning_is_still_discarded():
    """A properly closed block really is reasoning; it must not leak.

    English is what Qwen3 actually produces when it reasons on this prompt.
    """
    out = feed_all(["<think>", "Okay, let me parse this sentence",
                    "</think>", "你好"])
    assert out == "你好"
    assert "Okay" not in out


def test_known_limitation_chinese_reasoning_can_leak():
    """Documented, accepted, and impossible to fix while streaming.

    When the first non-whitespace character inside the block is Han, this
    filter cannot yet know whether a close tag is coming -- "<think>

你好"
    (answer, never closed) and "<think>你好吗</think>" (Chinese reasoning) are
    identical at that instant. The rule favours the case that occurs in
    practice: /no_think makes the block empty, and reasoning comes in English.

    If this ever starts mattering, the fix is to stop streaming out of think
    blocks and decide at flush -- at the cost of first-token latency on the
    ~70% of lines that arrive with no closing tag.
    """
    out = feed_all(["<think>", "你好吗", "</think>", "早上好"])
    assert "你好吗" in out          # leaks, by design


def test_truncated_english_reasoning_is_not_shown_as_a_subtitle():
    """Unclosed AND no Han: genuine reasoning cut off by max_new_tokens.

    Showing that as the translation would be worse than showing nothing.
    """
    assert feed_all(["<think>", "Okay, the user wants me to translate"]) == ""


@pytest.mark.parametrize("n", [1, 3, 7])
def test_unclosed_think_at_any_chunking(n):
    s = "<think>\n\n阶段间的复制。"
    chunks = [s[i:i + n] for i in range(0, len(s), n)]
    assert feed_all(chunks) == "阶段间的复制。"
