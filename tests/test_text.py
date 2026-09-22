"""Sentence splitting, glossary matching and context recycling."""
import pytest

from app.context import Context
from app.glossary import Glossary
from app.sentence_split import split_sentences


# ------------------------------------------------------------- splitting
def test_splits_on_terminators():
    got = split_sentences("This is one. And this is two! A third?")
    assert got == ["This is one.", "And this is two!", "A third?"]


def test_does_not_split_abbreviations():
    got = split_sentences("Dr. Smith joined us. He works on caches.")
    assert got == ["Dr. Smith joined us.", "He works on caches."]


def test_does_not_split_decimals():
    got = split_sentences("It rose by 12.5 percent last quarter.")
    assert len(got) == 1


def test_merges_tiny_fragments():
    got = split_sentences("Yes. No. Absolutely not the plan we agreed on.")
    assert all(len(s) >= 12 for s in got)


def test_hard_wraps_unpunctuated_wall():
    text = "word " * 120
    got = split_sentences(text, max_chars=100)
    assert len(got) > 1
    assert all(len(s) <= 100 for s in got)


def test_empty_input():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


# -------------------------------------------------------------- glossary
@pytest.fixture
def gloss(tmp_path):
    p = tmp_path / "g.tsv"
    p.write_text("# comment\nprefill\t预填充\nboil the ocean\t好高骛远\n"
                 "decode\t解码\n", encoding="utf-8")
    return Glossary(p)


def test_glossary_loads(gloss):
    assert len(gloss) == 3


def test_matches_case_insensitively(gloss):
    assert gloss.match("The Prefill stage") == {"Prefill": "预填充"}


def test_respects_word_boundaries(gloss):
    assert gloss.match("prefilling the buffer") == {}


def test_prefers_longest_match(gloss):
    hits = gloss.match("do not boil the ocean here")
    assert "boil the ocean" in hits


def test_prompt_line_only_for_hits(gloss):
    assert gloss.prompt_line("nothing relevant") == ""
    line = gloss.prompt_line("prefill and decode")
    assert line.startswith("[Terms] ")
    assert "预填充" in line and "解码" in line


def test_hot_reload(gloss, tmp_path):
    import os
    import time
    p = gloss.path
    time.sleep(0.01)
    p.write_text("prefill\t预填充\nlatency\t延迟\n", encoding="utf-8")
    os.utime(p, None)
    assert gloss.reload() is True
    assert "latency" in {k.lower() for k in gloss.match("latency matters")}


def test_disabled_glossary_matches_nothing(tmp_path):
    p = tmp_path / "g.tsv"
    p.write_text("prefill\t预填充\n", encoding="utf-8")
    g = Glossary(p, enabled=False)
    assert g.match("prefill") == {}


# --------------------------------------------------------------- context
def test_context_keeps_last_n_pairs():
    c = Context(pairs=2, reset_every=99)
    for i in range(5):
        c.add(f"en{i}", f"zh{i}")
    assert [en for en, _ in c.recent()] == ["en3", "en4"]


def test_context_reset_cycle():
    c = Context(pairs=3, reset_every=3)
    for i in range(2):
        c.add(f"en{i}", f"zh{i}")
    assert not c.needs_reset()
    c.add("en2", "zh2")
    assert c.needs_reset()
    c.mark_reset()
    assert not c.needs_reset()


def test_preamble_empty_when_no_history():
    assert Context().preamble() == ""


def test_preamble_contains_recent_pairs():
    c = Context(pairs=3)
    c.add("hello", "你好")
    assert "你好" in c.preamble()
