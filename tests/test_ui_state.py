"""Persisted window geometry.

`clamp_to` is the interesting part. Restoring a saved position blindly is a
good way to produce a window that runs but cannot be seen -- drag it to a
second monitor, unplug that monitor, restart -- and "runs but invisible" is
indistinguishable from "failed to launch" for the user.
"""
import json

from app.ui_state import MAX_FONT, MIN_FONT, MIN_HEIGHT, MIN_WIDTH, UiState

SCREEN = (0, 0, 1920, 1080)


def test_missing_file_gives_defaults(tmp_path):
    st = UiState.load(tmp_path / "nope.json")
    assert not st.has_geometry
    assert st.font_size is None


def test_corrupt_file_does_not_crash(tmp_path):
    p = tmp_path / "ui_state.json"
    p.write_text("{not json at all", encoding="utf-8")
    assert not UiState.load(p).has_geometry


def test_unknown_keys_are_ignored(tmp_path):
    """An older or newer build must not crash on an unexpected field."""
    p = tmp_path / "ui_state.json"
    p.write_text(json.dumps({"x": 10, "y": 20, "width": 800, "height": 200,
                             "font_size": 30, "from_the_future": True}),
                 encoding="utf-8")
    st = UiState.load(p)
    assert (st.x, st.y, st.width, st.height) == (10, 20, 800, 200)


def test_roundtrip(tmp_path):
    p = tmp_path / "ui_state.json"
    UiState(100, 200, 900, 250, 28).save(p)
    st = UiState.load(p)
    assert (st.x, st.y, st.width, st.height, st.font_size) == \
        (100, 200, 900, 250, 28)


def test_onscreen_geometry_is_left_alone():
    st = UiState(300, 400, 900, 200, 24).clamp_to(*SCREEN)
    assert (st.x, st.y, st.width, st.height) == (300, 400, 900, 200)


def test_window_saved_offscreen_comes_back():
    """The monitor it was on is gone; it must land somewhere reachable."""
    st = UiState(5000, 3000, 900, 200, 24).clamp_to(*SCREEN)
    assert st.x < 1920 and st.y < 1080
    # at least a grabbable sliver remains on screen
    assert st.x + st.width > 0 and st.y + st.height > 0


def test_negative_position_is_pulled_back():
    st = UiState(-4000, -900, 900, 200, 24).clamp_to(*SCREEN)
    assert st.x + st.width >= 80
    assert st.y >= 0


def test_oversized_window_is_capped_to_the_screen():
    st = UiState(0, 0, 99999, 99999, 24).clamp_to(*SCREEN)
    assert st.width <= 1920 and st.height <= 1080


def test_degenerate_size_is_raised_to_the_minimum():
    st = UiState(100, 100, 1, 1, 24).clamp_to(*SCREEN)
    assert st.width >= MIN_WIDTH and st.height >= MIN_HEIGHT


def test_font_is_clamped_both_ways():
    assert UiState(font_size=2).clamped_font() == MIN_FONT
    assert UiState(font_size=500).clamped_font() == MAX_FONT
    assert UiState(font_size=24).clamped_font() == 24
    assert UiState().clamped_font() is None


def test_clamp_without_geometry_is_a_noop():
    st = UiState(font_size=24)
    assert st.clamp_to(*SCREEN) is st


def test_save_to_unwritable_path_does_not_raise(tmp_path):
    """Losing window position must never take the app down with it."""
    UiState(1, 2, 3, 4, 5).save(tmp_path / "no" / "such" / "dir" / "s.json")
