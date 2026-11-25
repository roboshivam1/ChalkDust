"""Cue resolution (core/cues.py) and the whole-frame moment windows
(scenes/base.moment_windows) that animation components are timed by."""

from __future__ import annotations

import itertools

import pytest

from chalkdust.core.cues import (
    anchor_tokens,
    cue_problems,
    cue_times,
    find_phrase,
    tokenize,
    word_times,
)
from chalkdust.core.models import WordTiming
from chalkdust.scenes.base import MIN_MOMENT_SECONDS, Moment, moment_windows

N = ("Here is a curve. Pick a second point on it, and draw the line "
     "through both. Now slide the second point closer, and watch the line turn.")
MOMENTS = ["pose", "second", "slide"]


def test_tokens_drop_punctuation_and_count_sentences():
    toks = tokenize("Slide it, closer! Then 3.5 units. Can't stop")
    assert [t.key for t in toks] == ["slide", "it", "closer", "then", "35", "units", "cant", "stop"]
    assert [t.sentence for t in toks] == [0, 0, 0, 1, 1, 1, 2, 2]


def test_find_phrase_ignores_case_and_punctuation():
    toks = tokenize(N)
    assert find_phrase(toks, "Slide the SECOND point") == [t.key for t in toks].index("slide")
    assert find_phrase(toks, "second point", start=8) is not None
    assert find_phrase(toks, "tangent") is None


def test_cues_in_order_pass():
    assert cue_problems(N, MOMENTS, {"second": "second point on", "slide": "slide"}) == []


def test_cue_not_in_narration_is_refused_with_the_phrase():
    (p,) = cue_problems(N, MOMENTS, {"slide": "tangent line"})
    assert p.moment == "slide" and "'tangent line' is not in the narration" in p.message


def test_cue_out_of_order_is_refused():
    (p,) = cue_problems(N, MOMENTS, {"second": "watch the line", "slide": "pick a second"})
    assert "spoken before" in p.message


def test_cue_for_unknown_moment_is_refused():
    (p,) = cue_problems(N, MOMENTS, {"zoom": "curve"})
    assert "names no moment" in p.message


def test_cues_too_close_are_refused():
    (p,) = cue_problems(N, MOMENTS, {"second": "pick", "slide": "a second point"})
    assert "word(s) after" in p.message


def test_without_cues_sentences_anchor_moments():
    a = anchor_tokens(N, MOMENTS, {})
    keys = [t.key for t in tokenize(N)]
    assert a == {"pose": None, "second": keys.index("pick"), "slide": keys.index("now")}


def test_too_few_sentences_leave_moments_to_weights():
    assert anchor_tokens("One sentence only, no more", MOMENTS, {}) == {m: None for m in MOMENTS}


def test_word_times_follow_the_voice_and_interpolate_splits():
    text = "We can't stop now"
    # A voice that splits "can't" into two tokens and times every word.
    words = [WordTiming(word="We", start=0.1, end=0.3), WordTiming(word="ca", start=0.3, end=0.45),
             WordTiming(word="n't", start=0.45, end=0.6), WordTiming(word="stop", start=0.8, end=1.1),
             WordTiming(word="now", start=1.2, end=1.5)]
    t = word_times(text, words, 1.6)
    assert t[0] == 0.1 and t[2] == 0.8 and t[3] == 1.2
    assert 0.1 < t[1] < 0.8  # "can't" interpolated between its matched neighbours


def test_word_times_estimate_from_characters_without_a_voice():
    t = word_times("aaaa bbbb", [], 9.0)
    assert t == [0.0, 5.0]


def test_cue_times_use_the_cue_word():
    words = [WordTiming(word=w.text, start=0.4 * i, end=0.4 * i + 0.3)
             for i, w in enumerate(tokenize(N))]
    keys = [t.key for t in tokenize(N)]
    got = cue_times(N, MOMENTS, {"slide": "watch the line"}, words, 20.0)
    assert got == {"slide": pytest.approx(0.4 * keys.index("watch"))}


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_windows_always_sum_to_the_beat(fps):
    moments = [Moment("a", 1, 1.0), Moment("b", 2), Moment("c", 1, 0.5), Moment("d", 3)]
    for total in (fps * 3, fps * 9 + 7, fps * 25):
        for cut in itertools.product([None, 0, total // 3, total // 2, total - 1, total + 50],
                                     repeat=2):
            cues = {k: v for k, v in zip(("b", "d"), cut) if v is not None}
            lead, wins = moment_windows(moments, cues, total, fps)
            assert lead + sum(w.play + w.hold for w in wins) == total
            assert all(w.play >= 1 and w.hold >= 0 for w in wins)
            starts = [w.start for w in wins]
            assert starts == sorted(starts) and starts[0] == lead


def test_a_cued_moment_starts_on_its_frame():
    lead, wins = moment_windows([Moment("a"), Moment("b"), Moment("c")], {"b": 90, "c": 200}, 270, 30)
    assert lead == 0 and [w.start for w in wins] == [0, 90, 200]


def test_cues_too_close_are_pushed_to_the_floor_not_refused():
    fps = 30
    floor = -(-MIN_MOMENT_SECONDS * fps // 1)
    lead, wins = moment_windows([Moment("a"), Moment("b")], {"a": 30, "b": 31}, 270, fps)
    assert lead == 30 and wins[1].start - wins[0].start == floor


def test_motion_shorter_than_window_holds_the_rest():
    _, wins = moment_windows([Moment("a", 1, motion=1.0), Moment("b", 1)], {}, 300, 30)
    assert wins[0].play == 30 and wins[0].hold == 120
