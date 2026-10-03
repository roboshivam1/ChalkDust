"""NumberLineWalk: timing, schema refusals, and clean layout refusals.

Registry-wide layout invariants (examples clean, stress fits or refuses) live
in test_layout.py; this file pins what is specific to this component.
"""

from __future__ import annotations

import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.core.models import BeatSpec
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.number_line_walk import (
    NumberLineWalk,
    tick_values,
)
from chalkdust.validate.geometric import validate_beat

# One of every step kind, so every play() path is timed.
MIXED = {
    "range": [-5, 5],
    "steps": [
        {"at": -3, "label": "start"},
        {"to": 2},
        {"to": -1},
        {"interval": [0, None], "closed": [True, False], "label": "x ≥ 0"},
    ],
}


def _validate(params: dict):
    return validate_beat(BeatSpec(id="b01", narration="placeholder",
                                  component="NumberLineWalk", params=params))


class TestTiming:
    """The beat must last exactly as long as its audio (D-002), measured in
    rendered frames -- the thing that actually has to line up with the wav."""

    @pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
    def test_consumes_budget_to_the_frame(self, factor, tmp_path):
        comp = NumberLineWalk(MIXED)
        budget = comp.min_seconds() * factor
        fps = 15
        with tempconfig({"pixel_width": 128, "pixel_height": 72, "frame_rate": fps,
                         "media_dir": str(tmp_path), "write_to_movie": False,
                         "disable_caching": True, "verbosity": "WARNING",
                         "progress_bar": "none"}):
            scene = ChalkdustScene(comp, duration=budget)
            scene.render()
            elapsed = scene.renderer.time
            # Frames were really produced, so `time` counts frames rather than
            # summing requested run times (which would hide per-call rounding).
            assert not scene.renderer.skip_animations
        assert abs(elapsed - budget) <= 1 / fps, (elapsed, budget)


class TestSemanticHooks:
    def test_min_seconds_is_sum_of_phase_minimums(self):
        # intro 0.75 + mark 0.5 + 2 jumps 2.0 + interval 1.0 + hold 1.0
        assert NumberLineWalk(MIXED).min_seconds() == pytest.approx(5.25)

    def test_no_latex(self):
        assert NumberLineWalk(MIXED).latex_strings() == []


class TestSchema:
    """Inputs that cannot be drawn honestly are refused before anything builds."""

    @pytest.mark.parametrize("params", [
        pytest.param({"range": [0, 10], "steps": []}, id="empty-steps"),
        pytest.param({"range": [0, 10], "steps": [{"to": 3}]}, id="jump-before-mark"),
        pytest.param({"range": [0, 10], "steps": [{"at": 11}]}, id="out-of-range"),
        pytest.param({"range": [5, 5], "steps": [{"at": 5}]}, id="empty-range"),
        pytest.param({"range": [0, 100, 1], "steps": [{"at": 0}]}, id="too-many-ticks"),
        pytest.param({"range": [0, 1, 5], "steps": [{"at": 0}]}, id="too-few-ticks"),
        pytest.param({"range": [0, 10], "steps": [{"at": 3}, {"to": 3}]},
                     id="zero-length-jump"),
        pytest.param({"range": [0, 10], "steps": [{"interval": [None, None]}]},
                     id="interval-unbounded-both"),
        pytest.param({"range": [0, 10], "steps": [{"interval": [6, 2]}]},
                     id="interval-reversed"),
        pytest.param({"range": [0, 10],
                      "steps": [{"interval": [2, None], "closed": [True, True]}]},
                     id="closed-unbounded-end"),
        pytest.param({"range": [0, 10], "steps": [{"at": 1}] * 9}, id="too-many-steps"),
    ])
    def test_rejects(self, params):
        with pytest.raises(ValidationError):
            NumberLineWalk(params)


class TestLayout:
    def test_minimal_walk_validates_clean(self):
        report = _validate({"range": [0, 1], "steps": [{"at": 0}]})
        assert report.ok, f"\n{report}"

    def test_jump_too_small_to_draw_refuses_as_illegible(self):
        report = _validate({"range": [0, 1000], "steps": [{"at": 500}, {"to": 501}]})
        assert report.kinds() == {"illegible"}, f"\n{report}"

    def test_labels_that_cannot_clear_each_other_refuse_as_overflow(self):
        words = "a label far longer than any jump on a number line should carry"
        steps = [{"at": -9}] + [{"to": v, "label": words}
                                for v in (-4, 1, -2, 6, 9, -6, 3)]
        report = _validate({"range": [-10, 10], "steps": steps})
        assert report.kinds() == {"overflow"}, f"\n{report}"


class TestTicks:
    @pytest.mark.parametrize("rng,expected", [
        ([-2, 8], [float(v) for v in range(-2, 9)]),     # integer range: unit ticks
        ([0, 1], [k * 0.1 for k in range(11)]),          # fractional: 1/2/5 step
        ([0.5, 3.5, 1], [1.0, 2.0, 3.0]),                # explicit: multiples of it
    ])
    def test_tick_positions(self, rng, expected):
        assert tick_values(tuple(rng)) == pytest.approx(expected)
