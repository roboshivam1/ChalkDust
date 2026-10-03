"""NumberLineWalk: timing, schema refusals, and clean layout refusals.

Registry-wide layout invariants (examples clean, stress fits or refuses) live
in test_layout.py; this file pins what is specific to this component.
"""

from __future__ import annotations

import numpy as np
import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.core.models import BeatSpec
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.number_line_walk import (
    NumberLineWalk,
    tick_values,
)
from chalkdust.scenes.regions import LayoutError
from chalkdust.validate.geometric import LayoutProbe, validate_beat

# One of every step kind, so every play() path is timed.
MIXED = {
    "range": [-5, 5],
    "steps": [
        {"at": -3, "label": "start"},
        {"to": 2},
        {"to": -1},
        {"at": 4, "label": "reset"},   # second mark: walker slides, no new walker
        {"interval": [0, None], "closed": [True, False], "label": "x ≥ 0"},
    ],
}


FPS = 15
# intro + one phase per step + final hold
MIXED_PHASES = 1 + len(MIXED["steps"]) + 1


def _validate(params: dict):
    return validate_beat(BeatSpec(id="b01", narration="placeholder",
                                  component="NumberLineWalk", params=params))


def _render(params: dict, budget: float, tmp_path) -> float:
    """Render for real (frames produced, nothing written); return the elapsed
    scene time."""
    with tempconfig({"pixel_width": 128, "pixel_height": 72, "frame_rate": FPS,
                     "media_dir": str(tmp_path), "write_to_movie": False,
                     "disable_caching": True, "verbosity": "WARNING",
                     "progress_bar": "none"}):
        scene = ChalkdustScene(NumberLineWalk(params), duration=budget)
        scene.render()
        # Frames were really produced, so `time` counts frames rather than
        # summing requested run times (which would hide per-call rounding).
        assert not scene.renderer.skip_animations
        return scene.renderer.time


class TestTiming:
    """The beat must last exactly as long as its audio (D-002), measured in
    rendered frames -- the thing that actually has to line up with the wav."""

    @pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
    def test_consumes_budget_to_the_frame(self, factor, tmp_path):
        budget = NumberLineWalk(MIXED).min_seconds() * factor
        elapsed = _render(MIXED, budget, tmp_path)
        assert abs(elapsed - budget) <= 1 / FPS, (elapsed, budget)

    @pytest.mark.parametrize("frames", [MIXED_PHASES, MIXED_PHASES + 1],
                             ids=["one-frame-each", "a-phase-rounds-to-zero"])
    def test_phases_that_round_to_nothing_still_get_a_frame(self, frames, tmp_path):
        # At 8 frames the cumulative rounding gives the second mark 0 frames,
        # and Manim rejects run_time 0 with a bare ValueError. Every phase must
        # get at least one frame and the beat still end on its budget.
        budget = frames / FPS
        elapsed = _render(MIXED, budget, tmp_path)
        assert abs(elapsed - budget) <= 1 / FPS, (elapsed, budget)

    def test_budget_below_one_frame_per_phase_refuses_typed(self, tmp_path):
        # 0.05x min_seconds is 4 frames for 7 phases: nothing honest can be
        # shown, and the refusal must be a LayoutError the repair loop reads.
        budget = NumberLineWalk(MIXED).min_seconds() * 0.05
        with pytest.raises(LayoutError) as exc:
            _render(MIXED, budget, tmp_path)
        assert exc.value.kind == "overflow"


class TestSemanticHooks:
    def test_min_seconds_is_sum_of_phase_minimums(self):
        # intro 0.75 + 2 marks 1.0 + 2 jumps 2.0 + interval 1.0 + hold 1.0
        assert NumberLineWalk(MIXED).min_seconds() == pytest.approx(5.75)

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

    def test_walk_too_tangled_for_own_arc_labels_refuses_as_overflow(self):
        # Back and forth over one span, five jumps: the arcs cannot each carry
        # their own label inside the stage, so the beat refuses rather than
        # stacking a label over an arc it does not belong to.
        steps = [{"at": 0, "label": "start"}, {"to": 7}, {"to": -3}, {"to": 4},
                 {"to": -8}, {"to": 9}, {"interval": [-3, 4]}, {"to": 2}]
        report = _validate({"range": [-10, 10], "steps": steps})
        assert report.kinds() == {"overflow"}, f"\n{report}"

    def test_labels_that_cannot_clear_each_other_refuse_as_overflow(self):
        words = "a label far longer than any jump on a number line should carry"
        steps = [{"at": -9}] + [{"to": v, "label": words}
                                for v in (-4, 1, -2, 6, 9, -6, 3)]
        report = _validate({"range": [-10, 10], "steps": steps})
        assert report.kinds() == {"overflow"}, f"\n{report}"


class TestJumpLabelsReadAsTheirOwnArc:
    """A jump label must sit on its own arc: directly under the label's centre
    the first arc is its own, and no other arc is nearer the label. A backward
    jump nested under a forward one used to get its label nudged up over the
    forward arc, where it read as the forward jump's.

    Checked on the built scene with independent geometry (densely sampled
    curves), not with the component's own placement predicate.
    """

    @pytest.mark.parametrize("params,pairs", [
        pytest.param(NumberLineWalk.examples()[0],
                     {1: "jump 2->7", 2: "jump 7->4"},
                     id="example0-back-under-forward"),
        pytest.param({"range": [-1000, 1000],
                      "steps": [{"at": -1000}, {"to": 1000}, {"to": -1000},
                                {"to": 0, "label": "back to zero"}]},
                     {1: "jump −1000->1000", 2: "jump 1000->−1000",
                      3: "jump −1000->0"},
                     id="same-span-there-and-back"),
        pytest.param({"range": [-10, 10],
                      "steps": [{"at": 0}, {"to": 4}, {"to": -2}, {"to": 5},
                                {"to": -1}]},
                     {1: "jump 0->4", 2: "jump 4->−2", 3: "jump −2->5",
                      4: "jump 5->−1"},
                     id="zigzag"),
    ])
    def test_each_label_reads_as_its_own_arc(self, params, pairs):
        scene = LayoutProbe(NumberLineWalk(params), duration=10.0, strict=True)
        scene.construct()
        tagged = {getattr(m, "_chalk_label", None): m for m in scene.mobjects}
        curves = {name: np.array([tagged[name].point_from_proportion(t)
                                  for t in np.linspace(0, 1, 400)])
                  for name in pairs.values()}
        for step, own in pairs.items():
            lab = tagged[f"steps[{step}] label"]
            left, right = lab.get_left()[0], lab.get_right()[0]
            bottom, top = lab.get_bottom()[1], lab.get_top()[1]

            def below_centre(pts):
                col = pts[(np.abs(pts[:, 0] - lab.get_x()) < 0.05)
                          & (pts[:, 1] < bottom)]
                return col[:, 1].max() if len(col) else -np.inf

            def gap(pts):
                dx = np.maximum(0, np.maximum(left - pts[:, 0], pts[:, 0] - right))
                dy = np.maximum(0, np.maximum(bottom - pts[:, 1], pts[:, 1] - top))
                return np.hypot(dx, dy).min()

            assert max(curves, key=lambda n: below_centre(curves[n])) == own, step
            assert min(curves, key=lambda n: gap(curves[n])) == own, step


class TestTicks:
    @pytest.mark.parametrize("rng,expected", [
        ([-2, 8], [float(v) for v in range(-2, 9)]),     # integer range: unit ticks
        ([0, 1], [k * 0.1 for k in range(11)]),          # fractional: 1/2/5 step
        ([0.5, 3.5, 1], [1.0, 2.0, 3.0]),                # explicit: multiples of it
    ])
    def test_tick_positions(self, rng, expected):
        assert tick_values(tuple(rng)) == pytest.approx(expected)
