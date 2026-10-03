"""UnitBreakdown: timing, schema limits, the LaTeX error path, label layout.

Layout of examples() and stress() is covered by test_layout.py walking the
registry; these pin what that walk cannot see. LaTeX must be installed --
these compile real units and are never skipped without it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chalkdust.core.models import BeatSpec
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.unit_breakdown import LatexError, UnitBreakdown
from chalkdust.scenes.regions import bbox
from chalkdust.validate.geometric import LayoutProbe, validate_beat

NEWTON = UnitBreakdown.examples()[0]
MINIMAL = {"quantity": {"unit": "x"}, "decomposition": [{"unit": "y"}]}
FRAME = 1 / 60  # one frame at the final tier's 60 fps (D-006)


def _elapsed(params: dict, budget: float) -> float:
    """Build through the real scene with Manim's own clock running.

    skip_animations makes the renderer add each play's duration to its clock
    without drawing a frame, so this measures exactly what a render would
    spend, minus the encoding.
    """
    scene = ChalkdustScene(UnitBreakdown(params), duration=budget,
                           skip_animations=True)
    scene.setup()
    scene.construct()
    return scene.renderer.time


class TestTiming:
    """Animation consumes the beat's audio budget exactly (D-002)."""

    @pytest.mark.parametrize("factor", [0.5, 3.0])
    def test_consumes_budget_relative_to_minimum(self, factor):
        budget = UnitBreakdown(NEWTON).min_seconds() * factor
        assert _elapsed(NEWTON, budget) == pytest.approx(budget, abs=FRAME)

    @pytest.mark.parametrize("budget", [0.3, 25.0])
    def test_consumes_far_short_and_far_long_narration(self, budget):
        assert _elapsed(NEWTON, budget) == pytest.approx(budget, abs=FRAME)

    def test_minimal_input_consumes_budget(self):
        assert _elapsed(MINIMAL, 6.0) == pytest.approx(6.0, abs=FRAME)

    def test_min_seconds_grows_with_each_factor(self):
        one = UnitBreakdown(MINIMAL).min_seconds()
        three = UnitBreakdown(NEWTON).min_seconds()
        assert 0 < one < three


class TestSchema:
    """Inputs the param model must refuse before anything compiles."""

    @pytest.mark.parametrize("params", [
        {"quantity": {"unit": "x"}, "decomposition": []},
        {"quantity": {"unit": "x"}, "decomposition": [{"unit": "y"}] * 7},
        {"quantity": {"unit": "   "}, "decomposition": [{"unit": "y"}]},
        {"quantity": {"unit": "x", "label": "  "}, "decomposition": [{"unit": "y"}]},
        {"decomposition": [{"unit": "y"}]},
        {"quantity": {"unit": "x", "colour": "red"}, "decomposition": [{"unit": "y"}]},
    ], ids=["empty", "too-many", "blank-unit", "blank-label", "no-quantity",
            "unknown-term-field"])
    def test_rejects(self, params):
        with pytest.raises(ValidationError):
            UnitBreakdown(params)


class TestLatex:
    """Bad unit LaTeX is a typed finding the repair loop can act on, never a
    crash."""

    def test_latex_strings_is_the_one_compiled_expression(self):
        [expr] = UnitBreakdown(NEWTON).latex_strings()
        for term in [NEWTON["quantity"], *NEWTON["decomposition"]]:
            assert term["unit"] in expr

    def test_invalid_latex_is_a_typed_finding(self):
        params = {"quantity": {"unit": r"\mathrm{N"},
                  "decomposition": [{"unit": r"\notacommand{kg}"}]}
        report = validate_beat(BeatSpec(id="b01", narration="placeholder",
                                        component="UnitBreakdown", params=params))
        assert report.kinds() == {"latex"}, f"\n{report}"

    @pytest.mark.parametrize("params", [
        # Compiles, but closes the group early and splits into extra pieces.
        {"quantity": {"unit": "x }} {{ y"}, "decomposition": [{"unit": "z"}]},
        # Compiles to no glyphs at all.
        {"quantity": {"unit": "{}"}, "decomposition": [{"unit": "z"}]},
    ], ids=["mis-split", "no-glyphs"])
    def test_compiling_but_unusable_latex_raises_latex_error(self, params):
        probe = LayoutProbe(UnitBreakdown(params), duration=5.0)
        with pytest.raises(LatexError) as exc:
            probe.construct()
        assert exc.value.kind == "latex"


class TestLayout:
    def test_wide_labels_under_narrow_units_never_collide(self):
        params = {"quantity": {"unit": "a", "label": "a fairly long label"},
                  "decomposition": [{"unit": "b", "label": "another long one"},
                                    {"unit": "c", "label": "and a third label"}]}
        probe = LayoutProbe(UnitBreakdown(params), duration=5.0)
        probe.construct()
        [group] = probe.mobjects
        family = group.get_family()
        units = next(m for m in family if getattr(m, "_chalk_label", "") == "units")
        labels = [m for m in family
                  if getattr(m, "_chalk_label", "").startswith("label[")]
        assert len(labels) == 3
        boxes = [bbox(m) for m in labels]
        for i, a in enumerate(boxes):
            assert a.top < bbox(units).bottom
            for b in boxes[i + 1:]:
                assert not a.intersects(b)
