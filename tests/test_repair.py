"""Mechanical repair (SCENE_SPEC.md §9, step 1).

Uses unregistered test components that misplace content the way a buggy
component would, so the registry walks elsewhere never see them.
"""

from __future__ import annotations

import pytest
from manim import RIGHT, FadeIn, tempconfig

from chalkdust.core.models import Region
from chalkdust.scenes.components import Component, ComponentParams
from chalkdust.scenes.components.base import wrap, wrap_scale
from chalkdust.scenes.regions import (
    MIN_FONT_SIZE,
    fit_to_region,
    scale_with_tags,
)
from chalkdust.scenes.theme import body_text, get_theme, resolve_fonts
from chalkdust.validate.repair import (
    MAX_ATTEMPTS,
    RepairPlan,
    propose_fix,
    repair_component,
)


@pytest.fixture(autouse=True)
def _manim_scratch_in_tmp(tmp_path):
    """Every file Manim writes here goes to tmp_path, never the cwd: the
    probes respect a caller's media_dir (geometric.probe_media), and Text
    built outside a probe (propose_fix below) writes its SVGs under it too."""
    with tempconfig({"media_dir": str(tmp_path / "manim")}):
        yield


class _MisplacedParams(ComponentParams):
    text: str
    dx: float = 0.0               # absolute offset -- the bug under repair
    region: Region = Region.STAGE
    wrap_width: int | None = None
    fit: bool = False
    reposition_after_add: bool = False
    shrink: float = 1.0


class _Misplaced(Component):
    name = "_Misplaced"
    Params = _MisplacedParams

    def regions(self) -> set[Region]:
        return {self.params.region}

    def build(self, scene) -> None:
        p: _MisplacedParams = self.params
        text = body_text(wrap(p.text, p.wrap_width) if p.wrap_width else p.text,
                         scene.theme)
        if p.fit:
            fit_to_region(text, p.region)
        if p.shrink != 1.0:
            scale_with_tags(text, p.shrink)
        if p.reposition_after_add:
            scene.add(text)
            text.move_to(RIGHT * p.dx)   # overwrites anything applied at add
        else:
            text.shift(RIGHT * p.dx)
            scene.play(FadeIn(text), run_time=scene.budget(1)[0])


def _repair(**params):
    return repair_component(_Misplaced(params), "b01")


# Words averaging ~5 letters; lengths below are calibrated against body text
# at font_size 32 with wide margins, so they hold across font substitutions.
def _words(n_chars: int) -> str:
    return ("lorem ipsum dolor sit amet " * (n_chars // 27 + 1))[:n_chars].strip()


class TestNudge:
    def test_out_of_bounds_is_nudged_back_into_its_region(self):
        result = _repair(text="Off the edge", dx=8.0)
        assert result.history[0].kinds() == {"out_of_bounds"}
        assert result.ok and result.repaired
        fix = result.plan.fixes[0]
        assert fix.scale == 1.0 and fix.dx < 0

    def test_clean_beat_is_left_alone(self):
        result = _repair(text="Fits fine")
        assert result.ok and not result.repaired
        assert len(result.history) == 1


class TestScale:
    def test_too_wide_is_scaled_down_but_stays_legible(self):
        result = _repair(text=_words(75))  # ~15 units wide vs a 13-unit stage
        assert result.history[0].kinds() == {"out_of_bounds"}
        assert result.ok
        fix = result.plan.fixes[0]
        assert MIN_FONT_SIZE / 32 <= fix.scale < 1.0

    def test_never_shrinks_below_the_legibility_floor(self):
        # ~2.5x too wide: fitting would put text near font_size 13.
        result = _repair(text=_words(180))
        assert not result.ok
        assert result.plan == RepairPlan()          # a refusal renders unchanged
        assert result.report is result.history[0]   # and reports the original
        assert all("out_of_bounds" in r.kinds() for r in result.history)

    def test_propose_fix_refuses_rather_than_go_illegible(self):
        theme = resolve_fonts(get_theme("default"), warn=False)
        text = body_text(_words(180), theme)
        assert propose_fix(text, {Region.STAGE}) is None


class TestWrap:
    def test_too_tall_is_rewrapped_wider(self):
        # ~19 lines at width 20 overflow the stage; ~12 at width 30 fit.
        result = _repair(text=_words(360), wrap_width=20, fit=True)
        assert result.history[0].kinds() == {"overflow"}
        assert result.ok and result.plan.wrap_scale == 1.5

    def test_too_wide_is_rewrapped_narrower(self):
        # 50-char lines are too wide for half the stage at a legible size,
        # 75 are worse, 35 fit.
        result = _repair(text=_words(150), wrap_width=50, fit=True,
                         region=Region.STAGE_LEFT)
        assert result.ok and result.plan.wrap_scale == 0.7

    def test_wrap_scale_only_applies_inside_its_block(self):
        text = _words(200)
        before = wrap(text, 20)
        with wrap_scale(2.0):
            inside = wrap(text, 20)
        assert max(len(line) for line in before.splitlines()) <= 20
        assert 20 < max(len(line) for line in inside.splitlines()) <= 40
        assert wrap(text, 20) == before  # restored after the block


class TestBounds:
    def test_stops_when_a_fix_makes_no_progress(self):
        # The component moves its text after adding it, overwriting the
        # add-time fix: the second probe sees the same finding and stops.
        result = _repair(text="Stubborn", dx=9.0, reposition_after_add=True)
        assert not result.ok
        assert len(result.history) == 2
        assert result.plan == RepairPlan()

    def test_attempts_are_bounded(self):
        for params in ({"text": _words(180)}, {"text": "x", "dx": 9.0,
                                                "reposition_after_add": True}):
            assert len(_repair(**params).history) <= MAX_ATTEMPTS + 1

    def test_non_repairable_kinds_are_not_attempted(self):
        # Illegible text is a refusal (split the beat), never a repair.
        result = _repair(text="Too small", shrink=0.5)
        assert result.history[0].kinds() == {"illegible"}
        assert len(result.history) == 1 and not result.ok

    def test_plan_is_deterministic(self):
        a = _repair(text=_words(75), dx=3.0)
        b = _repair(text=_words(75), dx=3.0)
        assert a.ok and a.plan == b.plan
