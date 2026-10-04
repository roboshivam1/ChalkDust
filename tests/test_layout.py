"""Layout invariants across the whole component library.

These tests walk the registry, so every component added later is covered
automatically -- no test file to remember to update. A component that acts on
a carried artifact (Callout, ZoomHighlight) is walked too: each case is built
with the artifacts its Component.fixture_carry_in names on screen, as the
render builds a carry-in beat (SCENE_SPEC.md §6).
"""

from __future__ import annotations

import pytest

from chalkdust.continuity import fixture_beat
from chalkdust.core.models import Region
from chalkdust.scenes.components import get_component, registered_names
from chalkdust.scenes.regions import LayoutError, region_rect, safe_area
from chalkdust.validate.geometric import Report, validate_beat

# Kinds that represent a component correctly refusing overloaded content.
CLEAN_REFUSALS = {"overflow", "illegible", "invalid_latex", "unrenderable_text"}


def _validate(component: str, params: dict) -> Report:
    """validate_beat on one case, built as the pipeline builds its beat: a
    carry-in consumer's case with its fixture artifacts on screen first
    (continuity.fixture_beat; a plain beat for every other component)."""
    spec, recipes = fixture_beat(component, params)
    return validate_beat(spec, recipes=recipes)


def _cases(kind: str):
    """Flatten (component, params) pairs from every registered component."""
    out = []
    for name in registered_names():
        cls = get_component(name)
        for i, params in enumerate(getattr(cls, kind)()):
            out.append(pytest.param(name, params, id=f"{name}-{kind}{i}"))
    return out


class TestExamples:
    """Realistic content must validate clean."""

    @pytest.mark.parametrize("name,params", _cases("examples"))
    def test_validates_clean(self, name, params):
        report = _validate(name, params)
        assert report.ok, f"\n{report}"


class TestStress:
    """Abusive content must either fit or refuse cleanly -- never render broken,
    never raise something unrelated."""

    @pytest.mark.parametrize("name,params", _cases("stress"))
    def test_fits_or_refuses_cleanly(self, name, params):
        report = _validate(name, params)
        if report.ok:
            return  # handled it; fine
        bad = report.kinds() - CLEAN_REFUSALS
        assert not bad, (
            f"{name} failed with unexpected kind(s) {bad}; overloaded content "
            f"should refuse via {CLEAN_REFUSALS}\n{report}"
        )

    @pytest.mark.parametrize("name,params", _cases("stress"))
    def test_no_crash(self, name, params):
        # build_error means an exception that is not a LayoutError escaped --
        # an IndexError or TypeError from unhandled content volume.
        report = _validate(name, params)
        crashes = [f for f in report.findings if f.kind == "build_error"]
        assert not crashes, f"\n{report}"


class TestRegions:
    """The layout grid itself must be internally consistent."""

    def test_all_regions_inside_safe_area(self):
        safe = safe_area()
        for region in Region:
            assert safe.contains(region_rect(region)), f"{region} escapes safe area"

    def test_stage_halves_do_not_overlap(self):
        left = region_rect(Region.STAGE_LEFT)
        right = region_rect(Region.STAGE_RIGHT)
        assert not left.intersects(right)

    def test_title_and_lower_third_do_not_overlap_stage(self):
        stage = region_rect(Region.STAGE)
        for other in (Region.TITLE_BAR, Region.LOWER_THIRD):
            assert not stage.intersects(region_rect(other)), f"STAGE hits {other}"


class TestSchema:
    """Param validation must reject what it cannot honour."""

    @pytest.mark.parametrize("name", registered_names())
    def test_rejects_unknown_param(self, name):
        cls = get_component(name)
        examples = cls.examples()
        if not examples:
            pytest.skip(f"{name} declares no examples")
        with pytest.raises(Exception):
            cls({**examples[0], "definitely_not_a_real_param": 1})
