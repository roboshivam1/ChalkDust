"""Layout invariants across the whole component library.

These tests walk the registry, so every component added later is covered
automatically -- no test file to remember to update. A component that acts on
a carried artifact contributes its carried_examples() / carried_stress() too,
built with those artifacts on screen as the render builds them
(validate/fixtures.py, SCENE_SPEC.md §6).
"""

from __future__ import annotations

import pytest

from chalkdust.core.models import Region
from chalkdust.scenes.components import get_component, registered_names
from chalkdust.scenes.regions import LayoutError, region_rect, safe_area
from chalkdust.validate.fixtures import FixtureCase, fixture_cases, lent_builders
from chalkdust.validate.geometric import Report, validate_beat

# Kinds that represent a component correctly refusing overloaded content.
CLEAN_REFUSALS = {"overflow", "illegible", "invalid_latex"}


def _cases(kind: str):
    """Flatten (component, case) pairs from every registered component: its
    plain fixtures, then its carried ones (ids `-carried_<kind><i>`)."""
    out = []
    for name in registered_names():
        cases = fixture_cases(name, kind)
        plain = [c for c in cases if not c.carry_in]
        carried = [c for c in cases if c.carry_in]
        for i, case in enumerate(plain):
            out.append(pytest.param(name, case, id=f"{name}-{kind}{i}"))
        for i, case in enumerate(carried):
            out.append(pytest.param(name, case, id=f"{name}-carried_{kind}{i}"))
    return out


def _validate(name: str, case: FixtureCase) -> Report:
    """validate_beat on one fixture, its carried artifacts built first."""
    with lent_builders(name):
        return validate_beat(case.spec(name), recipes=case.carry_in)


class TestExamples:
    """Realistic content must validate clean."""

    @pytest.mark.parametrize("name,case", _cases("examples"))
    def test_validates_clean(self, name, case):
        report = _validate(name, case)
        assert report.ok, f"\n{report}"


class TestStress:
    """Abusive content must either fit or refuse cleanly -- never render broken,
    never raise something unrelated."""

    @pytest.mark.parametrize("name,case", _cases("stress"))
    def test_fits_or_refuses_cleanly(self, name, case):
        report = _validate(name, case)
        if report.ok:
            return  # handled it; fine
        bad = report.kinds() - CLEAN_REFUSALS
        assert not bad, (
            f"{name} failed with unexpected kind(s) {bad}; overloaded content "
            f"should refuse via {CLEAN_REFUSALS}\n{report}"
        )

    @pytest.mark.parametrize("name,case", _cases("stress"))
    def test_no_crash(self, name, case):
        # build_error means an exception that is not a LayoutError escaped --
        # an IndexError or TypeError from unhandled content volume.
        report = _validate(name, case)
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
        examples = fixture_cases(name, "examples")
        if not examples:
            pytest.skip(f"{name} declares no examples")
        with pytest.raises(Exception):
            cls({**examples[0].params, "definitely_not_a_real_param": 1})
