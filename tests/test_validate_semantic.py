"""Semantic validation (rung 2) and the Component hooks it reads."""

from __future__ import annotations

import pytest
from pydantic import Field

from chalkdust.core.models import BeatSpec, Region
from chalkdust.scenes.components import (
    Component,
    ComponentParams,
    get_component,
    registered_names,
)
from chalkdust.scenes.components.base import MIN_STEP_SECONDS
from chalkdust.validate.semantic import (
    MAX_BEAT_SECONDS,
    check_capacity,
    check_carry_in,
    check_duration,
    check_latex,
    check_region_conflicts,
    estimate_seconds,
    region_capacity,
    validate_semantic,
)


class _StepperParams(ComponentParams):
    steps: int = 4
    tex: list[str] = Field(default_factory=list)


class _Stepper(Component):
    """Overrides both hooks. Deliberately NOT registered, so the registry
    walks in other test files never see it."""

    name = "_Stepper"
    Params = _StepperParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def build(self, scene) -> None:  # never built by the semantic rung
        raise AssertionError("semantic checks must not build the scene")

    def min_seconds(self) -> float:
        return MIN_STEP_SECONDS * self.params.steps

    def latex_strings(self) -> list[str]:
        return list(self.params.tex)


def _spec(component: str, params: dict, narration: str = "placeholder narration",
          carry_in: list[str] | None = None) -> BeatSpec:
    return BeatSpec(id="b01", narration=narration, component=component,
                    params=params, carry_in=carry_in or [])


class TestHookDefaults:
    """The hooks are optional: a component that ignores them must get the
    permissive defaults, never an AttributeError or a refusal."""

    @pytest.mark.parametrize("name", registered_names())
    def test_defaults_are_permissive(self, name):
        cls = get_component(name)
        if not cls.examples():
            pytest.skip(f"{name} declares no examples")
        component = cls(cls.examples()[0])
        assert component.min_seconds() >= 0.0
        assert isinstance(component.latex_strings(), list)


class TestDuration:
    def test_estimate_uses_documented_pace(self):
        assert estimate_seconds(" ".join(["word"] * 160)) == pytest.approx(60.0)

    def test_narration_shorter_than_component_minimum_is_refused(self):
        findings = check_duration(_Stepper({"steps": 10}), seconds=3.0)
        assert [f.kind for f in findings] == ["duration"]

    def test_narration_covering_the_minimum_passes(self):
        assert check_duration(_Stepper({"steps": 10}), seconds=6.0) == []

    def test_beat_longer_than_one_idea_is_refused(self):
        findings = check_duration(_Stepper({}), seconds=MAX_BEAT_SECONDS + 1,
                                  measured=True)
        assert [f.kind for f in findings] == ["duration"]
        assert "measured" in findings[0].message

    def test_measured_duration_overrides_the_estimate(self):
        # 70 words estimates past MAX_BEAT_SECONDS; the real audio says 12s.
        spec = _spec("TitleCard", {"title": "Hi"}, narration=" ".join(["w"] * 70))
        assert "duration" in validate_semantic(spec).kinds()
        assert "duration" not in validate_semantic(spec, duration=12.0).kinds()


class TestCarryIn:
    def test_unregistered_artifact_is_refused(self):
        findings = check_carry_in(["bucket_array"], registered=set())
        assert [f.kind for f in findings] == ["carry_in"]

    def test_registered_artifact_passes(self):
        assert check_carry_in(["bucket_array"], registered={"bucket_array"}) == []

    def test_validate_semantic_threads_the_registry_through(self):
        spec = _spec("TitleCard", {"title": "Hi"}, carry_in=["bucket_array"])
        assert "carry_in" in validate_semantic(spec).kinds()
        ok = validate_semantic(spec, registered_artifacts={"bucket_array"})
        assert "carry_in" not in ok.kinds()


class TestRegionConflicts:
    def test_nested_regions_conflict(self):
        # Different names, same space: STAGE contains STAGE_LEFT.
        findings = check_region_conflicts({"a": {Region.STAGE},
                                           "b": {Region.STAGE_LEFT}})
        assert [f.kind for f in findings] == ["region_conflict"]

    @pytest.mark.parametrize("a,b", [
        (Region.STAGE_LEFT, Region.STAGE_RIGHT),
        (Region.TITLE_BAR, Region.STAGE),
        (Region.STAGE, Region.LOWER_THIRD),
    ])
    def test_disjoint_regions_do_not_conflict(self, a, b):
        assert check_region_conflicts({"a": {a}, "b": {b}}) == []

    def test_concurrent_claim_reaches_the_check(self):
        spec = _spec("TitleCard", {"title": "Hi"})
        report = validate_semantic(spec, concurrent={"overlay": {Region.STAGE_RIGHT}})
        assert "region_conflict" in report.kinds()


class TestCapacity:
    def test_overloaded_text_is_refused(self):
        title = " ".join(["overloaded"] * 400)  # 4000 glyphs on a title card
        findings = check_capacity(get_component("TitleCard")({"title": title}))
        assert [f.kind for f in findings] == ["capacity"]

    def test_union_counts_nested_regions_once(self):
        assert region_capacity({Region.STAGE, Region.STAGE_LEFT}) == \
            region_capacity({Region.STAGE})


class TestLatex:
    """Needs a working LaTeX install (MiKTeX here). Never skipped: a missing
    toolchain is a real failure of this rung."""

    def test_valid_expression_compiles(self):
        assert check_latex(_Stepper({"tex": [r"\frac{a}{b} = c"]})) == []

    def test_malformed_expression_is_refused(self):
        # An undefined control sequence. (Unbalanced braces are NOT a good
        # probe: Manim 0.21 wraps each MathTex in dvisvgm \special groups
        # whose closing brace absorbs a missing one, so build() would compile
        # them too.)
        findings = check_latex(_Stepper({"tex": [r"\frac{a}{b} = c",
                                                  r"\notarealmacro{x}"]}))
        # Assert the kind, not Manim's wording, which varies by failure mode.
        assert [f.kind for f in findings] == ["latex"]
        assert r"\notarealmacro{x}" in findings[0].message


class TestLibrary:
    """Every registered component's realistic examples must pass rung 2."""

    @pytest.mark.parametrize("name", registered_names())
    def test_examples_validate_clean(self, name):
        for params in get_component(name).examples():
            report = validate_semantic(_spec(name, params), duration=12.0)
            assert report.ok, f"\n{report}"
