"""Semantic validation (rung 2) and the Component hooks it reads."""

from __future__ import annotations

import pytest
from pydantic import Field

from chalkdust.continuity import fixture_beat
from chalkdust.core.models import BeatSpec, Region
from chalkdust.scenes.components import (
    Component,
    ComponentParams,
    get_component,
    registered_names,
)
from chalkdust.scenes.components import base as components_base
from chalkdust.scenes.components.base import MIN_STEP_SECONDS
from chalkdust.scenes.regions import INVALID_LATEX
from chalkdust.validate.semantic import (
    MAX_BEAT_SECONDS,
    check_capacity,
    check_carried_targets,
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


class _PointerParams(ComponentParams):
    target_id: str


class _Pointer(Component):
    """Acts on the carried artifact its params name, as Callout and
    ZoomHighlight do. Deliberately NOT registered."""

    name = "_Pointer"
    Params = _PointerParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def build(self, scene) -> None:  # never built by the semantic rung
        raise AssertionError("semantic checks must not build the scene")

    def carried_targets(self) -> list[str]:
        return [self.params.target_id]


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
        assert isinstance(component.carried_targets(), list)


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

    def test_target_not_carried_in_is_refused_by_name(self):
        # Registered by an earlier beat, but this beat does not carry it in:
        # carried() would raise CarryInError inside build(), a spec bug the
        # geometric rung could only report as a build_error.
        findings = check_carried_targets(_Pointer({"target_id": "causes"}),
                                         carry_in=["steps"])
        assert [f.kind for f in findings] == ["carry_in"]
        assert "'causes'" in findings[0].message

    def test_carried_target_passes(self):
        assert check_carried_targets(_Pointer({"target_id": "causes"}),
                                     carry_in=["causes"]) == []

    def test_validate_semantic_checks_carried_targets(self, monkeypatch):
        monkeypatch.setitem(components_base._REGISTRY, _Pointer.name, _Pointer)
        spec = _spec(_Pointer.name, {"target_id": "causes"})
        report = validate_semantic(spec, registered_artifacts={"causes"})
        assert report.kinds() == {"carry_in"}


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

    def test_raw_scene_source_is_not_counted_as_text(self):
        # RawScene params are Manim source, never drawn: ~4k glyphs of code
        # is a typical generated scene, not an overloaded frame.
        raw = get_component("RawScene")({"rationale": "a sweep the library lacks",
                                         "code": "sq = Square()\n" * 400})
        assert check_capacity(raw) == []

    def test_union_counts_nested_regions_once(self):
        assert region_capacity({Region.STAGE, Region.STAGE_LEFT}) == \
            region_capacity({Region.STAGE})


class TestLatex:
    """Needs a working LaTeX install (MiKTeX here). Never skipped: a missing
    toolchain is a real failure of this rung."""

    def test_valid_expression_compiles(self, tmp_path):
        assert check_latex(_Stepper({"tex": [r"\frac{a}{b} = c"]}),
                           media_dir=tmp_path) == []

    def test_malformed_expression_is_refused(self, tmp_path):
        # An undefined control sequence. (Unbalanced braces are NOT a good
        # probe: Manim 0.21 wraps each MathTex in dvisvgm \special groups
        # whose closing brace absorbs a missing one, so build() would compile
        # them too.)
        findings = check_latex(_Stepper({"tex": [r"\frac{a}{b} = c",
                                                  r"\notarealmacro{x}"]}),
                               media_dir=tmp_path)
        # Assert the kind, not Manim's wording, which varies by failure mode.
        # The kind is theme.math's, so rungs 2 and 3 share one vocabulary.
        assert [f.kind for f in findings] == [INVALID_LATEX]
        assert r"\notarealmacro{x}" in findings[0].message

    def test_scratch_never_lands_in_the_cwd(self, tmp_path, monkeypatch):
        # Register N-12: compiling wrote media/Tex into whatever directory
        # the rung ran from. Like the geometric probe, it defaults to work/.
        monkeypatch.chdir(tmp_path)
        assert check_latex(_Stepper({"tex": [r"x^2 + 1 = 0"]})) == []
        assert not (tmp_path / "media").exists()
        assert any((tmp_path / "work" / "manim" / "Tex").glob("*.svg"))


class TestLibrary:
    """Every registered component's realistic examples must pass rung 2."""

    @pytest.mark.parametrize("name", registered_names())
    def test_examples_validate_clean(self, name, tmp_path):
        for params in get_component(name).examples():
            # A carry-in consumer's case carries its fixture artifacts in,
            # registered by an earlier beat, as a real spec would.
            spec, _ = fixture_beat(name, params)
            report = validate_semantic(spec, registered_artifacts=spec.carry_in,
                                       duration=12.0, media_dir=tmp_path)
            assert report.ok, f"\n{report}"
