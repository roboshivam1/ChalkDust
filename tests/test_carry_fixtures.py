"""Carry-in consumer fixtures and BulletReveal's artifact builder
(SCENE_SPEC.md §6, §11 rule 6).

A carry-in consumer (Callout annotates a carried target, ZoomHighlight
focuses on part of one) cannot build from params alone: build() fetches its
target with continuity.carried(). Component.fixture_carry_in(params) names the
artifacts each examples()/stress() case stands on, and continuity.fixture_beat
turns the case into the beat the pipeline would build, so the registry walks
(tests/test_layout.py, tests/test_snapshots.py, the semantic TestLibrary)
cover a consumer like any other component. The consumers land after this
mechanism, so a tiny test-local one stands in for them here.

They build their fixtures against the real BulletReveal builder, and the
pipeline uses the same builder at every cut. A builder that only approximates
build() makes the list visibly jump between the producing beat's last frame
and the consuming beat's first.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import test_layout

from chalkdust import continuity
from chalkdust.continuity import ArtifactRecipe, CarryIn, carried
from chalkdust.core.models import BeatSpec, Region
from chalkdust.scenes.components import Component, ComponentParams, make_component
from chalkdust.scenes.components import base as components_base
from chalkdust.validate.geometric import LayoutProbe, validate_beat
from chalkdust.validate.semantic import validate_semantic
from chalkdust.validate.snapshot import (
    capture,
    diff_geometry,
    diff_structure,
    fingerprint,
    load,
    regenerate,
)

PARAMS = {"heading": "Three causes",
          "items": ["A weak hash function",
                    "A load factor left too high, so every chain keeps growing "
                    "until the table is resized",
                    "Adversarial keys chosen to collide"]}


def _by_label(scene, name):
    (mob,) = [m for m in scene.mobjects if getattr(m, "_chalk_label", None) == name]
    return mob


def test_carried_list_sits_exactly_where_its_beat_left_it():
    produced = LayoutProbe(make_component("BulletReveal", PARAMS), duration=8.0, strict=False)
    produced.construct()
    shown = _by_label(produced, "bullets")

    recipe = ArtifactRecipe(name="causes", producer="BulletReveal", params=PARAMS)
    consumer = CarryIn(make_component("TitleCard", {"title": "Next"}), [recipe])
    consumed = LayoutProbe(consumer, duration=8.0, strict=False)
    consumed.construct()
    carried = _by_label(consumed, "carried[causes]")

    # One part per item, in item order, each row (dot and text) at the same
    # place and size as on the producing beat's last frame.
    assert len(carried.submobjects) == len(shown.submobjects) == len(PARAMS["items"])
    for before, after in zip(shown.submobjects, carried.submobjects):
        np.testing.assert_allclose(after.get_center(), before.get_center(), atol=1e-6)
        np.testing.assert_allclose((after.width, after.height),
                                   (before.width, before.height), atol=1e-6)


# --- a carry-in consumer, local to these tests ---------------------------------

_LISTS = {"causes": PARAMS["items"],
          "steps": ["Hash the key", "Take it modulo the table size",
                    "Walk the chain", "Compare keys"]}


class _LiftParams(ComponentParams):
    target_id: str
    part: int


class _Lift(Component):
    """Brings one row of a carried list back to full strength -- acting on
    part of a carried visual, as ZoomHighlight and Callout do. Its build()
    raises CarryInError unless the beat carries its target in. Registered
    only for the tests that ask (the `lift` fixture)."""

    name = "_Lift"
    Params = _LiftParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def carried_targets(self) -> list[str]:
        return [self.params.target_id]

    def build(self, scene) -> None:
        row = carried(scene, self.params.target_id).submobjects[self.params.part]
        times = scene.budget(2, 1)
        scene.play(row.animate.set_opacity(1), run_time=times[0])
        scene.settle("row lifted")
        scene.wait(times[1])

    @classmethod
    def fixture_carry_in(cls, params: dict[str, Any]) -> list[ArtifactRecipe]:
        name = params["target_id"]
        return [ArtifactRecipe(name=name, producer="BulletReveal",
                               params={"items": _LISTS[name]})]

    @classmethod
    def examples(cls) -> list[dict[str, Any]]:
        return [{"target_id": "causes", "part": 1},
                {"target_id": "steps", "part": 3}]


@pytest.fixture
def lift(monkeypatch) -> str:
    monkeypatch.setitem(components_base._REGISTRY, _Lift.name, _Lift)
    return _Lift.name


def test_layout_walk_builds_each_case_with_its_target_carried_in(lift):
    # test_layout's own case list and validator, with _Lift registered.
    cases = [c.values for c in test_layout._cases("examples") if c.values[0] == lift]
    assert len(cases) == len(_Lift.examples())
    for name, params in cases:
        report = test_layout._validate(name, params)
        assert report.ok, f"\n{report}"


def test_a_consumer_built_bare_cannot_build(lift):
    # Why the walks need fixture_carry_in at all: without its target on
    # screen the consumer's build() raises CarryInError on every case --
    # a spec error the probe types as carry_in, not a build_error crash
    # (register D-G4b-1).
    spec = BeatSpec(id="b01", narration="placeholder narration", component=lift,
                    params=_Lift.examples()[0])
    report = validate_beat(spec)
    assert report.kinds() == {"carry_in"}
    assert "is not carried into this beat" in str(report)


def test_snapshot_records_and_matches_the_carried_frame(lift, tmp_path):
    # regenerate() and capture() are what tests/test_snapshots.py records and
    # compares with; the recorded frame includes the carried list.
    regenerate(lift, "carry-in consumer fixture", tmp_path)
    snap = load(lift, tmp_path)
    assert [c["params"] for c in snap["cases"]] == _Lift.examples()
    for case, params in zip(snap["cases"], _Lift.examples()):
        structure, geometry = capture(lift, params)
        assert diff_structure(case["structure"], structure) == []
        assert diff_geometry(case["geometry"][fingerprint()], geometry) == []
        (lifted,) = [op for op in structure["timeline"]
                     if op["op"] == "settle" and op["label"] == "row lifted"]
        labels = [m.get("label") for m in lifted["mobjects"]]
        assert f"carried[{params['target_id']}]" in labels


def test_semantic_walk_passes_a_consumer_case(lift, tmp_path):
    # The semantic TestLibrary's path: the case's carry_in, registered by an
    # earlier beat, satisfies the carried-target check.
    for params in _Lift.examples():
        spec, _ = continuity.fixture_beat(lift, params)
        report = validate_semantic(spec, registered_artifacts=spec.carry_in,
                                   duration=12.0, media_dir=tmp_path)
        assert report.ok, f"\n{report}"
