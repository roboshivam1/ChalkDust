"""Carry-in continuity (SCENE_SPEC.md §6).

No shipped component registers an artifact builder yet, so these tests lend
TitleCard a builder for the duration of each test (monkeypatched, never left
in the registry).
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
from manim import VGroup
from pydantic import ValidationError

from chalkdust import continuity
from chalkdust.continuity import (
    ArtifactRecipe,
    beat_component,
    build_artifact,
    carried,
    carry_in_fingerprint,
    resolve_carry_in,
)
from chalkdust.core.cache import beat_render_key
from chalkdust.core.models import BeatSpec, BuildContext, CarryInError, Quality, Region, VideoSpec
from chalkdust.render.worker import TIERS
from chalkdust.scenes.regions import bbox, region_rect
from chalkdust.scenes.theme import DEFAULT, title_text
from chalkdust.validate.geometric import LayoutProbe
from chalkdust.validate.repair import RepairPlan

# f2/f4's render-key terms (theme, tier, repair plan), fixed for these tests.
_TERMS = (asdict(DEFAULT), asdict(TIERS[Quality.DRAFT]), RepairPlan().key_data())


def _title_artifact(params, theme):
    return VGroup(title_text(params.title, theme))


@pytest.fixture
def title_builder(monkeypatch):
    monkeypatch.setitem(continuity._BUILDERS, "TitleCard", _title_artifact)


def _beat(bid, component="BulletReveal", params=None, **kw) -> BeatSpec:
    params = params if params is not None else {"items": ["one point"]}
    return BeatSpec(id=bid, narration="placeholder narration",
                    component=component, params=params, **kw)


def _video(*beats) -> VideoSpec:
    return VideoSpec(video_id="v", beats=beats)


def _producer(title="Bucket array"):
    return _beat("b01", "TitleCard", {"title": title}, registers="bucket_array")


def _consumer():
    return _beat("b02", carry_in=["bucket_array"])


def _carry_error(exc_info) -> CarryInError:
    err = exc_info.value.errors()[0]["ctx"]["error"]
    assert isinstance(err, CarryInError)
    return err


class TestValidation:
    def test_unknown_name_is_typed_error(self):
        with pytest.raises(ValidationError) as exc_info:
            _video(_producer(), _beat("b02", carry_in=["no_such_thing"]))
        err = _carry_error(exc_info)
        assert (err.beat_id, err.name) == ("b02", "no_such_thing")

    def test_name_registered_later_is_unknown(self):
        later = _beat("b02", "TitleCard", {"title": "x"}, registers="bucket_array")
        with pytest.raises(ValidationError) as exc_info:
            _video(_beat("b01", carry_in=["bucket_array"]), later)
        assert _carry_error(exc_info).beat_id == "b01"

    def test_duplicate_registration_rejected(self):
        twin = _beat("b02", "TitleCard", {"title": "y"}, registers="bucket_array")
        with pytest.raises(ValidationError) as exc_info:
            _video(_producer(), twin)
        assert _carry_error(exc_info).name == "bucket_array"

    def test_producer_without_builder_is_typed_error(self):
        # Valid spec, but BulletReveal has no artifact builder.
        video = _video(_beat("b01", registers="bucket_array"), _consumer())
        with pytest.raises(CarryInError):
            resolve_carry_in(video)


class TestResolve:
    def test_recipe_is_producers_component_and_params(self, title_builder):
        resolved = resolve_carry_in(_video(_producer(), _consumer()))
        assert resolved["b01"] == ()
        assert resolved["b02"] == (
            ArtifactRecipe(name="bucket_array", producer="TitleCard",
                           params={"title": "Bucket array"}),
        )


class TestCacheKey:
    def _key(self, producer_title: str) -> str:
        consumer = _consumer()
        recipes = (ArtifactRecipe(name="bucket_array", producer="TitleCard",
                                  params={"title": producer_title}),)
        return beat_render_key(consumer, 5.0, BuildContext(), *_TERMS,
                               carried=carry_in_fingerprint(recipes))

    def test_editing_producer_changes_consumer_key(self):
        assert self._key("Bucket array") != self._key("Bucket list")

    def test_same_recipe_same_key(self):
        assert self._key("Bucket array") == self._key("Bucket array")

    def test_carry_beat_without_fingerprint_refuses(self):
        with pytest.raises(ValueError, match="carry_in_fingerprint"):
            beat_render_key(_consumer(), 5.0, BuildContext(), *_TERMS)

    def test_plain_beat_key_unaffected(self):
        plain = _beat("b01")
        assert beat_render_key(plain, 5.0, BuildContext(), *_TERMS) == \
            beat_render_key(plain, 5.0, BuildContext(), *_TERMS, carried=None)


class TestScene:
    def _probe(self) -> LayoutProbe:
        recipes = resolve_carry_in(_video(_producer(), _consumer()))["b02"]
        probe = LayoutProbe(beat_component(_consumer(), recipes),
                            theme="default", duration=4.0, strict=False)
        probe.construct()
        return probe

    def test_rebuild_is_deterministic(self, title_builder):
        recipe = ArtifactRecipe(name="a", producer="TitleCard",
                                params={"title": "Bucket array"})
        a, b = build_artifact(recipe, DEFAULT), build_artifact(recipe, DEFAULT)
        pa = [m.points for m in a.family_members_with_points()]
        pb = [m.points for m in b.family_members_with_points()]
        assert len(pa) == len(pb) > 0
        assert all(np.array_equal(x, y) for x, y in zip(pa, pb))

    def test_carried_artifact_placed_in_stage_dimmed(self, title_builder):
        probe = self._probe()
        target = carried(probe, "bucket_array")
        assert target in probe.mobjects
        assert region_rect(Region.STAGE).contains(
            bbox(target))
        opacities = [m.get_fill_opacity() for m in target.family_members_with_points()]
        assert max(opacities) == pytest.approx(1 - continuity.DIM_DARKNESS)
        assert probe.layout_warnings == []

    def test_beats_own_component_still_builds(self, title_builder):
        probe = self._probe()
        # The carried artifact plus BulletReveal's bullets group.
        labels = {getattr(m, "_chalk_label", None) for m in probe.mobjects}
        assert {"carried[bucket_array]", "bullets"} <= labels

    def test_target_not_carried_is_typed_error(self, title_builder):
        with pytest.raises(CarryInError, match="no_such_thing"):
            carried(self._probe(), "no_such_thing")
