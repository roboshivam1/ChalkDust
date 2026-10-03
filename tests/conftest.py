"""Shared fixtures."""

from __future__ import annotations

import pytest
from manim import RIGHT, FadeIn, Restore, VGroup

from chalkdust import continuity
from chalkdust.continuity import carried
from chalkdust.core.models import Beat, BeatSpec, Region, VideoSpec
from chalkdust.scenes.components import Component, ComponentParams
from chalkdust.scenes.components import base as components_base
from chalkdust.scenes.theme import body_text, title_text


class _OffEdgeParams(ComponentParams):
    text: str
    dx: float = 0.0  # absolute offset -- the bug mechanical repair fixes


class _OffEdge(Component):
    """Places its text `dx` units right of centre: off the stage for large dx,
    which the geometric probe reports as out_of_bounds and repair can nudge
    back. Registered only for the test that asks for it."""

    name = "_OffEdge"
    Params = _OffEdgeParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def build(self, scene) -> None:
        text = body_text(self.params.text, scene.theme)
        text.shift(RIGHT * self.params.dx)
        scene.play(FadeIn(text), run_time=scene.budget(1)[0])


@pytest.fixture
def off_edge_beat(monkeypatch) -> Beat:
    """A 1 s beat that fails the geometric probe and passes after repair."""
    monkeypatch.setitem(components_base._REGISTRY, _OffEdge.name, _OffEdge)
    beat = Beat(spec=BeatSpec(id="b01", narration="placeholder narration",
                              component=_OffEdge.name,
                              params={"text": "Off the edge", "dx": 8.0}))
    beat.duration = 1.0
    return beat


# --- carry-in (SCENE_SPEC.md §6) --------------------------------------------


class _HighlightParams(ComponentParams):
    target_id: str


class _Highlight(Component):
    """Acts on a carried artifact the way ZoomHighlight will: its build()
    fails (CarryInError) unless the beat's carried artifact is on screen, so
    a clean build proves the caller built the beat with its recipes."""

    name = "_Highlight"
    Params = _HighlightParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def build(self, scene) -> None:
        target = carried(scene, self.params.target_id)
        scene.play(Restore(target), run_time=scene.budget(1)[0])


def _title_artifact(params, theme):
    return VGroup(title_text(params.title, theme))


@pytest.fixture
def carry_in_video(monkeypatch):
    """Factory: a two-beat video whose b02 carries in b01's artifact. TitleCard
    is lent an artifact builder and _Highlight is registered, both only for
    the test that asks."""
    monkeypatch.setitem(continuity._BUILDERS, "TitleCard", _title_artifact)
    monkeypatch.setitem(components_base._REGISTRY, _Highlight.name, _Highlight)

    def make(title: str = "Bucket array") -> VideoSpec:
        return VideoSpec(video_id="v", beats=(
            BeatSpec(id="b01", narration="placeholder narration",
                     component="TitleCard", params={"title": title},
                     registers="bucket_array"),
            BeatSpec(id="b02", narration="placeholder narration",
                     component=_Highlight.name, params={"target_id": "bucket_array"},
                     carry_in=["bucket_array"]),
        ))

    return make
