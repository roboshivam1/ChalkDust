"""Shared fixtures."""

from __future__ import annotations

import pytest
from manim import RIGHT, FadeIn

from chalkdust.core.models import Beat, BeatSpec, Region
from chalkdust.scenes.components import Component, ComponentParams
from chalkdust.scenes.components import base as components_base
from chalkdust.scenes.theme import body_text


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
