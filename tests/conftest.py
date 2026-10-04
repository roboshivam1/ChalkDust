"""Shared fixtures."""

from __future__ import annotations

import pytest
from manim import RIGHT, FadeIn, Restore, VGroup

from chalkdust import continuity
from chalkdust.continuity import carried
from chalkdust.core.models import Beat, BeatSpec, Region, VideoSpec
from chalkdust.scenes.components import Component, ComponentParams
from chalkdust.scenes.components import base as components_base
from chalkdust.scenes.regions import fit_to_region
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

    def carried_targets(self) -> list[str]:
        # Declared, as ZoomHighlight declares it: a consumer gets its carried
        # artifacts centred in STAGE (continuity.CarryIn).
        return [self.params.target_id]

    def build(self, scene) -> None:
        target = carried(scene, self.params.target_id)
        scene.play(Restore(target), run_time=scene.budget(1)[0])


class _Hold(Component):
    """The smallest consumer: holds its carried target on screen exactly as
    CarryIn placed it -- centred in STAGE, dimmed -- for the whole beat. A
    later beat that only carries an artifact in, without acting on it, must
    leave it a free STAGE region (register D-G4c-1), and no library component
    that claims one half of STAGE exists; so a producer's "carried into a
    later beat" test carries into this, and its artifact is the only thing
    on screen to check."""

    name = "_Hold"
    Params = _HighlightParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def carried_targets(self) -> list[str]:
        return [self.params.target_id]

    def build(self, scene) -> None:
        carried(scene, self.params.target_id)  # CarryInError unless carried in
        scene.wait(scene.budget(1)[0])


@pytest.fixture
def hold_consumer(monkeypatch) -> str:
    """Registers _Hold for the test that asks; returns its component name."""
    monkeypatch.setitem(components_base._REGISTRY, _Hold.name, _Hold)
    return _Hold.name


class _LeftNoteParams(ComponentParams):
    text: str
    spill: bool = False  # draw across all of STAGE, past its claimed half


class _LeftNote(Component):
    """Claims STAGE_LEFT only and does not act on what it carries in, so
    CarryIn fits its carried artifacts into STAGE_RIGHT (register D-G4c-1).
    With spill=True it breaks its own claim and centres its text in STAGE --
    over the carried artifact -- which the geometric probe must catch."""

    name = "_LeftNote"
    Params = _LeftNoteParams

    def regions(self) -> set[Region]:
        return {Region.STAGE_LEFT}

    def build(self, scene) -> None:
        text = body_text(self.params.text, scene.theme)
        text._chalk_label = "note"
        fit_to_region(text, Region.STAGE if self.params.spill else Region.STAGE_LEFT)
        scene.play(FadeIn(text), run_time=scene.budget(1)[0])


@pytest.fixture
def left_note(monkeypatch) -> str:
    """Registers _LeftNote for the test that asks; returns its component name."""
    monkeypatch.setitem(components_base._REGISTRY, _LeftNote.name, _LeftNote)
    return _LeftNote.name


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
