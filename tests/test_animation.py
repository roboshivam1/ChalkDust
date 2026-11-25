"""Animation components (scenes/components/animation.py): the contract checks
the layout tests do not cover.

- Continuity: the first frame of phase k+1 equals the last frame of phase k
  for the same setup, compared on the camera's own pixels (no encoder in
  between, so the comparison is exact). This is what lets consecutive beats
  join with no visible cut while still rendering independently (D-005).
- Every animation's examples() walk its phases in order with one setup, so
  the continuity test covers every phase boundary it has.
- Cue checks run at rung 2 from the text alone.
"""

from __future__ import annotations

import numpy as np
import pytest
from manim import tempconfig
from manim.utils.exceptions import EndSceneEarlyException

import chalkdust.scenes.components  # noqa: F401
from chalkdust.core.models import BeatSpec
from chalkdust.render.worker import long_path
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.animation import Animation
from chalkdust.scenes.components.base import _REGISTRY, make_component
from chalkdust.validate.semantic import check_continuity, validate_semantic

ANIMATIONS = sorted(n for n, c in _REGISTRY.items() if issubclass(c, Animation))
# Small and quick: continuity is a property of the mobjects, not the resolution.
CAMERA = {"pixel_width": 640, "pixel_height": 360, "frame_rate": 15,
          "skip_animations": True, "write_to_movie": False, "disable_caching": True,
          "verbosity": "WARNING"}


class _EntryScene(ChalkdustScene):
    """Captures the frame an animation phase starts on, then stops."""

    entry: np.ndarray | None = None

    def chalk_on_entry(self) -> None:
        self.renderer.update_frame(self)
        self.entry = np.array(self.renderer.get_frame())
        raise EndSceneEarlyException()


def _final_frame(component, tmp_path) -> np.ndarray:
    with tempconfig({**CAMERA, "media_dir": str(long_path(tmp_path))}):
        scene = ChalkdustScene(component, duration=9.0)
        scene.render()
        # The renderer keeps the last animation's static background, and
        # update_frame draws onto it: without this the final frame is the
        # picture drawn twice (antialiased edges thicken) and never matches.
        scene.renderer.static_image = None
        scene.renderer.update_frame(scene)
        return np.array(scene.renderer.get_frame())


def _entry_frame(component, tmp_path) -> np.ndarray:
    with tempconfig({**CAMERA, "media_dir": str(long_path(tmp_path))}):
        scene = _EntryScene(component, duration=9.0)
        scene.render()
        assert scene.entry is not None
        return scene.entry


def _chains(cls) -> list[tuple[dict, dict]]:
    ex = cls.examples()
    out = []
    for a, b in zip(ex, ex[1:]):
        ca, cb = cls(a), cls(b)
        if ca.setup() == cb.setup() and cb.phase_index() == ca.phase_index() + 1:
            out.append((a, b))
    return out


def test_there_are_animations():
    assert ANIMATIONS, "no Animation component registered"


@pytest.mark.parametrize("name", ANIMATIONS)
def test_examples_walk_every_phase_of_one_setup(name):
    cls = _REGISTRY[name]
    assert len(_chains(cls)) == len(cls.PHASES) - 1, (
        f"{name}.examples() must start with one example per phase, in phase order, "
        "all with the same setup, so every phase boundary is tested")


@pytest.mark.parametrize("name,pair", [(n, i) for n in ANIMATIONS
                                       for i in range(len(_REGISTRY[n].PHASES) - 1)])
def test_phase_starts_on_previous_phase_last_frame(name, pair, tmp_path):
    cls = _REGISTRY[name]
    a, b = _chains(cls)[pair]
    last = _final_frame(cls(a), tmp_path / "a")
    first = _entry_frame(cls(b), tmp_path / "b")
    diff = np.abs(last.astype(int) - first.astype(int)).max(axis=2)
    assert (diff > 8).sum() == 0, (
        f"{name}: phase {a['phase']!r} ends on a different frame than "
        f"{b['phase']!r} starts on ({int((diff > 8).sum())} pixels differ, max {diff.max()})")


def test_cue_not_in_narration_is_refused_at_rung_2():
    params = {**_REGISTRY["SecantToTangent"].examples()[1], "cues": {"line": "tangent of"}}
    spec = BeatSpec(id="b01", narration="Pick a second point. Draw the line through both. "
                    "The rise over the run is the slope.", component="SecantToTangent",
                    params=params)
    kinds = [f.kind for f in validate_semantic(spec).findings]
    assert kinds == ["cue"]


def test_continuing_a_phase_with_another_setup_is_refused():
    ex = _REGISTRY["SecantToTangent"].examples()
    a = BeatSpec(id="b01", narration="One. Two. Three.", component="SecantToTangent", params=ex[0])
    b = BeatSpec(id="b02", narration="One. Two. Three.", component="SecantToTangent",
                 params={**ex[1], "a": 2})
    (f,) = check_continuity(a, b)
    assert f.kind == "continuity" and "a" in f.message
    assert check_continuity(a, BeatSpec(id="b02", narration="x", component="SecantToTangent",
                                        params=ex[1])) == []


def test_starting_over_at_the_first_phase_is_allowed():
    ex = _REGISTRY["SecantToTangent"].examples()
    a = BeatSpec(id="b01", narration="x", component="SecantToTangent", params=ex[1])
    b = BeatSpec(id="b02", narration="x", component="SecantToTangent", params={**ex[0], "a": 2})
    assert check_continuity(a, b) == []


@pytest.mark.parametrize("name", ANIMATIONS)
def test_cue_names_are_the_phase_moments(name):
    for params in _REGISTRY[name].examples():
        c = make_component(name, params)
        assert c.moment_names() == [m.name for m in c.PHASES[params["phase"]]]
