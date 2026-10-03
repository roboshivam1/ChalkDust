"""The geometric probe (rung 3) itself."""

from __future__ import annotations

from manim import RIGHT

from chalkdust.core.models import Region
from chalkdust.scenes.components import Component, ComponentParams
from chalkdust.scenes.theme import body_text
from chalkdust.validate.geometric import LayoutProbe, run_probe


class _AnimatesWithoutAdding(Component):
    """Plays a non-introducer animation on a mobject it never add()ed --
    legal Manim: Scene.play adds it first. Not registered."""

    name = "_AnimatesWithoutAdding"
    Params = ComponentParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def build(self, scene) -> None:
        text = body_text("Off the edge", scene.theme).shift(RIGHT * 9)
        scene.play(text.animate.set_opacity(0.5), run_time=scene.beat_duration)


def test_probe_sees_what_a_render_adds():
    # A real render shows this mobject, so the probe must check it too.
    component = _AnimatesWithoutAdding({})
    probe = LayoutProbe(component, duration=4.0, strict=False)
    assert run_probe(probe, "b01").kinds() == {"out_of_bounds"}
