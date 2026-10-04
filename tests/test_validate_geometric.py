"""The geometric probe (rung 3) itself."""

from __future__ import annotations

from manim import RIGHT, tempconfig

from chalkdust.core.models import BeatSpec, Region
from chalkdust.scenes.components import Component, ComponentParams
from chalkdust.scenes.theme import body_text
from chalkdust.validate.geometric import LayoutProbe, run_probe, validate_beat


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


def _title_beat(title: str) -> BeatSpec:
    return BeatSpec(id="b01", narration="placeholder narration",
                    component="TitleCard", params={"title": title})


def test_probe_writes_nothing_into_the_cwd(tmp_path, monkeypatch):
    # Building Text writes Pango SVGs under Manim's media dir, ./media by
    # default. The probe must route that to the work dir it is given, or to
    # the default work/manim -- never media/ wherever validation was run.
    monkeypatch.chdir(tmp_path)
    given = tmp_path / "given"
    assert validate_beat(_title_beat("Routed scratch"), media_dir=given).ok
    assert validate_beat(_title_beat("Default scratch")).ok
    assert not (tmp_path / "media").exists()
    assert any((given / "texts").iterdir())
    assert any((tmp_path / "work" / "manim" / "texts").iterdir())


def test_probe_keeps_a_callers_media_dir(tmp_path, monkeypatch):
    # The pipeline and worker scope Manim to their work dir with tempconfig;
    # the probe must use that rather than substituting its own default.
    monkeypatch.chdir(tmp_path)
    with tempconfig({"media_dir": str(tmp_path / "callers")}):
        assert validate_beat(_title_beat("Caller scratch")).ok
    assert any((tmp_path / "callers" / "texts").iterdir())
    assert not (tmp_path / "work").exists()
