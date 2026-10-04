"""The geometric probe (rung 3) itself."""

from __future__ import annotations

import subprocess
import uuid

import pytest
from manim import RIGHT, tempconfig

from chalkdust.continuity import ArtifactRecipe
from chalkdust.core.models import BeatSpec, Region
from chalkdust.scenes import theme as theme_mod
from chalkdust.scenes.components import Component, ComponentParams
from chalkdust.scenes.theme import body_text
from chalkdust.validate.geometric import LayoutProbe, run_probe, validate_beat
from chalkdust.validate.repair import repair_beat


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


# --- typed failures from inside build() (register D-G4b-1, G4b-N6) ----------

_THREE_ROWS = ArtifactRecipe(name="causes", producer="BulletReveal",
                             params={"items": ["A weak hash", "A high load",
                                               "Chosen keys"]})


@pytest.mark.parametrize("rung", [validate_beat, repair_beat],
                         ids=["geometric", "repair"])
@pytest.mark.parametrize("name,params", [
    ("Callout", {"target_id": "causes", "text": "note", "part": 7}),
    ("ZoomHighlight", {"target_id": "causes", "callout": "note", "parts": [9]}),
])
def test_a_carry_in_error_in_build_is_a_carry_in_finding(rung, name, params,
                                                         tmp_path):
    # A part the carried artifact does not have raises CarryInError inside
    # build(). It is the spec's fault, typed as rung 2 types it -- never a
    # build_error (a crash) the repair loop would misread.
    spec = BeatSpec(id="b02", narration="placeholder narration", component=name,
                    params=params, carry_in=["causes"])
    with tempconfig({"media_dir": str(tmp_path)}):
        result = rung(spec, recipes=[_THREE_ROWS])
    report = getattr(result, "report", result)
    assert report.kinds() == {"carry_in"}, report
    assert "'causes'" in str(report)


def test_a_latex_timeout_in_build_is_a_toolchain_finding(tmp_path, monkeypatch):
    # TeX still running after its timeout, twice, is the machine's fault:
    # kind "toolchain", never invalid_latex or build_error.
    def hung(command):
        raise subprocess.TimeoutExpired(command, theme_mod.LATEX_CHECK_TIMEOUT)

    monkeypatch.setattr(theme_mod, "_compile", hung)
    monkeypatch.setattr(theme_mod, "_latex_verdicts", {})
    spec = BeatSpec(id="b01", narration="placeholder narration",
                    component="EquationDerivation",
                    params={"steps": ["x = 1", f"x = {uuid.uuid4().int}"]})
    report = validate_beat(spec, media_dir=tmp_path)
    assert report.kinds() == {"toolchain"}, report
