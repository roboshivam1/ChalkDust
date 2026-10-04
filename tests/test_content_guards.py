"""Content the fonts or TeX cannot honour refuses; it never renders broken.

Two root-cause guards (SCENE_SPEC.md §11 rule 1), pinned here once for every
component that inherits them:

  - LaTeX: MathTex silently repairs malformed source before compiling, so
    theme.check_latex_source compiles the raw source in the same template
    and environment. theme.math and the semantic rung both refuse through it.
  - Text: Pango draws nothing, or a missing-glyph box, for characters the
    resolved font lacks. The theme's text constructors refuse such text as
    kind "unrenderable_text"; rung 1 refuses code points no font could draw.

The maths tests compile LaTeX, so `latex` and `dvisvgm` must be on PATH.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess

import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.core.models import BeatSpec
from chalkdust.scenes import theme as theme_mod
from chalkdust.scenes.components import make_component
from chalkdust.scenes.regions import INVALID_LATEX, UNRENDERABLE_TEXT, LayoutError
from chalkdust.scenes.theme import (
    DEFAULT,
    _installed_fonts,
    body_text,
    check_latex_source,
    check_renderable,
    math,
    mono_text,
    resolve_fonts,
)
from chalkdust.validate.geometric import validate_beat
from chalkdust.validate.semantic import check_latex

# --- LaTeX ------------------------------------------------------------------

# The gate-4/5 oracle: each of these compiled through MathTex and rendered
# altered maths (.run/evidence/gate-45/texprobe/*.tex holds the rewritten
# sources), or failed only by accident of the rewrite.
MALFORMED = [r"\frac{1}{", r"x^{2", r"\undefinedmacro{x} = 1", r"\left( x + 1", r"}} = 1"]


@pytest.mark.parametrize("bad", MALFORMED)
def test_math_refuses_source_mathtex_would_repair(bad, tmp_path):
    with tempconfig({"media_dir": str(tmp_path)}), pytest.raises(LayoutError) as exc:
        math(bad, DEFAULT, what="step[1]")
    assert exc.value.kind == INVALID_LATEX
    assert "step[1]" in str(exc.value)


@pytest.mark.parametrize("good", [r"x &= 1 \\ &= 2",
                                  r"{{ x^2 + 6x }} + 5 &= 0",
                                  r"\left( x + 1 \right)^2"])
def test_valid_aligned_maths_still_compiles(good, tmp_path):
    # The check compiles in align*, as MathTex does: in plain math mode every
    # derivation with & would false-refuse.
    with tempconfig({"media_dir": str(tmp_path)}):
        assert math(good, DEFAULT).width > 0


def test_raw_latex_verdict_is_cached_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(theme_mod, "_latex_verdicts", {})
    with tempconfig({"media_dir": str(tmp_path)}):
        with pytest.raises(LayoutError):
            check_latex_source(r"\frac{1}{")
        records = list((tmp_path / "latex").glob("*.json"))
        assert len(records) == 1
        assert json.loads(records[0].read_text())["error"]
        # A new process starts with no in-memory verdicts; the record alone
        # must answer, without running LaTeX again.
        monkeypatch.setattr(theme_mod, "_latex_verdicts", {})

        def no_latex(*args, **kwargs):
            raise AssertionError("LaTeX ran despite a cached verdict")

        monkeypatch.setattr(subprocess, "run", no_latex)
        with pytest.raises(LayoutError) as exc:
            check_latex_source(r"\frac{1}{")
    assert exc.value.kind == INVALID_LATEX


def test_missing_latex_is_a_toolchain_error_not_invalid_latex(tmp_path, monkeypatch):
    # Regenerating the spec cannot fix a machine without LaTeX, so this must
    # not reach the repair loop as invalid_latex.
    monkeypatch.setattr(theme_mod, "_latex_verdicts", {})

    def absent(*args, **kwargs):
        raise FileNotFoundError("latex")

    monkeypatch.setattr(subprocess, "run", absent)
    with tempconfig({"media_dir": str(tmp_path)}), pytest.raises(RuntimeError):
        check_latex_source(r"x + 1")


def _maths_specs(bad: str) -> list[tuple[str, dict]]:
    """One spec per maths-taking component with `bad` in its maths field.
    GraphPlot is absent on purpose: its LaTeX is generated from parsed
    expressions, so a spec cannot put raw LaTeX into it."""
    return [
        ("EquationDerivation", {"steps": ["x = 1", bad]}),
        ("SplitCompare", {"left": {"title": "Before", "math": bad},
                          "right": {"title": "After", "math": "x"}}),
        ("FreeBodyDiagram", {"body": "block", "forces": [
            {"label": bad, "angle": 270, "magnitude": 10},
            {"label": "N", "angle": 90, "magnitude": 10}]}),
        ("SolutionStep", {"n": 1, "claim": "Solve for x", "work": [bad]}),
        ("AnswerBox", {"value": bad}),
    ]


@pytest.mark.parametrize("name,params", [
    pytest.param(name, params, id=f"{name}-{i}")
    for i, bad in enumerate(MALFORMED) for name, params in _maths_specs(bad)
])
def test_every_maths_component_refuses_malformed_latex(name, params, tmp_path):
    spec = BeatSpec(id="b01", narration="placeholder narration", component=name,
                    params=params)
    semantic = check_latex(make_component(name, params), media_dir=tmp_path)
    assert {f.kind for f in semantic} == {INVALID_LATEX}, semantic
    report = validate_beat(spec, media_dir=tmp_path)
    assert INVALID_LATEX in report.kinds(), report
    assert "build_error" not in report.kinds(), report


# --- text -------------------------------------------------------------------

# The verifiers' repros (c-number-line-walk, c-split-compare, c-step-trace,
# c-graph-plot, c-data-structure-viz): each drew nothing, drew tofu, dropped a
# word, crashed a layout or hung it.
UNDRAWABLE = {
    "zero-width-only": "\u200b",
    "hebrew": "\u05e9\u05dc\u05d5\u05dd",
    "arabic": "\u0639\u0631\u0628\u064a",
    "mixed-rtl": "x \u05e9\u05dc\u05d5\u05dd",
    "emoji-run": "\U0001F355 pizza \U0001F680",
    "private-use-e000": "\ue000",
    "private-use-f06d": "\uf06d",
    "cjk": "\u5f00\u59cb",
}


def _resolved():
    return resolve_fonts(DEFAULT, warn=False)


def _covers(font: str, s: str) -> bool:
    return not theme_mod.unsupported_characters(s, font)


@pytest.mark.parametrize("s", list(UNDRAWABLE.values()), ids=list(UNDRAWABLE))
def test_text_constructors_refuse_what_the_font_cannot_draw(s, tmp_path):
    theme = _resolved()
    if s.strip("\u200b") and _covers(theme.type.body_font, s):
        pytest.skip(f"{theme.type.body_font} draws {s!r} on this machine")
    with tempconfig({"media_dir": str(tmp_path)}), pytest.raises(LayoutError) as exc:
        body_text(s, theme)
    assert exc.value.kind == UNRENDERABLE_TEXT
    assert repr(theme.type.body_font) in str(exc.value)


def test_cjk_in_mono_refuses(tmp_path):
    theme = _resolved()
    if _covers(theme.type.mono_font, "\u5f00"):
        pytest.skip(f"{theme.type.mono_font} draws CJK on this machine")
    with tempconfig({"media_dir": str(tmp_path)}), pytest.raises(LayoutError) as exc:
        mono_text("x = \u5f00", theme)
    assert exc.value.kind == UNRENDERABLE_TEXT
    assert "U+5F00" in str(exc.value)


@pytest.mark.parametrize("s", ["office fluff", "caf\u00e9", "cafe\u0301",
                               "a\u200cb", "x\u00b2 \u2264 \u03b1 \u2192 \u221e"],
                         ids=["ligatures", "precomposed", "combining", "zwnj", "symbols"])
def test_ordinary_text_still_renders(s, tmp_path):
    with tempconfig({"media_dir": str(tmp_path)}):
        assert body_text(s, _resolved()).width > 0


@pytest.mark.parametrize("font,s", [("Calibri", "office fluff"),
                                    ("Cascadia Code", "a <= b -> c")])
def test_ligating_fonts_are_not_judged_by_glyph_count(font, s, tmp_path):
    # Calibri draws "ffi" as one glyph and Cascadia Code draws "<=" as one:
    # fewer glyphs than characters is correct shaping, not missing glyphs.
    if font not in _installed_fonts():
        pytest.skip(f"{font} is not installed")
    theme = dataclasses.replace(DEFAULT, type=dataclasses.replace(
        DEFAULT.type, body_font=font))
    with tempconfig({"media_dir": str(tmp_path)}):
        assert body_text(s, theme).width > 0


def test_check_renderable_is_reusable_on_indirect_text(tmp_path):
    # CodeWalk builds through Manim's Code, not the constructors; it calls the
    # same check on its source with the resolved mono font.
    font = _resolved().type.mono_font
    with tempconfig({"media_dir": str(tmp_path)}):
        check_renderable("for i in range(n):\n    total += i", font)
        if _covers(font, "\U0001F600"):
            pytest.skip(f"{font} draws emoji on this machine")
        with pytest.raises(LayoutError) as exc:
            check_renderable("print('\U0001F600')", font, what="source line 1")
    assert exc.value.kind == UNRENDERABLE_TEXT
    assert "source line 1" in str(exc.value)


def test_manims_fewer_glyphs_error_becomes_the_typed_refusal(monkeypatch):
    def fewer(*args, **kwargs):
        raise ValueError("Text 'x' rendered fewer glyph(s) than its non-space "
                         "characters even with disable_ligatures=True.")

    monkeypatch.setattr(theme_mod, "Text", fewer)
    monkeypatch.setattr(theme_mod, "check_renderable", lambda *a, **k: None)
    with pytest.raises(LayoutError) as exc:
        body_text("x", _resolved())
    assert exc.value.kind == UNRENDERABLE_TEXT


# Each repro in a merged component: a clean refusal, never a crash or a hang.
# model_construct skips rung 1, which refuses private-use first, so the theme
# guard itself is what is tested.
COMPONENT_REPROS = {
    "SplitCompare-zero-width-title": ("SplitCompare", {
        "left": {"title": "\u200b"}, "right": {"title": "After"}}),
    "SplitCompare-hebrew-body": ("SplitCompare", {
        "left": {"title": "Before", "body": UNDRAWABLE["hebrew"]},
        "right": {"title": "After"}}),
    "NumberLineWalk-hebrew-jump-label": ("NumberLineWalk", {
        "range": [0, 10], "steps": [{"at": 1}, {"to": 7, "label": UNDRAWABLE["hebrew"]}]}),
    "NumberLineWalk-arabic-mark-label": ("NumberLineWalk", {
        "range": [0, 10], "steps": [{"at": 1, "label": UNDRAWABLE["arabic"]}]}),
    "GraphPlot-arabic-marker": ("GraphPlot", {
        "functions": [{"expr": "x^2"}], "x_range": [-2, 2],
        "markers": [{"x": 1, "label": UNDRAWABLE["arabic"]}]}),
    "StepTrace-emoji-run": ("StepTrace", {
        "variables": ["x"], "frames": [[UNDRAWABLE["emoji-run"]]]}),
    "BoxFlow-cjk": ("BoxFlow", {
        "nodes": [{"id": "a", "label": UNDRAWABLE["cjk"]}, {"id": "b", "label": "End"}],
        "edges": [{"source": "a", "target": "b"}]}),
    "DataStructureViz-cjk-in-mono": ("DataStructureViz", {
        "kind": "array", "initial": [UNDRAWABLE["cjk"], "b"]}),
    "TitleCard-cjk": ("TitleCard", {"title": UNDRAWABLE["cjk"]}),
    "TitleCard-private-use": ("TitleCard", {"title": "Menu \ue000"}),
    "BulletReveal-hebrew": ("BulletReveal", {"items": [UNDRAWABLE["hebrew"]]}),
}


@pytest.mark.parametrize("name,params", list(COMPONENT_REPROS.values()),
                         ids=list(COMPONENT_REPROS))
def test_components_refuse_undrawable_text_cleanly(name, params, tmp_path):
    spec = BeatSpec.model_construct(id="b01", narration="placeholder narration",
                                    component=name, params=params, registers=None,
                                    carry_in=[])
    report = validate_beat(spec, media_dir=tmp_path)
    assert report.kinds() == {UNRENDERABLE_TEXT}, report


# --- rung 1 -----------------------------------------------------------------


@pytest.mark.parametrize("params", [
    {"title": "\ue000 menu"},
    {"title": "\uf06d"},
    {"items": ["fine", "\U000E0080"]},  # unassigned (past the Tags block)
    {"left": {"title": "ok", "body": "a\ud800b"}},  # a lone surrogate
], ids=["private-use-e000", "private-use-f06d", "unassigned", "surrogate"])
def test_rung1_refuses_code_points_no_font_can_draw(params):
    with pytest.raises(ValidationError):
        BeatSpec(id="b01", narration="placeholder narration",
                 component="TitleCard", params=params)


def test_rung1_refuses_private_use_in_narration():
    with pytest.raises(ValidationError, match="private-use"):
        BeatSpec(id="b01", narration="press \ue000 to start", component="TitleCard",
                 params={"title": "Start"})


def test_rung1_keeps_format_characters_for_the_component_to_judge():
    # ZWNJ and ZWJ are part of real words (Persian, Indic scripts), and a
    # zero-width-only param may be a component's "absent" (GeometryConstruct
    # reads such a note as no note); the theme refuses it only if drawn.
    spec = BeatSpec(id="b01", narration="mi\u200cxed na\u00adrration",
                    component="TitleCard",
                    params={"title": "a\u200cb", "subtitle": "x\u200dy",
                            "kicker": "\u200b"})
    assert spec.params["title"] == "a\u200cb"
