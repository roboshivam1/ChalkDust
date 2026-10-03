"""CodeWalk-specific behaviour.

Layout of examples() and stress() is covered by test_layout.py walking the
registry; this file pins what that walk cannot see: the timing contract
(clocked frames and one real draft render counted with ffprobe), the schema's
refusals, the glyph guard's refusals (text the resolved mono font cannot draw
one glyph per character, kind "unrenderable_text"), where the highlight
actually lands, and the carry-in artifact.

CodeWalk compiles no LaTeX (latex_strings() is empty), so there is no
invalid-LaTeX case here or in stress(); nothing on PATH beyond ffmpeg/ffprobe.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict

import numpy as np
import pytest
from manim import Code, tempconfig
from manim.animation.animation import prepare_animation
from pydantic import ValidationError

from chalkdust.continuity import ArtifactRecipe, build_artifact, resolve_carry_in
from chalkdust.core.models import BeatSpec, Quality, Region, VideoSpec
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import code_walk
from chalkdust.scenes.components.code_walk import (
    DIM_OPACITY,
    MAX_HIGHLIGHTS,
    CodeWalk,
)
from chalkdust.scenes.regions import UNRENDERABLE_TEXT, LayoutError, bbox, fit_to_region
from chalkdust.scenes.theme import DEFAULT, resolve_fonts
from chalkdust.validate.geometric import LayoutProbe, validate_beat

SOURCE = "def f(x):\n    y = x * 2\n    return y\n\nprint(f(3))"
EXAMPLES = CodeWalk.examples()
DRAFT = TIERS[Quality.DRAFT]
THEME = resolve_fonts(DEFAULT, warn=False)
MONO = THEME.type.mono_font  # Courier New where JetBrains Mono is missing


def _probe(component: CodeWalk) -> LayoutProbe:
    probe = LayoutProbe(component, duration=10.0)
    probe.construct()
    return probe


class _Clock(LayoutProbe):
    """A probe that records, per play() and wait(), the frames Manim would
    render for it -- the beat's real length, without encoding a frame. A
    play() with no explicit run_time counts at its animations' own default, so
    a forgotten run_time shows up as a frame mismatch."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.segments: list[float] = []

    def play(self, *animations, **kwargs) -> None:  # type: ignore[override]
        run_time = kwargs.get("run_time")
        if run_time is None:
            run_time = max(prepare_animation(a).run_time for a in animations)
        self.segments.append(run_time * self.fps)
        super().play(*animations, **kwargs)

    def wait(self, duration: float = 1.0, *args, **kwargs) -> None:  # type: ignore[override]
        self.segments.append(duration * self.fps)


def _forbid_building_code(monkeypatch) -> None:
    """Make any Code construction for a listing fail the test. The size
    check's own probe listing (_cell, cached per theme) is built first."""
    code_walk._cell(THEME)

    class _NoCode:
        default_paragraph_config = Code.default_paragraph_config

        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("Code was built for a listing the guard should refuse")

    monkeypatch.setattr(code_walk, "Code", _NoCode)


def _rows_hold(code: Code, source: str) -> bool:
    """Each source line has one glyph per visible character, every one of
    them on that line's row -- what a glyph-to-character shift breaks (the
    shifted glyphs land on the row above, while the counts can still match)."""
    nums = code.line_numbers
    pitch = abs(nums[0].get_y() - nums[1].get_y())
    for i, line in enumerate(source.split("\n")):
        glyphs = list(code.code_lines[i])
        if len(glyphs) != sum(not c.isspace() for c in line):
            return False
        top, bottom = nums[i].get_y() + pitch / 2, nums[i].get_y() - pitch / 2
        if any(g.get_bottom()[1] >= top or g.get_top()[1] <= bottom for g in glyphs):
            return False
    return True


def _clocked(params: dict, duration: float) -> _Clock:
    with tempconfig({"frame_rate": DRAFT.frame_rate}):
        clock = _Clock(CodeWalk(params), duration=duration)
        clock.construct()
    return clock


class TestTiming:
    """Animation lasts exactly the beat's frames, ceil(audio * fps) (D-002),
    whether narration runs far shorter or far longer than the walk wants."""

    @pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
    @pytest.mark.parametrize("params", EXAMPLES, ids=[f"ex{i}" for i in range(len(EXAMPLES))])
    def test_plays_exactly_the_beats_frames(self, params, factor):
        # 0.37 s off a whole frame, so the beat has to round up.
        duration = CodeWalk(params).min_seconds() * factor + 0.37 / DRAFT.frame_rate
        clock = _clocked(params, duration)
        frames = [round(f) for f in clock.segments]
        # Every segment is whole frames: nothing rounds on its own any more.
        assert clock.segments == pytest.approx(frames, abs=1e-9)
        assert sum(frames) == clock.beat_frames == math.ceil(duration * DRAFT.frame_rate)

    def test_draft_render_is_exactly_the_beat(self, tmp_path):
        # A real 480p15 render, frames counted by ffprobe: 3.879 s of audio is
        # ceil(3.879 * 15) = 59 frames across this example's ten segments.
        duration = 3.879
        with tempconfig({**asdict(DRAFT), "media_dir": str(tmp_path),
                         "disable_caching": True, "progress_bar": "none",
                         "verbosity": "WARNING", "output_file": "codewalk"}):
            scene = ChalkdustScene(CodeWalk(EXAMPLES[1]), duration=duration)
            scene.render()
            movie = scene.renderer.file_writer.movie_file_path
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
            capture_output=True, text=True, check=True).stdout
        frames = int(json.loads(out)["streams"][0]["nb_read_frames"])
        assert frames == math.ceil(duration * DRAFT.frame_rate) == 59

    def test_min_seconds_grows_with_each_highlight(self):
        bare = CodeWalk({"language": "python", "source": SOURCE})
        walked = CodeWalk({"language": "python", "source": SOURCE,
                           "highlights": [{"start": 1}, {"start": 2}]})
        assert bare.min_seconds() > 0
        per_step = (walked.min_seconds() - bare.min_seconds()) / 2
        assert per_step > 0

    def test_takes_no_latex(self):
        assert CodeWalk({"language": "python", "source": SOURCE}).latex_strings() == []


class TestSchema:
    """Input the component cannot honour fails at validation, as a pydantic
    error -- never as a crash inside build()."""

    @pytest.mark.parametrize("params", [
        {"language": "python", "source": ""},
        {"language": "python", "source": "  \n\t\n   "},
        {"language": "klingon", "source": "x = 1"},
        {"language": "python", "source": "x = 1", "highlights": [{"start": 2}]},
        {"language": "python", "source": SOURCE, "highlights": [{"start": 0}]},
        {"language": "python", "source": SOURCE,
         "highlights": [{"start": 3, "end": 2}]},
        {"language": "python", "source": SOURCE,
         "highlights": [{"start": 1, "colour": "red"}]},
        {"language": "python", "source": SOURCE,
         "highlights": [{"start": 1}] * (MAX_HIGHLIGHTS + 1)},
        {"language": "python", "source": SOURCE, "highlights": [{"start": True}]},
        {"language": "python", "source": SOURCE, "highlights": [{"start": "2"}]},
        {"language": "python", "source": SOURCE,
         "highlights": [{"start": 1, "end": "3"}]},
    ], ids=["empty", "whitespace", "unknown-language", "past-last-line",
            "line-zero", "reversed-span", "unknown-span-field", "too-many-steps",
            "bool-line", "string-line", "string-end"])
    def test_rejects(self, params):
        with pytest.raises(ValidationError):
            CodeWalk(params)

    @pytest.mark.parametrize("source", [
        "x = 1  # done \u2705",
        "print('\U0001F680')",
        "ok = True  # \U0001F44D\U0001F3FD",
        "s = 'a\u200bb'",
        "s = 'a\u200db'",
        "x = 1  \u200f# rtl",
        "# hyphen\u00adated",
    ], ids=["emoji-bmp", "emoji-astral", "emoji-skin-tone", "zero-width-space",
            "zero-width-joiner", "bidi-mark", "soft-hyphen"])
    def test_rejects_characters_a_listing_cannot_draw(self, source):
        # Code demands one glyph per non-space character; these never give
        # one and used to escape build() as a bare ValueError.
        with pytest.raises(ValidationError, match=r"line 1 contains U\+"):
            CodeWalk({"language": "python", "source": source})

    def test_decomposed_accent_is_composed_and_draws(self):
        # "e" + U+0301 is two characters Pango draws as one glyph, which Code
        # refused; NFC makes it the one character it looks like.
        comp = CodeWalk({"language": "python", "source": "s = 'cafe\u0301'"})
        assert comp.params.source == "s = 'caf\u00e9'"
        assert [getattr(m, "_chalk_label", None) for m in _probe(comp).mobjects] == ["code"]

    def test_blank_edges_dropped_so_numbers_match_screen(self):
        # Pygments strips leading newlines before lexing; if the spec kept
        # them, "line 3" in the spec would be line 1 on screen.
        comp = CodeWalk({"language": "python", "source": "\n\nx = 1\ny = 2  \n\n",
                         "highlights": [{"start": 2}]})
        assert comp.params.source == "x = 1\ny = 2"
        with pytest.raises(ValidationError):
            CodeWalk({"language": "python", "source": "\n\nx = 1\ny = 2\n\n",
                      "highlights": [{"start": 3}]})


class TestBuild:
    def test_too_wide_refuses_with_overflow(self):
        comp = CodeWalk({"language": "python",
                         "source": "# https://example.com/" + "segment-" * 20})
        with pytest.raises(LayoutError) as info:
            _probe(comp)
        assert info.value.kind == "overflow"
        assert "columns" in str(info.value)

    @pytest.mark.parametrize("source", [
        "\n".join(f"x{i} = {i}" for i in range(200)),
        "x = '" + "a" * 3000 + "'",
    ], ids=["200-lines", "3000-columns"])
    def test_huge_listing_refuses_overflow_before_code_is_built(self, source):
        # Past the frame's size Pango silently drops glyphs and Code raised a
        # bare ValueError; the size check now refuses first.
        with pytest.raises(LayoutError, match="legibility floor") as info:
            _probe(CodeWalk({"language": "python", "source": source}))
        assert info.value.kind == "overflow"

    def test_glyph_the_font_cannot_draw_refuses_unrenderable(self):
        # Past the schema's list: a combining grapheme joiner draws nothing.
        # It is refused as the theme refuses text its font cannot draw, not
        # as "illegible" (that kind means too small, which splitting the beat
        # would fix; this only a rewrite fixes).
        comp = CodeWalk({"language": "python", "source": "x = 'a\u034fb'"})
        with pytest.raises(LayoutError, match=r"U\+0061 U\+034F") as info:
            _probe(comp)
        assert info.value.kind == UNRENDERABLE_TEXT

    @pytest.mark.parametrize("source,named", [
        ("# \u4f60\u597d\u4e16\u754c\nname = '\u6771\u4eac'  # tokyo", "U+4F60"),
        ("# \u2200x \u2264 \u221e\nx = 1", "U+2200"),
        ("# \u0928\u092e\u0938\u094d\u0924\u0947\nx = 1", "U+0928"),
        ("# \u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\nx = 1", "U+0E2A"),
        ("# \u05e9\u05dc\u05d5\u05dd\nx = 1", "U+05E9"),
        ("# \u0645\u0631\u062d\u0628\u0627\nx = 1", "U+0645"),
        ("love = '\u2764\ufe0f'\nx = 1", "U+2764"),
        ("key = '1\ufe0f\u20e3'\nx = 1", "U+0031 U+FE0F U+20E3"),
        ("s = 'x\ufe0f'\nx = 1", "U+0078 U+FE0F"),
        ("\u0301x = 1\ny = 2", "U+0301"),
        ("p = '\ue000'\nx = 1", "U+E000"),
        ("u = '\u0378'\nx = 1", "U+0378"),
        ("m = '\U0001d400'\nx = 1", "U+1D400"),
    ], ids=["cjk", "maths-symbols", "devanagari", "thai", "hebrew", "arabic",
            "heart-vs16", "keycap", "variation-selector", "lone-mark",
            "private-use", "unassigned", "math-alnum"])
    def test_text_the_mono_font_cannot_draw_refuses_before_code(self, source, named,
                                                                monkeypatch):
        # Each of these drew missing-glyph boxes or nothing; Code then mapped
        # glyphs to characters off by the difference, dropped later lines and
        # left its " pA1" alignment glyphs in the frame, and validation said
        # ok. The theme's glyph guard refuses them before Code is built,
        # naming the character, its line and the resolved font.
        _forbid_building_code(monkeypatch)
        comp = CodeWalk({"language": "python", "source": source})
        with pytest.raises(LayoutError) as info:
            _probe(comp)
        assert info.value.kind == UNRENDERABLE_TEXT
        message = str(info.value)
        assert named in message and repr(MONO) in message
        assert re.match(r"CodeWalk source lines? 1\b", message)

    def test_context_shaping_the_probes_miss_is_still_refused(self, monkeypatch):
        # The last check is Code's own assumption, on the whole listing: one
        # glyph per visible character. Blind the cluster probe (as shaping in
        # context would) and the keycap still refuses, before Code is built.
        monkeypatch.setattr(code_walk, "_cluster_paths",
                            lambda cluster, font: sum(not c.isspace() for c in cluster))
        _forbid_building_code(monkeypatch)
        comp = CodeWalk({"language": "python", "source": "key = '1\ufe0f\u20e3'\nx = 1"})
        with pytest.raises(LayoutError,
                           match="draws 15 glyphs for its 12 visible characters") as info:
            _probe(comp)
        assert info.value.kind == UNRENDERABLE_TEXT

    @pytest.mark.parametrize("source", [
        "s = 'na\u00efve caf\u00e9 \u00df \u00f1'\nx = 1",
        "s = '\u03b1\u03b2\u03b3 \u03a9 \u03bb'\nx = 1",
        "s = '\u043f\u0440\u0438\u0432\u0435\u0442'\nx = 1",
        "s = 'q\u0301'\nx = 1",
        "v = 'Vi\u1ec7t \u01d8'\nx = 1",
    ], ids=["latin-1", "greek", "cyrillic", "q-acute", "vietnamese"])
    def test_text_the_font_draws_renders_on_its_rows(self, source):
        # The guard must not cost the scripts the font does draw: each glyph
        # lands on its own line's row, none shifted, nothing left over.
        probe = _probe(CodeWalk({"language": "python", "source": source,
                                 "highlights": [{"start": 1}]}))
        code, _bar = probe.mobjects
        assert _rows_hold(code, source)

    def test_minimal_listing_has_no_highlight_bar(self):
        probe = _probe(CodeWalk({"language": "c", "source": "x"}))
        assert [getattr(m, "_chalk_label", None) for m in probe.mobjects] == ["code"]

    def test_bar_covers_exactly_the_last_span_and_dims_the_rest(self):
        probe = _probe(CodeWalk({"language": "python", "source": SOURCE,
                                 "highlights": [{"start": 1}, {"start": 2, "end": 3}]}))
        code, bar = probe.mobjects
        assert getattr(bar, "_chalk_label") == "highlight"
        nums = code.line_numbers
        inside = {i for i, n in enumerate(nums)
                  if bar.get_bottom()[1] < n.get_y() < bar.get_top()[1]}
        assert inside == {1, 2}
        # Behind the text, or the wash tints the glyphs it is meant to frame.
        assert bar.z_index < code.code_lines.z_index
        lit = [nums[i].get_fill_opacity() for i in range(len(nums))]
        assert lit == pytest.approx([DIM_OPACITY, 1, 1, DIM_OPACITY, DIM_OPACITY])


class TestCarryIn:
    """A walked listing can be carried into a later beat (SCENE_SPEC.md §6),
    so a ZoomHighlight or Callout can act on code the viewer has just read."""

    def _artifact(self, params: dict) -> Code:
        recipe = ArtifactRecipe(name="listing", producer="CodeWalk", params=params)
        return build_artifact(recipe, resolve_fonts(DEFAULT, warn=False))

    def test_artifact_is_the_settled_last_frame(self):
        # Same listing, same bar on the last span, same dimming as the frame
        # the producing beat ended on -- the picture persists across the cut.
        params = EXAMPLES[1]
        code, bar = _probe(CodeWalk(params)).mobjects
        art = self._artifact(params)
        fit_to_region(art, Region.STAGE)
        art_bar = art.submobjects[-1]
        assert getattr(art_bar, "_chalk_label") == "highlight"
        for got, want in ((art.background, code.background), (art_bar, bar)):
            g, w = bbox(got), bbox(want)
            assert (g.x, g.y, g.width, g.height) == pytest.approx(
                (w.x, w.y, w.width, w.height), abs=1e-6)
        opacity = [n.get_fill_opacity() for n in art.line_numbers]
        assert opacity == pytest.approx([n.get_fill_opacity() for n in code.line_numbers])
        assert art_bar.z_index < art.code_lines.z_index

    def test_rebuild_is_deterministic(self):
        a, b = self._artifact(EXAMPLES[0]), self._artifact(EXAMPLES[0])
        pa = [m.points for m in a.family_members_with_points()]
        pb = [m.points for m in b.family_members_with_points()]
        assert len(pa) == len(pb) > 0
        assert all(np.array_equal(x, y) for x, y in zip(pa, pb))

    def test_artifact_refuses_an_oversized_listing_like_the_beat(self):
        params = {"language": "python",
                  "source": "\n".join(f"x{i} = {i}" for i in range(200))}
        with pytest.raises(LayoutError) as info:
            self._artifact(params)
        assert info.value.kind == "overflow"

    def test_artifact_refuses_unrenderable_text_like_the_beat(self):
        # The builder shares the guarded path: a carried listing cannot bring
        # back the garbled CJK listing the producing beat refuses.
        params = {"language": "python", "source": "# \u4f60\u597d\nx = 1"}
        with pytest.raises(LayoutError, match=r"U\+4F60") as info:
            self._artifact(params)
        assert info.value.kind == UNRENDERABLE_TEXT

    def test_carried_listing_validates_in_a_later_beat(self):
        video = VideoSpec(video_id="v", beats=(
            BeatSpec(id="b01", narration="placeholder narration",
                     component="CodeWalk", params=EXAMPLES[1], registers="listing"),
            BeatSpec(id="b02", narration="placeholder narration",
                     component="BulletReveal", params={"items": ["one point"]},
                     carry_in=["listing"]),
        ))
        recipes = resolve_carry_in(video)["b02"]
        assert recipes[0].producer == "CodeWalk"
        report = validate_beat(video.beats[1], duration=4.0, recipes=recipes)
        assert report.ok, f"\n{report}"
