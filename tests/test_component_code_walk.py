"""CodeWalk-specific behaviour.

Layout of examples() and stress() is covered by test_layout.py walking the
registry; this file pins what that walk cannot see: the timing contract
(clocked frames and one real draft render counted with ffprobe), the schema's
refusals, the glyph guard's refusals (text the resolved mono font cannot draw
one glyph per character, kind "unrenderable_text"), where the highlight
actually lands, the carry-in artifact, and that a consumer naming part k of
a carried listing (ZoomHighlight, Callout) lands on source line k + 1.

CodeWalk compiles no LaTeX (latex_strings() is empty), so there is no
invalid-LaTeX case here or in stress(); nothing on PATH beyond ffmpeg/ffprobe.
"""

from __future__ import annotations

import json
import math
import dataclasses
import re
import subprocess
import unicodedata
from dataclasses import asdict

import numpy as np
import pytest
from manim import Code, Group, Text, tempconfig
from manim.animation.animation import prepare_animation
from pydantic import ValidationError

from chalkdust.continuity import (
    ArtifactRecipe,
    beat_component,
    build_artifact,
    part_range_problems,
    resolve_carry_in,
)
from chalkdust.core.models import BeatSpec, Quality, Region, VideoSpec
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import code_walk, make_component
from chalkdust.scenes.components.code_walk import (
    DIM_OPACITY,
    MAX_HIGHLIGHTS,
    CodeWalk,
)
from chalkdust.scenes.components.zoom_highlight import FRAME_BUFF, MIN_ZOOM
from chalkdust.scenes.regions import UNRENDERABLE_TEXT, LayoutError, bbox, fit_to_region
from chalkdust.scenes.theme import DEFAULT, resolve_fonts
from chalkdust.validate.geometric import LayoutProbe, validate_beat

SOURCE = "def f(x):\n    y = x * 2\n    return y\n\nprint(f(3))"
EXAMPLES = CodeWalk.examples()
DRAFT = TIERS[Quality.DRAFT]
THEME = resolve_fonts(DEFAULT, warn=False)
MONO = THEME.type.mono_font  # Courier New where JetBrains Mono is missing
# Courier New, named outright: the fallback mono font on Windows and macOS,
# for facts that belong to it -- glyphs it draws in the wrong column -- and
# that the theme's JetBrains Mono does not reproduce.
COURIER = dataclasses.replace(
    THEME, type=dataclasses.replace(THEME.type, mono_font="Courier New"))
# Character sequences a programming font (JetBrains Mono, Fira Code) draws
# as one ligature through 'calt' unless the listing turns it off.
PROGRAMMING_LIGATURES = (
    "if lo <= hi and a >= b and a != b and x == y: mid = (lo + hi) >> 1\n"
    "f = lambda x: x ** 2  # -> => <- |> <| :: ... /* */ <!-- --> www\n"
    "n = 0x1F; s = '===' + '!==' + '<=>' + '>>=' + '<<' + '&&' + '||'"
)
# Precomposed letters past Latin-1 (o and u double acute, a and e ogonek,
# c s z caron, l stroke): one character, one glyph, one column each.
LATIN_EXT_A = "s = '\u0151\u0171 \u0105\u0119 \u010d\u0161\u017e \u0142'\nx = 1"


def _probe(component: CodeWalk) -> LayoutProbe:
    probe = LayoutProbe(component, duration=10.0)
    probe.construct()
    return probe


def _probe_in(component: CodeWalk, theme) -> LayoutProbe:
    probe = LayoutProbe(component, duration=10.0, theme=theme)
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

    @pytest.mark.parametrize("source,named,line", [
        ("s = 'q\u0301'\nx = 1", "U+0071 U+0301", 1),
        ('s = "\u0130".lower()  # \'i\u0307\'', "U+0069 U+0307", 1),
        ("s = 'x\u0301y'", "U+0078 U+0301", 1),
        ("x = ' \u0301'", "U+0020 U+0301", 1),
        ("x = 1\ns = 'a" + "\u0323\u0324\u0325\u0326" * 5 + "'", "U+0326", 2),
        ("z\u0300\u0301\u0302\u0303\u0304\u0305 = 1", "U+0305", 1),
        ("x = 'a\u034fb'", "U+0061 U+034F", 1),
        ("love = '\u2764\ufe0f'\nx = 1", "U+2764 U+FE0F", 1),
        ("key = '1\ufe0f\u20e3'\nx = 1", "U+0031 U+FE0F U+20E3", 1),
        ("s = 'x\ufe0f'\nx = 1", "U+0078 U+FE0F", 1),
        ("\u0301x = 1\ny = 2", "U+0301", 1),
        ("# \u0928\u092e\u0938\u094d\u0924\u0947\nx = 1", "U+094D", 1),
        ("# \u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\nx = 1", "U+0E31", 1),
    ], ids=["q-acute", "turkish-dotted-i", "acute-between-letters", "acute-on-space",
            "below-mark-stack", "stacked-marks", "grapheme-joiner", "heart-vs16",
            "keycap", "variation-selector", "lone-mark", "devanagari-virama",
            "thai-vowel-marks"])
    def test_rejects_combining_marks_naming_them_and_their_line(self, source, named,
                                                                line):
        # A nonspacing or enclosing mark NFC cannot fold into a precomposed
        # letter is drawn off its letter in Courier New, about one cell right:
        # q + U+0301 put the accent on the closing quote, 'x' + U+0301 + 'y'
        # read as 'xy' with the accent on the y, and "İ".lower() lost its dot
        # to the quote -- while validation said ok. Refused at rung 1, naming
        # each mark, the letter it sits on and the line.
        # q-acute moved here from the rows and columns "renders" tests, which
        # pinned that misrender as correct; stacked-marks from the column
        # check, and the joiner, VS16, keycap, variation selector, lone mark
        # and the marked Devanagari and Thai words from the font guard, all
        # of which now never see a mark.
        with pytest.raises(ValidationError) as info:
            CodeWalk({"language": "python", "source": source})
        message = str(info.value)
        assert f"source line {line} contains combining mark" in message
        assert named in message
        assert "precomposed letter" in message

    def test_combining_marks_on_several_lines_are_all_located(self):
        # The repair loop fixes the listing in one pass only if it hears of
        # every line: the first is spelled out, the rest are listed.
        with pytest.raises(ValidationError,
                           match=r"source line 2 contains .* Also on line 4\."):
            CodeWalk({"language": "python",
                      "source": "x = 1\ns = 'q\u0301'\ny = 2\nt = 'j\u0301'"})

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
        # Past the schema's list: U+3164 HANGUL FILLER is a letter that looks
        # blank, and Courier New draws a missing-glyph box for it. (The
        # combining grapheme joiner this case used to pin is now refused at
        # the schema, as a combining mark.) It is refused as the theme
        # refuses text its font cannot draw, not as "illegible" (that kind
        # means too small, which splitting the beat would fix; this only a
        # rewrite fixes).
        comp = CodeWalk({"language": "python", "source": "x = 'a\u3164b'"})
        with pytest.raises(LayoutError, match=r"U\+3164") as info:
            _probe(comp)
        assert info.value.kind == UNRENDERABLE_TEXT

    @pytest.mark.parametrize("source,named", [
        ("# \u4f60\u597d\u4e16\u754c\nname = '\u6771\u4eac'  # tokyo", "U+4F60"),
        ("# \u2230x \u225e \u221e\nx = 1", "U+2230"),
        ("# \u0915\u092e\u0932\nx = 1", "U+0915"),
        ("# \u0e2a\u0e1a\u0e32\u0e22\nx = 1", "U+0E2A"),
        ("# \u05e9\u05dc\u05d5\u05dd\nx = 1", "U+05E9"),
        ("# \u0645\u0631\u062d\u0628\u0627\nx = 1", "U+0645"),
        ("love = '\u2764'\nx = 1", "U+2764"),
        ("p = '\ue000'\nx = 1", "U+E000"),
        ("u = '\u0378'\nx = 1", "U+0378"),
        ("m = '\U0001d400'\nx = 1", "U+1D400"),
    ], ids=["cjk", "maths-symbols", "devanagari", "thai", "hebrew", "arabic",
            "heart", "private-use", "unassigned", "math-alnum"])
    def test_text_the_mono_font_cannot_draw_refuses_before_code(self, source, named,
                                                                monkeypatch):
        # Each of these drew missing-glyph boxes or nothing; Code then mapped
        # glyphs to characters off by the difference, dropped later lines and
        # left its " pA1" alignment glyphs in the frame, and validation said
        # ok. The theme's glyph guard refuses them before Code is built,
        # naming the character, its line and the resolved font. (The VS16
        # heart, keycap, variation selector and lone mark moved to the
        # schema's combining-mark refusal, as did the Devanagari and Thai
        # words, whose virama and vowel signs are marks; the bare heart and
        # unmarked Devanagari and Thai letters stay here. The maths case was
        # U+2200 FOR ALL, which only Courier New lacks: JetBrains Mono draws
        # it, and it renders. U+2230 VOLUME INTEGRAL is missing from both.)
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
        # context would) and U+1680 OGHAM SPACE MARK -- a space to every
        # per-character check, a dash in the line -- still refuses, before
        # Code is built. (This used to be the keycap, which the schema now
        # refuses as combining marks.)
        monkeypatch.setattr(code_walk, "_cluster_paths",
                            lambda cluster, font: sum(not c.isspace() for c in cluster))
        _forbid_building_code(monkeypatch)
        comp = CodeWalk({"language": "python", "source": "s = 'a\u1680b'\nx = 1"})
        with pytest.raises(LayoutError,
                           match="draws 10 glyphs for its 9 visible characters") as info:
            _probe(comp)
        assert info.value.kind == UNRENDERABLE_TEXT

    @pytest.mark.parametrize("source", [
        "s = 'na\u00efve caf\u00e9 \u00df \u00f1'\nx = 1",
        "s = '\u03b1\u03b2\u03b3 \u03a9 \u03bb'\nx = 1",
        "s = '\u043f\u0440\u0438\u0432\u0435\u0442'\nx = 1",
        LATIN_EXT_A,
        "v = 'Vi\u1ec7t \u01d8'\nx = 1",
    ], ids=["latin-1", "greek", "cyrillic", "latin-ext-a", "vietnamese"])
    def test_text_the_font_draws_renders_on_its_rows(self, source):
        # The guard must not cost the scripts the font does draw: each glyph
        # lands on its own line's row, none shifted, nothing left over.
        # (q + U+0301 was a case here: its rows hold, but the accent is drawn
        # on the closing quote, so it pinned a misrender. It is refused at the
        # schema now; precomposed Latin Extended-A takes its place.)
        probe = _probe(CodeWalk({"language": "python", "source": source,
                                 "highlights": [{"start": 1}]}))
        code, _bar = probe.mobjects
        assert _rows_hold(code, source)

    @pytest.mark.parametrize("source,named", [
        ("s = 'a˜b'", "U+02DC"),
        ("# ƁƂƃƄƅƆƇƈƉ ok", "U+0181"),
        ("s = 'a\u03fdb'", "U+03FD"),
    ], ids=["small-tilde", "latin-ext-b-stack", "reversed-lunate-sigma"])
    def test_glyphs_off_their_columns_refuse_unrenderable(self, source, named):
        # Every count passes for these in Courier New -- each character draws
        # one path alone and in context -- yet in a line the tilde lands on
        # the b, and the nine letters, like U+03FD and the b after it, pile
        # into one blob, while validation said ok. The column check on the
        # built listing refuses them, in the beat and in the carry-in
        # artifact. (Stacked combining marks were a case here; the schema
        # refuses them now, before any column is measured.) These are
        # Courier New's glyphs, so the listing is drawn in Courier New
        # whatever mono font is installed: JetBrains Mono gives the small
        # tilde its own cell and has no glyph at all for the others (the
        # next test holds the theme's font to the same guard).
        recipe = ArtifactRecipe(name="listing", producer="CodeWalk",
                                params={"language": "python", "source": source})
        for attempt in (lambda: _probe_in(CodeWalk(recipe.params), COURIER),
                        lambda: build_artifact(recipe, COURIER)):
            with pytest.raises(LayoutError) as info:
                attempt()
            assert info.value.kind == UNRENDERABLE_TEXT
            message = str(info.value)
            assert named in message and repr("Courier New") in message
            assert message.startswith("CodeWalk source line 1:")
            assert "column" in message

    @pytest.mark.parametrize("source,named", [
        ("s = 'a\u02dcb'", "U+02DC"),
        ("# \u0181\u0182\u0183\u0184\u0185\u0186\u0187\u0188\u0189 ok", "U+0181"),
        ("s = 'a\u03fdb'", "U+03FD"),
    ], ids=["small-tilde", "latin-ext-b-stack", "reversed-lunate-sigma"])
    def test_same_text_in_the_theme_font_is_refused_or_on_its_columns(
            self, source, named):
        # The same listings in the resolved mono font: whichever font that
        # is, each is either refused as text the font cannot draw, naming
        # the character and the font, or built -- and then the column check
        # passed in build() and every glyph sits on its own row. Never a
        # listing drawn with its glyphs off their characters.
        source += "\nx = 1"  # a second row, to measure the first against
        try:
            probe = _probe(CodeWalk({"language": "python", "source": source}))
        except LayoutError as exc:
            assert exc.kind == UNRENDERABLE_TEXT
            assert named in str(exc) and repr(MONO) in str(exc)
            return
        assert _rows_hold(probe.mobjects[0], source)

    @pytest.mark.parametrize("source", [
        LATIN_EXT_A,
        "s = 'cafe\u0301'\nx = 1",
        "# Tiếng Việt có dấu\nx = 1",
        next(case["source"] for case in CodeWalk.stress()
             if "мир" in case["source"]),
        "# → ← ↑ ↓ ↔\n# ┌─┐ │ └─┘\nx = 1",
        "s = “quoted” + ‘single’  # — em – en …\n"
        "n = 3 × 4 ÷ 2 − 1",
        "s = 'q\u0301'\nx = 1",
        PROGRAMMING_LIGATURES,
    ], ids=["latin-ext-a", "nfd-e-acute", "vietnamese", "latin-greek-cyrillic",
            "arrows-boxes", "typographic", "q-acute-refused",
            "programming-ligatures"])
    def test_text_the_font_spaces_renders_on_its_columns(self, source):
        # The column check must not cost text the font does space one per
        # column: measured on its own here, every glyph sits within half a
        # cell of its column, counted from the first glyph of its line. No
        # glyph is skipped: a combining mark NFC leaves standing is refused
        # by the schema, never measured -- q + U+0301 used to be measured
        # here with its mark skipped, which pinned the accent drawn on the
        # closing quote as correct -- so the listing that reaches build() is
        # one column and one glyph per character (the NFD accent arrives
        # composed).
        params = {"language": "python", "source": source}
        if any(unicodedata.category(c) in ("Mn", "Me")
               for c in unicodedata.normalize("NFC", source)):
            with pytest.raises(ValidationError, match="combining mark"):
                CodeWalk(params)
            return
        comp = CodeWalk(params)
        text = comp.params.source
        probe = _probe(comp)
        code = probe.mobjects[0]
        # One cell of the placed (fitted) listing: a natural-size probe's
        # digit pitch, scaled as line number 1 was scaled.
        ref = code_walk._code("00\n00", "text", THEME)
        advance = ref.code_lines[0][1].get_x() - ref.code_lines[0][0].get_x()
        advance *= code.line_numbers[0].height / ref.line_numbers[0].height
        for i, line in enumerate(text.split("\n")):
            glyphs = iter(code.code_lines[i])
            first = None
            for col, c in enumerate(line):
                if c.isspace():
                    continue
                x = next(glyphs).get_x()
                if first is None:
                    first = (x, col)
                assert abs((x - first[0]) / advance - (col - first[1])) <= 0.5, (i, c)

    def test_listing_and_its_probes_draw_with_ligatures_off(self, monkeypatch):
        # Code's disable_ligatures turns off liga/dlig/clig/hlig but not
        # calt, through which JetBrains Mono draws `<=`, `===`, `>>` as one
        # glyph, and Code refused ordinary source. Every Text a listing is
        # drawn with -- code, line numbers, alignment glyphs -- and every
        # probe of it (whole listing, cluster) must reach Pango with all of
        # CODE_FONT_FEATURES off, or the probe and the drawing disagree.
        seen: list[str] = []
        original = Text._text2svg

        def spy(self, color):
            seen.append(color)
            return original(self, color)

        monkeypatch.setattr(Text, "_text2svg", spy)
        features = f"font_features='{code_walk.CODE_FONT_FEATURES}"
        code = code_walk._code("if lo <= hi:\n    m = (lo + hi) >> 1", "python", THEME)
        assert len(seen) >= 3 and all(features in c for c in seen), seen
        assert len(code.code_lines[0]) == len("iflo<=hi:")
        assert len(code.code_lines[1]) == len("m=(lo+hi)>>1")
        seen.clear()
        code_walk._refuse_unrenderable("s = 'caf\u00e9' if a <= b else '==='", THEME)
        code_walk._cluster_paths.cache_clear()
        code_walk._cluster_paths("\u00e9", MONO)
        assert len(seen) >= 2 and all(features in c for c in seen), seen

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
        # Same panel, same wash on the last span, same dimming, every line
        # where it was -- the frame the producing beat ended on, so the
        # picture persists across the cut.
        params = EXAMPLES[1]
        code, bar = _probe(CodeWalk(params)).mobjects
        art = self._artifact(params)
        fit_to_region(art, Region.STAGE)
        washes = [m for m in art.get_family() if getattr(m, "_chalk_label", None) == "highlight"]
        assert len(washes) == 1  # EXAMPLES[1] ends on line 9 alone
        for got, want in ((_panel(art), code.background),
                          (Group(*washes), bar)):
            _same_box(got, want)
        for k, row in enumerate(art):
            _same_box(_ink(row), _beat_line(code, k))
            assert row[-2].get_fill_opacity() == pytest.approx(
                code.line_numbers[k].get_fill_opacity())
        # Behind the text, by draw order: no z-index survives (it is global to
        # a scene, so carried glyphs at z 1 would draw over a consumer's lens
        # card), and each wash comes before its line's number and glyphs.
        assert {m.z_index for m in art.get_family()} == {0}
        family = art.get_family()
        lit = next(row for row in art if row[0] in washes)
        assert family.index(lit[0]) < family.index(lit[-2]) < family.index(lit[-1])

    def test_wash_over_a_span_tiles_the_beats_bar(self):
        # The bar is cut into one wash per lit line so that no part spans
        # lines. Together they are the beat's bar exactly: edge to edge, no
        # gap and no overlap to show as a seam in the translucent wash.
        params = {"language": "python", "source": SOURCE,
                  "highlights": [{"start": 1}, {"start": 2, "end": 4}]}
        _code, bar = _probe(CodeWalk(params)).mobjects
        art = self._artifact(params)
        fit_to_region(art, Region.STAGE)
        lit = [k for k, row in enumerate(art)
               if getattr(row[0], "_chalk_label", None) == "highlight"]
        assert lit == [1, 2, 3]
        washes = [art[k][0] for k in lit]
        _same_box(Group(*washes), bar)
        for upper, lower in zip(washes, washes[1:]):
            assert upper.get_bottom()[1] == pytest.approx(lower.get_top()[1], abs=1e-9)

    def test_parts_are_the_source_lines_in_order(self):
        # Callout's `part` and ZoomHighlight's `parts` index the carried
        # artifact's top-level parts. On Code's own structure those were the
        # background, the whole number column and the whole code block; they
        # must be the lines, part k = line k + 1, and nothing else.
        params = EXAMPLES[1]
        art = self._artifact(params)
        lines = params["source"].split("\n")
        assert len(art.submobjects) == len(lines)
        for k, (row, line) in enumerate(zip(art, lines)):
            assert getattr(row, "_chalk_label") == f"line {k + 1}"
            *_, number, glyphs = row
            assert len(number) == len(str(k + 1))
            assert len(glyphs) == sum(not c.isspace() for c in line)
        # The parts' count is what the semantic rung checks indices against.
        recipe = ArtifactRecipe(name="listing", producer="CodeWalk", params=params)
        zoom = make_component("ZoomHighlight", {"target_id": "listing", "parts": [10],
                                                "callout": "past the end"})
        (problem,) = part_range_problems(zoom, [recipe], THEME)
        assert "builds with 10 part(s)" in problem

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

    def test_carried_listing_validates_in_a_later_beat(self, hold_consumer):
        # Carried into _Hold, a consumer: a beat that carries an artifact in
        # without acting on it must leave it a free STAGE region (D-G4c-1).
        video = VideoSpec(video_id="v", beats=(
            BeatSpec(id="b01", narration="placeholder narration",
                     component="CodeWalk", params=EXAMPLES[1], registers="listing"),
            BeatSpec(id="b02", narration="placeholder narration",
                     component=hold_consumer, params={"target_id": "listing"},
                     carry_in=["listing"]),
        ))
        recipes = resolve_carry_in(video)["b02"]
        assert recipes[0].producer == "CodeWalk"
        report = validate_beat(video.beats[1], duration=4.0, recipes=recipes)
        assert report.ok, f"\n{report}"


class TestConsumers:
    """A later beat acting on a carried listing lands on the line it names
    (SCENE_SPEC.md §6): source line n is part n - 1."""

    PARAMS = EXAMPLES[1]  # ten lines, the walk ending on line 9 alone

    def _consumer(self, component: str, params: dict) -> LayoutProbe:
        recipe = ArtifactRecipe(name="listing", producer="CodeWalk", params=self.PARAMS)
        spec = BeatSpec(id="b02", narration="placeholder narration", component=component,
                        params={"target_id": "listing", **params}, carry_in=["listing"])
        probe = LayoutProbe(beat_component(spec, (recipe,)), duration=8.0, strict=False)
        probe.construct()
        assert probe.layout_warnings == []
        return probe

    @pytest.mark.parametrize("k,marker", [
        (1, "focus frame"),  # line 2, a long dimmed line: too wide to magnify
        (8, "focus frame"),  # line 9, the lit line: its wash is part of it
        (9, "zoom lens"),    # line 10, "}": short enough to magnify
    ], ids=["line2-dim", "line9-lit", "line10-short"])
    def test_zoom_highlight_part_k_frames_exactly_line_k(self, k, marker):
        mobs = {getattr(m, "_chalk_label", None): m
                for m in self._consumer("ZoomHighlight",
                                        {"parts": [k], "callout": "this line"}).mobjects}
        target, shown = mobs["carried[listing]"], mobs[marker]
        row = target[k]
        # The focus is the whole line -- its number and every glyph -- where
        # the producing beat left it (CarryIn fits the listing to STAGE as
        # the beat did).
        code, _bar = _probe(CodeWalk(self.PARAMS)).mobjects
        _same_box(_ink(row), _beat_line(code, k))
        numbers = [r[-2].get_y() for r in target]
        if marker == "focus frame":
            # Framed around exactly that line: its box, FRAME_BUFF out, holds
            # line k's number and none of its neighbours'.
            want, got = bbox(row), bbox(shown)
            assert (got.x, got.y) == pytest.approx((want.x, want.y), abs=1e-6)
            assert (got.width, got.height) == pytest.approx(
                (want.width + 2 * FRAME_BUFF, want.height + 2 * FRAME_BUFF), abs=0.02)
            assert [i for i, y in enumerate(numbers) if got.bottom < y < got.top] == [k]
        else:
            # Magnified: the lens shows line k and only line k, scaled as a
            # whole -- same glyphs, same shape, zoom times the size.
            (mag,) = shown[1]
            zoom = mag.height / row.height
            assert zoom >= MIN_ZOOM
            a = [m.points for m in row.family_members_with_points()]
            b = [m.points for m in mag.family_members_with_points()]
            assert len(a) == len(b) > 0
            for pa, pb in zip(a, b):
                np.testing.assert_allclose(
                    (pb - mag.get_center()) / zoom, pa - row.get_center(), atol=1e-6)

    def test_callout_part_k_points_at_line_k(self):
        probe = self._consumer("Callout", {"part": 4, "text": "found it", "side": "right"})
        mobs = {getattr(m, "_chalk_label", None): m for m in probe.mobjects}
        row, arrow = mobs["carried[listing]"][4], mobs["callout arrow"]
        # The arrow ends at line 5, level with it, just off its right edge.
        assert arrow.get_end()[1] == pytest.approx(row.get_y(), abs=0.05)
        assert 0 < arrow.get_end()[0] - row.get_right()[0] < 0.3


def _same_box(got, want) -> None:
    g, w = bbox(got), bbox(want)
    assert (g.x, g.y, g.width, g.height) == pytest.approx((w.x, w.y, w.width, w.height),
                                                          abs=1e-6)


def _panel(art):
    """The carried listing's panel alone: the artifact's own points."""
    panel = art.copy()
    panel.submobjects = []
    return panel


def _ink(row) -> Group:
    """A carried line's number and glyphs, without its wash."""
    return Group(*row[-2:])


def _beat_line(code: Code, k: int) -> Group:
    """Line k + 1 of the producing beat's listing: its number and glyphs."""
    return Group(code.line_numbers[k], code.code_lines[k])
