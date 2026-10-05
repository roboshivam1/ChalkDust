"""CodeWalk: a source listing walked through by highlighting lines.

Built on Manim's `Code` mobject as reworked in the 0.19 line (code_string=,
formatter_style=, paragraph_config=). Examples written against the pre-0.19
API (`file_name=`, `style=`, `insert_line_no=`) do not run on 0.21.

Code is not prose, so it is never wrapped. Breaking a line mid-token makes the
lexer colour the continuation wrongly (a wrapped comment loses its `#` and is
highlighted as code), and a wrapped Python line reads as a different
indentation. A listing that is too long or too wide scales down until it hits
the legibility floor and then refuses with a LayoutError -- the fix is a
shorter excerpt, which is a decision for the script, not the renderer
(SCENE_SPEC.md §4, §11 rule 1).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from statistics import median

from manim import (
    Code,
    FadeIn,
    ManimColor,
    Rectangle,
    Text,
    Transform,
    VGroup,
    VMobject,
    interpolate_color,
)
from manim.mobject.text import text_mobject
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pygments.lexers import get_lexer_by_name
from pygments.style import Style
from pygments.token import Comment, Error, Keyword, Number, String, Token
from pygments.util import ClassNotFound

from chalkdust.continuity import artifact_builder
from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    Component,
    ComponentParams,
    label,
    register,
)
from chalkdust.scenes.regions import (
    DEFAULT_PADDING,
    MIN_FONT_SIZE,
    UNRENDERABLE_TEXT,
    LayoutError,
    fit_to_region,
    region_rect,
    tag_font_size,
)
from chalkdust.scenes.theme import (
    Palette,
    Theme,
    check_renderable,
    unsupported_characters,
)

# Schema-level density limit, like BulletReveal's six items: past eight stops
# the beat is a lecture, and the error points at the real fix (split the beat).
MAX_HIGHLIGHTS = 8

# Relative timing weights; scene.budget() turns them into seconds (D-002).
# The final hold is long because a listing with no highlights is mostly hold,
# and a reveal stretched over the whole beat reads as a stalled render.
REVEAL = 1
MOVE = 1    # highlight bar travels to the next span
DWELL = 3   # viewer reads the highlighted lines while narration explains them
HOLD = 3

# Below this many seconds per weight unit a step stops being readable: the
# 0.4 s bar move is a jump cut, the 1.2 s dwell is one line glanced at.
# min_seconds() is this times the total weight, i.e. the sum of every step's
# own minimum (SCENE_SPEC.md §8 rung 2).
MIN_SECONDS_PER_WEIGHT = 0.4

DIM_OPACITY = 0.35        # lines outside the highlighted span
HIGHLIGHT_OPACITY = 0.18  # accent wash behind the span; text stays on top
PANEL_LIFT = 0.06         # panel fill, this far from bg toward fg

# Line pitch over digit height for Code's default line_spacing. Only used for
# a one-line listing, where there is no second line number to measure from.
SINGLE_LINE_PITCH = 1.75

# How far, in mono cells, a glyph's ink centre may sit from its source column
# before the listing is refused (_refuse_misplaced). Measured in Courier New:
# text that draws correctly -- ASCII, Latin-1, Greek, Cyrillic, Vietnamese,
# typographic quotes, arrows and box drawing, whose corners hug a cell edge
# -- is at most 0.23 cells off; a glyph with no advance (U+02DC,
# U+0181-U+0188, U+03FD) moves every later glyph at least 1.04 cells (an
# ideographic space, drawn by a fallback font, 0.67). Half a cell is the
# line between "in its own cell" and "in a neighbour's". Combining marks
# never get here: the schema refuses them (_normalise_source).
COLUMN_TOLERANCE = 0.5

# Code checks that every non-space character became exactly one glyph and
# raises a bare ValueError when one did not. Two kinds of character never do,
# in any font: format characters (category Cf: zero-width space and joiner,
# bidi marks, soft hyphen) draw nothing, and emoji are drawn by Pango from a
# colour font that yields no outline. Both are refused at the schema, where
# the repair loop can read which character on which line to drop
# (SCENE_SPEC.md §8 rung 1). That is an early refusal for the common cases,
# not the guard: what the *resolved* mono font cannot draw (CJK in Courier
# New, a right-to-left letter, U+2764 HEAVY BLACK HEART) is only
# knowable from the font, and is refused before Code is built by
# _refuse_unrenderable, through the theme's glyph guard.
# The BMP characters with Emoji_Presentation=Yes (Unicode emoji-data.txt);
# every other emoji lives in the pictograph blocks U+1F000-U+1FAFF.
_EMOJI_BMP = frozenset(
    [0x231A, 0x231B, *range(0x23E9, 0x23ED), 0x23F0, 0x23F3, 0x25FD, 0x25FE,
     0x2614, 0x2615, *range(0x2648, 0x2654), 0x267F, 0x2693, 0x26A1, 0x26AA,
     0x26AB, 0x26BD, 0x26BE, 0x26C4, 0x26C5, 0x26CE, 0x26D4, 0x26EA, 0x26F2,
     0x26F3, 0x26F5, 0x26FA, 0x26FD, 0x2705, 0x270A, 0x270B, 0x2728, 0x274C,
     0x274E, 0x2753, 0x2754, 0x2755, 0x2757, 0x2795, 0x2796, 0x2797, 0x27B0,
     0x27BF, 0x2B1B, 0x2B1C, 0x2B50, 0x2B55]
)
_PICTOGRAPHS = range(0x1F000, 0x1FB00)


class LineSpan(BaseModel):
    """1-based, inclusive line range, numbered as the listing displays it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Strict: lax mode reads `true` as line 1 and "2" as line 2, so a
    # malformed highlight would light a line nobody asked for.
    start: int = Field(ge=1, strict=True)
    end: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def _ordered(self) -> LineSpan:
        if self.end is not None and self.end < self.start:
            raise ValueError(f"end ({self.end}) is before start ({self.start})")
        return self

    @property
    def last(self) -> int:
        return self.end if self.end is not None else self.start


class CodeWalkParams(ComponentParams):
    # A Pygments lexer alias ("python", "javascript", "c", "sql"). Required:
    # Pygments' guesser is unreliable on short excerpts.
    language: str
    source: str
    highlights: list[LineSpan] = Field(default_factory=list, max_length=MAX_HIGHLIGHTS)

    @field_validator("language")
    @classmethod
    def _known_language(cls, v: str) -> str:
        try:
            get_lexer_by_name(v)
        except ClassNotFound:
            raise ValueError(
                f"unknown language {v!r}; use a Pygments lexer alias such as "
                "'python', 'javascript', 'c' or 'sql'"
            ) from None
        return v

    @field_validator("source")
    @classmethod
    def _normalise_source(cls, v: str) -> str:
        """Compose accents, refuse undrawable characters and combining marks,
        and drop trailing whitespace and blank lines at either end.

        NFC first: a decomposed accent ("e" + U+0301) is two characters that
        Pango draws as one glyph, which Code refuses; the composed form looks
        the same and is one character, one glyph.

        Then any combining mark NFC left standing (categories Mn and Me) is
        refused, naming it and its line. No precomposed letter exists for it
        (q + U+0301, i + U+0307 from "İ".lower(), a variation selector, a
        keycap), and Courier New draws such a mark about one cell right of
        its letter -- on the closing quote, or on the next letter, so
        x + U+0301 + y reads as x + ý -- while every glyph count and
        column passes. Code listings almost never carry one, so refusing it
        at rung 1, where the repair loop can read what to drop, beats a
        listing that shows other text than its source (SCENE_SPEC.md §8,
        §11 rule 1).

        Pygments strips leading and trailing newlines before lexing (its
        `stripnl` default), so a leading blank line would silently shift every
        displayed line number against the highlights. Normalising here means
        the numbers the spec validates against are the numbers on screen.
        """
        lines = [line.rstrip() for line in unicodedata.normalize("NFC", v).splitlines()]
        for n, line in enumerate(lines, start=1):
            for ch in line:
                if _undrawable(ch):
                    name = unicodedata.name(ch, "unnamed character")
                    raise ValueError(
                        f"source line {n} contains U+{ord(ch):04X} {name}, which "
                        "a code listing cannot draw (emoji and zero-width or "
                        "format characters); remove it or spell it in ASCII"
                    )
        marked = [(n, line) for n, line in enumerate(lines, start=1)
                  if any(_combining(c) for c in line)]
        if marked:
            n, line = marked[0]
            raise ValueError(_mark_message(n, line, [m for m, _ in marked[1:]]))
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        if not lines:
            raise ValueError("source cannot be empty")
        return "\n".join(lines)

    @model_validator(mode="after")
    def _highlights_in_range(self) -> CodeWalkParams:
        n = self.source.count("\n") + 1
        for i, span in enumerate(self.highlights):
            if span.last > n:
                raise ValueError(
                    f"highlights[{i}] reaches line {span.last}; source has {n} lines"
                )
        return self


@register
class CodeWalk(Component):
    name = "CodeWalk"
    Params = CodeWalkParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def _weights(self) -> list[float]:
        return [REVEAL] + [MOVE, DWELL] * len(self.params.highlights) + [HOLD]

    def min_seconds(self) -> float:
        return MIN_SECONDS_PER_WEIGHT * sum(self._weights())

    def latex_strings(self) -> list[str]:
        return []

    def build(self, scene: ChalkdustScene) -> None:
        p: CodeWalkParams = self.params
        theme = scene.theme

        code = _listing(p, theme)
        try:
            fit_to_region(code, Region.STAGE)
        except LayoutError as exc:
            lines = p.source.split("\n")
            widest = max(len(line.expandtabs(4)) for line in lines)
            raise LayoutError(
                f"CodeWalk listing of {len(lines)} lines, widest {widest} "
                f"columns: {exc}",
                kind=exc.kind,
            ) from exc

        # What each source line's glyphs and number currently show at.
        opacity = [1.0] * len(code.line_numbers)

        times = scene.budget(*self._weights())
        scene.play(FadeIn(code), run_time=times[0])
        scene.settle("code revealed")

        bar = None
        for k, span in enumerate(p.highlights):
            move, dwell = times[1 + 2 * k], times[2 + 2 * k]
            # One bar travels span to span (Transform keeps the first mobject),
            # so its label carries no index; settle() names the step instead.
            target = label(_bar(code, span, theme.palette.accent), "highlight")
            if bar is None:
                bar = target
                anims = [FadeIn(bar)]
            else:
                anims = [Transform(bar, target)]

            # Animate the listing's own parts, never a wrapper group: Manim
            # adds any animated mobject that is not already on screen to the
            # scene, so a fresh VGroup per row would become a stray top-level
            # mobject on top of the listing.
            for i, want in enumerate(_row_opacities(code, span)):
                if want != opacity[i]:
                    anims += [m.animate.set_opacity(want) for m in _row(code, i)]
                    opacity[i] = want

            scene.play(*anims, run_time=move)
            scene.settle(f"highlight {k}")
            scene.wait(dwell)

        scene.wait(times[-1])

    @classmethod
    def examples(cls):
        return [
            {"language": "python",
             "source": "def bucket_index(key, capacity):\n"
                       "    h = hash(key)\n"
                       "    return h % capacity\n"
                       "\n"
                       "index = bucket_index(\"cat\", 8)",
             "highlights": [{"start": 2}, {"start": 3}, {"start": 5}]},
            {"language": "javascript",
             "source": "function binarySearch(arr, target) {\n"
                       "  let lo = 0, hi = arr.length - 1;\n"
                       "  while (lo <= hi) {\n"
                       "    const mid = (lo + hi) >> 1;\n"
                       "    if (arr[mid] === target) return mid;\n"
                       "    if (arr[mid] < target) lo = mid + 1;\n"
                       "    else hi = mid - 1;\n"
                       "  }\n"
                       "  return -1;\n"
                       "}",
             "highlights": [{"start": 2}, {"start": 3, "end": 4},
                            {"start": 5, "end": 7}, {"start": 9}]},
            {"language": "sql",
             "source": "SELECT name, COUNT(*) AS orders\n"
                       "FROM customers JOIN orders USING (customer_id)\n"
                       "GROUP BY name\n"
                       "HAVING COUNT(*) > 5;"},
        ]

    @classmethod
    def stress(cls):
        long_listing = "\n".join(
            f"    total_{i} = compute_partial_sum(values, start={i}, step=3)"
            for i in range(30)
        )
        return [
            # (a) 3x a realistic excerpt, every highlight slot used.
            {"language": "python",
             "source": "def accumulate(values):\n" + long_listing,
             "highlights": [{"start": 1}] + [
                 {"start": 4 * i + 2, "end": 4 * i + 4} for i in range(7)]},
            # (b) a 60-character identifier: no break point, has to scale.
            {"language": "python",
             "source": "rehashed_bucket_index_for_keys_that_collide_in_the_open_tabl"
                       " = 0\nprint(table)",
             "highlights": [{"start": 1}]},
            # (b) a URL in a comment, far wider than the stage at legible size.
            {"language": "python",
             "source": "# https://example.com/" + "a-path-segment-without-any-spaces/" * 3
                       + "index.html\nfetch()",
             "highlights": [{"start": 1}, {"start": 2}]},
            # (a) a whole 200-line file pasted in: refused on size before
            # Code is built (Pango would drop glyphs past the frame).
            {"language": "python",
             "source": "\n".join(f"x{i} = {i}" for i in range(200)),
             "highlights": [{"start": 1}]},
            # (c) minimal: one character, no highlights.
            {"language": "c", "source": "x"},
            # (c) a highlight that lands on a blank line mid-listing.
            {"language": "python", "source": "a = 1\n\nb = 2",
             "highlights": [{"start": 2}, {"start": 1, "end": 3}]},
            # (e) text the mono font may lack, each of which garbled the
            # listing while validation said ok: CJK (missing-glyph boxes),
            # a right-to-left comment (draws nothing), a symbol the schema
            # lets through (U+2764, no emoji presentation), a letter with no
            # advance (U+02DC lands on the next letter), a space that draws
            # a glyph in context (U+1680), and Unicode maths. Private-use
            # code points and combining marks (a keycap, a variation
            # selector, q + U+0301) never get this far: rung 1 refuses them.
            {"language": "python",
             "source": "# 你好世界\nname = '東京'  # tokyo",
             "highlights": [{"start": 2}]},
            {"language": "python", "source": "x = 1  # שלום עולם\ny = 2",
             "highlights": [{"start": 1}]},
            {"language": "python", "source": "love = '\u2764'\nprint(love)"},
            {"language": "python", "source": "s = 'a\u02dcb'\nprint(s)"},
            {"language": "python", "source": "s = 'a\u1680b'\nprint(s)"},
            {"language": "python", "source": "# ∀x ≤ ∞\nx = 1",
             "highlights": [{"start": 2}]},
            # (e) accented Latin, Greek and Cyrillic, which the font does draw:
            # these must render, every glyph on its own row.
            {"language": "python",
             "source": "café = 'naïve'  # ß ñ\nλ = 'αβγ Ω'\nмир = 'привет'",
             "highlights": [{"start": 2}, {"start": 3}]},
        ]


# --- private helpers --------------------------------------------------------


@lru_cache(maxsize=8)
def _code_style(palette: Palette) -> type[Style]:
    """A Pygments style built from the theme palette.

    Code takes its colours from a Pygments style, so this is how the theme
    reaches the glyphs -- no named Pygments style, no colour from the spec.
    Could be shared via theme.py if another component ever lexes code.
    """
    panel = interpolate_color(
        ManimColor(palette.bg), ManimColor(palette.fg), PANEL_LIFT
    ).to_hex()
    return type(
        "ChalkdustCodeStyle",
        (Style,),
        {
            "background_color": panel,
            "line_number_color": palette.muted,
            "styles": {
                Token: palette.fg,
                Comment: palette.muted,
                Keyword: palette.accent,
                String: palette.success,
                Number: palette.accent_alt,
                Error: palette.danger,
            },
        },
    )


def _undrawable(ch: str) -> bool:
    """A character Code cannot turn into exactly one glyph (see _EMOJI_BMP)."""
    cp = ord(ch)
    if unicodedata.category(ch) == "Cf":
        return True
    return cp in _EMOJI_BMP or (
        cp in _PICTOGRAPHS and unicodedata.category(ch) in ("So", "Sk")
    )


def _combining(ch: str) -> bool:
    """A nonspacing or enclosing mark (Mn, Me): drawn on the character before
    it, which is what a mono listing gets wrong (see _normalise_source)."""
    return unicodedata.category(ch) in ("Mn", "Me")


def _mark_message(n: int, line: str, more: list[int]) -> str:
    """The schema's refusal of the combining marks on source line `n`: each
    distinct mark with its name, and each letter + marks it sits in with
    their code points, so the fix (which character to drop) is readable."""
    marks = dict.fromkeys(c for c in line if _combining(c))
    named = ", ".join(f"U+{ord(c):04X} {unicodedata.name(c, 'unnamed mark')}"
                      for c in marks)
    where = ", ".join(
        f"{cl!r} ({' '.join(f'U+{ord(c):04X}' for c in cl)})"
        for cl in dict.fromkeys(cl for cl in _clusters(line) if any(map(_combining, cl)))
    )
    also = (f" Also on line{'s' if len(more) > 1 else ''} "
            f"{', '.join(map(str, more))}." if more else "")
    return (
        f"source line {n} contains combining mark{'s' if len(marks) > 1 else ''} "
        f"{named} in {where}, which no precomposed letter absorbs: a mono "
        f"listing draws such a mark off its letter, over the next column, or "
        f"not at all, so it would show other text than its source. Use a "
        f"precomposed letter (é, ñ, ü are fine) or spell it in ASCII.{also}"
    )


def _paragraph_config(theme: Theme) -> dict:
    """What Code passes to its Paragraphs: its defaults (line_spacing,
    disable_ligatures) under the theme's mono type."""
    return {**Code.default_paragraph_config,
            "font": theme.type.mono_font, "font_size": theme.type.mono}


# The OpenType features a listing is drawn with off: every one that lets a
# font draw several characters as one glyph. Code passes
# disable_ligatures=True, which ManimPango 0.7 turns into
# font_features='liga=0,dlig=0,clig=0,hlig=0' -- without 'calt', the
# contextual alternates through which the theme's JetBrains Mono (like Fira
# Code and Cascadia Code) draws its programming ligatures. With calt on,
# `<=`, `===`, `>>` and `->` each shape to fewer glyphs than characters and
# Code refuses ordinary source; with it off, every character is its own
# glyph in its own cell, which is what a listing means. A font without
# these features (Courier New) draws the same with them off.
CODE_FONT_FEATURES = "calt=0,liga=0,dlig=0,clig=0,hlig=0"


class _CodeText(Text):
    """Text drawn with CODE_FONT_FEATURES off: the one Text every listing
    glyph and every probe of one goes through.

    ManimPango 0.7's Text path takes no font features. It draws each run as
    Pango markup, <span color='{colour}'>{escaped text}</span>, with the
    run's colour written in as given, so the features ride in on the colour
    as a second attribute of that span. Pango still reads the colour, the
    text stays escaped, and the features become part of Text's SVG cache key
    with it, so no listing cached with ligatures on is reused.
    """

    def _text2svg(self, color):  # type: ignore[override]
        return super()._text2svg(
            f"{color}' font_features='{CODE_FONT_FEATURES}"
        )


@contextmanager
def _code_text() -> Iterator[None]:
    """Build Code's Paragraphs from _CodeText.

    Code takes no Text class: it builds each Paragraph, and Paragraph its
    Text, through text_mobject's module-global name `Text`. So for the
    duration of one Code build that name is _CodeText. Building a scene is
    single-threaded, and the name is restored however the build ends.
    """
    saved = text_mobject.Text
    text_mobject.Text = _CodeText
    try:
        yield
    finally:
        text_mobject.Text = saved


def _code(source: str, language: str, theme: Theme) -> Code:
    """Manim's Code in the theme's colours and mono type, at natural size,
    one glyph per character (CODE_FONT_FEATURES off)."""
    try:
        with _code_text():
            return Code(
                code_string=source,
                language=language,
                formatter_style=_code_style(theme.palette),
                add_line_numbers=True,
                background="rectangle",
                background_config={"stroke_color": theme.palette.muted},
                paragraph_config=_paragraph_config(theme),
            )
    except ValueError as exc:
        # Backstop behind _refuse_unrenderable, as theme._text keeps one behind
        # check_renderable: Code checks one glyph per non-space character and
        # raises a bare ValueError. Only that check is translated, to the
        # theme's kind for it; any other ValueError is a bug and propagates.
        if "rendered fewer glyph" not in str(exc):
            raise
        raise LayoutError(
            f"CodeWalk source cannot be drawn in font {theme.type.mono_font!r}: "
            "it shapes to fewer glyphs than it has characters. Rewrite it with "
            "characters that font draws.",
            kind=UNRENDERABLE_TEXT,
        ) from exc


@lru_cache(maxsize=8)
def _cell(theme: Theme) -> tuple[float, float]:
    """(line pitch, column advance) of a listing at the theme's mono size,
    measured off a two-line ASCII probe rather than assumed from the font."""
    probe = _code("0000000000\n0000000000", "text", theme)
    nums, first = probe.line_numbers, probe.code_lines[0]
    pitch = nums[0].get_y() - nums[1].get_y()
    advance = (first[-1].get_x() - first[0].get_x()) / (len(first) - 1)
    return pitch, advance


def _cells(line: str) -> float:
    """A lower bound on how many mono cells `line` takes.

    Printable ASCII is one Courier cell by construction, and East Asian wide
    characters are square, so at least one. Anything else counts half a cell:
    the narrowest glyph measured across Latin, Greek, Cyrillic, Hebrew, Arabic,
    Indic, Thai, symbols and punctuation is 0.87 of a cell, and a script the
    mono font lacks falls back to wider glyphs. Combining marks and whitespace
    other than the space count nothing.
    """
    total = 0.0
    for c in line:
        if " " <= c <= "~" or unicodedata.east_asian_width(c) in ("W", "F"):
            total += 1
        elif not c.isspace() and unicodedata.category(c)[0] != "M":
            total += 0.5
    return total


def _refuse_oversized(p: CodeWalkParams, theme: Theme) -> None:
    """Refuse, before Code is built, a listing that cannot fit the STAGE even
    at the legibility floor.

    Code draws each Paragraph on a Pango surface the size of the output frame
    (config.pixel_width x pixel_height) and silently drops the glyphs that
    fall off it, which surfaces as Code's bare glyph-count ValueError -- for a
    pasted 200-line file, or one 3000-column line, before fit_to_region ever
    gets to say overflow. Both bounds here are lower bounds on the listing's
    size (line-number centres span (lines - 1) pitches; columns are counted
    by _cells), so this never refuses a listing fit_to_region would accept.
    """
    pitch, advance = _cell(theme)
    floor = MIN_FONT_SIZE / theme.type.mono  # the smallest scale fit allows
    room = region_rect(Region.STAGE).inset(DEFAULT_PADDING)
    lines = p.source.split("\n")
    columns = max(_cells(line.expandtabs(4)) for line in lines)
    tall = (len(lines) - 1) * pitch * floor > room.height
    wide = (columns - 1) * advance * floor > room.width
    if tall or wide:
        raise LayoutError(
            f"CodeWalk listing of {len(lines)} lines, at least {columns:.0f} columns, "
            f"cannot fit the stage at the legibility floor (font_size "
            f"{MIN_FONT_SIZE:.0f}). Show a shorter excerpt, or split the beat.",
            kind="overflow",
        )


def _clusters(line: str) -> list[str]:
    """`line` split into base characters, each with the combining marks that
    follow it (a mark at the start of a line stands alone)."""
    out: list[str] = []
    for c in line:
        if out and unicodedata.category(c)[0] == "M":
            out[-1] += c
        else:
            out.append(c)
    return out


@lru_cache(maxsize=4096)
def _cluster_paths(cluster: str, font: str) -> int:
    """How many paths Pango draws for `cluster` alone in `font`, as a listing
    draws it (CODE_FONT_FEATURES off), or -1 if it refuses the string: the
    theme's per-character probe, for a base + marks."""
    try:
        return len(_CodeText(cluster, font=font).submobjects)
    except Exception:
        return -1


def _lines_with(source: str, found: Callable[[str], bool]) -> str:
    """'line 3' or 'lines 1, 4': the source lines `found` picks, for a message."""
    nums = [str(n) for n, line in enumerate(source.split("\n"), start=1) if found(line)]
    return f"line{'s' if len(nums) > 1 else ''} {', '.join(nums)}"


class _GlyphCount(_CodeText):
    """A listing's Text that keeps how many glyphs Pango drew, before Manim
    maps them onto characters (with disable_ligatures, that mapping drops
    extras)."""

    def _gen_chars(self):  # type: ignore[override]
        self.glyphs = len(self.submobjects)
        return super()._gen_chars()


def _refuse_unrenderable(source: str, theme: Theme) -> None:
    """Refuse, before Code is built, a listing the resolved mono font cannot
    draw as one glyph per character (kind "unrenderable_text", SCENE_SPEC.md
    §11 rule 1).

    Code maps the glyphs Pango draws back onto the source's characters by
    count. A character the font lacks is not an error to Pango: it draws a
    missing-glyph box -- a box plus one path per hex digit -- or nothing.
    Too few glyphs and Code raises a bare ValueError; too many and the
    mapping shifts silently: colours land on the wrong characters, later
    lines lose their glyphs, and the alignment glyphs Code strips from the
    last line (" pA1") stay in the frame -- while validate_beat says ok. So
    the listing is refused from the font, before anything is drawn.

    Three checks. The first two run over distinct pieces, so their cost is
    the alphabet, not the listing (both are cached per process), and they
    name what to rewrite:
      1. theme.check_renderable on the distinct characters -- the guard every
         theme text constructor applies. It names each character the font
         cannot draw, and the font.
      2. Each cluster that check does not settle -- a precomposed letter
         (the theme lets one draw as base + mark), or a base carrying spacing
         marks (Mc; the schema has already refused the nonspacing and
         enclosing ones) -- is drawn whole and must come out as exactly one
         path per non-space character, which is what Code assumes.
      3. Code's assumption itself, so shaping in context cannot slip past the
         probes: the whole listing, drawn as Code's Paragraph draws it (the
         same config, CODE_FONT_FEATURES off), must come out as exactly one
         glyph per non-space character. An ASCII listing skips it -- check 1
         drew each of its characters as one path, and nothing in a mono font
         changes that count but its ligatures and contextual alternates,
         which _CodeText turns off -- so only a listing that leaves ASCII
         pays for the extra Text.
    """
    font = theme.type.mono_font
    chars = "".join(dict.fromkeys(c for c in source if not c.isspace()))
    try:
        check_renderable(chars, font, what="CodeWalk source characters")
    except LayoutError as exc:
        bad = {c for c, _ in unsupported_characters(chars, font)}
        where = _lines_with(source, lambda line: not bad.isdisjoint(line))
        raise LayoutError(f"CodeWalk source {where}: {exc}", kind=exc.kind) from exc

    lines = source.split("\n")
    for cluster in dict.fromkeys(cl for line in lines for cl in _clusters(line)):
        if (len(unicodedata.normalize("NFD", cluster)) == 1
                and unicodedata.category(cluster)[0] != "M"):
            continue  # one plain character: check_renderable saw it draw one path
        want = sum(not c.isspace() for c in cluster)
        got = _cluster_paths(cluster, font)
        if got != want:
            points = " ".join(f"U+{ord(c):04X}" for c in cluster)
            where = _lines_with(source, lambda line: cluster in _clusters(line))
            drawn = "cannot be shaped" if got < 0 else f"draws {got} glyph(s)"
            raise LayoutError(
                f"CodeWalk source {where}: {cluster!r} ({points}) {drawn} in "
                f"font {font!r}, not {want} (one per character), so the "
                f"listing's glyphs would no longer line up with its characters. "
                f"Rewrite it with characters that font draws.",
                kind=UNRENDERABLE_TEXT,
            )

    if source.isascii():
        return
    want = sum(not c.isspace() for c in source)
    try:
        got: int | None = _GlyphCount(source, **_paragraph_config(theme)).glyphs
    except ValueError as exc:
        if "rendered fewer glyph" not in str(exc):
            raise
        got = None  # fewer than `want`; Manim does not say how many
    if got != want:
        drawn = "fewer glyphs" if got is None else f"{got} glyphs"
        raise LayoutError(
            f"CodeWalk source draws {drawn} for its {want} visible characters "
            f"in font {font!r} (its non-ASCII text is shaped in context), so "
            f"the listing's glyphs would no longer line up with its "
            f"characters. Rewrite the non-ASCII text in characters that font "
            f"draws one by one.",
            kind=UNRENDERABLE_TEXT,
        )


def _refuse_misplaced(code: Code, source: str, theme: Theme) -> None:
    """Refuse a built listing whose glyphs do not land on their mono columns
    (kind "unrenderable_text", SCENE_SPEC.md §11 rule 1).

    _refuse_unrenderable counts glyphs; this measures where they went. Some
    characters draw exactly one path alone and in context, so every count
    passes, yet Pango gives them no advance in a line: U+02DC SMALL TILDE
    lands on the next letter, and U+0181-U+0188 and U+03FD pile into one
    blob. A code listing's columns are part of what it says (indentation,
    alignment), so that is garbled text that validate_beat would call ok.

    The model is what a mono listing promises: every character after Code's
    tab expansion (tab_width 4, as Code is built here) takes one column. The
    schema refused the combining marks that would take none, so no mark is
    measured against a letter here. Each glyph's ink centre must sit within
    COLUMN_TOLERANCE cells of its column. Column 0 is taken from the glyphs
    whose place nothing but ASCII decides -- an ASCII glyph with only ASCII
    before it on its line, whose advance the mono font fixes -- not from
    every glyph: a median over the whole listing drifts toward a line's
    shifted tail and lets the shift it is meant to catch through.
    Runs on the listing at natural size, where _cell measured the cell, and
    costs arithmetic only -- no extra Text.
    """
    font = theme.type.mono_font
    _, advance = _cell(theme)
    lines = [line.expandtabs(4) for line in source.split("\n")]
    # (line number, column = index in the expanded line, offset in cells
    #  from that column, whether only ASCII decides where it sits)
    placed: list[tuple[int, int, float, bool]] = []
    for n, line in enumerate(lines, start=1):
        glyphs = iter(code.code_lines[n - 1])
        for j, c in enumerate(line):
            if c.isspace():
                continue  # Code keeps no glyph for whitespace
            # One glyph per visible character is what _refuse_unrenderable and
            # Code's own count check held the listing to, line by line.
            glyph = next(glyphs, None)
            if glyph is None:
                raise LayoutError(
                    f"CodeWalk source line {n} draws fewer glyphs than it has "
                    f"visible characters in font {font!r}. Rewrite it with "
                    f"characters that font draws one by one.",
                    kind=UNRENDERABLE_TEXT,
                )
            placed.append((n, j, glyph.get_x() / advance - j,
                           line[:j + 1].isascii()))
    if not placed:
        return
    # A listing with no such glyph (every line opens on non-ASCII) falls back
    # to each line's first glyph, which nothing before it can have moved.
    anchors = [off for _, _, off, ascii_only in placed if ascii_only] or [
        next(off for m, _, off, _ in placed if m == n)
        for n in dict.fromkeys(m for m, *_ in placed)
    ]
    origin = median(anchors)
    for i, (n, j, off, _) in enumerate(placed):
        if abs(off - origin) <= COLUMN_TOLERANCE:
            continue
        # Name the cause with the effect: the text from the glyph before the
        # stray one on its line (the character that took the wrong advance
        # is in there) through the stray glyph itself.
        start = placed[i - 1][1] if i and placed[i - 1][0] == n else j
        text = lines[n - 1][start:j + 1]
        points = " ".join(f"U+{ord(c):04X}" for c in text)
        raise LayoutError(
            f"CodeWalk source line {n}: {text!r} ({points}) does not advance "
            f"one column per character in font {font!r} -- "
            f"{lines[n - 1][j]!r} is drawn {abs(off - origin):.1f} cells from "
            f"its column, so the listing's glyphs collapse or shift against "
            f"its source. Rewrite it with characters that font spaces one per "
            f"column.",
            kind=UNRENDERABLE_TEXT,
        )


def _listing(p: CodeWalkParams, theme: Theme) -> Code:
    """The listing at its natural size, unplaced: shared by build() and the
    carry-in artifact so the two cannot drift apart -- and so the artifact
    refuses exactly what the beat refuses.

    Size first: it is arithmetic on the source, and it bounds the alphabet
    the glyph probe then has to draw (a pasted file of 100k distinct
    characters refuses as overflow at once, not after 100k probe Texts).
    Then the glyph guard, before Code is built; then, on the built listing,
    the column check -- the one failure only the laid-out glyphs show.
    """
    _refuse_oversized(p, theme)
    _refuse_unrenderable(p.source, theme)
    code = _code(p.source, p.language, theme)
    _refuse_misplaced(code, p.source, theme)
    code = label(code, "code")
    # Code builds its Paragraphs itself, bypassing theme.mono_text, so tag
    # them here or the legibility floor never sees this text.
    tag_font_size(code.code_lines, theme.type.mono)
    tag_font_size(code.line_numbers, theme.type.mono)
    # Text above the highlight bar; the bar is added later and would
    # otherwise draw over the glyphs it is meant to sit behind.
    code.code_lines.set_z_index(1)
    code.line_numbers.set_z_index(1)
    return code


def _row(code: Code, i: int) -> list[VMobject]:
    """Source line i's glyphs and its line number. An empty source line is an
    empty group with no points; it has nothing to dim, so it is left out."""
    return [m for m in (code.code_lines[i], code.line_numbers[i])
            if m.has_points() or m.family_members_with_points()]


def _row_opacities(code: Code, span: LineSpan) -> list[float]:
    """Per source line: full strength inside `span`, dimmed outside."""
    return [1.0 if span.start <= i + 1 <= span.last else DIM_OPACITY
            for i in range(len(code.line_numbers))]


@artifact_builder("CodeWalk")
def _artifact(p: CodeWalkParams, theme: Theme) -> _Listing:
    """The settled last frame, for a later beat to carry in (SCENE_SPEC.md §6):
    the listing, and when the walk had highlights, its last span lit and the
    other lines dimmed -- the picture the viewer was left with.

    Parts are the source lines, in order: submobjects[i] is line i + 1 as the
    listing numbers it, a VGroup of [wash,] line number, glyphs -- the wash
    only on the lines of the last span. That is what Callout's `part` and
    ZoomHighlight's `parts` index (SCENE_SPEC.md §6). Code's own top-level
    parts are the background, the whole number column and the whole code
    block, so indexing those focused a column, never a line.

    Nothing else may be a part: the consumers treat every other part as a
    sibling to recede or keep clear of, so a panel or bar spanning the lines
    would make each line's arrow "cross" it. The panel is therefore the
    artifact itself -- its own points draw it, under every line -- and the
    beat's bar is cut at the midpoints between line numbers into one wash per
    lit line, which tile it exactly.

    No z-index survives: z-index is global to a scene, so carried glyphs at
    z 1 would draw over a consumer's lens card. Family order keeps the
    layering instead (panel, then each line's wash under its glyphs).
    """
    code = _listing(p, theme)
    nums, lines = code.line_numbers, code.code_lines
    span = p.highlights[-1] if p.highlights else None
    washes = (_wash(code, span, theme.palette.accent) if span is not None else {})
    want = _row_opacities(code, span) if span is not None else [1.0] * len(nums)

    rows = []
    for i in range(len(nums)):
        for m in _row(code, i):
            m.set_opacity(want[i])
        number, glyphs = nums[i], lines[i]
        tag_font_size(number, theme.type.mono)
        tag_font_size(glyphs, theme.type.mono)
        parts = [washes[i]] if i in washes else []
        rows.append(label(VGroup(*parts, number, glyphs), f"line {i + 1}"))

    panel = _Listing()
    panel.set_points(code.background.points)
    panel.match_style(code.background)
    panel.add(*rows)
    panel.set_z_index(0)  # whole family
    return label(panel, "code")


class _Listing(VMobject):
    """A carried listing: its own points draw the panel, its submobjects are
    the lines. Manim counts a mobject with points as its own first element
    when indexing, iterating or measuring it (Mobject.__getitem__, __iter__,
    __len__), so listing[4] would be line 4, not line 5 -- and Callout points
    at plan[part]. Here those go over the lines alone, as on a VGroup."""

    def __getitem__(self, value):
        if isinstance(value, slice):
            return VGroup(*self.submobjects[value])
        return self.submobjects[value]

    def __iter__(self):
        return iter(self.submobjects)

    def __len__(self) -> int:
        return len(self.submobjects)


def _wash(code: Code, span: LineSpan, color: str) -> dict[int, Rectangle]:
    """_bar() behind `span`, cut into one rectangle per line (by 0-based line
    index). Each runs from the midpoint to the line number above to the
    midpoint to the one below, the span's ends half a pitch out as _bar's
    are, so together they cover exactly _bar's rectangle, without a gap or
    an overlap to show as a seam in the translucent wash."""
    bar = _bar(code, span, color)
    ys = [code.line_numbers[i].get_y() for i in range(span.start - 1, span.last)]
    cuts = [bar.get_top()[1],
            *((a + b) / 2 for a, b in zip(ys, ys[1:])),
            bar.get_bottom()[1]]
    washes = {}
    for k, (top, bottom) in enumerate(zip(cuts, cuts[1:])):
        piece = Rectangle(width=bar.width, height=top - bottom, fill_color=color,
                          fill_opacity=HIGHLIGHT_OPACITY, stroke_width=0)
        piece.move_to(bar).set_y((top + bottom) / 2)
        washes[span.start - 1 + k] = label(piece, "highlight")
    return washes


def _bar(code: Code, span: LineSpan, color: str) -> Rectangle:
    """Highlight bar behind `span`, measured from the line numbers.

    Line numbers are digits only -- no descenders, one per line even when the
    code line is empty -- so their centres are a stable row grid, unlike the
    code glyphs whose bounding boxes change with content.
    """
    nums = code.line_numbers
    if len(nums) > 1:
        pitch = (nums[0].get_y() - nums[-1].get_y()) / (len(nums) - 1)
    else:
        pitch = nums[0].height * SINGLE_LINE_PITCH

    top = nums[span.start - 1].get_y()
    bottom = nums[span.last - 1].get_y()
    content = VGroup(code.line_numbers, code.code_lines)
    pad = (pitch - nums[0].height) / 2

    bar = Rectangle(
        width=content.width + 2 * pad,
        height=top - bottom + pitch,
        fill_color=color,
        fill_opacity=HIGHLIGHT_OPACITY,
        stroke_width=0,
    )
    bar.move_to(content).set_y((top + bottom) / 2)
    return bar
