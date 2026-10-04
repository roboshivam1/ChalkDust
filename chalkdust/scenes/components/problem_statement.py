"""ProblemStatement: the question, formatted (JEE branch, SCENE_SPEC.md §5).

    A block of mass m is released from rest on a rough plane      <- text
    inclined at theta to the horizontal ...
    GIVEN                          FIND
    m = 2 kg                       the acceleration a of the block
    theta = 30 deg

Every field is prose with inline maths written $...$. Prose is theme body
text and maths is theme LaTeX, set on one shared baseline and flowed into
lines here -- Manim's Text neither wraps nor knows where its baseline is, so
both are measured rather than assumed.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from typing import Annotated, NamedTuple

import numpy as np
from manim import DL, LEFT, RIGHT, UP, Mobject, Text, VGroup
from pydantic import AfterValidator, Field

from chalkdust.continuity import artifact_builder
from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    MIN_STEP_SECONDS,
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
)
from chalkdust.scenes.theme import (
    Theme,
    body_cap_height,
    body_text,
    caption_text,
    check_renderable,
    emphasize,
    math,
    refuse_invalid_latex,
)

# The paragraph measure in average characters: a comfortable reading line,
# capped by the stage width for themes with wide type.
MEASURE_CHARS = 64

# Spacing in body cap heights, so it scales with the theme.
LINE_PITCH = 1.7     # baseline to baseline for an ordinary line of prose
LINE_CLEARANCE = 0.25  # minimum ink gap when tall inline maths needs more room
WORD_SPACE = 0.4     # between prose and an adjacent maths chunk
# Between maths and prose the source joins with no space ("$\mu$.",
# "$x$-axis"): TeX's italic correction, so a slanted letter's tail does not
# run into the punctuation after it.
ITALIC_GAP = 0.08
ITEM_GAP = 0.5       # extra space between two given items
HEADER_GAP = 0.5     # between a GIVEN/FIND header and its content
SECTION_GAP = 1.4    # between the statement and the given/find row
COLUMN_GAP = 1.6     # between the given column and the find column

# Prefix compiled ahead of every inline maths chunk. The H is a strut: its
# bottom is the TeX baseline, the one thing a MathTex bounding box cannot tell
# us. It is measured, then removed. \textstyle keeps fractions and limits at
# inline size -- MathTex sets display maths otherwise.
STRUT = r"\mathrm{H}\textstyle "

# The same idea for prose: each run is typeset as PROSE_STRUT + run, and the
# H's bottom marks the run's baseline -- Pango's Text has no baseline either.
# The space keeps the H a glyph of its own: a run that starts with a
# combining mark would compose onto it ("H" + U+0302 shapes as one glyph).
PROSE_STRUT = "H "

# Relative weights of each step's share of the beat (D-002), handed to
# scene.budget(). A weight of 1 -- one given -- is one reveal step, so it needs
# MIN_STEP_SECONDS to register. The statement's weight is split evenly over
# its lines, so the total -- and min_seconds() -- is known from the params
# alone, before any text is measured.
SECONDS_PER_WEIGHT = MIN_STEP_SECONDS
STATEMENT_WEIGHT = 3
GIVEN_WEIGHT = 1
FIND_WEIGHT = 2
HOLD_WEIGHT = 3


def _blank(c: str) -> bool:
    """A character that never draws ink: whitespace, a separator, or a format
    character such as a zero-width space or joiner -- which str.strip()
    keeps, so a field of them alone would otherwise pass as text."""
    return c.isspace() or unicodedata.category(c) in {"Cf", "Zs", "Zl", "Zp"}


def _inline(v: str) -> str:
    """Prose with $...$ maths. Rejected here rather than at build time,
    because an unbalanced $ would silently swap which parts are maths."""
    v = v.strip()
    if all(_blank(c) for c in v):
        raise ValueError("cannot be blank: it has no visible character")
    if any(unicodedata.category(c) == "Cc" and not c.isspace() for c in v):
        raise ValueError("contains a control character")
    if v.count("$") % 2:
        raise ValueError(
            "unbalanced $: inline maths is written $...$ and a literal "
            "dollar sign is not supported"
        )
    if any(not chunk.strip() for chunk in v.split("$")[1::2]):
        raise ValueError("empty inline maths ($$)")
    return v


InlineText = Annotated[str, AfterValidator(_inline)]


class ProblemStatementParams(ComponentParams):
    text: InlineText
    # Capped at 6, like BulletReveal: a problem with more givens than that is
    # a data table, and failing at the schema points at the real fix.
    given: list[InlineText] = Field(default_factory=list, max_length=6)
    find: InlineText


# --- inline flow ------------------------------------------------------------


class _Token(NamedTuple):
    text: str
    is_math: bool
    space_before: bool  # whether the source had whitespace before it


def _tokens(s: str) -> list[_Token]:
    """Split into prose words and maths chunks, remembering where the source
    had spaces: in "the $x$-axis" nothing separates the maths from "-axis"."""
    out: list[_Token] = []
    space = False
    for i, part in enumerate(s.split("$")):
        if i % 2:
            out.append(_Token(part.strip(), True, space))
            space = False
            continue
        words = part.split()
        if not words:
            space = space or bool(part)
            continue
        lead = space or part[0].isspace()
        out += [_Token(w, False, lead if j == 0 else True) for j, w in enumerate(words)]
        space = part[-1].isspace()
    return out


def _maths(chunk: str) -> str:
    """The exact LaTeX string compiled for one inline maths chunk."""
    return STRUT + chunk


def _braces_balance(chunk: str) -> bool:
    """Every unescaped { closed by a later }, none left open. \\{ and \\} are
    literal braces and do not count."""
    depth, i = 0, 0
    while i < len(chunk):
        c = chunk[i]
        if c == "\\":
            i += 2
            continue
        depth += (c == "{") - (c == "}")
        if depth < 0:
            return False
        i += 1
    return depth == 0


@lru_cache(maxsize=16)
def _char_width(theme: Theme) -> float:
    """Average prose character width, for estimating where lines break. The
    estimate is then checked against real widths, so it only has to be close."""
    sample = "The quick brown fox jumps over the lazy dog"
    return float(body_text(sample, theme).width) / len(sample)


@lru_cache(maxsize=16)
def _math_size(theme: Theme) -> float:
    """The maths size whose capitals match the prose's. Computer Modern and
    the theme's sans differ in proportion, so at equal nominal sizes inline
    maths reads a size smaller than the words around it."""
    tex_cap = math(r"\mathrm{H}", theme, size=theme.type.body).height
    return theme.type.body * body_cap_height(theme) / tex_cap


def _baselined(mob: Mobject, strut: Mobject) -> Mobject:
    """Remove the strut glyph and put the baseline it marked at y=0."""
    baseline = strut.get_bottom()[1]
    for m in mob.get_family():
        if strut in m.submobjects:
            m.remove(strut)
            break
    return mob.shift(UP * -baseline)


@lru_cache(maxsize=16)
def _strut_outline(theme: Theme) -> np.ndarray:
    """The prose strut's outline, built alone, relative to its lower-left
    corner -- what _prose_strut() looks for in a run typeset behind it."""
    [glyph] = body_text(PROSE_STRUT, theme).family_members_with_points()
    return glyph.points - glyph.get_corner(DL)


def _prose_strut(t: Text, theme: Theme) -> Mobject:
    """The strut glyph in `t` = body_text(PROSE_STRUT + run), found by shape.

    Not t[0]: the order of Text's submobjects is Pango's order of drawn
    paths, not the string's. A missing-glyph box is several paths, and with
    one in the run Manim listed a piece of the box first -- the H stayed on
    screen and the baseline was read off the box (verify-rb2). So: every
    glyph whose outline is the H built alone (same font and size, so the
    same points up to translation), and of those the leftmost -- the strut
    leads the string, and its H sets the line left-to-right. A run's own H
    ("Hence") matches too, but always to the strut's right.
    """
    ref = _strut_outline(theme)
    tol = 1e-6 * body_cap_height(theme)
    found = [g for g in t.family_members_with_points()
             if g.points.shape == ref.shape
             and np.allclose(g.points - g.get_corner(DL), ref, rtol=0, atol=tol)]
    if not found:
        # Unreachable for text the theme's glyph guard lets through: the
        # strut is set apart by a space and nothing kerns its outline.
        raise RuntimeError(f"ProblemStatement: no baseline strut in {t.original_text!r}")
    return min(found, key=lambda g: g.get_left()[0])


class _Flow:
    """Lays one InlineText out into lines no wider than `width`."""

    def __init__(self, s: str, theme: Theme, width: float, what: str,
                 max_height: float = float("inf")) -> None:
        # `what` names the spec field ("text", "given[1]", "find") in a
        # refusal, so the repair loop knows which one to regenerate.
        # `max_height` is the tallest stack of lines that could still fit
        # STAGE at the legibility floor (see _layout); past it, lines()
        # refuses rather than typesetting the rest of a paragraph that can
        # only be refused once fit_to_region sees it.
        self.source, self.theme, self.width, self.what = s, theme, width, what
        self.max_height = max_height
        self.cap = body_cap_height(theme)
        self.tokens = _tokens(s)
        self.maths = {i: self._compile(t.text)
                      for i, t in enumerate(self.tokens) if t.is_math}

    def _compile(self, chunk: str) -> Mobject:
        # theme.math refuses LaTeX that does not compile as LayoutError kind
        # "invalid_latex". Its renders-nothing check cannot fire here -- the
        # strut always draws -- so a chunk that adds no glyph of its own
        # (${}$, $\quad$) is refused the same way, by the same function.
        what = f"ProblemStatement {self.what}"
        if not _braces_balance(chunk):
            # Manim closes its own wrapper group with a brace right after the
            # source, so "\frac{1}{2" compiles -- swallowing the wrapper's
            # brace -- and is never reported by LaTeX itself.
            raise refuse_invalid_latex(what, chunk, "has unbalanced braces")
        m = math(_maths(chunk), self.theme, size=_math_size(self.theme), what=what)
        glyphs = m.family_members_with_points()
        if len(glyphs) < 2:
            raise refuse_invalid_latex(what, chunk, "renders nothing")
        return _baselined(m, glyphs[0])

    def _estimate(self, i: int, scale: float = 1.0) -> float:
        """Token i's width: maths is built, so exact; prose is the pangram
        average times `scale`, the ratio real prose has shown so far."""
        t = self.tokens[i]
        if t.is_math:
            return self.maths[i].width
        return len(t.text) * _char_width(self.theme) * scale

    def _fill(self, start: int, scale: float) -> list[int]:
        """Greedy: the tokens from `start` that fit one line on estimated
        widths -- always at least one, so a token wider than the measure
        gets a line of its own and fit_to_region scales it or refuses."""
        space = WORD_SPACE * self.cap
        idx: list[int] = []
        x = 0.0
        for i in range(start, len(self.tokens)):
            gap = space if idx and self.tokens[i].space_before else 0.0
            w = self._estimate(i, scale)
            if idx and x + gap + w > self.width:
                break
            idx.append(i)
            x += gap + w
        return idx

    def _prose_widths(self, idx: list[int], line: VGroup) -> tuple[float, float] | None:
        """(real, estimated at scale 1) width of the prose in a built line:
        the line's width less what the estimate already has exactly (maths)
        or fixes (word spaces). None for a line with no prose to measure."""
        space = WORD_SPACE * self.cap
        est = fixed = 0.0
        for n, i in enumerate(idx):
            if n and self.tokens[i].space_before:
                fixed += space
            if self.tokens[i].is_math:
                fixed += self.maths[i].width
            else:
                est += self._estimate(i)
        real = line.width - fixed
        return (real, est) if est > 0 and real > 0 else None

    def _split_long_words(self) -> None:
        """Hard-break prose words wider than a line, as wrap() does for the
        other components. Maths is never broken; fit_to_region decides."""
        per_line = max(1, int(self.width / _char_width(self.theme)))
        out, maths = [], {}
        for i, t in enumerate(self.tokens):
            if t.is_math:
                maths[len(out)] = self.maths[i]
                out.append(t)
            elif len(t.text) > per_line:
                pieces = [t.text[k:k + per_line] for k in range(0, len(t.text), per_line)]
                out += [_Token(p, False, t.space_before if k == 0 else False)
                        for k, p in enumerate(pieces)]
            else:
                out.append(t)
        self.tokens, self.maths = out, maths

    def _prose(self, run: str) -> Mobject:
        """One run of prose words, baseline at y=0, strut removed."""
        # The theme's glyph guard, on exactly the run and naming the field:
        # body_text() below would refuse the same characters, but quoting
        # the strut-prefixed string the spec never contained.
        check_renderable(run, self.theme.type.body_font,
                         what=f"ProblemStatement {self.what}")
        t = body_text(PROSE_STRUT + run, self.theme)
        out = _baselined(t, _prose_strut(t, self.theme))
        # What the mobject now draws -- the snapshot records this string.
        out.original_text = run
        return out

    def _line(self, idx: list[int]) -> VGroup:
        """Build one line, baseline at y=0, left edge at x=0."""
        pieces: list[tuple[Mobject, bool]] = []
        is_maths = {id(m) for m in self.maths.values()}
        run: list[str] = []
        run_space = False

        def flush() -> None:
            if run:
                piece = self._prose("".join(run))
                # A run of zero-width characters alone draws nothing; it
                # takes no place on the line (and no word space).
                if piece.family_members_with_points():
                    pieces.append((piece, run_space))
                run.clear()

        for n, i in enumerate(idx):
            t = self.tokens[i]
            if t.is_math:
                flush()
                pieces.append((self.maths[i], t.space_before and n > 0))
            else:
                if not run:
                    run_space = t.space_before and n > 0
                    run.append(t.text)
                else:
                    run.append((" " if t.space_before else "") + t.text)
        flush()

        x, after_maths = 0.0, False
        for mob, spaced in pieces:
            if spaced:
                x += WORD_SPACE * self.cap
            elif after_maths and id(mob) not in is_maths:
                x += ITALIC_GAP * self.cap
            after_maths = id(mob) in is_maths
            mob.shift(RIGHT * (x - mob.get_left()[0]))
            x = mob.get_right()[0]
        return VGroup(*(m for m, _ in pieces))

    def lines(self) -> list[VGroup]:
        """The lines, each with its baseline at y=0, for _stack()."""
        self._split_long_words()
        # Line by line: fill greedily on estimated widths, build, and if the
        # line is still too wide, re-fill it from the width it really set.
        # The rest of the paragraph is planned from the next unplaced token
        # each time, never from an up-front plan: correcting a fixed plan by
        # handing each overlong line's last token down pushed the excess onto
        # every later line, which then rebuilt once per token it handed on --
        # quadratic in the text, so an all-caps or wide-lettered statement
        # took minutes to build and a huge one hours to refuse (verify-w4-1).
        # The pangram average underestimates such text, so the ratio real
        # prose has shown so far scales the estimate for every later line:
        # wide text then costs about one build per line, and ordinary prose
        # (ratio at or under 1) breaks where it always did.
        pitch = LINE_PITCH * self.cap
        built: list[VGroup] = []
        start, scale = 0, 1.0
        seen_real = seen_est = 0.0
        while start < len(self.tokens):
            idx = self._fill(start, scale)
            line = self._line(idx)
            while line.width > self.width and len(idx) > 1:
                widths = self._prose_widths(idx, line)
                shorter = (self._fill(start, max(scale, widths[0] / widths[1]))
                           if widths else idx)
                # Never the same line twice: if the re-fill cannot shorten
                # it (a ratio from very little prose), drop the last token.
                idx = shorter if len(shorter) < len(idx) else idx[:-1]
                line = self._line(idx)
            start = idx[-1] + 1
            widths = self._prose_widths(idx, line)
            if widths:
                seen_real, seen_est = seen_real + widths[0], seen_est + widths[1]
                scale = max(1.0, seen_real / seen_est)
            if not line.family_members_with_points():
                continue
            built.append(line)
            if (len(built) - 1) * pitch > self.max_height:
                # Baselines are at least LINE_PITCH apart (_stack), so these
                # lines alone are taller than anything fit_to_region could
                # shrink to legibility: refuse now, with the kind it would
                # use, rather than build the rest only to be refused.
                raise LayoutError(
                    f"ProblemStatement {self.what} needs more than {len(built) - 1} "
                    f"lines, which cannot fit the stage at a legible size. "
                    f"Shorten it or split the problem over beats.",
                    kind="overflow")
        # Characters the theme font cannot draw never get here: the theme's
        # glyph guard refuses them in _prose() as unrenderable_text. Zero-width
        # characters do, and draw nothing; a line of nothing but those was
        # dropped above, rather than stacked as a blank line with an empty
        # reveal step. The schema guarantees each field a visible character,
        # so a field left with no line at all would be a gap in that chain; it is
        # refused with the guard's kind rather than laid out as an empty group
        # at the origin (which the settle check reported as an overlap).
        if not built:
            raise LayoutError(
                f"ProblemStatement {self.what} {self.source!r} draws nothing in "
                f"font {self.theme.type.body_font!r}. Write the text the viewer "
                f"should read.", kind=UNRENDERABLE_TEXT)
        return built


def _stack(rows: list[list[VGroup]], cap: float) -> list[VGroup]:
    """Stack rows of lines, each built with its baseline at y=0, top to bottom.

    Baseline to baseline at a constant pitch, so a superscript or a
    descender never changes the rhythm -- more only where tall maths would
    otherwise touch the line above. Rows (given items) get ITEM_GAP extra.
    The first baseline stays at y=0; returns one group per row.
    """
    baseline, prev_descent = 0.0, None
    for lines in rows:
        for j, line in enumerate(lines):
            ascent, descent = line.get_top()[1], -line.get_bottom()[1]
            if prev_descent is not None:
                baseline -= (ITEM_GAP * cap if j == 0 else 0.0) + max(
                    LINE_PITCH * cap, prev_descent + LINE_CLEARANCE * cap + ascent)
            line.shift(UP * baseline)
            prev_descent = descent
    return [VGroup(*lines) for lines in rows]


class _Layout(NamedTuple):
    """The settled visual, at full opacity and not yet fitted to STAGE."""

    blocks: list[VGroup]        # statement, [given], find: the exclusive groups
    lines: list[VGroup]         # statement lines, revealed one by one
    given_head: Mobject | None
    given_items: list[VGroup]
    find_head: Mobject
    find_item: VGroup


def _layout(p: ProblemStatementParams, theme: Theme) -> _Layout:
    """Statement over a GIVEN | FIND row. Shared by build() and the carry-in
    artifact builder, so a carried problem is exactly the one that was shown."""
    cap = body_cap_height(theme)
    stage = region_rect(Region.STAGE).inset(DEFAULT_PADDING)
    measure = min(stage.width, MEASURE_CHARS * _char_width(theme))
    # GIVEN and FIND share the measure as two columns.
    column = (measure - COLUMN_GAP * cap) / 2
    # The tallest any one field's lines could be and still fit. FIND's
    # caption is always in the layout, so fit_to_region can shrink it at most
    # to MIN_FONT_SIZE / caption before the legibility floor refuses it;
    # STAGE's height over that is a bound no field may pass (_Flow.lines).
    tallest = stage.height * theme.type.caption / MIN_FONT_SIZE

    lines = _Flow(p.text, theme, measure, "text", tallest).lines()
    [statement] = _stack([lines], cap)
    label(statement, "statement")
    for i, line in enumerate(lines):
        label(line, f"statement[{i}]")

    def section(header: str, items: list[VGroup], x: float,
                top: float) -> tuple[VGroup, Mobject]:
        """A captioned column, header over its items, top-left at (x, top).
        Both headers are caps-only captions, so side-by-side GIVEN and
        FIND columns line up without being aligned to each other."""
        head = caption_text(header, theme)
        head.next_to(items[0], UP, buff=HEADER_GAP * cap, aligned_edge=LEFT)
        block = VGroup(head, *items)
        block.shift(UP * (top - block.get_top()[1])
                    + RIGHT * (x - block.get_left()[0]))
        return block, head

    left = statement.get_left()[0]
    top = statement.get_bottom()[1] - SECTION_GAP * cap
    blocks = [statement]
    given_items = _stack([_Flow(g, theme, column, f"given[{i}]", tallest).lines()
                          for i, g in enumerate(p.given)], cap)
    for i, item in enumerate(given_items):
        label(item, f"given[{i}]")

    # FIND sits beside the givens, or under the statement when there are
    # none. A given whose maths is too wide to break overruns its column;
    # then FIND drops below the givens at the full measure, so the overrun
    # can never reach it.
    given_head = None
    find_x, find_top, find_width = left, top, measure
    if given_items:
        given_block, given_head = section("GIVEN", given_items, left, top)
        blocks.append(label(given_block, "given"))
        if given_block.get_right()[0] <= left + column + 1e-6:
            find_x, find_width = left + column + COLUMN_GAP * cap, column
        else:
            find_top = given_block.get_bottom()[1] - SECTION_GAP * cap
    [find_item] = _stack([_Flow(p.find, theme, find_width, "find", tallest).lines()],
                         cap)
    label(find_item, "find value")
    emphasize(find_item, theme)
    find_block, find_head = section("FIND", [find_item], find_x, find_top)
    blocks.append(label(find_block, "find"))
    return _Layout(blocks, lines, given_head, given_items, find_head, find_item)


@artifact_builder("ProblemStatement")
def _artifact(params: ProblemStatementParams, theme: Theme) -> Mobject:
    """The stated problem, for a later beat to carry in (SCENE_SPEC.md §6) --
    typically kept on screen, dimmed, while the solution is worked."""
    return VGroup(*_layout(params, theme).blocks)


# --- the component ----------------------------------------------------------


@register
class ProblemStatement(Component):
    name = "ProblemStatement"
    Params = ProblemStatementParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def _weights(self, statement_lines: int = 1) -> list[float]:
        n_given = len(self.params.given)
        return ([STATEMENT_WEIGHT / statement_lines] * statement_lines
                + [GIVEN_WEIGHT] * n_given + [FIND_WEIGHT, HOLD_WEIGHT])

    def min_seconds(self) -> float:
        return SECONDS_PER_WEIGHT * sum(self._weights())

    def latex_strings(self) -> list[str]:
        p: ProblemStatementParams = self.params
        return [_maths(t.text)
                for field in (p.text, *p.given, p.find)
                for t in _tokens(field) if t.is_math]

    def build(self, scene: ChalkdustScene) -> None:
        lay = _layout(self.params, scene.theme)
        fit_to_region(label(VGroup(*lay.blocks), "ProblemStatement"), Region.STAGE)

        # Separate top-level mobjects so the settle check asserts the
        # statement and the columns never overlap; revealed by opacity, as
        # BulletReveal does, so the checks see the final geometry.
        for block in lay.blocks:
            for mob in block.get_family():
                mob.set_opacity(0)
            scene.add(block)
        scene.exclusive(*lay.blocks)

        # Whole-frame run times that sum to the beat exactly (D-002); the
        # scene makes Manim render each as exactly that many frames.
        times = iter(scene.budget(*self._weights(len(lay.lines))))
        for line in lay.lines:
            scene.play(line.animate.set_opacity(1), run_time=next(times))
        for i, item in enumerate(lay.given_items):
            reveal = [lay.given_head, item] if i == 0 else [item]
            scene.play(*(m.animate.set_opacity(1) for m in reveal), run_time=next(times))
        scene.play(lay.find_head.animate.set_opacity(1),
                   lay.find_item.animate.set_opacity(1), run_time=next(times))
        scene.settle("problem stated")
        scene.wait(next(times))

    @classmethod
    def examples(cls):
        return [
            {"text": r"A block of mass $m$ is released from rest at the top of a "
                     r"rough plane inclined at $\theta$ to the horizontal. The "
                     r"coefficient of kinetic friction between the block and the "
                     r"plane is $\mu$.",
             "given": [r"$m = 2\,\mathrm{kg}$", r"$\theta = 30^\circ$",
                       r"$\mu = 0.2$", r"$g = 10\,\mathrm{m\,s^{-2}}$"],
             "find": r"the acceleration $a$ of the block down the plane"},
            {"text": r"Evaluate the definite integral "
                     r"$\int_0^{\pi/2} \frac{\sin x}{\sin x + \cos x}\,dx$.",
             "find": r"the value of the integral $I$"},
            {"text": r"A particle moves along the $x$-axis so that its position "
                     r"at time $t$ seconds is $x(t) = t^3 - 6t^2 + 9t$ metres.",
             "given": [r"$x(t) = t^3 - 6t^2 + 9t$", r"$t \ge 0$"],
             "find": r"the times at which the particle is momentarily at rest, "
                     r"and the distance it travels between them"},
        ]

    @classmethod
    def stress(cls):
        incline = cls.examples()[0]
        return [
            # Three times a realistic statement, six long givens, a long find.
            {"text": " ".join([incline["text"]] * 3),
             "given": [r"the mass of the block on the incline is $m = 2\,\mathrm{kg}$ "
                       r"measured on a calibrated balance"] * 6,
             "find": "the acceleration of the block down the plane, the normal "
                     "reaction on it, and the work done against friction over "
                     r"the first $2\,\mathrm{m}$ of its slide"},
            # Unwrappable tokens: a 60-char identifier and a URL in prose, and
            # a 60-char maths chunk that cannot be broken at all.
            {"text": "The constant " + "k" * 60 + " is defined at "
                     "https://example.org/" + "a" * 40 + " for this problem.",
             "given": [r"$\mathrm{" + "X" * 60 + "}$"],
             "find": "the_value_of_" + "q" * 47},
            # Unbreakable givens wider than their column but not the stage:
            # FIND must drop below them, not collide.
            {"text": incline["text"],
             "given": [r"$v = \sqrt{2gh} = \sqrt{2 \times 10 \times 5} = "
                       r"10\,\mathrm{m\,s^{-1}}$",
                       r"$g = 10\,\mathrm{m\,s^{-2}}$"],
             "find": incline["find"]},
            # Minimal: one character each, no givens.
            {"text": "x", "find": "y"},
            {"text": "$x$", "given": ["$y$"], "find": "$z$"},
            # Invalid LaTeX, which must refuse as invalid_latex: an unknown
            # command, an unclosed brace, and maths that compiles to nothing.
            {"text": r"The net force on the block is $\notacommand{F}$.",
             "find": "the acceleration"},
            {"text": incline["text"], "given": [r"$\mu = \frac{1}{2$"],
             "find": incline["find"]},
            {"text": "x", "find": r"the value of $\quad$"},
            # Glyph-less prose: emoji and right-to-left letters the theme font
            # draws as nothing. The theme's glyph guard refuses each as
            # unrenderable_text, naming the field -- not laid out as an empty
            # group at the origin, and not silently missing a word.
            {"text": "\U0001F600" * 10, "find": "y"},
            {"text": "The block \u05d0\u05d1 rests on the plane.", "find": "y"},
            # Text that draws but is hostile to the baseline strut: zero-width
            # spaces long enough to hard-break into lines of their own (dropped:
            # they draw nothing), and words opening with a combining mark (which
            # must not compose onto the strut and vanish with it).
            {"text": "\u200b" * 200 + " then $x$ rests",
             "given": ["\u0302a = 1"], "find": "\u0301y"},
            # Private-use characters (Symbol-font Greek pasted from a PDF) are
            # refused at rung 1 by BeatSpec before any component is built, so
            # they cannot be a case here; the component's tests pin that, and
            # the theme's guard beneath it.
        ]
