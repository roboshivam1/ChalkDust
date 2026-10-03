"""UnitBreakdown: a quantity's unit decomposed into its factors.

    N    =    kg    .    m     .   s^-2
  force      mass      length    per second squared

The unit row is compiled as ONE LaTeX expression, so TeX sets every baseline
(a superscript never drags its term off the line). The row is then re-spaced
horizontally into columns as wide as each term's label, so labels can never
collide however short the unit above them is.

The settled row (units plus labels) is also this component's carry-in artifact
(SCENE_SPEC.md §6): `_layout` is a pure function of params and theme, shared by
build() and the registered artifact builder, so a later beat rebuilds exactly
what this one showed.
"""

from __future__ import annotations

from dataclasses import dataclass

from manim import RIGHT, UP, Mobject, Text, VGroup
from pydantic import Field, field_validator

from chalkdust.continuity import artifact_builder
from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    Component,
    ComponentParams,
    label,
    register,
    wrap,
)
from chalkdust.scenes.regions import LayoutError, fit_to_region
from chalkdust.scenes.theme import (
    Theme,
    body_cap_height,
    body_text,
    emphasize,
    math,
    refuse_invalid_latex,
)

LABEL_WRAP = 16

# Spacing in body cap heights, so it scales with the theme.
OPERATOR_PAD = 0.9  # each side of "=" / "\cdot", beyond the column edge
LABEL_DROP = 2.2    # lowest unit glyph down to the labels' common baseline

# Relative weights of each step's share of the beat (D-002). One unit of
# weight must be at least this long for the step to stay legible, so the
# minimum beat length is simply the weight total times this.
SECONDS_PER_WEIGHT = 0.4
QUANTITY_WEIGHT = 2
FACTOR_WEIGHT = 2
HOLD_WEIGHT = 3


def _first_baseline(text: Text, first_line: Text) -> float:
    """y of the baseline of `text`'s first line, measured on `first_line`: that
    line alone, built with the same font and size.

    Bounding-box tops vary with ascenders and bottoms with descenders, so
    neither gives a stable line to hang text from. On a single line every
    glyph sits on one baseline, and the median glyph bottom is that baseline,
    outvoting the odd descender (the "g" in "length"). It is measured on the
    line alone because nothing maps a multi-line Text's glyphs back to its
    lines: Pango draws no glyph for a zero-width or format character and
    composes "e" plus a combining acute into one, so slicing the first n
    glyphs for n characters (rb2) took glyphs from the second line and hung
    the label a whole line too high, into the unit row. The first line's ink
    is the top of the whole label (the next line starts a full line pitch
    lower), so its baseline sits the same distance below `text`'s top as it
    does below `first_line`'s.
    """
    bottoms = sorted(g.get_bottom()[1] for g in first_line.submobjects
                     if g.has_points())
    # first_line came from a theme constructor, which refuses text that draws
    # nothing, so there is always a glyph to vote.
    depth = first_line.get_top()[1] - bottoms[len(bottoms) // 2]
    return float(text.get_top()[1] - depth)


class UnitTerm(ComponentParams):
    # LaTeX, math mode, e.g. r"\mathrm{kg}" or r"\mathrm{s}^{-2}".
    unit: str
    # Optional plain-text meaning shown under the unit ("mass"). Not in the
    # SCENE_SPEC.md §5 key params, but a decomposition the viewer cannot read
    # the dimensions of is just a string of symbols.
    label: str | None = None

    @field_validator("unit")
    @classmethod
    def _unit_not_blank(cls, v: str) -> str:
        # A blank term compiles to nothing and leaves an orphan operator.
        v = v.strip()
        if not v:
            raise ValueError("unit cannot be blank")
        return v

    @field_validator("label")
    @classmethod
    def _label_not_blank(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("label cannot be blank; omit it instead")
        return v


class UnitBreakdownParams(ComponentParams):
    quantity: UnitTerm
    # Capped at 6, like BulletReveal: past that the row is a wall of symbols,
    # and failing at the schema points at the real fix (split the beat).
    decomposition: list[UnitTerm] = Field(min_length=1, max_length=6)


# What a refusal names, per term, so the repair loop can point the LLM at the
# exact spec field to regenerate (SCENE_SPEC.md §9).
def _fields(p: UnitBreakdownParams, attr: str = "unit") -> list[tuple[str, UnitTerm]]:
    return [(f"quantity.{attr}", p.quantity),
            *((f"decomposition[{i}].{attr}", t) for i, t in enumerate(p.decomposition))]


def _expression(p: UnitBreakdownParams) -> str:
    """The single LaTeX string we compile.

    Each term sits in Manim's {{ }} group notation, so the compiled MathTex
    comes back as alternating term / operator submobjects.
    """
    parts = ["{{ " + p.quantity.unit + " }}", "="]
    for i, term in enumerate(p.decomposition):
        if i:
            parts.append(r"\cdot")
        parts.append("{{ " + term.unit + " }}")
    return " ".join(parts)


def _breaks_grouping(unit: str) -> bool:
    r"""Whether `unit` would upset the {{ }} split of the joined row: braces
    that do not balance, or a "{{" opening at the top level (which Manim
    reads as a group boundary). Escaped \{ and \} are literal braces."""
    depth, i = 0, 0
    while i < len(unit):
        c = unit[i]
        if c == "\\":
            i += 2  # a control symbol: \{, \} or \\ is never a group brace
            continue
        if c == "{":
            if depth == 0 and unit.startswith("{{", i):
                return True
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                return True
        i += 1
    return depth != 0


def _compile_row(p: UnitBreakdownParams, theme: Theme) -> VGroup:
    """The unit row as one MathTex of 2n+1 term/operator pieces.

    Every failure refuses through theme.math's path, kind "invalid_latex",
    never a raw exception: bad unit LaTeX is bad content from the spec.
    """
    fields = _fields(p)
    expr = _expression(p)
    try:
        row = math(expr, theme, what="UnitBreakdown units")
    except LayoutError:
        # Name the culprit when one unit fails on its own; the joined row's
        # message alone would leave the repair loop guessing which term.
        # Costs extra compiles only on this failure path.
        for what, term in fields:
            math(term.unit, theme, what=what)
        raise
    pieces = row.submobjects
    # A unit with unbalanced braces, or a {{ }} group of its own, can compile
    # yet split into a different number of groups, and terms and operators
    # would then be mismatched -- a label under the wrong unit.
    if len(pieces) != 2 * len(fields) - 1:
        culprit = next(((w, t.unit) for w, t in fields if _breaks_grouping(t.unit)),
                       ("UnitBreakdown units", expr))
        raise refuse_invalid_latex(
            *culprit,
            f"splits the row into {len(pieces)} pieces, not "
            f"{2 * len(fields) - 1}: balance its braces and drop any {{{{ }}}} group")
    for (what, term), piece in zip(fields, pieces[0::2]):
        if not any(m.has_points() for m in piece.get_family()):
            raise refuse_invalid_latex(what, term.unit, "renders nothing")
    return row


@dataclass
class _Row:
    group: VGroup                 # the settled visual, labelled "UnitBreakdown"
    terms: list[Mobject]          # quantity first, then each factor
    operators: list[Mobject]      # "=", then a "\cdot" between factors
    labels: list[Mobject | None]  # per term; None where the term has no label


def _layout(p: UnitBreakdownParams, theme: Theme) -> _Row:
    """Build and arrange the row, unplaced: a pure function of params and theme
    (no scene, no animation), so the carry-in builder can share it."""
    cap = body_cap_height(theme)
    row = _compile_row(p, theme)
    pieces = row.submobjects
    terms, operators = pieces[0::2], pieces[1::2]
    emphasize(terms[0], theme)

    # Labels go through the theme's text constructor named by spec field, so
    # a label the body font cannot draw, wholly or in part (a lone zero-width
    # space, an emoji, a right-to-left script, CJK the font has no glyphs for),
    # refuses there as "unrenderable_text" naming the field (SCENE_SPEC.md
    # §11 rule 1), before anything is laid out around it.
    label_fields = _fields(p, "label")
    labels = [
        label(body_text(wrap(t.label, LABEL_WRAP), theme, theme.palette.muted,
                        what=what), f"label[{i}]")
        if t.label else None
        for i, (what, t) in enumerate(label_fields)
    ]

    # Re-space horizontally only: TeX already set the baselines, and a pure
    # x-shift keeps them.
    x = terms[0].get_left()[0]
    for i, (term, lab) in enumerate(zip(terms, labels)):
        if i:
            op = operators[i - 1]
            op.shift(RIGHT * (x + OPERATOR_PAD * cap - op.get_left()[0]))
            x = op.get_right()[0] + OPERATOR_PAD * cap
        width = max(term.width, lab.width if lab is not None else 0.0)
        term.shift(RIGHT * (x + width / 2 - term.get_center()[0]))
        x += width
    # Columns are as wide as their labels, so a narrow unit over a wide label
    # leaves its column's edge far from its glyphs; an operator left at the
    # edge reads as belonging to one side ("m  ·     s^-2"). Centre each one in
    # the visible gap between its neighbours' glyphs. The gap is at least
    # 2 * OPERATOR_PAD caps plus the operator, so this only ever moves it
    # within the space the columns already reserved.
    for left, op, right in zip(terms, operators, terms[1:]):
        mid = (left.get_right()[0] + right.get_left()[0]) / 2
        op.shift(RIGHT * (mid - op.get_center()[0]))

    # Labels share one baseline for their first line, so a two-line label
    # hangs down rather than pushing its neighbours, and "mass" sits level
    # with "length" despite having no ascenders.
    baseline = min(term.get_bottom()[1] for term in terms) - LABEL_DROP * cap
    present = []
    for term, lab, (what, t) in zip(terms, labels, label_fields):
        if lab is None:
            continue
        first_line = body_text(wrap(t.label, LABEL_WRAP).split("\n", 1)[0],
                               theme, theme.palette.muted, what=what)
        lab.shift(RIGHT * (term.get_center()[0] - lab.get_center()[0])
                  + UP * (baseline - _first_baseline(lab, first_line)))
        present.append(lab)

    label(row, "units")
    group = label(VGroup(row, *present), "UnitBreakdown")
    return _Row(group, list(terms), list(operators), labels)


@artifact_builder("UnitBreakdown")
def _artifact(params: UnitBreakdownParams, theme: Theme) -> Mobject:
    """The settled decomposition, for a later beat's carry_in (SCENE_SPEC.md
    §6). Unplaced and at full opacity; CarryIn fits and dims it."""
    return _layout(params, theme).group



@register
class UnitBreakdown(Component):
    name = "UnitBreakdown"
    Params = UnitBreakdownParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def _weights(self) -> list[float]:
        n = len(self.params.decomposition)
        return [QUANTITY_WEIGHT] + [FACTOR_WEIGHT] * n + [HOLD_WEIGHT]

    def min_seconds(self) -> float:
        return SECONDS_PER_WEIGHT * sum(self._weights())

    def latex_strings(self) -> list[str]:
        """The one expression build() compiles, {{ }} groups included, so the
        semantic rung compiles exactly what the build will."""
        return [_expression(self.params)]

    def build(self, scene: ChalkdustScene) -> None:
        row = _layout(self.params, scene.theme)
        fit_to_region(row.group, Region.STAGE)

        # One top-level mobject, revealed by opacity, so the settle checks see
        # the theme's font-size tags on the whole row (as BulletReveal does).
        steps = [[row.terms[0]] + ([row.labels[0]] if row.labels[0] is not None else [])]
        for op, term, lab in zip(row.operators, row.terms[1:], row.labels[1:]):
            steps.append([op, term] + ([lab] if lab is not None else []))
        for mob in row.group.get_family():
            mob.set_opacity(0)
        scene.add(row.group)

        # Every run time comes from budget(): whole frames summing exactly to
        # the beat (D-002), so nothing here rounds or snaps on its own.
        times = scene.budget(*self._weights())
        for step, t in zip(steps, times):
            scene.play(*(m.animate.set_opacity(1) for m in step), run_time=t)
        scene.settle("units decomposed")
        scene.wait(times[-1])

    @classmethod
    def examples(cls):
        return [
            {"quantity": {"unit": r"\mathrm{N}", "label": "force"},
             "decomposition": [
                 {"unit": r"\mathrm{kg}", "label": "mass"},
                 {"unit": r"\mathrm{m}", "label": "length"},
                 {"unit": r"\mathrm{s}^{-2}", "label": "per second squared"},
             ]},
            {"quantity": {"unit": r"\mathrm{J}", "label": "energy"},
             "decomposition": [
                 {"unit": r"\mathrm{N}", "label": "force"},
                 {"unit": r"\mathrm{m}", "label": "distance"},
             ]},
            {"quantity": {"unit": r"\mathrm{W}"},
             "decomposition": [{"unit": r"\mathrm{J}"},
                               {"unit": r"\mathrm{s}^{-1}"}]},
        ]

    @classmethod
    def stress(cls):
        return [
            # Max factors, every label far longer than a real one.
            {"quantity": {"unit": r"\mathrm{Pa}\,\mathrm{s}",
                          "label": "dynamic viscosity of the working fluid"},
             "decomposition": [
                 {"unit": r"\mathrm{kg}^{2}",
                  "label": "mass of the sample measured twice over"},
                 {"unit": r"\mathrm{m}^{-1}",
                  "label": "per unit length along the channel"},
                 {"unit": r"\mathrm{s}^{-1}",
                  "label": "per unit time across the run"},
                 {"unit": r"\mathrm{mol}^{-1}",
                  "label": "per mole of dissolved substance"},
                 {"unit": r"\mathrm{K}^{1/2}",
                  "label": "square root of absolute temperature"},
                 {"unit": r"\mathrm{cd}\,\mathrm{sr}",
                  "label": "luminous intensity over solid angle"},
             ]},
            # Unwrappable tokens: a 60-char identifier as label and as unit.
            {"quantity": {"unit": r"\mathrm{" + "X" * 60 + "}",
                          "label": "https://example.org/" + "a" * 40},
             "decomposition": [{"unit": r"\mathrm{kg}",
                                "label": "dimensional_analysis_" + "q" * 39}]},
            # Unwrappable tokens in the labels only, under ordinary units:
            # wrap() hard-breaks them, so the row should still fit.
            {"quantity": {"unit": r"\mathrm{N}",
                          "label": "https://example.org/" + "a" * 40},
             "decomposition": [{"unit": r"\mathrm{kg}",
                                "label": "dimensional_analysis_" + "q" * 39},
                               {"unit": r"\mathrm{m}\,\mathrm{s}^{-2}"}]},
            # Minimal: one factor, no labels.
            {"quantity": {"unit": "x"}, "decomposition": [{"unit": "y"}]},
            # Minimal with labels: one-character units and labels.
            {"quantity": {"unit": "x", "label": "a"},
             "decomposition": [{"unit": "y", "label": "b"}]},
            # Invalid LaTeX: refuses as "invalid_latex", naming the unit.
            {"quantity": {"unit": r"\mathrm{N}"},
             "decomposition": [{"unit": r"\notacommand{kg}"}]},
            # An unclosed brace inside a unit: refuses as "invalid_latex".
            {"quantity": {"unit": r"\mathrm{N"},
             "decomposition": [{"unit": r"\mathrm{kg}"}]},
            # Compiles, but closes its {{ }} group early and mis-splits.
            {"quantity": {"unit": "x }} {{ y"}, "decomposition": [{"unit": "z"}]},
            # Compiles to no glyphs: a unit that renders nothing.
            {"quantity": {"unit": r"\mathrm{N}"}, "decomposition": [{"unit": r"\quad"}]},
            # A label Pango shapes to zero glyphs (a lone zero-width space
            # survives strip()): the theme refuses it as "unrenderable_text",
            # naming the label.
            {"quantity": {"unit": r"\mathrm{N}", "label": "\u200b"},
             "decomposition": [{"unit": r"\mathrm{kg}", "label": "mass"}]},
            # Partly drawable labels: an emoji run before Latin words (rb2's
            # repro, once hung a line too high into the unit row), a
            # right-to-left word, and CJK the body font draws as missing-glyph
            # boxes. Each refuses as "unrenderable_text" naming its label.
            # (A private-use character never gets this far: rung 1 refuses it
            # in any param, SCENE_SPEC.md §8.)
            {"quantity": {"unit": r"\mathrm{N}",
                          "label": "\U0001F600" * 8 + " force and more words"},
             "decomposition": [{"unit": r"\mathrm{kg}", "label": "mass"}]},
            {"quantity": {"unit": r"\mathrm{N}", "label": "force"},
             "decomposition": [{"unit": r"\mathrm{kg}",
                                "label": "\u0643\u062a\u0644\u0629 mass"}]},
            {"quantity": {"unit": r"\mathrm{N}", "label": "force"},
             "decomposition": [{"unit": r"\mathrm{kg}", "label": "mass \u8d28\u91cf"}]},
            # Drawable labels whose glyphs do not map one-to-one onto their
            # characters: zero-width spaces, a word joiner and combining
            # accents draw no glyph of their own. They must render, every
            # first line on the shared baseline.
            {"quantity": {"unit": r"\mathrm{N}",
                          "label": "\u200b" * 8 + "force and more words"},
             "decomposition": [{"unit": r"\mathrm{kg}",
                                "label": "me\u0301tre\u2060kilo\u0301 and more"},
                               {"unit": r"\mathrm{m}", "label": "length"}]},
        ]
