"""UnitBreakdown: a quantity's unit decomposed into its factors.

    N    =    kg    .    m     .   s^-2
  force      mass      length    per second squared

The unit row is compiled as ONE LaTeX expression, so TeX sets every baseline
(a superscript never drags its term off the line). The row is then re-spaced
horizontally into columns as wide as each term's label, so labels can never
collide however short the unit above them is.
"""

from __future__ import annotations

from manim import DOWN, RIGHT, UP, VGroup
from pydantic import Field, field_validator

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
from chalkdust.scenes.theme import body_cap_height, body_text, emphasize, math

LABEL_WRAP = 16

# Spacing in body cap heights, so it scales with the theme.
OPERATOR_PAD = 0.9  # each side of "=" / "\cdot", beyond the column edge
LABEL_GAP = 1.2     # between the lowest unit glyph and the labels' common top

# Relative weights of each step's share of the beat (D-002). One unit of
# weight must be at least this long for the step to stay legible, so the
# minimum beat length is simply the weight total times this.
SECONDS_PER_WEIGHT = 0.4
QUANTITY_WEIGHT = 2
FACTOR_WEIGHT = 2
HOLD_WEIGHT = 3


class LatexError(LayoutError):
    """Unit LaTeX that does not compile, or that mis-splits into terms.

    A LayoutError subclass because that is the only exception the geometric
    probe preserves the kind of (validate/geometric.py) -- anything else is
    reported as a build_error crash, which this is not: it is bad content
    from the spec, and the repair loop should regenerate it (SCENE_SPEC.md §9).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="latex")


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


@register
class UnitBreakdown(Component):
    name = "UnitBreakdown"
    Params = UnitBreakdownParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def _expression(self) -> str:
        """The single LaTeX string we compile.

        Each term sits in Manim's {{ }} group notation, so the compiled
        MathTex comes back as alternating term / operator submobjects.
        """
        p: UnitBreakdownParams = self.params
        parts = ["{{ " + p.quantity.unit + " }}", "="]
        for i, term in enumerate(p.decomposition):
            if i:
                parts.append(r"\cdot")
            parts.append("{{ " + term.unit + " }}")
        return " ".join(parts)

    def _weights(self) -> list[float]:
        n = len(self.params.decomposition)
        return [QUANTITY_WEIGHT] + [FACTOR_WEIGHT] * n + [HOLD_WEIGHT]

    def min_seconds(self) -> float:
        return SECONDS_PER_WEIGHT * sum(self._weights())

    def latex_strings(self) -> list[str]:
        return [self._expression()]

    def build(self, scene: ChalkdustScene) -> None:
        p: UnitBreakdownParams = self.params
        theme = scene.theme
        cap = body_cap_height(theme)
        expr = self._expression()

        try:
            row = math(expr, theme)
        except ValueError as exc:
            # Manim raises ValueError when LaTeX rejects the source. Its
            # RuntimeError (no log file at all) is a broken installation, not
            # bad content, so it is deliberately left to propagate.
            raise LatexError(f"UnitBreakdown LaTeX failed to compile: {expr!r} ({exc})") from exc

        terms_in = [p.quantity, *p.decomposition]
        pieces = row.submobjects
        # Unbalanced braces in a unit can compile yet split into a different
        # number of groups, and then terms and operators would be mismatched.
        if len(pieces) != 2 * len(terms_in) - 1 or not all(
            any(m.has_points() for m in piece.get_family()) for piece in pieces
        ):
            raise LatexError(
                f"UnitBreakdown LaTeX did not split into {len(terms_in)} visible "
                f"terms: {expr!r}. Check each unit's braces."
            )
        terms, operators = pieces[0::2], pieces[1::2]
        emphasize(terms[0], theme)

        labels = [
            label(body_text(wrap(t.label, LABEL_WRAP), theme, theme.palette.muted),
                  f"label[{i}]")
            if t.label else None
            for i, t in enumerate(terms_in)
        ]

        # Re-space horizontally only: TeX already set the baselines, and a
        # pure x-shift keeps them.
        x = terms[0].get_left()[0]
        for i, (term, lab) in enumerate(zip(terms, labels)):
            if i:
                op = operators[i - 1]
                op.shift(RIGHT * (x + OPERATOR_PAD * cap - op.get_left()[0]))
                x = op.get_right()[0] + OPERATOR_PAD * cap
            width = max(term.width, lab.width if lab is not None else 0.0)
            term.shift(RIGHT * (x + width / 2 - term.get_center()[0]))
            x += width

        # Labels hang from one common line, so a two-line label never pushes
        # its neighbours down.
        floor = min(term.get_bottom()[1] for term in terms) - LABEL_GAP * cap
        present = []
        for term, lab in zip(terms, labels):
            if lab is None:
                continue
            lab.shift(RIGHT * (term.get_center()[0] - lab.get_center()[0])
                      + UP * (floor - lab.get_top()[1]))
            present.append(lab)

        label(row, "units")
        breakdown = label(VGroup(row, *present), "UnitBreakdown")
        fit_to_region(breakdown, Region.STAGE)

        # One top-level mobject, revealed by opacity, so the settle checks see
        # the theme's font-size tags on the whole row (as BulletReveal does).
        steps = [[terms[0]] + ([labels[0]] if labels[0] is not None else [])]
        for op, term, lab in zip(operators, terms[1:], labels[1:]):
            steps.append([op, term] + ([lab] if lab is not None else []))
        for mob in breakdown.get_family():
            mob.set_opacity(0)
        scene.add(breakdown)

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
            # Minimal: one factor, no labels.
            {"quantity": {"unit": "x"}, "decomposition": [{"unit": "y"}]},
        ]
