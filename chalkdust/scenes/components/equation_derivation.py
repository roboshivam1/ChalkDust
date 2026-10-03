"""EquationDerivation: step-by-step algebra, each step grown out of the last.

Steps stack downward and stay on screen, so the viewer can always see where
the current line came from. Each new step is produced by TransformMatchingTex
from a COPY of the previous one: the old line stays put and the parts the two
lines share visibly travel down into the new one. Authors mark those shared
parts with Manim's double-brace groups, e.g.

    "{{ x^2 + 6x }} + 5 = 0"  ->  "{{ x^2 + 6x }} + 9 = 4"

Without markup a step still transforms (unmatched pieces cross-fade); the
markup only makes the motion more explanatory.

Annotations ("divide both sides by 2") sit in a margin column to the right of
the equations, each vertically centred on the step it explains.
"""

from __future__ import annotations

from typing import Annotated

from manim import DOWN, LEFT, RIGHT, FadeIn, MathTex, TransformMatchingTex, VGroup, Write
from pydantic import Field, StringConstraints, model_validator

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
from chalkdust.scenes.theme import Theme, body_cap_height, body_text, math

# Blank strings are rejected at the schema rung: an empty step compiles to
# nothing, and an empty annotation is a margin note that says nothing.
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

NOTE_WRAP = 24      # characters per annotation line; keeps the margin column narrow

# Spacing, in body cap heights so it scales with the theme (see bullet_reveal).
ROW_GAP = 1.6       # vertical space between one step's row and the next
NOTE_GAP = 2.5      # horizontal space between the equation column and the notes

# Minimum legible duration of each kind of segment, in seconds. These double
# as the budget weights: build() scales all of them by the same factor, so the
# beat stays legible exactly when the narration is at least min_seconds() long
# (SCENE_SPEC.md §8 rung 2 compares the two).
WRITE_S = 1.0       # writing the first step
TRANSFORM_S = 1.0   # growing step i out of step i-1
ANNOTATE_S = 0.5    # fading in a step's annotation
DWELL_S = 0.75      # holding a freshly completed step still long enough to read it


class Annotation(ComponentParams):
    # Index into `steps` (0-based) of the step this note explains.
    step: int = Field(ge=0)
    text: NonBlank


class EquationDerivationParams(ComponentParams):
    # LaTeX, one string per line of the derivation. Capped at 12 -- three times
    # a realistic 4-step beat -- to bound LaTeX compile cost on runaway input.
    # Legibility is NOT what the cap protects: a derivation too tall for the
    # stage refuses at layout time with a LayoutError, well before 12 steps
    # when they are long.
    steps: list[NonBlank] = Field(min_length=1, max_length=12)
    annotations: list[Annotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _annotations_point_at_steps(self) -> EquationDerivationParams:
        seen: set[int] = set()
        for a in self.annotations:
            if a.step >= len(self.steps):
                raise ValueError(
                    f"annotation for step {a.step} but steps are indexed "
                    f"0..{len(self.steps) - 1}"
                )
            # One note per step: the margin row is sized for one, and two notes
            # on one line is a sign the step should be split.
            if a.step in seen:
                raise ValueError(f"step {a.step} has more than one annotation")
            seen.add(a.step)
        return self


@register
class EquationDerivation(Component):
    name = "EquationDerivation"
    Params = EquationDerivationParams

    def regions(self) -> set[Region]:
        # The equations and their margin notes share the full stage.
        return {Region.STAGE}

    # --- hooks for the semantic rung (SCENE_SPEC.md §8 rung 2) ---------------

    def latex_strings(self) -> list[str]:
        """Every LaTeX string build() compiles, so the semantic rung can compile
        each standalone and fail fast before a scene is constructed."""
        return list(self.params.steps)

    def min_seconds(self) -> float:
        """Shortest narration at which every segment still gets its minimum."""
        return sum(self._segments())

    def _segments(self) -> list[float]:
        """Minimum durations of every timed segment, in the order build() plays
        them. Shared by build() and min_seconds() so they cannot drift apart."""
        p: EquationDerivationParams = self.params
        noted = {a.step for a in p.annotations}
        out: list[float] = []
        for i in range(len(p.steps)):
            out.append(WRITE_S if i == 0 else TRANSFORM_S)
            if i in noted:
                out.append(ANNOTATE_S)
            out.append(DWELL_S)
        return out

    # --- build ---------------------------------------------------------------

    def build(self, scene: ChalkdustScene) -> None:
        p: EquationDerivationParams = self.params
        theme = scene.theme
        cap = body_cap_height(theme)

        eqs = [_step_tex(i, s, theme) for i, s in enumerate(p.steps)]
        notes = {
            a.step: label(body_text(wrap(a.text, NOTE_WRAP), theme, theme.palette.muted),
                          f"annotation[{a.step}]")
            for a in p.annotations
        }

        # Stack rows downward. A row is a step plus its note, centred on each
        # other, so a row is as tall as whichever of the two is taller and a
        # three-line note can never collide with the next step.
        rows: list[VGroup] = []
        for i, eq in enumerate(eqs):
            note = notes.get(i)
            if note is not None:
                note.match_y(eq)
            row = VGroup(eq, *([note] if note is not None else []))
            if rows:
                row.next_to(rows[-1], DOWN, buff=cap * ROW_GAP)
                # Steps share a centre line, like a derivation on a board.
                eq.match_x(eqs[0])
            rows.append(row)

        # Notes form one left-aligned margin column clear of the widest step.
        column = VGroup(*eqs)
        for i, note in notes.items():
            note.next_to(column, RIGHT, buff=cap * NOTE_GAP)
            note.match_y(eqs[i])

        # One fit for everything, so steps and notes scale together and keep
        # their alignment. Raises LayoutError (overflow) when the derivation is
        # too long or too wide to stay legible: the fix is splitting the beat.
        derivation = label(VGroup(*eqs, *notes.values()), "derivation")
        fit_to_region(derivation, Region.STAGE)

        # Every step and note is a top-level scene mobject once revealed, so
        # the settle checks see each one -- including its tracked font size.
        scene.exclusive(*eqs, *notes.values())

        times = iter(scene.budget(*self._segments()))
        for i, eq in enumerate(eqs):
            if i == 0:
                scene.play(Write(eq), run_time=next(times))
            else:
                # Transform from a copy: TransformMatchingTex removes its source
                # from the scene when it finishes, and the previous step must
                # stay. The target is the fitted, font-size-tagged step itself,
                # so what remains on screen carries an honest size tag.
                scene.play(TransformMatchingTex(eqs[i - 1].copy(), eq),
                           run_time=next(times))
            if i in notes:
                scene.play(FadeIn(notes[i], shift=RIGHT * cap), run_time=next(times))
            scene.settle(f"step {i}")
            scene.wait(next(times))

    # --- fixtures ------------------------------------------------------------

    @classmethod
    def examples(cls):
        return [
            # Completing the square, with double-brace groups so shared terms
            # travel between lines.
            {"steps": [r"{{ x^2 + 6x }} + 5 = 0",
                       r"{{ x^2 + 6x }} + 9 = 4",
                       r"{{ (x + 3)^2 }} = 4",
                       r"x + 3 = \pm 2",
                       r"x = -1 \text{ or } x = -5"],
             "annotations": [{"step": 1, "text": "add 4 to both sides"},
                             {"step": 2, "text": "factor the perfect square"},
                             {"step": 3, "text": "take square roots"}]},
            # Rearranging a kinematics equation; a tall fraction on the last row.
            {"steps": [r"v^2 = u^2 + 2as",
                       r"2as = v^2 - u^2",
                       r"s = \frac{v^2 - u^2}{2a}"],
             "annotations": [{"step": 2, "text": "divide both sides by 2a"}]},
            # No annotations at all.
            {"steps": [r"\sin^2\theta + \cos^2\theta = 1",
                       r"\frac{\sin^2\theta}{\sin^2\theta} + "
                       r"\frac{\cos^2\theta}{\sin^2\theta} = \frac{1}{\sin^2\theta}",
                       r"1 + \cot^2\theta = \csc^2\theta"]},
        ]

    @classmethod
    def stress(cls):
        # Invalid LaTeX is pinned in tests/test_component_equation_derivation.py
        # rather than here: it refuses with kind "invalid_latex", which the
        # registry-wide stress test does not (yet) count as a clean refusal.
        long_step = r"\frac{a_1 x^2 + b_1 x + c_1}{d_1} + \frac{a_2 x^2 + b_2 x + c_2}{d_2} = 0"
        return [
            # 3x realistic volume: the schema maximum of steps, each long and
            # tall, every one annotated at length.
            {"steps": [long_step] * 12,
             "annotations": [{"step": i, "text": "combine the two fractions "
                              "over a common denominator and simplify"}
                             for i in range(12)]},
            # Unwrappable tokens: a 60-character identifier in the maths and a
            # space-free URL in the margin note.
            {"steps": [r"\mathrm{" + "identifier" * 6 + "} = 1",
                       r"x = 1"],
             "annotations": [{"step": 0, "text": "https://example.com/" + "a" * 60}]},
            # Minimal: a single one-character step, no annotations.
            {"steps": ["x"]},
            # Minimal with a note: one step, one short note.
            {"steps": ["x = 1"], "annotations": [{"step": 0, "text": "a"}]},
        ]


def _step_tex(i: int, source: str, theme: Theme) -> MathTex:
    """Compile one step through the theme's maths constructor.

    Manim reports a LaTeX failure as a bare ValueError from deep inside its
    compile step. Converting it here gives the repair loop a typed refusal it
    can dispatch on (regenerate the spec) instead of a crash, and the probe
    reports it as a finding of that kind.
    """
    try:
        eq = math(source, theme)
    except ValueError as exc:
        raise LayoutError(
            f"step[{i}] is not valid LaTeX: {source!r} ({exc})",
            kind="invalid_latex",
        ) from exc
    if eq.width == 0 and eq.height == 0:
        # Compiles, draws nothing (e.g. only spacing commands). Placing it would
        # leave a silent gap in the derivation.
        raise LayoutError(
            f"step[{i}] renders nothing: {source!r}", kind="invalid_latex"
        )
    return label(eq, f"step[{i}]")
