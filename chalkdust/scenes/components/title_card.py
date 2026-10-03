"""TitleCard: opening, closing, and section-break cards."""

from __future__ import annotations

from typing import Annotated

from manim import DOWN, FadeIn, VGroup
from pydantic import StringConstraints

from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    MIN_STEP_SECONDS,
    Component,
    ComponentParams,
    label,
    register,
    wrap,
)
from chalkdust.scenes.regions import fit_to_region
from chalkdust.scenes.theme import caption_text, title_text, body_text


# Blank text is refused at the schema rung. A blank title builds an empty card
# that passes every layout check, and a blank kicker or subtitle spends a
# reveal on nothing. Leave an optional part out instead of passing "".
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class TitleCardParams(ComponentParams):
    title: NonBlank
    subtitle: NonBlank | None = None
    # Small line above the title -- series name, chapter, "Part 2".
    kicker: NonBlank | None = None


@register
class TitleCard(Component):
    name = "TitleCard"
    Params = TitleCardParams

    def regions(self) -> set[Region]:
        # A title card owns the whole stage; nothing else shares the frame.
        return {Region.STAGE}

    def _weights(self) -> list[float]:
        """Budget weights: one per part, plus a hold. Shared by build() and
        min_seconds() so they cannot drift apart."""
        p: TitleCardParams = self.params
        n_parts = 1 + bool(p.kicker) + bool(p.subtitle)
        return [1] * n_parts + [3]

    def min_seconds(self) -> float:
        # Each part's reveal (weight 1) gets at least MIN_STEP_SECONDS.
        return MIN_STEP_SECONDS * sum(self._weights())

    def build(self, scene: ChalkdustScene) -> None:
        p: TitleCardParams = self.params
        theme = scene.theme

        parts = []
        if p.kicker:
            parts.append(caption_text(p.kicker.upper(), theme, theme.palette.accent))
        parts.append(title_text(wrap(p.title, 28), theme))
        if p.subtitle:
            parts.append(body_text(wrap(p.subtitle, 44), theme, theme.palette.muted))

        card = label(VGroup(*parts), "TitleCard")
        # buff is larger below the kicker than between title and subtitle,
        # so the hierarchy reads without needing a rule or divider.
        card.arrange(DOWN, buff=0.35)
        fit_to_region(card, Region.STAGE)

        # Stagger the reveal so the eye lands on the title, not everything at
        # once. Weights are relative; budget() converts them to real seconds.
        times = scene.budget(*self._weights())

        for part, t in zip(parts, times):
            scene.play(FadeIn(part, shift=DOWN * 0.2), run_time=t)
        scene.settle("title card revealed")
        scene.wait(times[-1])

    @classmethod
    def examples(cls):
        return [
            {"title": "Why Hash Maps Degrade"},
            {"kicker": "CS Fundamentals", "title": "Binary Search",
             "subtitle": "Halving the problem, every step"},
        ]

    @classmethod
    def stress(cls):
        return [
            # Titles far longer than any sane beat would carry.
            {"title": "An Extraordinarily Long Title That No Reasonable "
                      "Editor Would Ever Approve For A Card"},
            {"kicker": "A KICKER THAT IS ITSELF FAR TOO LONG TO SIT ABOVE "
                       "ANYTHING",
             "title": "Compounding The Problem With A Long Title As Well",
             "subtitle": "And a subtitle that keeps going well past the point "
                         "where anyone would still be reading it attentively"},
            {"title": "Supercalifragilisticexpialidocious" * 3},  # unwrappable
            # Unwrappable tokens in the kicker and subtitle too.
            {"kicker": "https://example.com/" + "a" * 40, "title": "Hashing",
             "subtitle": "x" * 90},
            # Minimal content: one character per part.
            {"title": "A"},
            {"kicker": "K", "title": "A", "subtitle": "b"},
        ]
