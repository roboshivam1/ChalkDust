"""TitleCard: opening, closing, and section-break cards."""

from __future__ import annotations

from manim import DOWN, FadeIn, VGroup

from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    Component,
    ComponentParams,
    label,
    register,
    wrap,
)
from chalkdust.scenes.regions import fit_to_region
from chalkdust.scenes.theme import caption_text, title_text, body_text


class TitleCardParams(ComponentParams):
    title: str
    subtitle: str | None = None
    # Small line above the title -- series name, chapter, "Part 2".
    kicker: str | None = None


@register
class TitleCard(Component):
    name = "TitleCard"
    Params = TitleCardParams

    def regions(self) -> set[Region]:
        # A title card owns the whole stage; nothing else shares the frame.
        return {Region.STAGE}

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
        weights = [1] * len(parts) + [3]  # one per part, plus a hold
        times = scene.budget(*weights)

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
        ]
