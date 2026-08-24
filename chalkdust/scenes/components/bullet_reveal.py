"""BulletReveal: a heading plus sequentially revealed points."""

from __future__ import annotations

from typing import Literal

from manim import DOWN, LEFT, ORIGIN, UP, Dot, FadeIn, VGroup
from pydantic import Field

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
from chalkdust.scenes.theme import body_cap_height, body_text, heading_text

WRAP_WIDTH = 46

# Vertical rhythm, expressed in cap heights so it scales with the theme.
LINE_HEIGHT = 1.7   # one line of text, anchor to anchor
PARA_GAP = 0.9      # additional space between bullets
DOT_GAP = 0.28      # horizontal space between dot and text (absolute units)


class BulletRevealParams(ComponentParams):
    heading: str | None = None
    # Capped at 6. A density limit that fires at schema validation -- cheaper
    # than the legibility check, and its error points at the real fix (split
    # the beat) rather than at a font size.
    items: list[str] = Field(min_length=1, max_length=6)
    reveal: Literal["sequential", "all"] = "sequential"


@register
class BulletReveal(Component):
    name = "BulletReveal"
    Params = BulletRevealParams

    def regions(self) -> set[Region]:
        r = {Region.STAGE}
        if self.params.heading:
            r.add(Region.TITLE_BAR)
        return r

    def build(self, scene: ChalkdustScene) -> None:
        p: BulletRevealParams = self.params
        theme = scene.theme
        cap = body_cap_height(theme)

        heading = None
        if p.heading:
            heading = label(heading_text(wrap(p.heading, 34), theme), "heading")
            fit_to_region(heading, Region.TITLE_BAR)

        rows, dots, line_counts = [], [], []
        for i, item in enumerate(p.items):
            wrapped = wrap(item, WRAP_WIDTH)
            line_counts.append(wrapped.count("\n") + 1)

            text = body_text(wrapped, theme)
            dot = Dot(radius=0.07, color=theme.palette.accent)
            dot.next_to(text, LEFT, buff=DOT_GAP)
            # Sit the dot on the optical centre of the FIRST line, measured
            # down from the top by half a cap height. Independent of whether
            # the row happens to contain descenders.
            dot.set_y(text.get_top()[1] - cap / 2)

            row = label(VGroup(dot, text), f"bullet[{i}]")
            # Common left edge, so the dots form a straight column.
            row.align_to(ORIGIN, LEFT)
            rows.append(row)
            dots.append(dot)

        # Stack manually at a constant pitch instead of arrange(), which would
        # space by bounding box and reintroduce the descender problem. The dot
        # is already at the row's anchor, so we align dot y positions.
        cursor = 0.0
        for row, dot, n_lines in zip(rows, dots, line_counts):
            row.shift(UP * (cursor - dot.get_y()))
            cursor -= cap * (LINE_HEIGHT * n_lines + PARA_GAP)

        bullets = label(VGroup(*rows), "bullets")
        fit_to_region(bullets, Region.STAGE)

        # Add the group up front so the scene holds one top-level mobject the
        # overlap check can reason about; reveal by animating opacity.
        if p.reveal == "sequential":
            for row in rows:
                row.set_opacity(0)
        scene.add(bullets)

        if heading is not None:
            scene.exclusive(heading, bullets)

        weights = ([1] if heading is not None else []) + [2] * len(rows) + [2]
        times = scene.budget(*weights)
        idx = 0

        if heading is not None:
            scene.play(FadeIn(heading), run_time=times[idx])
            idx += 1

        if p.reveal == "all":
            scene.play(FadeIn(bullets), run_time=sum(times[idx:-1]))
        else:
            for row in rows:
                scene.play(row.animate.set_opacity(1), run_time=times[idx])
                idx += 1

        scene.settle("bullets revealed")
        scene.wait(times[-1])

    @classmethod
    def examples(cls):
        return [
            {"items": ["One jump becomes a walk"]},
            {"heading": "Three causes",
             "items": ["A weak hash function",
                       "A load factor left too high",
                       "Adversarial keys chosen to collide"]},
            {"items": [f"Point number {i}" for i in range(6)], "reveal": "all"},
        ]

    @classmethod
    def stress(cls):
        return [
            # Max items, each far longer than a real bullet.
            {"heading": "A heading that runs considerably longer than it should",
             "items": ["This bullet carries a great deal more text than any "
                       "single point in a well-constructed beat should ever "
                       "hold, and it continues at length"] * 6},
            # One unwrappable token.
            {"items": ["antidisestablishmentarianism" * 4]},
            {"heading": "Short", "items": ["a"] * 6},  # minimal content
        ]
