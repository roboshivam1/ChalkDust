"""Callout: annotate something already on screen.

The thing being annotated is a carried artifact (SCENE_SPEC.md §6): an earlier
beat registered it, this beat lists it in `carry_in`, and `target_id` names it.
CarryIn has already rebuilt it, fitted it to STAGE and dimmed it before build()
runs, so a Callout never constructs its target -- it only makes room beside it,
brings it back to full strength, and points at it.

    b02: BulletReveal  registers="causes"
    b05: Callout       carry_in=["causes"],
                       params={"target_id": "causes", "part": 2,
                               "text": "The one attackers exploit", "side": "right"}
"""

from __future__ import annotations

import unicodedata
from typing import Annotated, Any, Literal

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    FadeIn,
    GrowArrow,
    Mobject,
    MoveToTarget,
)
from pydantic import AfterValidator, Field, StringConstraints

from chalkdust.continuity import DIM_DARKNESS, ArtifactRecipe, carried
from chalkdust.core.models import CarryInError, Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    MIN_STEP_SECONDS,
    Component,
    ComponentParams,
    label,
    register,
    wrap,
)
from chalkdust.scenes.regions import (
    DEFAULT_PADDING,
    LayoutError,
    Rect,
    bbox,
    fit_to_region,
    region_rect,
)
from chalkdust.scenes.theme import body_text

# Wrap widths in characters. Beside the target the label shares the stage's
# width with it, so lines stay short; above or below it gets the full width.
WRAP_BESIDE = 22
WRAP_ACROSS = 46

# The label's band never takes more than this share of the stage along the
# side axis. Past it, the target would have to shrink further than the label is
# worth -- the label is the annotation, the target is the subject.
MAX_BAND = 0.42

# Space between the target and its label, along the side axis. Wide enough
# that the arrow spanning it reads as an arrow rather than a tick.
GAP = 0.8
ARROW_BUFF = 0.1

# Steps a viewer must register: make room, point, read.
_STEPS = 3

_SIDES = {"left": LEFT, "right": RIGHT, "above": UP, "below": DOWN}

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Characters that draw nothing: controls, format characters (zero-width
# space, BOM, word joiner, direction marks) and separators. str.strip() keeps
# the format characters, so NonBlank alone lets a label of them through, and
# a label that draws nothing has nothing to place or point from.
_INVISIBLE = {"Cc", "Cf", "Zs", "Zl", "Zp"}


def _has_glyphs(s: str) -> str:
    if all(c.isspace() or unicodedata.category(c) in _INVISIBLE for c in s):
        raise ValueError("text has no visible characters (only spaces, "
                         "zero-width or direction marks); write the label")
    return s


Visible = Annotated[NonBlank, AfterValidator(_has_glyphs)]


class CalloutParams(ComponentParams):
    # A carry-in name, not a mobject: the beat must list it in `carry_in`.
    target_id: NonBlank
    text: Visible
    side: Literal["left", "right", "above", "below"] = "right"
    # Optional: point at one top-level part of the target (the n-th bullet,
    # the n-th array cell) instead of the whole artifact. Without it a callout
    # on a six-bullet list can only say something about the list as a whole,
    # which is rarely what the narration is doing. Indexes the producer's
    # artifact-builder structure, which builders document as stable (§6).
    part: int | None = Field(default=None, ge=0)


@register
class Callout(Component):
    name = "Callout"
    Params = CalloutParams

    def regions(self) -> set[Region]:
        # Target and label share the stage between them.
        return {Region.STAGE}

    def min_seconds(self) -> float:
        return MIN_STEP_SECONDS * _STEPS

    def latex_strings(self) -> list[str]:
        return []  # plain text only; nothing to compile

    def carried_targets(self) -> list[str]:
        return [self.params.target_id]

    def build(self, scene: ChalkdustScene) -> None:
        p: CalloutParams = self.params
        theme = scene.theme
        target = carried(scene, p.target_id)  # CarryInError if not carried in

        # CarryIn centres every carried artifact in STAGE, and the label and
        # arrow are placed against the target alone: anything else carried in
        # would sit under them. A Callout annotates one artifact; refuse the
        # rest by name rather than draw over it.
        others = sorted(set(getattr(scene, "_chalk_carried", {})) - {p.target_id})
        if others:
            raise LayoutError(
                f"Callout annotates {p.target_id!r} alone, but this beat also "
                f"carries in {others}, which the label and arrow would cover; "
                f"carry in only {p.target_id!r}",
                kind="overlap",
            )

        parts = target.submobjects
        if p.part is not None and p.part >= len(parts):
            raise CarryInError(
                f"Callout points at part {p.part} of {p.target_id!r}, which has "
                f"{len(parts)} part(s) (0..{len(parts) - 1})",
                beat_id=None, name=p.target_id,
            )

        direction = _SIDES[p.side]
        beside = p.side in ("left", "right")
        stage = region_rect(Region.STAGE)

        text = label(
            body_text(wrap(p.text, WRAP_BESIDE if beside else WRAP_ACROSS), theme),
            "callout",
        )
        if not text.family_members_with_points():
            # The schema refuses invisible-only text; this is the font's
            # word (a character it draws as nothing) and it ends the same way.
            raise LayoutError(
                f"Callout label {p.text!r} draws no glyphs in the theme's font; "
                f"write the label in visible characters",
                kind="illegible",
            )

        # The plan: the target's full-strength state (CarryIn saved it before
        # dimming), moved and if need be shrunk to leave a band for the label.
        # MoveToTarget animates the carried target into it, so the viewer sees
        # the picture they already know make room rather than jump.
        plan = target.saved_state.copy()
        if p.part is not None:
            for i, sub in enumerate(plan.submobjects):
                if i != p.part:
                    sub.fade(DIM_DARKNESS)  # the rest stays context

        label_rect, target_rect = _split(stage, p.side, _extent(text, beside))
        fit_to_region(text, label_rect)  # raises overflow if it cannot stay legible
        fit_to_region(plan, target_rect)

        pointed = plan[p.part] if p.part is not None else plan
        _place_beside(text, plan, pointed, direction, stage.inset(DEFAULT_PADDING))

        # Centre target + label together along the side axis, so the pair sits
        # in the middle of the stage instead of hugging opposite edges.
        pair = _bbox_union(plan, text)
        if beside:
            offset = np.array([stage.x - pair.x, 0.0, 0.0])
        else:
            offset = np.array([0.0, stage.y - pair.y, 0.0])
        plan.shift(offset)
        text.shift(offset)

        end = pointed.get_edge_center(direction)
        # Leave the label from the point on its facing edge nearest the part,
        # so the arrow runs straight across the gap even when the label had
        # to be clamped off-level to stay inside the stage.
        start = text.get_edge_center(-direction)
        span = bbox(text)
        if beside:
            start[1] = float(np.clip(end[1], span.bottom, span.top))
        else:
            start[0] = float(np.clip(end[0], span.left, span.right))
        if p.part is not None:
            # A straight arrow to an inner part can run through its siblings
            # (side="below" on the first bullet of a list). That is a broken
            # frame, not a stylistic choice: refuse it and name the fix.
            for i, sub in enumerate(plan.submobjects):
                if i != p.part and _segment_hits(start, end, bbox(sub)):
                    raise LayoutError(
                        f"Callout arrow to part {p.part} of {p.target_id!r} "
                        f"would cross part {i}; choose a side facing part "
                        f"{p.part} directly",
                        kind="overlap",
                    )
        arrow = label(
            Arrow(start, end, buff=ARROW_BUFF, color=theme.palette.accent),
            "callout arrow",
        )

        target.target = plan
        scene.exclusive(target, text)

        t_room, t_point, t_read = scene.budget(2, 2, 3)
        scene.play(MoveToTarget(target), run_time=t_room)
        # The move hands the target the plan's points, not its font-size
        # tags; the plan's are the honest ones after fit_to_region shrank it.
        for mob, planned in zip(target.get_family(), plan.get_family()):
            if hasattr(planned, "_chalk_font_size"):
                mob._chalk_font_size = planned._chalk_font_size  # type: ignore[attr-defined]
        scene.play(GrowArrow(arrow), FadeIn(text), run_time=t_point)
        scene.settle("callout shown")
        scene.wait(t_read)

    # --- fixtures -----------------------------------------------------------
    # Every case acts on a carried list. fixture_carry_in() hands the registry
    # walks (test_layout, test_snapshots, the semantic TestLibrary) the recipe
    # for the case's target, so each case is built exactly as the pipeline
    # builds a carry-in beat: BulletReveal's artifact on screen, dimmed, first.

    @classmethod
    def fixture_carry_in(cls, params: dict[str, Any]) -> list[ArtifactRecipe]:
        items = _FIXTURE_LISTS.get(params.get("target_id"))
        if items is None:
            return []
        return [ArtifactRecipe(name=params["target_id"], producer="BulletReveal",
                               params={"items": items})]

    @classmethod
    def examples(cls):
        return [
            {"target_id": "causes", "part": 2, "side": "right",
             "text": "The one an attacker controls"},
            {"target_id": "steps", "side": "below",
             "text": "Every lookup pays for all four steps"},
            {"target_id": "steps", "part": 0, "side": "left",
             "text": "Constant time"},
        ]

    @classmethod
    def stress(cls):
        return [
            # (a) 3x realistic volume, beside the last of six full-width rows.
            {"target_id": "crowded", "part": 5, "side": "right",
             "text": "This annotation runs to roughly three times the length "
                     "any callout should, because the model decided to explain "
                     "the whole idea in the label instead of in the narration "
                     "where it belongs"},
            # (a) far past capacity: must refuse as overflow, not shrink.
            {"target_id": "crowded", "side": "above",
             "text": "An annotation that will not fit. " * 30},
            # (b) unwrappable tokens: a URL and a 60-char identifier.
            {"target_id": "single", "side": "below",
             "text": "https://example.com/" + "a" * 60},
            {"target_id": "single", "side": "left", "text": "x" * 60},
            # (c) minimal: one-character label on a one-character target.
            {"target_id": "single", "part": 0, "text": "a"},
        ]


# The lists the fixtures annotate, by carry-in name (BulletReveal items).
_FIXTURE_LISTS: dict[str, list[str]] = {
    "causes": ["A weak hash function", "A load factor left too high",
               "Adversarial keys chosen to collide"],
    "steps": ["Hash the key", "Find the bucket", "Walk the chain", "Compare keys"],
    # As crowded as a legible target gets: six full-width bullets.
    "crowded": ["Each of these bullets runs a full line"] * 6,
    "single": ["a"],
}


# --- private helpers --------------------------------------------------------


def _extent(mob: Mobject, beside: bool) -> float:
    """The label's natural size along the side axis."""
    return mob.width if beside else mob.height


def _split(stage: Rect, side: str, label_extent: float) -> tuple[Rect, Rect]:
    """Cut the stage into (label band, target area) along the side axis.

    The band is as wide as the label needs (plus fit_to_region's padding) up
    to MAX_BAND of the stage; the target gets the rest, less GAP.
    """
    beside = side in ("left", "right")
    total = stage.width if beside else stage.height
    band = min(label_extent + 2 * DEFAULT_PADDING, MAX_BAND * total)
    rest = total - band - GAP
    # Sign of the label band's offset from the stage centre.
    sign = 1.0 if side in ("right", "above") else -1.0
    band_c = sign * (total - band) / 2
    rest_c = -sign * (total - rest) / 2
    if beside:
        return (Rect(stage.x + band_c, stage.y, band, stage.height),
                Rect(stage.x + rest_c, stage.y, rest, stage.height))
    return (Rect(stage.x, stage.y + band_c, stage.width, band),
            Rect(stage.x, stage.y + rest_c, stage.width, rest))


def _place_beside(text: Mobject, plan: Mobject, pointed: Mobject,
                  direction: np.ndarray, inner: Rect) -> None:
    """Put the label GAP away from the whole target on `direction`, level with
    the pointed-at part across the side axis, kept inside the stage."""
    text.next_to(plan, direction, buff=GAP)
    beside = direction[1] == 0
    if beside:
        text.set_y(pointed.get_y())
        lo, hi = inner.bottom - text.get_bottom()[1], inner.top - text.get_top()[1]
        text.shift(UP * float(np.clip(0.0, lo, hi)))
    else:
        text.set_x(pointed.get_x())
        lo, hi = inner.left - text.get_left()[0], inner.right - text.get_right()[0]
        text.shift(RIGHT * float(np.clip(0.0, lo, hi)))


def _bbox_union(a: Mobject, b: Mobject) -> Rect:
    """Bounding box of two mobjects together."""
    ra, rb = bbox(a), bbox(b)
    left, right = min(ra.left, rb.left), max(ra.right, rb.right)
    bottom, top = min(ra.bottom, rb.bottom), max(ra.top, rb.top)
    return Rect((left + right) / 2, (bottom + top) / 2, right - left, top - bottom)


def _segment_hits(start: np.ndarray, end: np.ndarray, rect: Rect,
                  samples: int = 64) -> bool:
    """Whether the segment start->end passes through the interior of `rect`.

    Sampled rather than solved: the arrow is drawn once per beat, so a closed
    form buys nothing here and is easier to get subtly wrong at the edges.
    """
    inner = rect.inset(0.02)
    for t in np.linspace(0.0, 1.0, samples):
        x, y = (start + (end - start) * t)[:2]
        if inner.left < x < inner.right and inner.bottom < y < inner.top:
            return True
    return False

