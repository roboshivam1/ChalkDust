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
from chalkdust.scenes.theme import Theme, body_text

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

# Where the label goes when the requested side does not work, in order: the
# opposite side first (it faces the same part from the other way, so an end
# part blocked from one side is clear from the other), then across. Right
# before left and below before above, in reading order.
_FALLBACK = {
    "right": ("left", "below", "above"),
    "left": ("right", "below", "above"),
    "below": ("above", "right", "left"),
    "above": ("below", "right", "left"),
}

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
        # sits under them, and no side of the target is clear of it. The stage
        # holds one annotated artifact; more is more content than it can show
        # legibly, so it refuses as overflow (a clean refusal the repair loop
        # answers by rewriting the beat) and names what to drop.
        others = sorted(set(getattr(scene, "_chalk_carried", {})) - {p.target_id})
        if others:
            raise LayoutError(
                f"Callout annotates {p.target_id!r} alone, but this beat also "
                f"carries in {others}; the stage has no room beside "
                f"{p.target_id!r} that is clear of them, so the label and arrow "
                f"would cover them. Carry in only {p.target_id!r}.",
                kind="overflow",
            )

        parts = target.submobjects
        if p.part is not None and p.part >= len(parts):
            raise CarryInError(
                f"Callout points at part {p.part} of {p.target_id!r}, which has "
                f"{len(parts)} part(s) (0..{len(parts) - 1})",
                beat_id=None, name=p.target_id,
            )

        # `side` is where the label goes when that side works. When it does
        # not -- the label or the made-room target would fall below the
        # legibility floor there, or a straight arrow from there to an inner
        # part would cross its siblings (side="below" on the first bullet) --
        # the next side in _FALLBACK that works is used instead: the same
        # mechanical repair as wrapping or shrinking (SCENE_SPEC.md §9 step
        # 1), and a clear frame rather than a refusal over a placement hint
        # (§11 rule 1). Only when no side works does the beat refuse.
        stage = region_rect(Region.STAGE)
        failures: list[str] = []
        too_big: LayoutError | None = None
        for side in (p.side, *_FALLBACK[p.side]):
            try:
                laid = _lay_out(target, p, side, theme, stage)
            except LayoutError as exc:
                if exc.kind != "overflow":
                    raise  # the text itself (unrenderable_text): no side helps
                too_big = too_big or exc
                failures.append(f"{side}: too little room to stay legible")
                continue
            if isinstance(laid, str):
                failures.append(f"{side}: {laid}")
                continue
            break
        else:
            if too_big is not None and all(f.endswith("legible") for f in failures):
                # Too much label for any side: the requested side's own
                # message says by how much, and the fix is the content.
                raise too_big
            raise LayoutError(
                f"Callout on {p.target_id!r} has no side that works ("
                + "; ".join(failures)
                + "). Shorten the label, or point at a part on the target's edge.",
                kind="overflow",
            )
        plan, text, start, end = laid

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
            # A side that cannot work: below six rows, an arrow up to the
            # FIRST crosses the other five. Must move to a clear side.
            {"target_id": "crowded", "part": 0, "side": "below",
             "text": "Only the first of these six rows"},
            # (e) text the theme's font cannot draw. Each must refuse as
            # unrenderable_text (theme.check_renderable), never render as a
            # bare arrow to an empty or tofu-filled label: right-to-left
            # letters and emoji draw nothing, CJK draws missing-glyph boxes,
            # and a blank braille cell is a visible-category character the
            # font has no glyph for. (Format-character-only and private-use
            # text never get this far: the schema and rung 1 refuse them.)
            {"target_id": "steps", "part": 1,
             "text": "\u0647\u0630\u0627 \u0647\u0648 \u0627\u0644\u0645\u0641\u062a\u0627\u062d"},
            {"target_id": "steps", "text": "\U0001F600\U0001F525 hot path"},
            {"target_id": "steps", "side": "below", "text": "\u54c8\u5e0c\u8868"},
            {"target_id": "single", "text": "\u2800\u2800"},
            # (d) Callout takes no maths, so there is no LaTeX to be invalid:
            # LaTeX-looking text is plain text and must be drawn literally.
            {"target_id": "steps", "part": 3, "side": "left",
             "text": r"\frac{1}{ $x^2$ }} {50%} #_^ \\"},
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


def _lay_out(target: Mobject, p: CalloutParams, side: str, theme: Theme,
             stage: Rect) -> tuple[Mobject, Mobject, np.ndarray, np.ndarray] | str:
    """Plan the callout with its label on `side`.

    Returns (plan, label, arrow start, arrow end). `plan` is the target's
    full-strength state (CarryIn saved it before dimming), moved and if need
    be shrunk to leave a band for the label, with every part but the pointed
    one left dim; MoveToTarget animates the carried target into it, so the
    viewer sees the picture they already know make room rather than jump.

    Returns why not, as a string, when a straight arrow from this side would
    cross a sibling part. Raises LayoutError("overflow") when the label or
    the target cannot stay legible in its share of the stage.
    """
    direction = _SIDES[side]
    beside = side in ("left", "right")

    text = label(
        body_text(wrap(p.text, WRAP_BESIDE if beside else WRAP_ACROSS), theme,
                  what="Callout label"),
        "callout",
    )
    if not text.family_members_with_points():
        # The theme refuses a Text with no points first, as
        # "unrenderable_text"; this keeps placement from crashing on one all
        # the same (an empty label has no centre to align on).
        raise LayoutError(
            f"Callout label {p.text!r} draws no glyphs in the theme's font; "
            f"write the label in visible characters",
            kind="illegible",
        )

    plan = target.saved_state.copy()
    if p.part is not None:
        for i, sub in enumerate(plan.submobjects):
            if i != p.part:
                sub.fade(DIM_DARKNESS)  # the rest stays context

    label_rect, target_rect = _split(stage, side, _extent(text, beside))
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
        # (side="below" on the first bullet of a list): a broken frame, so
        # this side does not work.
        for i, sub in enumerate(plan.submobjects):
            if i != p.part and _segment_hits(start, end, bbox(sub)):
                return f"the arrow to part {p.part} would cross part {i}"
    return plan, text, start, end


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

