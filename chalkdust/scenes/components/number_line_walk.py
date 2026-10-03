"""NumberLineWalk: discrete stepping and intervals on a number line.

A walk is a list of steps over one number line:

    {"at": -2, "label": "start"}            put the walker on a value
    {"to": 3}                               jump there along an arc ("+5")
    {"interval": [1, null], "closed": [false, false], "label": "x > 1"}

Jumps arc above the line with their label at the apex; marks and intervals
label below the tick numbers. Every label is placed by measuring, not by
convention: it is nudged away from arcs and other labels until clear, and if
that cannot be done inside the stage the beat refuses (SCENE_SPEC.md §11.1).

All text is Pango (theme constructors), never LaTeX. Number-line labels are
short ("+3", "x < 5", "start") and Unicode covers them, so this component has
no LaTeX compile cost and no invalid-LaTeX failure mode.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    ArcBetweenPoints,
    Circle,
    Create,
    Dot,
    FadeIn,
    Line,
    MoveAlongPath,
    NumberLine,
    VGroup,
    VMobject,
    config,
)
from pydantic import Field, model_validator

from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
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
from chalkdust.scenes.theme import body_text, caption_text

# Density limits that fire at schema validation, where the error points at
# the real fix (split the beat) rather than at a font size.
MAX_STEPS = 8
MAX_TICKS = 40
# Auto tick spacing. A walk over an integer range of up to UNIT_TICK_SPAN
# gets unit ticks -- discrete stepping is counted in ones, [-10, 10] included.
# Anything else gets a 1/2/5 step aiming at about TARGET_TICKS intervals.
UNIT_TICK_SPAN = (4, 20)
TARGET_TICKS = 10

LABEL_WRAP = 18

# Seconds each phase needs to stay legible. Doubles as the relative weights
# the beat budget is split by, so at exactly min_seconds() every phase gets
# its minimum and longer narration stretches all of them evenly (D-002).
SECONDS = {"intro": 0.75, "mark": 0.5, "jump": 1.0, "interval": 1.0, "hold": 1.0}

# Geometry, in Manim units, relative to the line.
EDGE_ROOM = 0.6        # stage width kept free at each end for edge labels
TICK_LABEL_BUFF = 0.2
LABEL_BUFF = 0.15      # label to its anchor (arc apex, tick band)
LABEL_PAD = 0.06       # clearance a placed label keeps from everything else
NUDGE = 0.08           # placement search step
MAX_NUDGES = 60        # ~5 units of search; past that the stage is full
ARC_RISE_RATIO = 0.3   # arc height per unit of chord
MIN_ARC_RISE = 0.35
MAX_ARC_RISE = 1.1
BACK_RISE_FACTOR = 0.65  # a jump back over the same span must not coincide
MIN_JUMP_WIDTH = 0.3   # below this the arrow tip is wider than the arc
ARC_SAMPLES = 120
WALKER_RADIUS = 0.1
ENDPOINT_RADIUS = 0.09
INTERVAL_STROKE = 8


class MarkStep(ComponentParams):
    """Place the walker on a value. A walk must begin with one."""

    at: float
    label: str | None = None


class JumpStep(ComponentParams):
    """Jump from the walker's position to `to`. The label defaults to the
    signed distance ("+5", "−2"), which is what a stepping beat narrates."""

    to: float
    label: str | None = None


class IntervalStep(ComponentParams):
    """Highlight a span. A null end is unbounded (drawn to the line's end with
    an arrow); `closed` defaults to closed at every finite end."""

    interval: tuple[float | None, float | None]
    closed: tuple[bool, bool] | None = None
    label: str | None = None

    def is_closed(self) -> tuple[bool, bool]:
        if self.closed is not None:
            return self.closed
        lo, hi = self.interval
        return (lo is not None, hi is not None)


Step = MarkStep | JumpStep | IntervalStep


class NumberLineWalkParams(ComponentParams):
    # [min, max] or [min, max, tick]. Ticks sit at multiples of the tick step;
    # omitted, a 1/2/5 step is chosen. Never a coordinate: these are values
    # on the line, the component decides where the line goes.
    range: tuple[float, float] | tuple[float, float, float]
    steps: list[Step] = Field(min_length=1, max_length=MAX_STEPS)

    @model_validator(mode="after")
    def _coherent(self) -> NumberLineWalkParams:
        lo, hi = self.range[0], self.range[1]
        if not lo < hi:
            raise ValueError(f"range min {lo} must be below max {hi}")
        ticks = tick_values(self.range)
        if len(ticks) < 2:
            raise ValueError("the tick step leaves fewer than two ticks on the line")
        if len(ticks) > MAX_TICKS + 1:
            raise ValueError(
                f"{len(ticks)} ticks (max {MAX_TICKS + 1}); use a larger tick step"
            )

        def inside(v: float, what: str) -> None:
            if not lo <= v <= hi:
                raise ValueError(f"{what} {v} is outside the range [{lo}, {hi}]")

        pos: float | None = None
        for i, step in enumerate(self.steps):
            if isinstance(step, MarkStep):
                inside(step.at, f"steps[{i}].at")
                pos = step.at
            elif isinstance(step, JumpStep):
                if pos is None:
                    raise ValueError(
                        f"steps[{i}] jumps before the walker is placed; "
                        "begin the walk with an {'at': ...} step"
                    )
                inside(step.to, f"steps[{i}].to")
                if step.to == pos:
                    raise ValueError(f"steps[{i}] jumps to where the walker already is")
                pos = step.to
            else:
                a, b = step.interval
                if a is None and b is None:
                    raise ValueError(f"steps[{i}] interval needs at least one finite end")
                for v in (a, b):
                    if v is not None:
                        inside(v, f"steps[{i}].interval end")
                if a is not None and b is not None and not a < b:
                    raise ValueError(f"steps[{i}] interval [{a}, {b}] is empty")
                closed = step.is_closed()
                if (a is None and closed[0]) or (b is None and closed[1]):
                    raise ValueError(f"steps[{i}] closes an unbounded end")
        return self


def tick_values(rng: tuple[float, ...]) -> list[float]:
    """Tick positions: multiples of the tick step inside [min, max]."""
    lo, hi = rng[0], rng[1]
    step = rng[2] if len(rng) == 3 else _auto_step(lo, hi)
    if step <= 0:
        return []
    first = math.ceil(lo / step - 1e-9)
    last = math.floor(hi / step + 1e-9)
    return [k * step for k in range(first, last + 1)]


def _auto_step(lo: float, hi: float) -> float:
    span = hi - lo
    if (float(lo).is_integer() and float(hi).is_integer()
            and UNIT_TICK_SPAN[0] <= span <= UNIT_TICK_SPAN[1]):
        return 1.0
    return _nice_step(span / TARGET_TICKS)


def _nice_step(raw: float) -> float:
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * mag >= raw - 1e-12:
            return m * mag
    return 10 * mag


def _fmt(v: float) -> str:
    """A tick number as a reader writes it: no '.0', a true minus sign."""
    s = str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.6g}"
    return s.replace("-", "−")


def _signed(delta: float) -> str:
    return ("+" if delta > 0 else "−") + _fmt(abs(delta))


@register
class NumberLineWalk(Component):
    name = "NumberLineWalk"
    Params = NumberLineWalkParams

    def regions(self) -> set[Region]:
        # The line spans the full stage width; nothing can share it.
        return {Region.STAGE}

    # --- semantic-rung hooks (SCENE_SPEC.md §8 rung 2) ----------------------

    def min_seconds(self) -> float:
        return sum(self._weights())

    def latex_strings(self) -> list[str]:
        # Every label is Pango text; see the module docstring.
        return []

    def _weights(self) -> list[float]:
        kinds = [_kind(s) for s in self.params.steps]
        return [SECONDS["intro"]] + [SECONDS[k] for k in kinds] + [SECONDS["hold"]]

    # --- build --------------------------------------------------------------

    def build(self, scene: ChalkdustScene) -> None:
        p: NumberLineWalkParams = self.params
        theme = scene.theme
        lo, hi = p.range[0], p.range[1]

        inner = region_rect(Region.STAGE).inset(DEFAULT_PADDING)
        line = NumberLine(
            x_range=[lo, hi, hi - lo],
            length=inner.width - 2 * EDGE_ROOM,
            include_ticks=False,
            color=theme.palette.muted,
        )
        ticks_values = tick_values(p.range)
        tick_marks = VGroup(*(line.get_tick(v) for v in ticks_values))
        tick_marks.set_color(theme.palette.muted)
        axis = label(VGroup(line, tick_marks), "number line")
        # Labels may hang past the line ends by EDGE_ROOM, never further.
        bounds = (line.get_left()[0] - EDGE_ROOM, line.get_right()[0] + EDGE_ROOM)

        tick_labels = label(
            self._tick_labels(line, ticks_values, theme), "tick labels")
        below_top = tick_labels.get_bottom()[1] - LABEL_BUFF
        placed = [bbox(tick_labels)]

        # Pass 1: every arc, so labels placed in pass 2 can avoid arcs that
        # appear later in the walk as well as earlier ones.
        arcs, paths = {}, {}
        pos: float | None = None
        for i, step in enumerate(p.steps):
            if isinstance(step, MarkStep):
                pos = step.at
            elif isinstance(step, JumpStep):
                arcs[i], paths[i] = self._arc(line, pos, step.to, theme)
                pos = step.to
        obstacles = (np.vstack([_samples(a) for a in arcs.values()])
                     if arcs else np.empty((0, 3)))

        # Pass 2: visuals and labels, placed in walk order.
        visuals: dict[int, list[VMobject]] = {}
        labels: dict[int, VMobject] = {}
        walker = None
        palette_cycle = [theme.palette.accent_alt, theme.palette.success]
        n_intervals = 0
        pos = None
        for i, step in enumerate(p.steps):
            text: str | None = step.label
            if isinstance(step, MarkStep):
                if walker is None:
                    walker = Dot(line.n2p(step.at), radius=WALKER_RADIUS,
                                 color=theme.palette.accent)
                anchor, above = line.n2p(step.at), False
                colour = theme.palette.fg
                pos = step.at
            elif isinstance(step, JumpStep):
                text = text or _signed(step.to - pos)
                anchor = (line.n2p(pos) + line.n2p(step.to)) / 2
                anchor = anchor + UP * (arcs[i].get_top()[1] - anchor[1])
                above = True
                colour = theme.palette.accent
                pos = step.to
            else:
                colour = palette_cycle[n_intervals % len(palette_cycle)]
                n_intervals += 1
                visuals[i] = self._interval(line, step, colour, theme)
                anchor = visuals[i][0].get_center()
                above = False

            if not text:
                continue
            mob = label(body_text(wrap(text, LABEL_WRAP), theme, colour),
                        f"steps[{i}] label")
            if above:
                mob.next_to(anchor, UP, buff=LABEL_BUFF)
            else:
                mob.set_x(anchor[0])
                mob.align_to(np.array([0.0, below_top, 0.0]), UP)
            _clamp_x(mob, bounds)
            _place(mob, placed, obstacles, UP if above else DOWN)
            placed.append(bbox(mob))
            labels[i] = mob

        everything = VGroup(axis, tick_labels, *arcs.values(), *paths.values(),
                            *(m for ms in visuals.values() for m in ms),
                            *labels.values())
        if walker is not None:
            everything.add(walker)
            walker.set_z_index(1)  # rides on top of arcs and intervals
        fit_to_region(everything, Region.STAGE)

        scene.exclusive(tick_labels, *labels.values())
        times = _frame_times(scene, self._weights())

        scene.play(Create(axis), FadeIn(tick_labels), run_time=times[0])
        walker_shown = False
        for i, (step, t) in enumerate(zip(p.steps, times[1:-1])):
            anims = []
            if isinstance(step, MarkStep):
                if walker_shown:
                    anims.append(walker.animate.move_to(line.n2p(step.at)))
                else:
                    anims.append(FadeIn(walker, scale=0.5))
                    walker_shown = True
            elif isinstance(step, JumpStep):
                anims += [Create(arcs[i]), MoveAlongPath(walker, paths[i])]
            else:
                anims += [Create(visuals[i][0]), *(FadeIn(m) for m in visuals[i][1:])]
            if i in labels:
                anims.append(FadeIn(labels[i]))
            scene.play(*anims, run_time=t)

        scene.settle("walk complete")
        scene.wait(times[-1])

    # --- pieces -------------------------------------------------------------

    def _tick_labels(self, line: NumberLine, values: list[float], theme) -> VGroup:
        """Numbers under the ticks, thinned to every 1st/2nd/5th/... tick so
        neighbours never touch. The stride is anchored on zero when zero is
        on the line, so 0 is always labelled."""
        texts = [caption_text(_fmt(v), theme) for v in values]
        spacing = line.get_unit_size() * (values[1] - values[0])
        widest = max(t.width for t in texts)
        stride = next(s for s in (1, 2, 4, 5, 10, 20, 40, 50)
                      if widest + 2 * LABEL_PAD <= s * spacing)
        base = min(range(len(values)), key=lambda k: abs(values[k]))
        shown = VGroup()
        for k, (v, t) in enumerate(zip(values, texts)):
            if (k - base) % stride == 0:
                t.next_to(line.n2p(v), DOWN, buff=TICK_LABEL_BUFF)
                shown.add(t)
        return shown

    def _arc(self, line: NumberLine, start: float, end: float, theme):
        """An arrowed arc above the line from start to end, plus the bare arc
        the walker travels along (the drawn one is shortened by its tip)."""
        a, b = line.n2p(start), line.n2p(end)
        chord = float(np.linalg.norm(b - a))
        if chord < MIN_JUMP_WIDTH:
            raise LayoutError(
                f"jump {_fmt(start)} -> {_fmt(end)} spans only {chord:.2f} units "
                f"(floor {MIN_JUMP_WIDTH}); narrow the range or use bigger steps.",
                kind="illegible",
            )
        rise = min(MAX_ARC_RISE, max(MIN_ARC_RISE, ARC_RISE_RATIO * chord))
        backward = end < start
        if backward:
            rise *= BACK_RISE_FACTOR
        # Chord and sagitta fix the arc's angle; the sign keeps it above the
        # line whichever way the walker travels.
        angle = 4 * math.atan(2 * rise / chord) * (1 if backward else -1)
        arc = ArcBetweenPoints(a, b, angle=angle, color=theme.palette.accent)
        arc.add_tip(tip_length=0.18, tip_width=0.18)
        path = ArcBetweenPoints(a, b, angle=angle)
        return label(arc, f"jump {_fmt(start)}->{_fmt(end)}"), path

    def _interval(self, line: NumberLine, step: IntervalStep, colour: str,
                  theme) -> list[VMobject]:
        """Thick bar on the line; filled dot at a closed end, hollow at an
        open one, arrow at an unbounded one."""
        lo, hi = line.x_range[0], line.x_range[1]
        a, b = step.interval
        bar = Line(line.n2p(lo if a is None else a), line.n2p(hi if b is None else b),
                   color=colour, stroke_width=INTERVAL_STROKE)
        if a is None:
            bar.add_tip(tip_length=0.2, tip_width=0.2, at_start=True)
        if b is None:
            bar.add_tip(tip_length=0.2, tip_width=0.2)
        out: list[VMobject] = [label(bar, "interval")]
        for v, closed in zip((a, b), step.is_closed()):
            if v is None:
                continue
            if closed:
                end = Dot(line.n2p(v), radius=ENDPOINT_RADIUS, color=colour)
            else:
                end = Circle(radius=ENDPOINT_RADIUS, color=colour, stroke_width=3)
                end.set_fill(theme.palette.bg, opacity=1).move_to(line.n2p(v))
            end.set_z_index(1)
            out.append(end)
        return out

    # --- fixtures -----------------------------------------------------------

    @classmethod
    def examples(cls):
        return [
            # 2 + 5 - 3, the classic stepping picture.
            {"range": [-2, 8],
             "steps": [{"at": 2, "label": "start"}, {"to": 7}, {"to": 4}]},
            # Counting by threes from zero.
            {"range": [0, 12, 1],
             "steps": [{"at": 0}, {"to": 3}, {"to": 6}, {"to": 9},
                       {"to": 12, "label": "4 jumps of 3"}]},
            # An inequality and a bounded interval.
            {"range": [-5, 5],
             "steps": [{"interval": [1, None], "closed": [False, False],
                        "label": "x > 1"},
                       {"interval": [-4, -1], "label": "−4 ≤ x ≤ −1"}]},
        ]

    @classmethod
    def stress(cls):
        long_label = ("a label carrying far more words than any jump on a "
                      "number line should ever need to say out loud")
        return [
            # (a) every step slot used, every label ~3x a realistic one.
            {"range": [-10, 10],
             "steps": [{"at": -9, "label": long_label},
                       {"to": -4, "label": long_label},
                       {"to": 1, "label": long_label},
                       {"to": -2, "label": long_label},
                       {"to": 6, "label": long_label},
                       {"to": 9, "label": long_label},
                       {"interval": [-6, 3], "label": long_label},
                       {"interval": [None, -7], "closed": [False, True],
                        "label": long_label}]},
            # (a) every step slot used with realistic labels, back and forth
            # over shared spans so arcs nest and labels must stack.
            {"range": [-10, 10],
             "steps": [{"at": 0, "label": "start"}, {"to": 7}, {"to": -3},
                       {"to": 4}, {"to": -8}, {"to": 9},
                       {"interval": [-3, 4], "label": "visited twice"},
                       {"to": 2, "label": "end"}]},
            # (a) a wide range, auto ticks thinned, full-width jumps.
            {"range": [-1000, 1000],
             "steps": [{"at": -1000}, {"to": 1000}, {"to": -1000},
                       {"to": 0, "label": "back to zero"}]},
            # (b) an unwrappable token as a label.
            {"range": [0, 10],
             "steps": [{"at": 1},
                       {"to": 9, "label": "https://example.com/" + "x" * 40}]},
            # The walker re-placed mid-walk: a second mark slides it along
            # the line instead of fading in a second walker.
            {"range": [-5, 5],
             "steps": [{"at": 2}, {"to": 5}, {"at": -3, "label": "reset"},
                       {"to": 1}]},
            # (c) minimal: one mark, no label, the smallest sensible range.
            {"range": [0, 1], "steps": [{"at": 0}]},
            # A jump too small to draw at this range: must refuse, not smear.
            {"range": [0, 1000], "steps": [{"at": 500}, {"to": 501}]},
        ]


def _kind(step: Step) -> Literal["mark", "jump", "interval"]:
    if isinstance(step, MarkStep):
        return "mark"
    if isinstance(step, JumpStep):
        return "jump"
    return "interval"


def _samples(arc: VMobject) -> np.ndarray:
    """Points along an arc (and its tip) for collision tests. Bezier control
    points alone sit off the curve, so they would report false hits."""
    pts = [arc.point_from_proportion(t) for t in np.linspace(0, 1, ARC_SAMPLES)]
    for tip in arc.get_tips():
        pts.extend(tip.get_vertices())
    return np.array(pts)


def _clamp_x(mob: VMobject, bounds: tuple[float, float]) -> None:
    """Pull a label back inside the line's horizontal extent. A label wider
    than that is left centred; fit_to_region then refuses it as illegible."""
    left, right = bounds
    if mob.width > right - left:
        return
    if mob.get_left()[0] < left:
        mob.shift(RIGHT * (left - mob.get_left()[0]))
    elif mob.get_right()[0] > right:
        mob.shift(LEFT * (mob.get_right()[0] - right))


def _place(mob: VMobject, placed: list[Rect], obstacles: np.ndarray,
           direction: np.ndarray) -> None:
    """Nudge `mob` along `direction` until it clears every placed label and
    every arc. Measured, so a crowded walk degrades to a taller stack -- and
    then to a refusal from fit_to_region -- never to overlapping text."""
    for _ in range(MAX_NUDGES):
        box = bbox(mob)
        pad = Rect(box.x, box.y, box.width + 2 * LABEL_PAD, box.height + 2 * LABEL_PAD)
        hits_label = any(pad.intersects(r) for r in placed)
        hits_arc = bool(len(obstacles)) and bool(np.any(
            (obstacles[:, 0] > pad.left) & (obstacles[:, 0] < pad.right)
            & (obstacles[:, 1] > pad.bottom) & (obstacles[:, 1] < pad.top)))
        if not (hits_label or hits_arc):
            return
        mob.shift(direction * NUDGE)
    raise LayoutError(
        f"{getattr(mob, '_chalk_label', 'label')} cannot be placed clear of the "
        "other labels and arcs; split this beat or shorten its labels.",
        kind="overflow",
    )


def _frame_times(scene: ChalkdustScene, weights: list[float]) -> list[float]:
    """scene.budget(), snapped to whole frames.

    Manim renders a play() as ceil(run_time * fps) frames and a wait() as
    floor(...), so un-snapped run times drift by up to a frame per call and a
    multi-step beat ends audibly off its audio. Rounding the cumulative
    boundaries keeps the total within half a frame of the budget. The
    epsilons land each call on exactly its frame count despite float error.
    """
    fps = config.frame_rate
    edges = np.round(np.cumsum(scene.budget(*weights)) * fps)
    frames = np.diff(np.concatenate([[0.0], edges]))
    eps = 1e-6
    out = [(f - eps) / fps for f in frames[:-1]]   # play(): ceil
    out.append((frames[-1] + eps) / fps)            # wait(): floor
    return out
