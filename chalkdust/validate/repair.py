"""Mechanical repair (SCENE_SPEC.md §9, step 1): no LLM, bounded, re-validated.

The geometric probe says what is wrong; this module proposes the cheapest
mechanical change that might fix it, rebuilds the beat with that change, and
lets the probe judge again. Repair only ever proposes -- re-validation decides.
That is what keeps it safe: a fix that makes things worse just fails the next
probe, and a beat that cannot be repaired comes back exactly as it was.

Three levers, matching the spec:

  scale to fit  shrink a top-level mobject that is too big for its region --
                never below MIN_FONT_SIZE. Text that would need to go smaller
                stays a refusal: the fix is splitting the beat (§4).
  nudge         translate it the shortest distance back inside its region.
  wrap text     rebuild with every wrap() width scaled, for content that
                fit_to_region refused as overflow.

A repair is a RepairPlan, not a mutated scene. Beats render independently and
are cached by the hash of their spec (D-004, D-005), so the render has to
reproduce the repaired layout from the spec alone: the render worker runs
repair_beat() and builds a RepairedScene with the plan it returns. The plan
itself (RepairPlan.key_data) is a term of the beat's render key, so a change
to the repair code that changes what is drawn also changes the key.

LIMITATIONS, same register as geometric.py's:
  - Fixes are keyed on the order in which distinct mobjects are first added to
    the scene. Build is deterministic, so the probe and the render add in the
    same order -- LayoutProbe.play mirrors Scene.play for exactly this reason.
  - A fix is applied when its mobject is first added. A component that then
    repositions it absolutely (move_to, next_to) overwrites the fix; the loop
    sees no progress, stops, and the finding stands.
  - Transform targets enter the scene through Scene.replace, not add, so they
    are never fixed. Probe and render behave the same, so this costs repairs,
    never correctness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from manim import Mobject

from chalkdust.continuity import ArtifactRecipe, beat_component
from chalkdust.core.models import BeatSpec, Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import Component
from chalkdust.scenes.components.base import wrap_scale
from chalkdust.scenes.regions import (
    DEFAULT_PADDING,
    MIN_FONT_SIZE,
    Rect,
    bbox,
    region_rect,
    safe_area,
    scale_with_tags,
    smallest_font_size,
)
from chalkdust.scenes.theme import Theme
from chalkdust.validate.geometric import Finding, LayoutProbe, Report, run_probe

# Re-probes after the first validation. Each costs one build; the whole
# SCENE_SPEC.md §9 loop is bounded at three rungs, and this is only rung one.
MAX_ATTEMPTS = 3

# wrap() width multipliers tried, in order, on overflow. Wider first: the
# usual overflow is too many lines for the region's height. 0.7 then covers
# content that is too wide. 1.5x a component's designed width still sits
# inside a readable line length.
WRAP_SCALES = (1.5, 0.7)

# Finding kinds a mechanical lever can address. Everything else -- illegible,
# overlap, build_error -- passes straight to the next rung (§9: regenerate).
REPAIRABLE = {"out_of_bounds", "overflow"}


@dataclass(frozen=True)
class Fix:
    """Scale about the mobject's own centre, then shift."""

    scale: float = 1.0
    dx: float = 0.0
    dy: float = 0.0

    def then(self, other: Fix) -> Fix:
        # Scaling about its own centre leaves the centre where it is, so
        # composed scales multiply and shifts simply add.
        return Fix(self.scale * other.scale, self.dx + other.dx, self.dy + other.dy)

    def apply(self, mob: Mobject) -> None:
        if self.scale != 1.0:
            scale_with_tags(mob, self.scale)
        mob.shift(np.array([self.dx, self.dy, 0.0]))


@dataclass(frozen=True)
class RepairPlan:
    """Everything needed to rebuild a beat in its repaired form.

    fixes maps first-add index (0 = first distinct mobject added) to a Fix.
    """

    wrap_scale: float = 1.0
    fixes: dict[int, Fix] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.wrap_scale == 1.0 and not self.fixes

    def key_data(self) -> dict:
        """The plan as plain data for the render cache key (D-004).

        Rounded like the key's duration term, so float noise in a recomputed
        plan does not cause spurious misses; 1e-6 units is far below a pixel.
        """
        return {
            "wrap_scale": round(self.wrap_scale, 6),
            "fixes": {str(i): [round(f.scale, 6), round(f.dx, 6), round(f.dy, 6)]
                      for i, f in sorted(self.fixes.items())},
        }

    def with_fixes(self, new: dict[int, Fix]) -> RepairPlan:
        merged = dict(self.fixes)
        for index, fix in new.items():
            merged[index] = merged[index].then(fix) if index in merged else fix
        return RepairPlan(self.wrap_scale, merged)


class _AppliesPlan:
    """Scene mixin: apply a RepairPlan while the component builds."""

    def __init__(self, component: Component, plan: RepairPlan | None = None,
                 **kwargs) -> None:
        super().__init__(component, **kwargs)  # type: ignore[call-arg]
        # Set after Scene.__init__, which initialises state of its own.
        self.repair_plan = plan or RepairPlan()
        # Holding the mobjects keeps their ids from being reused by new ones.
        self._first_added: list[Mobject] = []
        self._add_index: dict[int, int] = {}

    def add(self, *mobjects: Mobject):
        # getattr: tolerate an add() issued before __init__ has finished.
        index_of = getattr(self, "_add_index", None)
        if index_of is not None:
            for mob in mobjects:
                if id(mob) in index_of:
                    continue
                index_of[id(mob)] = len(self._first_added)
                self._first_added.append(mob)
                fix = self.repair_plan.fixes.get(index_of[id(mob)])
                if fix is not None:
                    fix.apply(mob)
        return super().add(*mobjects)  # type: ignore[misc]

    def construct(self) -> None:
        with wrap_scale(self.repair_plan.wrap_scale):
            super().construct()  # type: ignore[misc]


class RepairedScene(_AppliesPlan, ChalkdustScene):
    """ChalkdustScene that builds with a repair plan applied.

    The render integration point: RepairedScene(component, plan=result.plan,
    theme=..., duration=...) wherever a ChalkdustScene would be built. With an
    empty plan it behaves exactly like ChalkdustScene.
    """


class RepairProbe(_AppliesPlan, LayoutProbe):
    """A LayoutProbe that builds with a plan and proposes the next one."""

    def __init__(self, component: Component, plan: RepairPlan | None = None,
                 **kwargs) -> None:
        super().__init__(component, plan=plan, **kwargs)
        self.proposed: dict[int, Fix] = {}

    def settle(self, label: str = "settle point") -> None:
        regions = self.component.regions()  # type: ignore[attr-defined]
        for mob in self.mobjects:
            index = self._add_index.get(id(mob))
            if index is None or index in self.proposed:
                continue
            fix = propose_fix(mob, regions)
            if fix is not None:
                self.proposed[index] = fix
        super().settle(label)


def propose_fix(mob: Mobject, regions: set[Region]) -> Fix | None:
    """Scale-and-nudge that brings an out-of-bounds mobject inside its region.

    None when the mobject is already inside the safe area, or when fitting it
    would push its text below the legibility floor -- that stays a refusal.
    """
    box = bbox(mob)
    if safe_area().contains(box):
        return None

    target = _target_rect(box, regions).inset(DEFAULT_PADDING)
    scale = min([1.0] + [t / b for t, b in ((target.width, box.width),
                                            (target.height, box.height)) if b > 0])
    smallest = smallest_font_size(mob)
    if smallest is not None and smallest * scale < MIN_FONT_SIZE:
        return None

    return Fix(
        scale=scale,
        dx=_nudge(box.x, box.width * scale, target.left, target.right),
        dy=_nudge(box.y, box.height * scale, target.bottom, target.top),
    )


def _target_rect(box: Rect, regions: set[Region]) -> Rect:
    """The declared region the mobject overlaps most; the safe area if none.

    Sorted first so ties break the same way in every process -- Region is a
    str enum, and set order follows the per-process string hash seed.
    """
    rects = [region_rect(r) for r in sorted(regions, key=lambda r: r.value)]
    best = max(rects, key=lambda r: (r.overlap_area(box), r.area), default=None)
    if best is None or best.overlap_area(box) == 0:
        return safe_area()
    return best


def _nudge(centre: float, size: float, lo: float, hi: float) -> float:
    """Shortest shift putting [centre-size/2, centre+size/2] inside [lo, hi]."""
    if centre - size / 2 < lo:
        return lo - (centre - size / 2)
    if centre + size / 2 > hi:
        return hi - (centre + size / 2)
    return 0.0


# --- the loop ---------------------------------------------------------------


@dataclass
class RepairResult:
    """Outcome of mechanical repair on one beat.

    On success, `plan` is what to render with and `report` is clean. On
    failure, both are the UNREPAIRED ones: a failed repair must not change
    what renders, nor what the next rung is told. Every attempt is kept in
    `history`, first entry unrepaired.
    """

    beat_id: str
    plan: RepairPlan
    report: Report
    history: list[Report]

    @property
    def ok(self) -> bool:
        return self.report.ok

    @property
    def repaired(self) -> bool:
        return self.ok and not self.plan.empty


def repair_beat(spec: BeatSpec, theme: Theme | str = "default",
                duration: float = 8.0,
                recipes: Sequence[ArtifactRecipe] = ()) -> RepairResult:
    """Validate one beat and mechanically repair it if it fails. Never raises.

    `recipes` are the beat's carried artifacts (SCENE_SPEC.md §6): the probe
    builds with them on screen, as the render does (continuity.beat_component).
    """
    try:
        component = beat_component(spec, recipes)
    except Exception as exc:
        report = Report(spec.id, [Finding("build_error", f"{type(exc).__name__}: {exc}")])
        return RepairResult(spec.id, RepairPlan(), report, [report])
    return repair_component(component, spec.id, theme=theme, duration=duration)


def repair_component(component: Component, beat_id: str,
                     theme: Theme | str = "default",
                     duration: float = 8.0) -> RepairResult:
    """The bounded loop: probe, propose, re-probe, at most MAX_ATTEMPTS times."""
    plan = RepairPlan()
    report, proposed = _probe(component, beat_id, plan, theme, duration)
    history = [report]
    wraps = list(WRAP_SCALES)

    for _ in range(MAX_ATTEMPTS):
        if report.ok or not report.kinds() & REPAIRABLE:
            break
        if proposed:
            candidate = plan.with_fixes(proposed)
        elif wraps:
            # Fresh fixes: the geometry they were computed for is gone.
            candidate = RepairPlan(wrap_scale=wraps.pop(0))
        else:
            break

        new_report, new_proposed = _probe(component, beat_id, candidate, theme, duration)
        history.append(new_report)
        if new_report.findings == report.findings:
            break  # no progress -- e.g. the component overwrote the fix
        plan, report, proposed = candidate, new_report, new_proposed

    if report.ok:
        return RepairResult(beat_id, plan, report, history)
    return RepairResult(beat_id, RepairPlan(), history[0], history)


def _probe(component: Component, beat_id: str, plan: RepairPlan, theme: Theme | str,
           duration: float) -> tuple[Report, dict[int, Fix]]:
    probe = RepairProbe(component, plan=plan, theme=theme, duration=duration,
                        strict=False)
    return run_probe(probe, beat_id), probe.proposed
