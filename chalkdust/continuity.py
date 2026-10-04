"""Continuity between independently rendered beats (SCENE_SPEC.md §6, D-005).

A beat may not depend on mobject state left behind by the previous beat, so
persistence is declarative:

    b02: DataStructureViz, registers="bucket_array"
    b03: SplitCompare,     carry_in=["bucket_array"]   (refused: see below)
    b04: ZoomHighlight,    carry_in=["bucket_array"], target_id="bucket_array"

A carried artifact is never serialised. It is RE-BUILT in the consuming beat
from the producing beat's component name + params (its construction params),
by a builder function the producing component registers. Same inputs, same
mobject -- which is what keeps the render cache honest, provided the recipe is
part of the consuming beat's key (`carry_in_fingerprint`, core/cache.py).

Flow:
    recipes = resolve_carry_in(video_spec)[beat.id]      # spec level
    key     = beat_render_key(spec, dur, ctx, theme, tier, repair,
                              carried=carry_in_fingerprint(recipes))
    scene   = ChalkdustScene(beat_component(spec, recipes), ...)
    # inside a component's build():  target = carried(scene, "bucket_array")

Where a carried artifact goes depends on whether the beat's component acts
on it (register D-G4c-1):

  - A consumer (a component whose carried_targets() is non-empty: Callout,
    ZoomHighlight) gets every carried artifact centred in STAGE, dimmed, and
    lays the frame out around them itself.
  - Any other component does not know the artifact is there, so it would
    draw over it. Its carried artifacts go, dimmed, into the STAGE region it
    does not claim (carry_region): STAGE_RIGHT beside a STAGE_LEFT
    component, and the reverse. SCENE_SPEC.md §4 (no two claimants share a
    region) and §11 rule 1 (never render broken) govern §6's illustrative
    "rendered in STAGE, dimmed". A component that claims STAGE, or both
    halves, leaves no region free, and the semantic rung refuses the beat as
    a region_conflict before any speech or render (carry_region_problems).

This module deliberately does not import the component package at load time:
components import `artifact_builder` and `carried` from here, so a top-level
import the other way would be circular.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from manim import Mobject
from pydantic import BaseModel

from chalkdust.core.cache import content_hash
from chalkdust.core.models import BeatSpec, CarryInError, Region, VideoSpec
from chalkdust.core.version import COMPONENT_LIBRARY_VERSION
from chalkdust.scenes.regions import LayoutError, Rect, fit_to_region, region_rect
from chalkdust.scenes.theme import Theme

if TYPE_CHECKING:
    from chalkdust.scenes.base import ChalkdustScene
    from chalkdust.scenes.components.base import Component, ComponentParams

# How strongly a carried artifact recedes. It is context for the new beat, not
# its subject; a component that wants it back at full strength plays
# Restore(target) -- the undimmed state is saved before dimming.
DIM_DARKNESS = 0.65

# Names the placement rule for the render key (carry_in_fingerprint). Where a
# carried artifact lands is a function of the consuming beat's component and
# params -- already in the key -- and of this rule. Change the rule, change
# this, and every carry-in render re-keys (D-004).
PLACEMENT_RULE = "consumer:stage/other:free-stage-region"

ArtifactBuilder = Callable[["ComponentParams", Theme], Mobject]


class ArtifactRecipe(BaseModel, frozen=True):
    """Everything that determines a carried artifact. Frozen and fully
    serialisable because it feeds the consuming beat's cache key (D-004)."""

    name: str
    producer: str  # component name of the registering beat
    params: dict[str, Any]  # that beat's params: the construction params


# --- builder registry -------------------------------------------------------
# Component name -> function that rebuilds that component's artifact mobject.
# Populated at import time by the producing component's module.

_BUILDERS: dict[str, ArtifactBuilder] = {}


def artifact_builder(component: str) -> Callable[[ArtifactBuilder], ArtifactBuilder]:
    """Register how `component` rebuilds its artifact for a later beat.

        @artifact_builder("DataStructureViz")
        def _artifact(params: DataStructureVizParams, theme: Theme) -> Mobject:
            return _array_mobject(params, theme)   # NOT added to any scene

    The builder receives the producing beat's validated params and the theme,
    and must be a pure function of them: no randomness, no scene state, no
    animation. Return the settled visual, unpositioned -- placement is ours.
    Build text through theme.py so the legibility check still sees it.
    """

    def decorate(fn: ArtifactBuilder) -> ArtifactBuilder:
        if component in _BUILDERS:
            raise ValueError(f"artifact builder for {component!r} already registered")
        _BUILDERS[component] = fn
        return fn

    return decorate


# --- spec level -------------------------------------------------------------


def resolve_carry_in(video: VideoSpec) -> dict[str, tuple[ArtifactRecipe, ...]]:
    """Beat id -> the recipes of the artifacts it carries in, in carry_in order.

    VideoSpec validation already guarantees every name is registered by an
    earlier beat. What it cannot know -- it lives below the scenes boundary --
    is whether the producing component can rebuild an artifact at all, so that
    is checked here, with the same typed error.
    """
    import chalkdust.scenes.components  # noqa: F401  registers every builder

    producers = {b.registers: b for b in video.beats if b.registers is not None}
    resolved: dict[str, tuple[ArtifactRecipe, ...]] = {}
    for beat in video.beats:
        recipes = []
        for name in beat.carry_in:
            src = producers[name]
            if src.component not in _BUILDERS:
                raise CarryInError(
                    f"{beat.id} carries in {name!r}, registered by {src.id} "
                    f"({src.component}), but {src.component} cannot rebuild an "
                    f"artifact; components that can: {sorted(_BUILDERS)}",
                    beat_id=beat.id, name=name,
                )
            recipes.append(ArtifactRecipe(name=name, producer=src.component,
                                          params=src.params))
        resolved[beat.id] = tuple(recipes)
    return resolved


def fixture_beat(component: str, params: dict[str, Any],
                 ) -> tuple[BeatSpec, tuple[ArtifactRecipe, ...]]:
    """One examples()/stress() case as the beat the pipeline would build.

    A carry-in consumer (Callout, ZoomHighlight) cannot build from params
    alone: build() fetches its target with carried(), which raises on a bare
    build. Its Component.fixture_carry_in(params) names the artifacts the case
    stands on, so the case becomes a beat carrying them in (the spec) plus
    what resolve_carry_in would hand its render (the recipes). For every other
    component the recipes are empty and the beat is the plain one.

    The single entry point the registry walks share -- tests/test_layout.py,
    the snapshot capture (validate/snapshot.py) and the semantic TestLibrary
    -- so a consumer's fixtures are built one way everywhere, and that way is
    beat_component(spec, recipes), the render's own.
    """
    from chalkdust.scenes.components import get_component

    recipes = tuple(get_component(component).fixture_carry_in(params))
    spec = BeatSpec(id="b01", narration="placeholder narration",
                    component=component, params=params,
                    carry_in=[r.name for r in recipes])
    return spec, recipes


def referenced_parts(component: Component) -> tuple[str, list[int]] | None:
    """The carried target a consumer indexes into, and the part indices it
    names; None when it names no parts (it acts on the whole artifact).

    The consumers' shared param contract (Callout, ZoomHighlight): `target_id`
    names the carried artifact, and `part` (one index) or `parts` (several)
    index its top-level parts -- the producer's artifact-builder structure,
    which builders document as stable (SCENE_SPEC.md §6).
    """
    params = component.params
    target = getattr(params, "target_id", None)
    if target is None or target not in component.carried_targets():
        return None
    indices = [i for i in (getattr(params, "part", None),) if i is not None]
    indices += list(getattr(params, "parts", None) or ())
    return (target, indices) if indices else None


def part_range_problems(component: Component, recipes: Sequence[ArtifactRecipe],
                        theme: Theme) -> list[str]:
    """Part indices a consumer names that its carried target does not have.

    The part count is a property of the producer's artifact, so the artifact
    is rebuilt from its recipe -- exactly as CarryIn would -- and counted.
    Without this the bad index surfaces only inside build(), as a CarryInError
    the probe used to report as build_error (register D-G4b-1). A target the
    beat does not carry in is check_carried_targets' finding, not this one;
    an artifact that cannot be rebuilt is left for the probe to report.
    """
    named = referenced_parts(component)
    if named is None:
        return []
    target, indices = named
    recipe = next((r for r in recipes if r.name == target), None)
    if recipe is None:
        return []
    try:
        n = len(build_artifact(recipe, theme).submobjects)
    except Exception:
        return []
    bad = [i for i in indices if i >= n]
    if not bad:
        return []
    return [
        f"{component.name} points at part(s) {bad} of {target!r}, which "
        f"{recipe.producer} builds with {n} part(s) (indices 0..{n - 1}). "
        f"Point at a part that exists, or drop the index to act on the whole "
        f"artifact."
    ]


def carry_in_fingerprint(recipes: Sequence[ArtifactRecipe]) -> str:
    """The cache-key term for a beat's carried artifacts.

    Covers the producing component, its construction params, and the library
    version whose builder code turns them into a mobject. Order is kept:
    carry_in order is draw order, and row order in a free region.

    Placement is the rest of what decides the frame. It is a function of the
    consuming beat's component and params (whether it is a consumer, and
    carry_region of its regions()), which beat_render_key already hashes as
    the beat's own spec, and of the rule itself: PLACEMENT_RULE.
    """
    return content_hash(
        "carry_in",
        [r.model_dump(mode="json") for r in recipes],
        COMPONENT_LIBRARY_VERSION,
        PLACEMENT_RULE,
    )


# --- placement --------------------------------------------------------------


def is_consumer(component: Component) -> bool:
    """Whether `component` acts on what it carries in (Callout, ZoomHighlight).

    Declared, not detected: carried_targets() is every name build() fetches
    with carried(). A consumer lays the frame out around all of its carried
    artifacts itself (ZoomHighlight bands the others, Callout refuses them);
    every other component gets them placed clear of its own regions.
    """
    return bool(component.carried_targets())


def carry_region(claimed: Iterable[Region]) -> Region | None:
    """The STAGE region a component claiming `claimed` leaves free, or None.

    STAGE_RIGHT beside a STAGE_LEFT component and the reverse; all of STAGE
    for a component that claims neither half (one that lives in TITLE_BAR or
    LOWER_THIRD only). None when it claims STAGE or both halves: there is
    nowhere on the stage the artifact would not sit under the component
    (SCENE_SPEC.md §4).
    """
    claimed = set(claimed)
    if Region.STAGE in claimed:
        return None
    left, right = Region.STAGE_LEFT in claimed, Region.STAGE_RIGHT in claimed
    if left and right:
        return None
    if left:
        return Region.STAGE_RIGHT
    if right:
        return Region.STAGE_LEFT
    return Region.STAGE


def carry_slots(region: Region, count: int) -> list[Rect]:
    """`count` equal rows of `region`, top to bottom, one per carried
    artifact in carry_in order. §6 keeps the carry-in set small; past one
    artifact each gets a share of the free region, and the legibility floor
    decides whether that share is enough."""
    rect = region_rect(region)
    h = rect.height / count
    return [Rect(rect.x, rect.top - (i + 0.5) * h, rect.width, h)
            for i in range(count)]


def carry_region_problems(component: Component, carry_in: Sequence[str]) -> list[str]:
    """Why `component` cannot have `carry_in` drawn beside it; [] when it can.

    The semantic rung's half of the placement rule (CarryIn.build is the
    other half): a component that does not act on its carried artifacts and
    claims the whole stage leaves them nowhere to go but underneath it --
    the overlap register D-G4c-1 found in every such beat.
    """
    if not carry_in or is_consumer(component):
        return []
    claimed = component.regions()
    if carry_region(claimed) is not None:
        return []
    names = ", ".join(repr(n) for n in carry_in)
    regions = ", ".join(sorted(r.value for r in claimed))
    one = len(carry_in) == 1
    return [
        f"{component.name} carries in {names} but does not act on "
        f"{'it' if one else 'them'}. A carried artifact the component does "
        f"not act on is drawn dimmed in a STAGE region the component leaves "
        f"free (stage_left or stage_right), and {component.name} claims "
        f"{regions}, which leaves none: {names} would sit under it. Drop "
        f"{names} from this beat's carry_in, or act on "
        f"{'it' if one else 'them'} with ZoomHighlight or Callout."
    ]


# --- scene level ------------------------------------------------------------


def build_artifact(recipe: ArtifactRecipe, theme: Theme) -> Mobject:
    """Rebuild one carried artifact, unplaced and at full strength."""
    from chalkdust.scenes.components import get_component

    params = get_component(recipe.producer).Params.model_validate(recipe.params)
    mob = _BUILDERS[recipe.producer](params, theme)
    mob._chalk_label = f"carried[{recipe.name}]"  # type: ignore[attr-defined]
    return mob


class CarryIn:
    """Wraps a beat's component so its carried artifacts are on screen first.

    Satisfies the same build(scene)/regions() contract as a Component, so it
    goes straight into ChalkdustScene or LayoutProbe. Artifacts appear at
    t=0 with no animation: the previous beat already showed them, and a cut
    should feel like the picture persisted.

    Placement (see the module docstring): a consumer's artifacts are centred
    in STAGE; anyone else's are fitted, one row each, into the STAGE region
    the component leaves free (carry_region), and tagged
    `_chalk_carry_clear` so the geometric probe checks that nothing of the
    component's is drawn over them (geometric.LayoutProbe.settle).
    """

    def __init__(self, component: Component, recipes: Sequence[ArtifactRecipe]) -> None:
        self.component = component
        self.recipes = tuple(recipes)

    def placement(self) -> Region | None:
        """Where the carried artifacts go: STAGE for a consumer, else the
        region the component leaves free; None when it leaves none."""
        if is_consumer(self.component):
            return Region.STAGE
        return carry_region(self.component.regions())

    def regions(self) -> set[Region]:
        return self.component.regions() | {self.placement() or Region.STAGE}

    def build(self, scene: ChalkdustScene) -> None:
        consumer = is_consumer(self.component)
        region = self.placement()
        if region is None:
            # The semantic rung refuses this beat before anything is built;
            # this is the backstop for a caller that skipped it, typed as
            # that rung types it.
            names = [r.name for r in self.recipes]
            raise LayoutError(carry_region_problems(self.component, names)[0],
                              kind="region_conflict")
        slots = ([region_rect(region)] * len(self.recipes) if consumer
                 else carry_slots(region, len(self.recipes)))
        artifacts: dict[str, Mobject] = {}
        for recipe, slot in zip(self.recipes, slots, strict=True):
            mob = build_artifact(recipe, scene.theme)
            try:
                fit_to_region(mob, slot)
            except LayoutError as exc:
                if consumer:
                    raise
                # Same kind (overflow: a clean refusal), but naming the
                # artifact and the region it was given, not just the text.
                raise LayoutError(
                    f"carried {recipe.name!r} ({recipe.producer}) does not fit "
                    f"legibly in {region.value}, the STAGE region "
                    f"{self.component.name} leaves free for it: {exc}",
                    kind=exc.kind,
                ) from exc
            mob.save_state()  # full strength, for Restore(target)
            # fade() scales each part's existing opacity; set_opacity() would
            # flatten them and fill in shapes that are meant to be outlines.
            mob.fade(DIM_DARKNESS)
            if not consumer:
                mob._chalk_carry_clear = True  # type: ignore[attr-defined]
            scene.add(mob)
            artifacts[recipe.name] = mob
        scene._chalk_carried = artifacts  # type: ignore[attr-defined]
        self.component.build(scene)


def beat_component(spec: BeatSpec, recipes: Sequence[ArtifactRecipe]) -> Component | CarryIn:
    """The object to hand ChalkdustScene for this beat."""
    from chalkdust.scenes.components import make_component

    component = make_component(spec.component, spec.params)
    return CarryIn(component, recipes) if recipes else component


def carried(scene: ChalkdustScene, name: str) -> Mobject:
    """The carried artifact `name`, already placed and dimmed (CarryIn).

    For components that act on something already on screen (ZoomHighlight,
    Callout): their `target_id` is a carry-in name. Raises CarryInError if
    the beat did not carry `name` in -- a spec bug, not a layout one.
    """
    artifacts: dict[str, Mobject] = getattr(scene, "_chalk_carried", {})
    if name not in artifacts:
        raise CarryInError(
            f"{name!r} is not carried into this beat; carried: {sorted(artifacts)}. "
            "Add it to the beat's carry_in.",
            beat_id=None, name=name,
        )
    return artifacts[name]
