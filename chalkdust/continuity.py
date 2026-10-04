"""Continuity between independently rendered beats (SCENE_SPEC.md §6, D-005).

A beat may not depend on mobject state left behind by the previous beat, so
persistence is declarative:

    b02: DataStructureViz, registers="bucket_array"
    b03: SplitCompare,     carry_in=["bucket_array"]   (rendered in STAGE, dimmed)
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

This module deliberately does not import the component package at load time:
components import `artifact_builder` and `carried` from here, so a top-level
import the other way would be circular.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from manim import Mobject
from pydantic import BaseModel

from chalkdust.core.cache import content_hash
from chalkdust.core.models import BeatSpec, CarryInError, Region, VideoSpec
from chalkdust.core.version import COMPONENT_LIBRARY_VERSION
from chalkdust.scenes.regions import fit_to_region
from chalkdust.scenes.theme import Theme

if TYPE_CHECKING:
    from chalkdust.scenes.base import ChalkdustScene
    from chalkdust.scenes.components.base import Component, ComponentParams

# How strongly a carried artifact recedes. It is context for the new beat, not
# its subject; a component that wants it back at full strength plays
# Restore(target) -- the undimmed state is saved before dimming.
DIM_DARKNESS = 0.65

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
    carry_in order is draw order.
    """
    return content_hash(
        "carry_in",
        [r.model_dump(mode="json") for r in recipes],
        COMPONENT_LIBRARY_VERSION,
    )


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
    """

    def __init__(self, component: Component, recipes: Sequence[ArtifactRecipe]) -> None:
        self.component = component
        self.recipes = tuple(recipes)

    def regions(self) -> set[Region]:
        return self.component.regions() | {Region.STAGE}

    def build(self, scene: ChalkdustScene) -> None:
        artifacts: dict[str, Mobject] = {}
        for recipe in self.recipes:
            mob = build_artifact(recipe, scene.theme)
            fit_to_region(mob, Region.STAGE)
            mob.save_state()  # full strength, for Restore(target)
            # fade() scales each part's existing opacity; set_opacity() would
            # flatten them and fill in shapes that are meant to be outlines.
            mob.fade(DIM_DARKNESS)
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
    """The carried artifact `name`, already placed in STAGE and dimmed.

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
