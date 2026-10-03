"""Component base class and registry.

A component owns its own layout. It receives validated params, declares which
regions it occupies, and builds itself into a ChalkdustScene. It never receives
a coordinate, colour, or font size -- those come from the theme and the region
system (SCENE_SPEC.md §1).
"""

from __future__ import annotations

import textwrap
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar

from manim import Mobject
from pydantic import BaseModel, ConfigDict

from chalkdust.core.models import Region

if TYPE_CHECKING:
    from chalkdust.continuity import ArtifactRecipe
    from chalkdust.scenes.base import ChalkdustScene

# Shortest time one reveal step can take and still register with a viewer --
# anything faster reads as a flash, not a reveal. Components overriding
# min_seconds() typically return MIN_STEP_SECONDS * <number of steps>.
MIN_STEP_SECONDS = 0.5


class ComponentParams(BaseModel):
    """Base for every component's parameter model.

    `extra="forbid"` is deliberate. When the model emits a param we do not
    support, we want a loud validation failure, not silent ignoring -- silent
    ignoring produces a video that renders fine and does not do what the spec
    asked for, which is far harder to debug.

    `frozen=True` because params feed the render cache key (D-004).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class Component(ABC):
    """One scene primitive.

    Subclasses set `name` (as referenced in a BeatSpec) and `Params`, then
    implement `regions()` and `build()`.
    """

    name: ClassVar[str]
    Params: ClassVar[type[ComponentParams]]
    # Set (to the reason) only by a component that builds no fixed visual of
    # its own, so it has nothing for the snapshot harness to record
    # (SCENE_SPEC.md §11 rule 6). Such a component must say what guards it
    # against Manim drift instead. Today: RawScene only.
    snapshot_exempt: ClassVar[str | None] = None

    def __init__(self, params: ComponentParams | dict[str, Any]) -> None:
        if isinstance(params, dict):
            # Raises ValidationError on unknown or malformed params.
            params = self.Params.model_validate(params)
        elif not isinstance(params, self.Params):
            raise TypeError(
                f"{type(self).__name__} expects {self.Params.__name__}, "
                f"got {type(params).__name__}"
            )
        self.params = params

    @abstractmethod
    def regions(self) -> set[Region]:
        """Regions this component occupies.

        Used by the compiler to reject two simultaneously-active components
        claiming the same space.
        """

    @abstractmethod
    def build(self, scene: "ChalkdustScene") -> None:
        """Construct and animate. Must consume exactly the scene's time budget."""

    # --- semantic-rung hooks ------------------------------------------------
    # Optional. Read by validate/semantic.py before anything is built
    # (SCENE_SPEC.md §8, rung 2). The defaults are permissive on purpose: a
    # component that does not override them is never refused on their account.

    def min_seconds(self) -> float:
        """Shortest narration, in seconds, these params can animate without
        rushing.

        The beat's audio duration is the component's whole time budget (D-002);
        if the narration is shorter than this, every step gets squeezed below
        what a viewer can follow. Typically MIN_STEP_SECONDS * steps, where a
        step is one reveal the viewer must register.
        """
        return 0.0

    def latex_strings(self) -> list[str]:
        """Every LaTeX string build() will compile, exactly as passed to
        MathTex (math mode).

        The semantic rung compiles each one standalone, so a malformed
        expression fails with its own source in the message -- not as a
        build_error from deep inside build().
        """
        return []

    def carried_targets(self) -> list[str]:
        """Every carry-in name build() will fetch with continuity.carried().

        For components that act on an artifact already on screen (Callout,
        ZoomHighlight: their `target_id`). The semantic rung refuses a beat
        whose carry_in does not list each one, as kind "carry_in" -- before
        anything is built, rather than as a CarryInError from inside build().
        """
        return []

    # --- test fixtures ------------------------------------------------------
    # Each component declares its own cases so the shared test suite covers
    # every component automatically as the library grows (SCENE_SPEC.md §11).

    @classmethod
    def examples(cls) -> list[dict[str, Any]]:
        """Realistic params that MUST validate clean."""
        return []

    @classmethod
    def stress(cls) -> list[dict[str, Any]]:
        """Deliberately abusive params -- roughly 3x realistic content volume.

        These must either validate clean or fail with a LayoutError. What they
        must never do is render something broken, or raise an unrelated
        exception like IndexError.
        """
        return []

    @classmethod
    def fixture_carry_in(cls, params: dict[str, Any]) -> list[ArtifactRecipe]:
        """The carried artifacts one examples()/stress() case is built with.

        A carry-in consumer (SCENE_SPEC.md §6) cannot build without its target
        on screen, so the registry walks (test_layout, test_snapshots, the
        semantic TestLibrary) build each case with these recipes carried in,
        exactly as the pipeline would. Empty for every other component.
        """
        return []


# --- registry ---------------------------------------------------------------
# Maps the `component` string in a BeatSpec to a class. Populated by the
# @register decorator at import time.

_REGISTRY: dict[str, type[Component]] = {}


def register(cls: type[Component]) -> type[Component]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must define a `name`")
    if cls.name in _REGISTRY:
        raise ValueError(f"component {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def get_component(name: str) -> type[Component]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown component {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def make_component(name: str, params: dict[str, Any]) -> Component:
    """Build a component instance from a BeatSpec's component + params."""
    return get_component(name)(params)


def registered_names() -> list[str]:
    """Every component name. Feeds the LLM prompt in Phase 2."""
    return sorted(_REGISTRY)


# --- shared helpers ---------------------------------------------------------


def label(mob: Mobject, text: str) -> Mobject:
    """Tag a mobject so layout errors name it usefully."""
    mob._chalk_label = text  # type: ignore[attr-defined]
    return mob


def wrap(s: str, width: int = 42) -> str:
    """Hard-wrap text. Manim's Text does not wrap on its own -- a long string
    becomes one very wide line that gets scaled into illegibility.

    `width` is scaled by the active wrap_scale(), 1.0 outside a repair."""
    return textwrap.fill(s.strip(), width=max(1, round(width * _WRAP_SCALE.get())))


# Mechanical repair (validate/repair.py, SCENE_SPEC.md §9) rebuilds a beat that
# overflowed with every wrap() width scaled -- wider lines for content that is
# too tall, narrower for content that is too wide. It lives here rather than in
# validate/ because scenes must not import the validator.
_WRAP_SCALE: ContextVar[float] = ContextVar("chalkdust_wrap_scale", default=1.0)


@contextmanager
def wrap_scale(factor: float) -> Iterator[None]:
    """Scale every wrap() width by `factor` for the duration of the block."""
    token = _WRAP_SCALE.set(factor)
    try:
        yield
    finally:
        _WRAP_SCALE.reset(token)
