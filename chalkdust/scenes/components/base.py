"""Component base class and registry.

A component owns its own layout. It receives validated params, declares which
regions it occupies, and builds itself into a ChalkdustScene. It never receives
a coordinate, colour, or font size -- those come from the theme and the region
system (SCENE_SPEC.md §1).
"""

from __future__ import annotations

import textwrap
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from manim import Mobject
from pydantic import BaseModel, ConfigDict

from chalkdust.core.models import Region

if TYPE_CHECKING:
    from chalkdust.scenes.base import ChalkdustScene


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
    becomes one very wide line that gets scaled into illegibility."""
    return textwrap.fill(s.strip(), width=width)
