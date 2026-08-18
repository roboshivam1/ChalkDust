"""ChalkdustScene: the base every component renders into.

Two jobs:

  1. Apply the theme (background colour, resolved fonts).
  2. Run layout assertions at settle points -- the moments when animation has
     stopped and the frame should be inspectable. This is where layout bugs
     get caught, at build time, before a frame is ever rendered.

A component that never calls settle() is a component whose layout is never
checked, so `construct` runs a final settle unconditionally.
"""

from __future__ import annotations

from typing import Protocol

from manim import Mobject, Scene

from chalkdust.scenes.regions import (
    LayoutError,
    assert_in_safe_area,
    assert_legible,
    bbox,
)
from chalkdust.scenes.theme import Theme, get_theme, resolve_fonts


class Component(Protocol):
    """Minimal contract. The real base class arrives with the component
    library; this keeps base.py independent of it."""

    def build(self, scene: "ChalkdustScene") -> None: ...


class ChalkdustScene(Scene):
    """Renders exactly one beat.

    Constructed programmatically rather than via the manim CLI, so we can pass
    the component, theme, and measured audio duration straight in.

    NOTE ON ATTRIBUTE NAMES: Manim's Scene owns `self.duration` and overwrites
    it on every play() call. Anything we add here is namespaced to avoid
    colliding with internals we do not control.
    """

    def __init__(
        self,
        component: Component,
        theme: Theme | str = "default",
        duration: float = 5.0,
        strict: bool = True,
        **kwargs,
    ) -> None:
        # Call this FIRST. Scene.__init__ initialises internal state and will
        # clobber any attribute we set beforehand.
        super().__init__(**kwargs)

        raw_theme = get_theme(theme) if isinstance(theme, str) else theme
        self.theme = resolve_fonts(raw_theme)
        self.component = component
        # Audio duration drives animation timing. Components divide this
        # budget; they never hardcode run times (D-002).
        self.beat_duration = float(duration)
        # strict=False downgrades assertion failures to recorded warnings,
        # used when rendering a degraded beat we already know is imperfect.
        self.strict = strict
        self.layout_warnings: list[str] = []

    def setup(self) -> None:
        self.camera.background_color = self.theme.palette.bg

    def construct(self) -> None:
        self.component.build(self)
        # Unconditional final check: no component can opt out of validation.
        self.settle("end of beat")

    # --- timing -------------------------------------------------------------

    def budget(self, *weights: float) -> list[float]:
        """Split the beat's audio duration into run times by relative weight.

            fade_in, hold, fade_out = scene.budget(1, 4, 1)
        """
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to more than zero")
        if self.beat_duration <= 0:
            # Without this, a zero budget produces run_time=0 and Manim raises
            # deep inside compile_animation_data, far from the real cause.
            raise ValueError(
                f"beat_duration is {self.beat_duration}; must be positive. "
                "Was the speech stage run before rendering?"
            )
        return [self.beat_duration * w / total for w in weights]

    # --- validation ---------------------------------------------------------

    def settle(self, label: str = "settle point") -> None:
        """Assert the current frame is valid.

        Call after any animation that leaves the scene in a state a viewer will
        actually look at. Cheap -- pure geometry on mobjects already in memory,
        no rendering involved.
        """
        for mob in self.mobjects:
            try:
                assert_in_safe_area(mob, label=f"{label}: {_name(mob)}")
                assert_legible(mob, label=f"{label}: {_name(mob)}")
            except LayoutError as exc:
                if self.strict:
                    raise
                self.layout_warnings.append(str(exc))

        self._check_pairwise_overlap(label)

    def _check_pairwise_overlap(self, label: str) -> None:
        """Flag overlap between top-level mobjects marked mutually exclusive.

        Overlap is only a bug when unintended, so it is opt-in via
        `exclusive()`. Checking everything would fire constantly on legitimate
        composition (labels on axes, callouts on diagrams).
        """
        tagged = [m for m in self.mobjects if getattr(m, "_chalk_exclusive", False)]
        for i, a in enumerate(tagged):
            for b in tagged[i + 1:]:
                if bbox(a).intersects(bbox(b)):
                    msg = f"{label}: {_name(a)} overlaps {_name(b)}"
                    if self.strict:
                        raise LayoutError(msg)
                    self.layout_warnings.append(msg)

    # --- convenience --------------------------------------------------------

    def exclusive(self, *mobs: Mobject) -> None:
        """Mark mobjects that must never overlap each other."""
        for m in mobs:
            m._chalk_exclusive = True  # type: ignore[attr-defined]


def _name(mob: Mobject) -> str:
    """Readable identifier for error messages. Components can set
    `_chalk_label` to make failures easier to trace."""
    return getattr(mob, "_chalk_label", type(mob).__name__)
