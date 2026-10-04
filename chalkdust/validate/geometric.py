"""Geometric validation without rendering (validation rung 3).

Components position mobjects at build time; animations only reveal them. That
lets us run build() with every animation snapped straight to its final state,
inspect the resulting geometry, and never encode a frame -- roughly two orders
of magnitude cheaper than a draft render.

LIMITATION worth knowing: an animation that MOVES a mobject rather than
revealing it is applied here in one step, so intermediate positions are never
checked. A component that flies something across the frame could clip the edge
mid-flight and pass this probe. The draft-render settle checks remain the
authoritative gate; this is the cheap pre-filter that catches most failures
before compute is spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from manim.animation.animation import prepare_animation

from chalkdust.core.models import BeatSpec
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.regions import LayoutError
from chalkdust.scenes.components import make_component


@dataclass(frozen=True)
class Finding:
    """One failure the repair loop can dispatch on (SCENE_SPEC.md §9).

    kinds, geometric (rung 3, this module):
      out_of_bounds | overlap | illegible | overflow | build_error
    kinds, semantic (rung 2, semantic.py):
      duration        -- narration too short for the component's steps, or
                         longer than one beat may run
      carry_in        -- references an artifact no earlier beat registered
      region_conflict -- two simultaneously active claimants share space
      capacity        -- more text than the claimed regions can hold legibly
      latex           -- a LaTeX string does not compile standalone
    """

    kind: str
    message: str


@dataclass
class Report:
    beat_id: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    def kinds(self) -> set[str]:
        return {f.kind for f in self.findings}

    def __str__(self) -> str:
        if self.ok:
            return f"{self.beat_id}: ok"
        lines = [f"{self.beat_id}: {len(self.findings)} finding(s)"]
        lines += [f"  [{f.kind}] {f.message}" for f in self.findings]
        return "\n".join(lines)


class LayoutProbe(ChalkdustScene):
    """A ChalkdustScene that applies animations instantly and writes nothing."""

    def play(self, *animations, **kwargs) -> None:  # type: ignore[override]
        # Same order of operations as Scene.play, so the probe adds exactly
        # what a real render adds, in the same sequence: first the mobject of
        # every non-introducer animation not yet on screen
        # (compile_animation_data), then each introducer just before it begins
        # (begin_animations). Mechanical repair keys its fixes on that order.
        # prepare_animation converts `.animate` builders into Animations.
        anims = [prepare_animation(a) for a in animations]
        self.add_mobjects_from_animations(anims)
        for anim in anims:
            if anim.is_introducer():
                self.add(anim.mobject)
            anim.begin()
        for anim in anims:
            anim.interpolate(1.0)   # jump to final state
            anim.finish()
            anim.clean_up_from_scene(self)   # removes mobjects for FadeOut etc.

    def wait(self, *args, **kwargs) -> None:  # type: ignore[override]
        return None


def validate_beat(spec: BeatSpec, theme: str = "default",
                  duration: float = 8.0) -> Report:
    """Check one beat's layout. Never raises -- failures come back as findings."""
    try:
        component = make_component(spec.component, spec.params)
    except Exception as exc:
        return Report(spec.id, [Finding("build_error", f"{type(exc).__name__}: {exc}")])

    # strict=False so the scene collects every finding instead of stopping at
    # the first. The repair loop wants the full picture in one pass.
    probe = LayoutProbe(component, theme=theme, duration=duration, strict=False)
    return run_probe(probe, spec.id)


def run_probe(probe: LayoutProbe, beat_id: str) -> Report:
    """Build `probe` (constructed with strict=False) and collect its findings.

    Split out of validate_beat so repair.py can probe with a scene that
    applies a repair plan, under exactly the same error handling.
    """
    report = Report(beat_id=beat_id)
    try:
        probe.construct()
    except LayoutError as exc:
        # A LayoutError raised during build() -- typically by fit_to_region --
        # never reaches settle()'s strict=False handling, so preserve its kind
        # here. Folding it into build_error would lose the information the
        # repair loop dispatches on.
        report.findings.append(Finding(exc.kind, str(exc)))
        return report
    except Exception as exc:
        # Anything else is a genuine crash: the component hit content it did
        # not anticipate and failed in a way it did not intend.
        report.findings.append(Finding("build_error", f"{type(exc).__name__}: {exc}"))
        return report

    report.findings.extend(Finding(k, m) for k, m in probe.layout_warnings)
    return report


def validate_specs(specs: list[BeatSpec], theme: str = "default") -> list[Report]:
    return [validate_beat(s, theme=theme) for s in specs]
