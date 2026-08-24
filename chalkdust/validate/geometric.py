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
    kind: str      # out_of_bounds | overlap | illegible | overflow | build_error
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
        for anim in animations:
            # prepare_animation converts `.animate` builders into Animations.
            anim = prepare_animation(anim)
            anim.begin()
            anim.interpolate(1.0)   # jump to final state
            anim.finish()
            if anim.is_introducer():
                self.add(anim.mobject)
            anim.clean_up_from_scene(self)   # removes mobjects for FadeOut etc.

    def wait(self, *args, **kwargs) -> None:  # type: ignore[override]
        return None


def validate_beat(spec: BeatSpec, theme: str = "default",
                  duration: float = 8.0) -> Report:
    """Check one beat's layout. Never raises -- failures come back as findings."""
    report = Report(beat_id=spec.id)
    try:
        component = make_component(spec.component, spec.params)
    except Exception as exc:
        report.findings.append(Finding("build_error", f"{type(exc).__name__}: {exc}"))
        return report

    # strict=False so the scene collects every finding instead of stopping at
    # the first. The repair loop wants the full picture in one pass.
    probe = LayoutProbe(component, theme=theme, duration=duration, strict=False)
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
