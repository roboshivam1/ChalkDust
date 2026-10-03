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

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from manim import config, tempconfig
from manim.animation.animation import prepare_animation

from chalkdust.continuity import ArtifactRecipe, beat_component
from chalkdust.core.models import BeatSpec
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.regions import LayoutError


@dataclass(frozen=True)
class Finding:
    """One failure the repair loop can dispatch on (SCENE_SPEC.md §9).

    kinds, geometric (rung 3, this module):
      out_of_bounds | overlap | illegible | overflow | invalid_latex |
      build_error
    kinds, semantic (rung 2, semantic.py):
      duration        -- narration too short for the component's steps, or
                         longer than one beat may run
      carry_in        -- references an artifact no earlier beat registered,
                         or acts on one the beat does not carry in
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


# Where the probe's Manim scratch goes when the caller has not said. Building a
# scene creates Manim's media tree, and building Text or MathTex writes SVGs
# into it; left at Manim's default that is ./media in whatever directory the
# probe happens to run from. This mirrors the pipeline's default work dir
# (work/, git-ignored) and its media_dir under it.
DEFAULT_PROBE_MEDIA_DIR = Path("work") / "manim"


@contextmanager
def probe_media(media_dir: Path | str | None = None) -> Iterator[None]:
    """Scope Manim's media_dir for a probe.

    An explicit `media_dir` wins. Otherwise a caller that already redirected
    Manim (the pipeline's tempconfig, the worker's) is left alone, and only
    Manim's own default -- ./media in the cwd -- is replaced by
    DEFAULT_PROBE_MEDIA_DIR.
    """
    if media_dir is None:
        if Path(config.media_dir) != Path("media"):
            yield
            return
        media_dir = DEFAULT_PROBE_MEDIA_DIR
    with tempconfig({"media_dir": str(media_dir)}):
        yield


class LayoutProbe(ChalkdustScene):
    """A ChalkdustScene that applies animations instantly and writes nothing
    into the cwd: its Manim scratch goes to `media_dir` (see probe_media)."""

    def __init__(self, *args, media_dir: Path | str | None = None, **kwargs) -> None:
        # Scene.__init__ builds the file writer, which creates the media tree.
        with probe_media(media_dir):
            super().__init__(*args, **kwargs)
        self.probe_media_dir = media_dir

    def construct(self) -> None:
        # build() creates Text and MathTex, which write their SVGs under it.
        with probe_media(self.probe_media_dir):
            super().construct()

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
                  duration: float = 8.0,
                  recipes: Sequence[ArtifactRecipe] = (),
                  media_dir: Path | str | None = None) -> Report:
    """Check one beat's layout. Never raises -- failures come back as findings.

    `recipes` are the beat's carried artifacts (SCENE_SPEC.md §6,
    continuity.resolve_carry_in): they are built on screen first, as in the
    render, so a beat is checked against the frame it will actually draw.
    `media_dir` is where Manim's scratch goes (see probe_media); never the cwd.
    """
    try:
        component = beat_component(spec, recipes)
    except Exception as exc:
        return Report(spec.id, [Finding("build_error", f"{type(exc).__name__}: {exc}")])

    # strict=False so the scene collects every finding instead of stopping at
    # the first. The repair loop wants the full picture in one pass.
    probe = LayoutProbe(component, theme=theme, duration=duration, strict=False,
                        media_dir=media_dir)
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


def validate_specs(specs: list[BeatSpec], theme: str = "default",
                   recipes: Mapping[str, Sequence[ArtifactRecipe]] | None = None,
                   media_dir: Path | str | None = None,
                   ) -> list[Report]:
    """validate_beat over `specs`; `recipes` is resolve_carry_in(video_spec)."""
    recipes = recipes or {}
    return [validate_beat(s, theme=theme, recipes=recipes.get(s.id, ()),
                          media_dir=media_dir) for s in specs]
