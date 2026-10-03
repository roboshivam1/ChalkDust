"""Render one beat to a known path.

Manim writes its output, partial movie files and text/LaTeX scratch under a
media dir, by name. We want beats in the cache, keyed by content (D-004), so
this drives the render inside a caller-given work dir, moves the output into
the cache, then deletes the partial movie files that produced it.

Every beat is built with its mechanical repair plan applied (SCENE_SPEC.md §9
step 1): the worker runs repair_beat() and renders a RepairedScene, which with
an empty plan is exactly a ChalkdustScene. The plan is recomputed from the
spec, not handed in, because beats render independently (D-005) and the plan
is a term of the cache key (D-004).

Two kinds of beat take a different path:

  - A beat that carries artifacts in (SCENE_SPEC.md §6) is built with
    continuity.beat_component(spec, recipes), so its carried artifacts are on
    screen -- in the repair probe and in the render alike -- and its key holds
    carry_in_fingerprint(recipes). render_video resolves the recipes once per
    video; render_beat takes them as an argument.
  - A RawScene beat (SCENE_SPEC.md §7) is routed to raw_scene.render_raw_beat
    before any repair or scene build: its code runs out of process, and any
    failure degrades it to BulletReveal instead of failing the beat.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from manim import tempconfig

from chalkdust.continuity import (
    ArtifactRecipe,
    beat_component,
    carry_in_fingerprint,
    resolve_carry_in,
)
from chalkdust.core.cache import Cache, beat_render_key
from chalkdust.core.models import Beat, BuildContext, Quality, Video
from chalkdust.scenes.components.raw_scene import RawScene, render_raw_beat
from chalkdust.scenes.theme import Theme, get_theme, resolve_fonts
from chalkdust.validate.repair import RepairedScene, RepairPlan, repair_beat


@dataclass(frozen=True)
class RenderTier:
    """Concrete output settings for one quality tier (D-006).

    The single definition of what "draft" and "final" mean: the worker renders
    with these and the cache key includes them, so retuning a tier can never
    serve clips made at the old settings. Field names are Manim config keys.
    """

    pixel_width: int
    pixel_height: int
    frame_rate: int


# Two-tier rendering (ARCHITECTURE.md §4): everything validates at 480p15, and
# 1080p60 runs only once every gate has passed.
TIERS: dict[Quality, RenderTier] = {
    Quality.DRAFT: RenderTier(pixel_width=854, pixel_height=480, frame_rate=15),
    Quality.FINAL: RenderTier(pixel_width=1920, pixel_height=1080, frame_rate=60),
}


def long_path(path: Path | str) -> Path:
    """`path` absolute, with every Windows 8.3 short name expanded.

    Manim hands its media dir to LaTeX, and TeX reads `~` in a path as an
    active character: under C:/Users/LOKAVY~1/... (the short form %TEMP%
    carries) every MathTex fails to compile (register N-3). os.path.realpath
    expands short names component by component, also for a path that does
    not exist yet; on macOS and Linux it only makes the path absolute and
    resolves symlinks. Every dir this module hands Manim goes through here.
    """
    return Path(os.path.realpath(path))


def _require_duration(beat: Beat) -> float:
    if beat.duration is None:
        raise ValueError(
            f"{beat.id} has no duration; the speech stage must run first"
        )
    return beat.duration


def plan_repair(beat: Beat, theme: Theme, ctx: BuildContext, work_dir: Path,
                recipes: Sequence[ArtifactRecipe] = ()) -> RepairPlan:
    """The mechanical repair plan this beat renders with; empty when the beat
    is clean, and also when repair cannot fix it -- the render's own strict
    settle checks then raise, as before (SCENE_SPEC.md §11 rule 1).

    The probe builds a Manim scene, which creates its media dirs, so it runs
    under `work_dir` like the render itself: nothing lands in the cwd, and
    LaTeX compiled here lands in the Tex/ cache the render then reuses.

    `recipes` are the beat's carried artifacts: the probe builds with them on
    screen, exactly as the render will, so fixes are planned against the
    frame that is actually drawn.
    """
    duration = _require_duration(beat)
    work_dir = long_path(work_dir)
    with tempconfig({**asdict(TIERS[ctx.quality]), "media_dir": str(work_dir)}):
        return repair_beat(beat.spec, theme, duration, recipes=recipes).plan


def render_key(beat: Beat, theme: Theme, ctx: BuildContext,
               plan: RepairPlan, recipes: Sequence[ArtifactRecipe] = ()) -> str:
    """The beat's cache key. `theme` must already be font-resolved -- the key
    has to describe the fonts that will actually be drawn, not the ones the
    theme asked for. `plan` is the one the render applies (plan_repair).

    `recipes` are the beat's resolved carry-ins (continuity.resolve_carry_in);
    a beat with a carry_in and no recipes raises, rather than keying a render
    that ignores what it carries."""
    return beat_render_key(
        beat.spec, _require_duration(beat), ctx, asdict(theme),
        asdict(TIERS[ctx.quality]), plan.key_data(),
        carried=carry_in_fingerprint(recipes) if recipes else None,
    )


def _render(beat: Beat, theme: Theme, ctx: BuildContext, cache: Cache,
            work_dir: Path, recipes: Sequence[ArtifactRecipe] = ()) -> tuple[Path, bool]:
    """Render under an already-resolved theme. Returns (path, rendered), where
    `rendered` is False on a cache hit.

    The same resolved theme object and the same repair plan feed both the key
    and the scene, so the key describes exactly what is drawn. The plan is
    computed even when the clip turns out to be cached: it is part of the key.
    """
    work_dir = long_path(work_dir)
    plan = plan_repair(beat, theme, ctx, work_dir, recipes)
    key = render_key(beat, theme, ctx, plan, recipes)
    slot = cache.slot("beats", key, ".mp4")
    rendered = not slot.exists

    if rendered:
        # Every beat's scene is named ChalkdustScene and, with Manim's cache
        # off, partials are named uncached_0000N.mp4 -- so each beat gets its
        # own video dir, or parallel workers would overwrite each other's
        # partials. media_dir itself stays shared: its texts/ and Tex/ are
        # Manim's own content-addressed text and LaTeX caches, worth keeping
        # across beats (ARCHITECTURE.md §3).
        beat_dir = work_dir / "videos" / f"beat_{key}"
        settings = {
            **asdict(TIERS[ctx.quality]),
            "media_dir": str(work_dir),
            "video_dir": str(beat_dir),
            "output_file": f"beat_{key}",
            "disable_caching": True,  # Manim's own cache duplicates ours
        }
        # tempconfig scopes these settings, so parallel workers later will not
        # stomp on each other's global config.
        with tempconfig(settings):
            scene = RepairedScene(
                beat_component(beat.spec, recipes),
                plan=plan,
                theme=theme,
                duration=beat.duration,  # type: ignore[arg-type]
            )
            scene.render()
            produced = Path(scene.renderer.file_writer.movie_file_path)

        # shutil.move, not Path.replace: the work dir is the caller's choice
        # and may sit on a different drive from the cache.
        shutil.move(produced, slot.tmp)
        slot.commit()
        # The clip is in the cache; its partial movie files are dead weight.
        # A failed render never gets here, so its partials stay for debugging.
        shutil.rmtree(beat_dir)

    beat.render_path = slot.path
    return slot.path, rendered


def render_beat(beat: Beat, theme: str, ctx: BuildContext, cache: Cache,
                work_dir: Path, recipes: Sequence[ArtifactRecipe] = ()) -> Path:
    """Render one beat, or return the cached file if it exists.

    Manim's scratch output goes under `work_dir`; nothing is written to the
    current directory. `recipes` are the beat's resolved carry-ins
    (continuity.resolve_carry_in(video_spec)[beat.id]); required for a beat
    with a carry_in. A RawScene beat goes to render_raw_beat, which takes the
    theme by name and degrades instead of raising.
    """
    work_dir = long_path(work_dir)
    if beat.spec.component == RawScene.name:
        return render_raw_beat(beat, theme, ctx, cache, work_dir)
    path, _ = _render(beat, resolve_fonts(get_theme(theme)), ctx, cache, work_dir,
                      recipes)
    return path


def render_video(video: Video, ctx: BuildContext, cache: Cache, work_dir: Path,
                 verbose: bool = True) -> Video:
    work_dir = long_path(work_dir)
    # Resolved once per video: one font check, one substitution warning, and
    # the same theme in every beat's key.
    theme = resolve_fonts(get_theme(video.spec.theme))
    # Once per video: which artifacts each beat carries, and how to rebuild
    # them (SCENE_SPEC.md §6). Raises CarryInError if a producer cannot.
    recipes = resolve_carry_in(video.spec)
    for beat in video.beats:
        if beat.spec.component == RawScene.name:
            render_raw_beat(beat, video.spec.theme, ctx, cache, work_dir)
            status = "degraded" if beat.degraded else "raw"
        else:
            _, rendered = _render(beat, theme, ctx, cache, work_dir, recipes[beat.id])
            status = "render" if rendered else "cached"
        if verbose:
            print(f"  [{status}] {beat.id}")
    return video
