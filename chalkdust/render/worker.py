"""Render one beat to a known path.

Manim writes its output, partial movie files and text/LaTeX scratch under a
media dir, by name. We want beats in the cache, keyed by content (D-004), so
this drives the render inside a caller-given work dir, moves the output into
the cache, then deletes the partial movie files that produced it.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from manim import tempconfig

from chalkdust.core.cache import Cache, beat_render_key
from chalkdust.core.models import Beat, BuildContext, Quality, Video
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import make_component
from chalkdust.scenes.theme import Theme, get_theme, resolve_fonts


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


def render_key(beat: Beat, theme: Theme, ctx: BuildContext) -> str:
    """The beat's cache key. `theme` must already be font-resolved -- the key
    has to describe the fonts that will actually be drawn, not the ones the
    theme asked for."""
    if beat.duration is None:
        raise ValueError(
            f"{beat.id} has no duration; the speech stage must run first"
        )
    return beat_render_key(
        beat.spec, beat.duration, ctx, asdict(theme), asdict(TIERS[ctx.quality])
    )


def _render(beat: Beat, theme: Theme, ctx: BuildContext, cache: Cache,
            work_dir: Path) -> tuple[Path, bool]:
    """Render under an already-resolved theme. Returns (path, rendered), where
    `rendered` is False on a cache hit.

    The same resolved theme object feeds both the key and the scene, so the
    key describes exactly what is drawn.
    """
    key = render_key(beat, theme, ctx)
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
            scene = ChalkdustScene(
                make_component(beat.spec.component, beat.spec.params),
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
                work_dir: Path) -> Path:
    """Render one beat, or return the cached file if it exists.

    Manim's scratch output goes under `work_dir`; nothing is written to the
    current directory.
    """
    path, _ = _render(beat, resolve_fonts(get_theme(theme)), ctx, cache, work_dir)
    return path


def render_video(video: Video, ctx: BuildContext, cache: Cache, work_dir: Path,
                 verbose: bool = True) -> Video:
    # Resolved once per video: one font check, one substitution warning, and
    # the same theme in every beat's key.
    theme = resolve_fonts(get_theme(video.spec.theme))
    for beat in video.beats:
        _, rendered = _render(beat, theme, ctx, cache, work_dir)
        if verbose:
            print(f"  [{'render' if rendered else 'cached'}] {beat.id}")
    return video
