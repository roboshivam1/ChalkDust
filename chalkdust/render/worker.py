"""Render one beat to a known path.

Manim writes into media/videos/<quality>/ by name. We want beats in the cache,
keyed by content (D-004), so this drives the render then moves the output.
"""

from __future__ import annotations

from pathlib import Path

from manim import config, tempconfig

from chalkdust.core.cache import Cache, beat_render_key
from chalkdust.core.models import Beat, BuildContext, Quality, Video
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components import make_component

QUALITY_FLAGS = {
    Quality.DRAFT: {"pixel_width": 854, "pixel_height": 480, "frame_rate": 15},
    Quality.FINAL: {"pixel_width": 1920, "pixel_height": 1080, "frame_rate": 60},
}


def render_beat(beat: Beat, theme: str, ctx: BuildContext, cache: Cache) -> Path:
    """Render one beat, or return the cached file if it exists."""
    if beat.duration is None:
        raise ValueError(
            f"{beat.id} has no duration; the speech stage must run first"
        )

    key = beat_render_key(beat.spec, beat.duration, ctx)
    slot = cache.slot("beats", key, ".mp4")

    if not slot.exists:
        settings = {
            **QUALITY_FLAGS[ctx.quality],
            "output_file": f"beat_{key}",
            "disable_caching": True,  # Manim's own cache duplicates ours
        }
        # tempconfig scopes these settings, so parallel workers later will not
        # stomp on each other's global config.
        with tempconfig(settings):
            scene = ChalkdustScene(
                make_component(beat.spec.component, beat.spec.params),
                theme=theme,
                duration=beat.duration,
            )
            scene.render()
            produced = Path(scene.renderer.file_writer.movie_file_path)

        produced.replace(slot.tmp)
        slot.commit()

    beat.render_path = slot.path
    return slot.path


def render_video(video: Video, ctx: BuildContext, cache: Cache,
                 verbose: bool = True) -> Video:
    for beat in video.beats:
        key = beat_render_key(beat.spec, beat.duration, ctx)  # type: ignore[arg-type]
        was_cached = cache.slot("beats", key, ".mp4").exists
        render_beat(beat, video.spec.theme, ctx, cache)
        if verbose:
            print(f"  [{'cached' if was_cached else 'render'}] {beat.id}")
    return video
