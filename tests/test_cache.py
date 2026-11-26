"""Render cache key completeness (D-004).

An input missing from the key serves a stale clip without any error, so each
determining input gets a test that changes only it and expects a new key.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from manim import tempconfig

from chalkdust.core.cache import Cache
from chalkdust.core.models import Beat, BeatSpec, BuildContext, Quality
from chalkdust.render import worker
from chalkdust.render.worker import RenderTier, render_beat, render_key
from chalkdust.scenes import theme as theme_mod
from chalkdust.scenes.theme import DEFAULT, THEMES, get_theme, resolve_fonts
from chalkdust.validate import repair as repair_mod
from chalkdust.validate.repair import RepairPlan


@pytest.fixture(autouse=True)
def _quiet_manim():
    with tempconfig({"verbosity": "WARNING", "progress_bar": "none"}):
        yield


def _beat() -> Beat:
    beat = Beat(spec=BeatSpec(id="b01", narration="placeholder narration",
                              component="TitleCard", params={"title": "Cache key"}))
    beat.duration = 1.0
    return beat


def test_theme_content_changes_rendered_clip(tmp_path, monkeypatch):
    cache, work = Cache(tmp_path / "cache"), tmp_path / "work"
    ctx = BuildContext(quality=Quality.DRAFT)

    first = render_beat(_beat(), "default", ctx, cache, work)
    # Same theme NAME, different content: the key must see what the theme
    # draws, not what it is called.
    recoloured = replace(DEFAULT, palette=replace(DEFAULT.palette, bg="#FFFFFF"))
    monkeypatch.setitem(THEMES, "default", recoloured)
    second = render_beat(_beat(), "default", ctx, cache, work)

    assert second != first, "a different theme was served the cached clip"


def test_resolved_font_changes_render_key(monkeypatch):
    # Pango substitutes a missing font silently, so the same theme renders
    # differently on a machine without its fonts. The key must follow.
    ctx = BuildContext()
    monkeypatch.setattr(theme_mod, "_installed_fonts",
                        lambda: {"Archivo", "Inter", "JetBrains Mono"})
    as_designed = render_key(_beat(), resolve_fonts(get_theme("default")), ctx,
                             RepairPlan())
    monkeypatch.setattr(theme_mod, "_installed_fonts",
                        lambda: {"Arial", "Courier New"})
    substituted = render_key(_beat(), resolve_fonts(get_theme("default")), ctx,
                             RepairPlan())

    assert as_designed != substituted


def test_tier_settings_change_render_key(monkeypatch):
    # Retuning what "draft" means must not serve clips made at the old size.
    ctx = BuildContext(quality=Quality.DRAFT)
    theme = resolve_fonts(get_theme("default"), warn=False)
    before = render_key(_beat(), theme, ctx, RepairPlan())
    monkeypatch.setitem(worker.TIERS, Quality.DRAFT,
                        RenderTier(pixel_width=640, pixel_height=360, frame_rate=15))

    assert render_key(_beat(), theme, ctx, RepairPlan()) != before


def test_repair_logic_change_changes_render_key(tmp_path, monkeypatch, off_edge_beat):
    # The render applies a mechanical repair plan, so the plan determines the
    # pixels. Changing only the repair code -- here, how far inside its region
    # a nudge lands -- must not serve the clip drawn with the old plan.
    ctx = BuildContext(quality=Quality.DRAFT)
    theme = resolve_fonts(get_theme("default"), warn=False)
    work = tmp_path / "work"
    before = render_key(off_edge_beat, theme, ctx,
                        worker.plan_repair(off_edge_beat, theme, ctx, work))
    monkeypatch.setattr(repair_mod, "DEFAULT_PADDING", repair_mod.DEFAULT_PADDING + 0.5)
    after = render_key(off_edge_beat, theme, ctx,
                       worker.plan_repair(off_edge_beat, theme, ctx, work))

    assert after != before
