"""BoxFlow: behaviour the registry-wide layout tests do not pin.

test_layout.py already proves examples() validate clean and stress() fits or
refuses cleanly. This file pins timing against the audio budget (clocked and
in a real draft render), the schema's refusals, the typed overflow path, the
routing guarantee that is the point of the component -- no edge ever runs
through a box -- and the carry-in artifact (SCENE_SPEC.md §6).
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict

import numpy as np
import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.continuity import (
    ArtifactRecipe,
    beat_component,
    build_artifact,
    carried,
    resolve_carry_in,
)
from chalkdust.core.models import BeatSpec, Quality, VideoSpec
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.box_flow import MAX_EDGES, MAX_NODES, BoxFlow
from chalkdust.scenes.regions import LayoutError, bbox
from chalkdust.scenes.theme import DEFAULT
from chalkdust.validate.geometric import LayoutProbe, validate_beat

PIPELINE, FAN, LOOP = BoxFlow.examples()
AT_CAPS, OVERLOADED, LONG_EDGES = BoxFlow.stress()[:3]
DRAFT = TIERS[Quality.DRAFT]
FPS = DRAFT.frame_rate  # 15 (D-006): the coarsest frame grid we render at


def _node(i: str, lab: str | None = None) -> dict:
    return {"id": i, "label": lab or i.upper()}


def _edge(s: str, t: str) -> dict:
    return {"source": s, "target": t}


TWO_WAY = {"nodes": [_node("a"), _node("b")],
           "edges": [_edge("a", "b"), _edge("b", "a")]}


class _Clocked(ChalkdustScene):
    """Records the run_time the component asks for on every play() and
    wait(), and the run_time Manim then clocks for it."""

    def play(self, *animations, **kwargs):  # type: ignore[override]
        # wait() reaches here too, carrying its duration on a Wait animation
        # rather than as run_time.
        self.asked.append(kwargs.get("run_time", animations[0].run_time))
        super().play(*animations, **kwargs)
        # Set by ChalkdustScene.get_run_time, which runs under
        # skip_animations too: (n + 0.5) / fps for a play of n frames.
        self.clocked.append(self.duration)


def _run(params: dict, budget: float, tmp_path) -> _Clocked:
    """Build at draft fps with skip_animations: every play() and wait() goes
    through the same run-time arithmetic as a render, minus the encoding."""
    settings = {"media_dir": str(tmp_path), "frame_rate": FPS,
                "disable_caching": True, "verbosity": "WARNING"}
    with tempconfig(settings):
        scene = _Clocked(BoxFlow(params), duration=budget, skip_animations=True)
        scene.asked, scene.clocked = [], []
        scene.setup()
        scene.construct()
    return scene


def _probe(params: dict, tmp_path) -> LayoutProbe:
    with tempconfig({"media_dir": str(tmp_path)}):
        probe = LayoutProbe(BoxFlow(params), duration=8.0)
        probe.construct()
    return probe


TIMED = pytest.mark.parametrize("params", [PIPELINE, FAN, LOOP, AT_CAPS],
                                ids=["pipeline", "fan", "loop", "at-caps"])
# Narration far shorter and far longer than the animation wants.
FACTORS = pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])


class TestTiming:
    """Animation lasts exactly the beat's audio, in whole frames (D-002)."""

    @TIMED
    @FACTORS
    def test_asks_whole_frames_summing_to_the_beat(self, params, factor,
                                                   tmp_path):
        # Every run time is a scene.budget() share as given: whole frames,
        # together exactly beat_frames. A private rounding or snapping step
        # in the component would show here as fractional frames.
        scene = _run(params, factor * BoxFlow(params).min_seconds(), tmp_path)
        frames = [t * FPS for t in scene.asked]
        assert all(f == pytest.approx(round(f), abs=1e-9) for f in frames)
        assert sum(round(f) for f in frames) == scene.beat_frames

    @TIMED
    @FACTORS
    def test_clocks_exactly_the_beats_frames(self, params, factor, tmp_path):
        # What Manim plays: the base scene renders a clocked run time of
        # (n + 0.5) / fps as exactly n frames (get_time_progression).
        budget = factor * BoxFlow(params).min_seconds()
        scene = _run(params, budget, tmp_path)
        assert sum(int(t * FPS) for t in scene.clocked) == scene.beat_frames
        assert scene.beat_frames == math.ceil(round(budget * FPS, 6))

    def test_no_step_shorter_than_a_frame_at_half_budget(self, tmp_path):
        # The most step-heavy fixture (caps, flow on) at half its minimum:
        # if any step fell below a frame, Manim would silently lengthen it.
        budget = 0.5 * BoxFlow(AT_CAPS).min_seconds()
        assert min(_run(AT_CAPS, budget, tmp_path).asked) >= 1 / FPS

    def test_draft_render_is_exactly_the_beat(self, tmp_path):
        # A real 480p15 encode, frames counted by ffprobe: 3.879 s of audio
        # is ceil(3.879 * 15) = 59 frames, across AT_CAPS's 17 plays (the
        # most of any fixture, so the most chances to drift).
        with tempconfig({**asdict(DRAFT), "media_dir": str(tmp_path),
                         "disable_caching": True, "progress_bar": "none",
                         "verbosity": "WARNING", "output_file": "frames"}):
            scene = ChalkdustScene(BoxFlow(AT_CAPS), duration=3.879)
            scene.render()
            movie = scene.renderer.file_writer.movie_file_path
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
             "v:0", "-show_entries", "stream=nb_read_frames", "-of", "json",
             str(movie)], capture_output=True, text=True, check=True).stdout
        frames = int(json.loads(out)["streams"][0]["nb_read_frames"])
        assert frames == math.ceil(3.879 * FPS) == 59

    def test_flow_adds_time(self):
        still = BoxFlow({**FAN, "animate_flow": False}).min_seconds()
        assert BoxFlow(FAN).min_seconds() > still > 0

    def test_takes_no_latex(self):
        assert BoxFlow(LOOP).latex_strings() == []


class TestSchema:
    """Inputs the diagram cannot honour fail at validation, not at render."""

    @pytest.mark.parametrize("params", [
        {"nodes": []},
        {"nodes": [_node(f"n{i}") for i in range(MAX_NODES + 1)]},
        {"nodes": [_node(f"n{i}") for i in range(MAX_NODES)],
         "edges": [_edge(f"n{i}", f"n{j}") for i in range(MAX_NODES)
                   for j in range(MAX_NODES)][:MAX_EDGES + 1]},
        {"nodes": [_node("a"), _node("a", "Other")]},
        {"nodes": [{"id": "  ", "label": "A"}]},
        {"nodes": [{"id": "a", "label": ""}]},
        {"nodes": [_node("a")], "edges": [_edge("a", "ghost")]},
        {"nodes": [_node("a"), _node("b")],
         "edges": [_edge("a", "b"), _edge("a", "b")]},
        {"nodes": [{"id": "a", "label": "A", "colour": "red"}]},
        {"nodes": [_node("a")], "edges": [{**_edge("a", "a"), "weight": 2}]},
    ], ids=["no-nodes", "too-many-nodes", "too-many-edges", "duplicate-id",
            "blank-id", "blank-label", "unknown-endpoint", "duplicate-edge",
            "node-extra-field", "edge-extra-field"])
    def test_rejects(self, params):
        with pytest.raises(ValidationError):
            BoxFlow(params)

    def test_minimal_diagram_builds(self, tmp_path):
        probe = _probe({"nodes": [_node("x")]}, tmp_path)
        assert not probe.layout_warnings


class TestOverflow:
    def test_overloaded_diagram_raises_layout_error(self, tmp_path):
        with tempconfig({"media_dir": str(tmp_path)}):
            probe = LayoutProbe(BoxFlow(OVERLOADED), duration=8.0)
            with pytest.raises(LayoutError) as exc:
                probe.construct()
        assert exc.value.kind == "overflow"


def _parts(probe: LayoutProbe):
    by = {getattr(m, "_chalk_label", ""): m for m in probe.mobjects}
    boxes = {k[5:-1]: v[0] for k, v in by.items() if k.startswith("node[")}
    edges = [by[f"edge[{k}]"] for k in range(len(
        [k for k in by if k.startswith("edge[")]))]
    return boxes, edges


class TestRouting:
    """Edges never run through a box -- the property the layered layout exists
    to guarantee (see box_flow module docstring)."""

    @pytest.mark.parametrize("params", [PIPELINE, FAN, LOOP, AT_CAPS,
                                        LONG_EDGES, TWO_WAY],
                             ids=["pipeline", "fan", "loop", "at-caps",
                                  "long-edges", "two-way"])
    def test_no_edge_enters_a_box(self, params, tmp_path):
        boxes, edges = _parts(_probe(params, tmp_path))
        eps = 0.01
        for k, edge in enumerate(edges):
            pts = edge[0].get_anchors()
            for p0, p1 in zip(pts, pts[1:]):
                for s in np.linspace(0, 1, 41):
                    x, y = (p0 + s * (p1 - p0))[:2]
                    for name, box in boxes.items():
                        r = bbox(box)
                        inside = (r.left + eps < x < r.right - eps
                                  and r.bottom + eps < y < r.top - eps)
                        assert not inside, f"edge[{k}] passes through {name}"

    @pytest.mark.parametrize("params", [FAN, LOOP, LONG_EDGES, TWO_WAY],
                             ids=["fan", "loop", "long-edges", "two-way"])
    def test_every_arrow_lands_on_its_true_target(self, params, tmp_path):
        # Back edges route through the layers in reverse; the tip must still
        # sit on the box the spec named as target, not the one it left.
        boxes, edges = _parts(_probe(params, tmp_path))
        for k, e in enumerate(params["edges"]):
            apex = edges[k][1].get_vertices()[0]
            r = bbox(boxes[e["target"]])
            on_edge = (r.left - 1e-6 <= apex[0] <= r.right + 1e-6
                       and r.bottom - 1e-6 <= apex[1] <= r.top + 1e-6)
            assert on_edge, f"edge[{k}] tip misses {e['target']}"

    def test_two_way_pair_runs_parallel(self, tmp_path):
        _, (ab, ba) = _parts(_probe(TWO_WAY, tmp_path))
        ab_y = {round(p[1], 6) for p in ab[0].get_anchors()}
        ba_y = {round(p[1], 6) for p in ba[0].get_anchors()}
        assert not ab_y & ba_y


def _carry_video(params: dict) -> VideoSpec:
    """b01 draws `params` and registers it; b02 carries it in."""
    return VideoSpec(video_id="v", beats=(
        BeatSpec(id="b01", narration="placeholder narration",
                 component="BoxFlow", params=params, registers="system"),
        BeatSpec(id="b02", narration="placeholder narration",
                 component="BulletReveal", params={"items": ["one point"]},
                 carry_in=["system"]),
    ))


class TestCarryIn:
    """A later beat can carry the diagram in (SCENE_SPEC.md §6)."""

    def test_carried_diagram_is_the_picture_its_beat_settled_on(self,
                                                                tmp_path):
        video = _carry_video(FAN)
        recipes = resolve_carry_in(video)["b02"]
        with tempconfig({"media_dir": str(tmp_path)}):
            consumer = LayoutProbe(beat_component(video.beats[1], recipes),
                                   duration=4.0)
            consumer.construct()
        art = carried(consumer, "system")
        settled = {getattr(m, "_chalk_label", ""): m
                   for m in _probe(FAN, tmp_path).mobjects}
        parts = {getattr(m, "_chalk_label", ""): m for m in art.submobjects}
        names = [f"node[{n['id']}]" for n in FAN["nodes"]] + [
            f"edge[{k}]" for k in range(len(FAN["edges"]))]
        assert sorted(parts) == sorted(names)
        for name in names:
            a, b = bbox(parts[name]), bbox(settled[name])
            assert (a.x, a.y, a.width, a.height) == pytest.approx(
                (b.x, b.y, b.width, b.height), abs=1e-6), name

    def test_rebuild_is_deterministic(self):
        recipe = ArtifactRecipe(name="system", producer="BoxFlow", params=LOOP)
        a, b = build_artifact(recipe, DEFAULT), build_artifact(recipe, DEFAULT)
        pa = [m.points for m in a.family_members_with_points()]
        pb = [m.points for m in b.family_members_with_points()]
        assert len(pa) == len(pb) > 0
        assert all(np.array_equal(x, y) for x, y in zip(pa, pb))

    def test_carry_in_beat_validates_clean(self):
        video = _carry_video(LOOP)
        report = validate_beat(video.beats[1], duration=4.0,
                               recipes=resolve_carry_in(video)["b02"])
        assert report.ok, f"\n{report}"
