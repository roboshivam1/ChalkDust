"""BoxFlow: behaviour the registry-wide layout tests do not pin.

test_layout.py already proves examples() validate clean and stress() fits or
refuses cleanly. This file pins timing against the audio budget, the schema's
refusals, the typed overflow path, and the routing guarantee that is the point
of the component: no edge ever runs through a box.
"""

from __future__ import annotations

import numpy as np
import pytest
from manim import tempconfig
from pydantic import ValidationError

from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.box_flow import MAX_EDGES, MAX_NODES, BoxFlow
from chalkdust.scenes.regions import LayoutError, bbox
from chalkdust.validate.geometric import LayoutProbe

PIPELINE, FAN, LOOP = BoxFlow.examples()
AT_CAPS, OVERLOADED, LONG_EDGES = BoxFlow.stress()[:3]
FPS = 15  # draft (D-006); the coarsest frame grid we render at


def _node(i: str, lab: str | None = None) -> dict:
    return {"id": i, "label": lab or i.upper()}


def _edge(s: str, t: str) -> dict:
    return {"source": s, "target": t}


TWO_WAY = {"nodes": [_node("a"), _node("b")],
           "edges": [_edge("a", "b"), _edge("b", "a")]}


class _Clocked(ChalkdustScene):
    """Records every run_time the component asks for, and what Manim then
    plays: the run_time it settled on and whether it froze the frame."""

    def play(self, *animations, **kwargs):  # type: ignore[override]
        # wait() reaches here too, carrying its duration on a Wait animation
        # rather than as run_time.
        self.asked.append(kwargs.get("run_time", animations[0].run_time))
        super().play(*animations, **kwargs)
        # Both are set by compile_animation_data / begin_animations, which
        # run under skip_animations too.
        self.played.append((self.duration,
                            self.is_current_animation_frozen_frame()))


def _frames(run_time: float, frozen: bool) -> int:
    """Frames Manim writes for one play, by its own arithmetic: a frozen wait
    is int(run_time / dt) frames (CairoRenderer.freeze_current_frame), any
    other play one frame per np.arange(0, run_time, dt) step
    (Scene.get_time_progression). Mirrored exactly rather than as
    ceil(run_time * fps), which would read a frozen hold one frame high."""
    dt = 1 / FPS
    if frozen:
        return int(run_time / dt)
    return len(np.arange(0, run_time, dt))


def _run(params: dict, budget: float, tmp_path) -> _Clocked:
    """Build with skip_animations: Manim then advances its clock by each
    play's run_time exactly -- after bumping any run_time shorter than one
    frame up to one frame, which is precisely the overrun to catch."""
    settings = {"media_dir": str(tmp_path), "frame_rate": FPS,
                "disable_caching": True, "verbosity": "WARNING"}
    with tempconfig(settings):
        scene = _Clocked(BoxFlow(params), duration=budget, skip_animations=True)
        scene.asked, scene.played = [], []
        scene.setup()
        scene.construct()
    return scene


def _probe(params: dict, tmp_path) -> LayoutProbe:
    with tempconfig({"media_dir": str(tmp_path)}):
        probe = LayoutProbe(BoxFlow(params), duration=8.0)
        probe.construct()
    return probe


class TestTiming:
    """Animation consumes exactly the beat's audio budget (D-002)."""

    @pytest.mark.parametrize("params", [PIPELINE, FAN, LOOP, AT_CAPS],
                             ids=["pipeline", "fan", "loop", "at-caps"])
    @pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
    def test_consumes_budget_exactly(self, params, factor, tmp_path):
        budget = factor * BoxFlow(params).min_seconds()
        scene = _run(params, budget, tmp_path)
        assert scene.renderer.time == pytest.approx(budget, abs=1 / FPS)

    @pytest.mark.parametrize("params", [PIPELINE, FAN, LOOP, AT_CAPS],
                             ids=["pipeline", "fan", "loop", "at-caps"])
    @pytest.mark.parametrize("factor", [0.5, 3.0], ids=["short", "long"])
    def test_renders_exactly_the_budget_in_frames(self, params, factor,
                                                  tmp_path):
        # The clock above sums float run_times and so matches by
        # construction; the video is whole frames per play. This is the
        # number the audio is muxed against (PRD G2).
        budget = factor * BoxFlow(params).min_seconds()
        played = _run(params, budget, tmp_path).played
        assert sum(_frames(t, frozen) for t, frozen in played)             == round(budget * FPS)

    def test_no_step_shorter_than_a_frame_at_half_budget(self, tmp_path):
        # The most step-heavy fixture (caps, flow on) at half its minimum:
        # if any step fell below a frame, Manim would silently lengthen it.
        budget = 0.5 * BoxFlow(AT_CAPS).min_seconds()
        assert min(_run(AT_CAPS, budget, tmp_path).asked) >= 1 / FPS

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
