"""DataStructureViz: timing, schema refusals, final state, determinism, carry-in.

Layout safety of examples() and stress() is covered by tests/test_layout.py,
which walks the registry. This file pins what that walk cannot see.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict

import numpy as np
import pytest
from manim import CyclicReplace, ManimColor, Text, VMobject, tempconfig
from pydantic import ValidationError

from chalkdust import continuity
from chalkdust.continuity import ArtifactRecipe, beat_component, carried, resolve_carry_in
from chalkdust.core.models import BeatSpec, Quality, Region, VideoSpec
from chalkdust.render.worker import TIERS
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.data_structure_viz import (
    BOARDS,
    EDGE_CLEAR,
    SWAP_LIFT,
    DataStructureViz,
    _plan,
)
from chalkdust.scenes.regions import LayoutError, bbox, region_rect, safe_area
from chalkdust.scenes.theme import get_theme, resolve_fonts
from chalkdust.validate.geometric import LayoutProbe, validate_beat

DRAFT = TIERS[Quality.DRAFT]
FPS = DRAFT.frame_rate
EXAMPLES = DataStructureViz.examples()
THEME = resolve_fonts(get_theme("default"), warn=False)


def _render(params: dict, duration: float, tmp_path) -> ChalkdustScene:
    """Build through a real ChalkdustScene, rasterising every frame but
    writing no movie, so renderer.time is the length the video would have."""
    with tempconfig({"pixel_width": 160, "pixel_height": 90, "frame_rate": FPS,
                     "write_to_movie": False, "disable_caching": True,
                     "progress_bar": "none", "verbosity": "WARNING",
                     "media_dir": str(tmp_path)}):
        scene = ChalkdustScene(DataStructureViz(params), duration=duration)
        scene.setup()
        scene.construct()
    return scene


def _probe(params: dict) -> LayoutProbe:
    scene = LayoutProbe(DataStructureViz(params), duration=8.0)
    scene.construct()
    return scene


def _values(scene, exclude: set[str] = frozenset()) -> list[Text]:
    """Text mobjects on screen at the end of the beat."""
    return _texts(scene.mobjects, exclude)


def _texts(mobs, exclude: set[str] = frozenset()) -> list[Text]:
    return [m for top in mobs for m in top.get_family()
            if isinstance(m, Text) and m.text not in exclude]


def _frames(scene: ChalkdustScene) -> int:
    """Frames the renderer wrote: add_frame advances renderer.time by exactly
    1/fps per frame, for plays and frozen waits alike."""
    return round(scene.renderer.time * FPS)


# --- timing (D-002) -------------------------------------------------------------

# One case per kind, plus the 12-operation stress case where per-call frame
# drift would add up fastest.
TIMED = [pytest.param(p, id=p["kind"]) for p in EXAMPLES] + [
    pytest.param(DataStructureViz.stress()[1], id="array-12-ops"),
]


@pytest.mark.parametrize("factor", [0.5, 3.0])
@pytest.mark.parametrize("params", TIMED)
def test_rendered_frames_equal_the_beat(params, factor, tmp_path):
    # 0.5x squeezes every step; 3x is past ANIM_STRETCH, so pauses appear.
    # Either way the clip is exactly ceil(audio * fps) frames (D-002).
    budget = factor * DataStructureViz(params).min_seconds()
    scene = _render(params, budget, tmp_path)
    assert _frames(scene) == scene.beat_frames == math.ceil(round(budget * FPS, 6))


def test_draft_render_frame_count_matches_audio(tmp_path):
    # A real 480p15 encode, counted by ffprobe: 11.27 s of audio is 170 frames.
    duration = 11.27
    with tempconfig({**asdict(DRAFT), "media_dir": str(tmp_path),
                     "disable_caching": True, "progress_bar": "none",
                     "verbosity": "WARNING", "output_file": "dsv"}):
        scene = ChalkdustScene(DataStructureViz(EXAMPLES[0]), duration=duration)
        scene.render()
        movie = scene.renderer.file_writer.movie_file_path
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(movie)],
        capture_output=True, text=True, check=True).stdout
    frames = int(json.loads(out)["streams"][0]["nb_read_frames"])
    assert frames == math.ceil(duration * FPS) == 170


def test_budget_too_short_for_the_steps_refuses_cleanly(tmp_path):
    # 14 segments (intro + 12 ops + hold) cannot each get a frame in 0.5 s
    # (8 frames) at 15 fps.
    with pytest.raises(LayoutError) as exc:
        _render(DataStructureViz.stress()[1], 0.5, tmp_path)
    assert exc.value.kind == "overflow"


def test_min_seconds_is_the_sum_of_step_minimums():
    # intro 1.0 + highlight 0.6 + swap 1.0 + highlight 0.6 + set 0.8 + hold 0.5
    assert DataStructureViz(EXAMPLES[0]).min_seconds() == pytest.approx(4.5)


def test_no_latex():
    for params in EXAMPLES:
        assert DataStructureViz(params).latex_strings() == []


# --- schema refusals ------------------------------------------------------------

REJECTED = {
    "array-empty": {"kind": "array", "initial": []},
    "array-over-cap": {"kind": "array", "initial": list(range(13))},
    "array-null": {"kind": "array", "initial": [1, None]},
    "blank-value": {"kind": "array", "initial": ["  "]},
    # A value is one line: a box sized for one line cannot hold more.
    "newline-in-cell": {"kind": "array", "initial": ["a\nb\nc\nd", "x"]},
    "newline-in-node": {"kind": "tree", "initial": ["top\nmid\nbot", 1, 2]},
    "swap-out-of-range": {"kind": "array", "initial": [1, 2],
                          "operations": [{"op": "swap", "at": [0, 2]}]},
    "swap-with-self": {"kind": "array", "initial": [1, 2],
                       "operations": [{"op": "swap", "at": [1, 1]}]},
    "string-index": {"kind": "array", "initial": [1, 2],
                     "operations": [{"op": "highlight", "at": ["0"]}]},
    "op-wrong-kind": {"kind": "stack", "initial": [1],
                      "operations": [{"op": "swap", "at": [0, 0]}]},
    "op-unknown-field": {"kind": "array", "initial": [1, 2],
                         "operations": [{"op": "swap", "at": [0, 1], "index": 0}]},
    "pop-empty": {"kind": "stack", "initial": [], "operations": [{"op": "pop"}]},
    "push-over-cap": {"kind": "stack", "initial": list(range(8)),
                      "operations": [{"op": "push", "value": 9}]},
    "highlight-popped": {"kind": "stack", "initial": [1, 2],
                         "operations": [{"op": "pop"}, {"op": "highlight", "at": [1]}]},
    "tree-null-root": {"kind": "tree", "initial": [None, 1]},
    "tree-orphan": {"kind": "tree", "initial": [1, None, 2, 3]},
    "tree-duplicate": {"kind": "tree", "initial": [1, 2, "2"]},
    "tree-occupied": {"kind": "tree", "initial": [1, 2],
                      "operations": [{"op": "insert", "parent": 1, "side": "left",
                                      "value": 3}]},
    "tree-too-deep": {"kind": "tree", "initial": list(range(32))},
    "tree-not-adjacent": {"kind": "tree", "initial": [1, 2, 3, 4],
                          "operations": [{"op": "traverse", "edge": [1, 4]}]},
    "graph-as-list": {"kind": "graph", "initial": ["A", "B"]},
    "array-as-graph": {"kind": "array", "initial": {"nodes": ["A"]}},
    "graph-unknown-endpoint": {"kind": "graph",
                               "initial": {"nodes": ["A"], "edges": [["A", "B"]]}},
    "graph-self-loop": {"kind": "graph", "initial": {"nodes": ["A"], "edges": [["A", "A"]]}},
    "graph-duplicate-edge": {"kind": "graph",
                             "initial": {"nodes": ["A", "B"],
                                         "edges": [["A", "B"], ["B", "A"]]}},
    "graph-missing-edge": {"kind": "graph",
                           "initial": {"nodes": ["A", "B", "C"], "edges": [["A", "B"]]},
                           "operations": [{"op": "traverse", "edge": ["A", "C"]}]},
}


@pytest.mark.parametrize("params", REJECTED.values(), ids=REJECTED.keys())
def test_schema_rejects(params):
    with pytest.raises(ValidationError):
        DataStructureViz(params)


def test_insert_below_deepest_level_rejected():
    spine = [{"op": "insert", "parent": i, "side": "left", "value": i + 1}
             for i in range(1, 6)]
    with pytest.raises(ValidationError, match="levels"):
        DataStructureViz({"kind": "tree", "initial": [1], "operations": spine})


# --- empty and minimal ------------------------------------------------------------


def test_empty_stack_shows_only_its_container():
    scene = _probe({"kind": "stack", "initial": []})
    assert not scene.layout_warnings
    assert _values(scene) == []


def test_unwrappable_token_refuses_as_overflow():
    with pytest.raises(LayoutError) as exc:
        _probe(DataStructureViz.stress()[5])
    assert exc.value.kind == "overflow"


# --- final state matches the simulated operations --------------------------------


def test_array_swap_and_set_leave_expected_values():
    scene = _probe(EXAMPLES[0])  # [29,10,14,37,13] swap 0,1 then a[4]=99
    cells = sorted(_values(scene, exclude={str(i) for i in range(5)}),
                   key=lambda m: m.get_x())
    assert [m.text for m in cells] == ["10", "29", "14", "37", "99"]


def test_stack_pop_removes_the_top():
    scene = _probe(EXAMPLES[1])  # main, push fib(3), push fib(2), pop
    stack = sorted(_values(scene), key=lambda m: m.get_y())
    assert [m.text for m in stack] == ["main", "fib(3)"]


def test_tree_insert_lands_right_of_and_below_its_parent():
    scene = _probe(EXAMPLES[2])  # insert 6 as right child of 3
    at = {m.text: m.get_center() for m in _values(scene)}
    assert at["6"][0] > at["3"][0] and at["6"][1] < at["3"][1]
    assert at["6"][0] < at["8"][0]  # still in 8's left subtree


# --- geometry the probe cannot see ------------------------------------------------


def test_long_swap_arc_stays_low_and_in_frame():
    # Swap across all 12 cells. The probe jumps straight to the end state, so
    # mid-flight height is only bounded by the arc rule; check it directly.
    comp = DataStructureViz(DataStructureViz.stress()[1])
    board = comp.board(THEME)
    row_y = board.cells[0].get_y()
    swap = next(a for a in board.animations(0, comp.params.operations[0])
                if isinstance(a, CyclicReplace))
    swap.begin()
    swap.interpolate(0.5)
    for mob in swap.mobject:
        assert safe_area().contains(bbox(mob))
        assert abs(mob.get_y() - row_y) <= SWAP_LIFT * board.cell_h * 1.05


def _assert_edges_clear(board, r: float, cap: float) -> None:
    """Every edge clears every node it does not join by EDGE_CLEAR cap heights
    beyond the rim; `r` and `cap` at the board's current scale."""
    for key, edge in board.edges.items():
        a, b = edge.get_start_and_end()
        for name, node in board.nodes.items():
            if name in key:
                continue
            p = node.get_center()
            t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
            gap = np.linalg.norm(p - (a + t * (b - a))) - r
            assert gap >= EDGE_CLEAR * cap * 0.99, (sorted(key), name, gap / cap)


@pytest.mark.parametrize("n", [8, 9, 10])
def test_graph_edges_clear_the_nodes_they_skip(n):
    # Skip-one edges on an n-node ring of one-digit labels. Unenlarged, each
    # chord passes inside r + EDGE_CLEAR cap heights of the node it skips
    # (about 1.09r from its centre at n = 10) -- grazing its rim, so the
    # skipped node reads as joined. The ring must grow until every edge clears
    # every node it does not join by EDGE_CLEAR cap heights beyond the rim.
    # That geometry is in cap heights, so it is the same under every font.
    names = [str(i) for i in range(n)]
    params = {"kind": "graph", "initial": {
        "nodes": names, "edges": [[names[i], names[(i + 2) % n]] for i in range(n)]}}
    grown = BOARDS["graph"](THEME, _plan(DataStructureViz(params).params), [])
    assert grown.grown
    _assert_edges_clear(grown, grown.r, grown.cap)
    # Fitting it to STAGE either keeps that clearance in proportion or refuses
    # as overflow, naming the enlarged ring. Whether it fits depends on the
    # mono font's cap height per point: the grown ring's size is fixed in cap
    # heights, and the theme's JetBrains Mono has a 28% taller cap than the
    # fallback Courier New, so its 9- and 10-node rings would render at 21.4pt
    # and 17.4pt, under the 22pt floor, while Courier New's still clear it.
    # The 8-node ring fits under both, and must.
    try:
        board = DataStructureViz(params).board(THEME)
    except LayoutError as exc:
        assert n > 8, exc
        assert exc.kind == "overflow"
        assert "graph ring was enlarged" in str(exc)
        return
    r = board.nodes["0"][0].width / 2          # radius after fitting to the stage
    _assert_edges_clear(board, r, board.cap * r / board.r)


def test_construction_is_deterministic():
    # Carry-in re-instantiates artifacts from params (SCENE_SPEC.md §6).
    for params in EXAMPLES:
        a = DataStructureViz(params).board(THEME).layout
        b = DataStructureViz(params).board(THEME).layout
        np.testing.assert_array_equal(a.get_all_points(), b.get_all_points())


# --- carry-in (SCENE_SPEC.md §6) ---------------------------------------------------

# The spec's own example: a hash table's buckets, "cat" hashed into bucket 4.
BUCKETS = {"kind": "array", "initial": ["-"] * 8,
           "operations": [{"op": "highlight", "at": [4]},
                          {"op": "set", "at": 4, "value": "cat"}]}


def _bucket_video(consumer: str) -> VideoSpec:
    """b02 registers "bucket_array"; b03, the `consumer` (the hold_consumer
    fixture's), carries it in, as in SCENE_SPEC.md §6. A beat that carries an
    artifact in without acting on it must leave it a free STAGE region
    (D-G4c-1)."""
    return VideoSpec(video_id="v", beats=(
        BeatSpec(id="b01", narration="placeholder narration", component="TitleCard",
                 params={"title": "Hash tables"}),
        BeatSpec(id="b02", narration="placeholder narration",
                 component="DataStructureViz", params=BUCKETS,
                 registers="bucket_array"),
        BeatSpec(id="b03", narration="placeholder narration", component=consumer,
                 params={"target_id": "bucket_array"},
                 carry_in=["bucket_array"]),
    ))


def _artifact(params: dict):
    recipe = ArtifactRecipe(name="a", producer="DataStructureViz", params=params)
    return continuity.build_artifact(recipe, THEME)


def test_registered_artifact_is_carried_into_a_later_beat(hold_consumer):
    video = _bucket_video(hold_consumer)
    recipes = resolve_carry_in(video)
    assert recipes["b03"] == (ArtifactRecipe(name="bucket_array",
                                             producer="DataStructureViz",
                                             params=BUCKETS),)
    assert validate_beat(video.beats[2], duration=4.0, recipes=recipes["b03"]).ok

    probe = LayoutProbe(beat_component(video.beats[2], recipes["b03"]),
                        duration=4.0)
    probe.construct()
    target = carried(probe, "bucket_array")
    assert target in probe.mobjects
    assert region_rect(Region.STAGE).contains(bbox(target))
    # Dimmed: the values (opaque text) are faded by DIM_DARKNESS.
    assert max(m.get_fill_opacity() for m in _texts([target])) == \
        pytest.approx(1 - continuity.DIM_DARKNESS)


# What each example leaves on screen once its operations have run.
FINAL = [
    ["10", "29", "14", "37", "99"],    # swap 0,1 then a[4] = 99
    ["main", "fib(3)"],                # push, push, pop
    ["1", "3", "6", "8", "10", "14"],  # 6 inserted
    ["A", "B", "C", "D", "E"],         # graph ops only change emphasis
]


@pytest.mark.parametrize("params,expected", list(zip(EXAMPLES, FINAL)),
                         ids=[p["kind"] for p in EXAMPLES])
def test_artifact_is_the_final_state(params, expected):
    art = _artifact(params)
    if params["kind"] == "array":
        # Index captions are not values; order cells left to right.
        values = sorted(_texts([art], exclude={str(i) for i in range(5)}),
                        key=lambda m: m.get_x())
        assert [m.text for m in values] == expected
    elif params["kind"] == "stack":
        values = sorted(_texts([art]), key=lambda m: m.get_y())  # bottom first
        assert [m.text for m in values] == expected
    else:
        assert sorted(m.text for m in _texts([art])) == sorted(expected)


def test_artifact_carries_no_focus_styling():
    # Emphasis belongs to the beat that played the operations, not to the
    # later beat carrying the structure in.
    accent = ManimColor(THEME.palette.accent).to_hex()
    for params in EXAMPLES:
        shapes = [m for m in _artifact(params).get_family()
                  if isinstance(m, VMobject) and not isinstance(m, Text)
                  and m.has_points()]
        assert shapes
        assert all(m.get_stroke_color().to_hex() != accent for m in shapes)


def test_artifact_rebuild_is_deterministic():
    for params in EXAMPLES:
        a, b = _artifact(params), _artifact(params)
        np.testing.assert_array_equal(a.get_all_points(), b.get_all_points())
