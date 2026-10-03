"""DataStructureViz: timing, schema refusals, final state, determinism.

Layout safety of examples() and stress() is covered by tests/test_layout.py,
which walks the registry. This file pins what that walk cannot see.
"""

from __future__ import annotations

import numpy as np
import pytest
from manim import CyclicReplace, Text, tempconfig
from pydantic import ValidationError

from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.data_structure_viz import SWAP_LIFT, DataStructureViz
from chalkdust.scenes.regions import LayoutError, bbox, safe_area
from chalkdust.scenes.theme import get_theme, resolve_fonts
from chalkdust.validate.geometric import LayoutProbe

FPS = 15
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
    return [m for top in scene.mobjects for m in top.get_family()
            if isinstance(m, Text) and m.text not in exclude]


# --- timing (D-002) -------------------------------------------------------------

# One case per kind, plus the 12-operation stress case where per-call frame
# drift would add up fastest.
TIMED = [pytest.param(p, id=p["kind"]) for p in EXAMPLES] + [
    pytest.param(DataStructureViz.stress()[1], id="array-12-ops"),
]


@pytest.mark.parametrize("factor", [0.5, 3.0])
@pytest.mark.parametrize("params", TIMED)
def test_rendered_length_matches_budget(params, factor, tmp_path):
    budget = factor * DataStructureViz(params).min_seconds()
    scene = _render(params, budget, tmp_path)
    assert abs(scene.renderer.time - budget) <= 1 / FPS


def test_budget_too_short_for_the_steps_refuses_cleanly(tmp_path):
    # 13 plays (intro + 12 ops) cannot each get a frame in 0.5s at 15fps.
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


def test_construction_is_deterministic():
    # Carry-in re-instantiates artifacts from params (SCENE_SPEC.md §6).
    for params in EXAMPLES:
        a = DataStructureViz(params).board(THEME).layout
        b = DataStructureViz(params).board(THEME).layout
        np.testing.assert_array_equal(a.get_all_points(), b.get_all_points())
