"""DataStructureViz: an array, stack, binary tree or graph, plus operations on it.

The whole beat -- every value a `set` will write, every cell a `push` will add,
every node an `insert` will grow -- is laid out once, up front, from params
alone, and fitted to the stage as a single group. Two consequences:

  * Nothing reflows mid-beat. A later push never shrinks the stack that is
    already on screen, and the frame the viewer settles on is the one that was
    checked against the legibility floor.
  * Construction is a pure function of params and theme, which is what lets a
    later beat re-instantiate this structure as a carry-in artifact rather than
    serialising mobjects (SCENE_SPEC.md §6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Annotated, Literal

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Animation,
    Circle,
    CyclicReplace,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    VGroup,
    Mobject,
    VMobject,
)
from pydantic import Field, StringConstraints, model_validator

from chalkdust.continuity import artifact_builder
from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    Component,
    ComponentParams,
    label,
    register,
)
from chalkdust.scenes.regions import LayoutError, fit_to_region
from chalkdust.scenes.theme import Theme, caption_text, mono_text

# --- density caps -------------------------------------------------------------
# Enforced at schema validation, like BulletReveal's item cap: cheaper than the
# legibility check, and the error names the real fix (split the beat) instead
# of a font size. Each cap is the most that fits the stage UNSCALED with
# realistic (two- to three-character) values; longer values still reach
# fit_to_region and refuse there.
MAX_ARRAY = 12        # index captions sit at 24pt, so arrays tolerate almost no scaling
MAX_STACK = 8
MAX_TREE_LEVELS = 5   # 16 slots on the bottom level
MAX_GRAPH_NODES = 10
MAX_GRAPH_EDGES = 20
MAX_OPERATIONS = 12

# --- geometry, in mono cap heights so it scales with the theme ---------------
CELL_PAD = 0.8        # space between a value and its cell wall
NODE_GAP = 1.0        # horizontal gap between bottom-level tree slots
LEVEL_GAP = 2.0       # vertical gap between tree levels
GRAPH_GAP = 2.0       # minimum gap between neighbouring graph nodes
GRAPH_ASPECT = 1.6    # the stage is wide; spread the ring to use it
STACK_WALL_GAP = 0.4  # air between the stacked cells and the container
SWAP_LIFT = 0.8       # peak arc height of a swap, in cell heights
INDEX_GAP = 0.6       # cell bottom to index caption

STROKE = 3.0          # stroke width for cells, nodes and edges
FOCUS_FILL = 0.25     # fill opacity of the accent wash on focused items

# --- timing (D-002) -----------------------------------------------------------
# Minimum seconds at which each step stays legible. These are also the budget
# weights, so at exactly min_seconds() every step gets exactly its minimum.
INTRO_MIN = 1.0
HOLD_MIN = 0.5
OP_MIN = {
    "highlight": 0.6,
    "swap": 1.0,
    "set": 0.8,
    "push": 0.8,
    "pop": 0.8,
    "insert": 1.0,
    "traverse": 0.8,
}
# An animation never runs longer than this multiple of its minimum. Narration
# beyond that becomes a pause on the settled frame after each step while the
# narrator talks over it -- a six-second swap reads as broken, not as calm.
ANIM_STRETCH = 2.0

KINDS_OPS = {
    "array": {"highlight", "swap", "set"},
    "stack": {"highlight", "push", "pop"},
    "tree": {"highlight", "insert", "traverse"},
    "graph": {"highlight", "traverse"},
}


# --- params -------------------------------------------------------------------

# A value shown in a cell or a node. Empty and whitespace-only strings are
# refused: they have no glyphs, so Manim cannot position them, and a cell that
# should read "empty" is clearer with an explicit "-" or "null" anyway.
Key = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] | int


class Highlight(ComponentParams):
    """Focus items: indices for array/stack (stack counts from the bottom),
    node labels for tree/graph."""

    op: Literal["highlight"]
    at: list[Key] = Field(min_length=1)


class Swap(ComponentParams):
    op: Literal["swap"]
    at: tuple[int, int]


class SetValue(ComponentParams):
    op: Literal["set"]
    at: int
    value: Key


class Push(ComponentParams):
    op: Literal["push"]
    value: Key


class Pop(ComponentParams):
    op: Literal["pop"]


class Insert(ComponentParams):
    """Add a child under an existing tree node. Explicit parent and side rather
    than BST insertion, so expression trees and heaps work too."""

    op: Literal["insert"]
    parent: Key
    side: Literal["left", "right"]
    value: Key


class Traverse(ComponentParams):
    """Walk an existing edge: it stays lit as a trail, the far node takes focus."""

    op: Literal["traverse"]
    edge: tuple[Key, Key]


Operation = Annotated[
    Highlight | Swap | SetValue | Push | Pop | Insert | Traverse,
    Field(discriminator="op"),
]


class GraphInitial(ComponentParams):
    # Nodes sit on a ring in list order, clockwise from the top. Listing them
    # so that most edges join ring neighbours keeps edges from crossing.
    nodes: list[Key] = Field(min_length=1, max_length=MAX_GRAPH_NODES)
    # Undirected. Traversal order is carried by the operations, not the edges.
    edges: list[tuple[Key, Key]] = Field(default_factory=list,
                                         max_length=MAX_GRAPH_EDGES)


class DataStructureVizParams(ComponentParams):
    kind: Literal["array", "stack", "tree", "graph"]
    # array/stack: values, bottom of the stack first. tree: level order with
    # null holes (LeetCode style). graph: {"nodes": [...], "edges": [[u, v]]}.
    initial: GraphInitial | list[Key | None]
    operations: list[Operation] = Field(default_factory=list,
                                        max_length=MAX_OPERATIONS)

    @model_validator(mode="after")
    def _operations_are_possible(self) -> DataStructureVizParams:
        # Simulating here turns "pop an empty stack" into a schema error the
        # repair loop can hand back to the model, instead of a build crash.
        _plan(self)
        return self


# --- the plan: a pure simulation of params ------------------------------------


@dataclass
class _Plan:
    """What the structure will ever contain, derived from params alone."""

    values: list[str] = field(default_factory=list)   # array cells / stack, initial
    sized: list[str] = field(default_factory=list)    # every string ever displayed
    max_depth: int = 0                                 # stack
    slots: dict[int, str] = field(default_factory=dict)  # tree: level-order slot -> label
    levels: int = 0                                    # tree
    nodes: list[str] = field(default_factory=list)     # graph
    edges: list[tuple[str, str]] = field(default_factory=list)


def _s(k: str | int) -> str:
    return str(k)


def _index(ref: str | int, size: int, what: str) -> int:
    if not isinstance(ref, int) or isinstance(ref, bool):
        raise ValueError(f"{what}: index must be an integer, got {ref!r}")
    if not 0 <= ref < size:
        raise ValueError(f"{what}: index {ref} out of range for {size} item(s)")
    return ref


def _depth(slot: int) -> int:
    return int(math.log2(slot + 1))


def _plan(p: DataStructureVizParams) -> _Plan:
    """Validate operations against the structure and record its full extent.

    Raises ValueError (surfacing as a pydantic ValidationError) on anything
    impossible: an op the kind does not support, an index out of range, a pop
    from an empty stack, an insert into an occupied slot, a walk along a
    missing edge.
    """
    plan = _Plan()
    graph = isinstance(p.initial, GraphInitial)
    if graph != (p.kind == "graph"):
        raise ValueError(
            "initial must be {'nodes': [...], 'edges': [...]} for a graph "
            "and a list for every other kind"
        )
    for i, op in enumerate(p.operations):
        if op.op not in KINDS_OPS[p.kind]:
            raise ValueError(
                f"operations[{i}]: {op.op!r} is not an operation on a {p.kind}; "
                f"allowed: {sorted(KINDS_OPS[p.kind])}"
            )

    if p.kind in ("array", "stack"):
        if any(v is None for v in p.initial):
            raise ValueError(f"null is only meaningful in a tree, not a {p.kind}")
        plan.values = [_s(v) for v in p.initial]
        plan.sized = list(plan.values)
        if p.kind == "array":
            _plan_array(p, plan)
        else:
            _plan_stack(p, plan)
    elif p.kind == "tree":
        _plan_tree(p, plan)
    else:
        _plan_graph(p, plan)
    return plan


def _plan_array(p: DataStructureVizParams, plan: _Plan) -> None:
    n = len(plan.values)
    if not 1 <= n <= MAX_ARRAY:
        raise ValueError(f"an array needs 1 to {MAX_ARRAY} values, got {n}")
    for i, op in enumerate(p.operations):
        what = f"operations[{i}] {op.op}"
        if isinstance(op, Highlight):
            for ref in op.at:
                _index(ref, n, what)
        elif isinstance(op, Swap):
            a, b = (_index(r, n, what) for r in op.at)
            if a == b:
                raise ValueError(f"{what}: cannot swap index {a} with itself")
        else:  # SetValue
            _index(op.at, n, what)
            plan.sized.append(_s(op.value))


def _plan_stack(p: DataStructureVizParams, plan: _Plan) -> None:
    depth = len(plan.values)
    if depth > MAX_STACK:
        raise ValueError(f"a stack holds at most {MAX_STACK} values, got {depth}")
    plan.max_depth = depth
    for i, op in enumerate(p.operations):
        what = f"operations[{i}] {op.op}"
        if isinstance(op, Push):
            depth += 1
            if depth > MAX_STACK:
                raise ValueError(f"{what}: stack would exceed {MAX_STACK} values")
            plan.sized.append(_s(op.value))
            plan.max_depth = max(plan.max_depth, depth)
        elif isinstance(op, Pop):
            if depth == 0:
                raise ValueError(f"{what}: the stack is empty")
            depth -= 1
        else:  # Highlight
            for ref in op.at:
                _index(ref, depth, what)


def _plan_tree(p: DataStructureVizParams, plan: _Plan) -> None:
    init = p.initial
    max_slots = 2 ** MAX_TREE_LEVELS - 1
    if not init or init[0] is None:
        raise ValueError("a tree needs a root: initial[0] cannot be empty or null")
    if len(init) > max_slots:
        raise ValueError(
            f"a tree has at most {MAX_TREE_LEVELS} levels ({max_slots} level-order "
            f"slots), got {len(init)}"
        )
    for slot, v in enumerate(init):
        if v is None:
            continue
        if slot and (slot - 1) // 2 not in plan.slots:
            raise ValueError(f"initial[{slot}] = {v!r} has no parent (its parent slot is null)")
        _claim_label(plan, slot, _s(v), f"initial[{slot}]")

    for i, op in enumerate(p.operations):
        what = f"operations[{i}] {op.op}"
        labels = {lab: s for s, lab in plan.slots.items()}
        if isinstance(op, Highlight):
            for ref in op.at:
                _node(labels, ref, what)
        elif isinstance(op, Insert):
            parent = _node(labels, op.parent, what)
            slot = 2 * parent + (1 if op.side == "left" else 2)
            if slot in plan.slots:
                raise ValueError(
                    f"{what}: {op.parent!r} already has a {op.side} child "
                    f"{plan.slots[slot]!r}"
                )
            if slot >= max_slots:
                raise ValueError(f"{what}: tree would exceed {MAX_TREE_LEVELS} levels")
            _claim_label(plan, slot, _s(op.value), what)
        else:  # Traverse
            a, b = (_node(labels, r, what) for r in op.edge)
            if (a - 1) // 2 != b and (b - 1) // 2 != a:
                raise ValueError(
                    f"{what}: {op.edge[0]!r} and {op.edge[1]!r} are not parent and child"
                )
    plan.sized = list(plan.slots.values())
    plan.levels = _depth(max(plan.slots)) + 1


def _claim_label(plan: _Plan, slot: int, lab: str, what: str) -> None:
    # Labels double as node identities, so they must be unique.
    if lab in plan.slots.values():
        raise ValueError(f"{what}: duplicate node label {lab!r}")
    plan.slots[slot] = lab


def _node(labels: dict[str, int], ref: str | int, what: str) -> int:
    if _s(ref) not in labels:
        raise ValueError(f"{what}: no node labelled {ref!r}")
    return labels[_s(ref)]


def _plan_graph(p: DataStructureVizParams, plan: _Plan) -> None:
    g: GraphInitial = p.initial  # type: ignore[assignment]
    plan.nodes = [_s(n) for n in g.nodes]
    if len(set(plan.nodes)) != len(plan.nodes):
        raise ValueError("graph node labels must be unique")
    known = set(plan.nodes)
    seen: set[frozenset[str]] = set()
    for u, v in g.edges:
        u, v = _s(u), _s(v)
        if u not in known or v not in known:
            raise ValueError(f"edge ({u!r}, {v!r}) names a node that does not exist")
        if u == v:
            raise ValueError(f"edge ({u!r}, {v!r}) is a self-loop")
        if frozenset((u, v)) in seen:
            raise ValueError(f"edge ({u!r}, {v!r}) is listed twice")
        seen.add(frozenset((u, v)))
        plan.edges.append((u, v))
    for i, op in enumerate(p.operations):
        what = f"operations[{i}] {op.op}"
        if isinstance(op, Highlight):
            for ref in op.at:
                if _s(ref) not in known:
                    raise ValueError(f"{what}: no node labelled {ref!r}")
        else:  # Traverse
            if frozenset(_s(r) for r in op.edge) not in seen:
                raise ValueError(f"{what}: there is no edge {op.edge[0]!r}-{op.edge[1]!r}")
    plan.sized = list(plan.nodes)


# --- boards: the mobjects for one kind, and how each op animates them ---------


class _Board:
    """All mobjects for one structure, laid out relative to each other.

    `layout` holds everything the beat will ever show and is what gets fitted
    to the stage; `shown` is the subset on screen when the beat opens;
    `final()` is the subset on screen once every operation has run.
    """

    def __init__(self, theme: Theme, plan: _Plan, ops: list) -> None:
        self.theme = theme
        self.plan = plan
        self.ops = ops
        self.cap = float(mono_text("H", theme).height)
        self.layout = VGroup()
        self.shown = VGroup()
        self.focused: list[VMobject] = []

    # shared pieces

    def _value(self, s: str) -> VMobject:
        return label(mono_text(s, self.theme), f"value {s!r}")

    def _widest(self) -> float:
        texts = [mono_text(s, self.theme) for s in self.plan.sized]
        return max([t.width for t in texts] + [0.0])

    def _shape_style(self, shape: VMobject) -> VMobject:
        return shape.set_stroke(self.theme.palette.fg, width=STROKE).set_fill(
            self.theme.palette.accent, opacity=0
        )

    def final(self) -> VGroup:
        """The structure after every operation, in neutral styling.

        Replays the operations' effect on WHAT is shown (values swapped or
        overwritten, cells pushed and popped, nodes inserted) without
        animating, on a freshly built board. The focus accent and traverse
        trails are deliberately left out: they belong to the beat that played
        the operations, and a later beat carrying this structure in decides
        its own emphasis. Call at most once per board.
        """
        raise NotImplementedError

    def refocus(self, targets: list[VMobject]) -> list[Animation]:
        """Move the accent from the previous op's items to this op's.

        Targets already in focus are re-animated to the same state rather
        than skipped, so highlighting the same item twice still yields an
        animation -- an empty play() raises inside Manim.
        """
        pal = self.theme.palette
        anims = [
            m.animate.set_stroke(pal.fg).set_fill(opacity=0)
            for m in self.focused if m not in targets
        ]
        anims += [
            m.animate.set_stroke(pal.accent).set_fill(pal.accent, opacity=FOCUS_FILL)
            for m in targets
        ]
        self.focused = list(targets)
        return anims

    def enter_focused(self, shape: VMobject) -> list[Animation]:
        """Focus a shape that is about to fade in.

        It is styled directly instead of animated: an .animate on the shape
        and a FadeIn on the group containing it, in one play(), would fight
        over the same mobject.
        """
        pal = self.theme.palette
        anims = self.refocus([])
        shape.set_stroke(pal.accent).set_fill(pal.accent, opacity=FOCUS_FILL)
        self.focused = [shape]
        return anims


class _ArrayBoard(_Board):
    def __init__(self, theme, plan, ops):
        super().__init__(theme, plan, ops)
        h = self.cap * (1 + 2 * CELL_PAD)
        w = max(self._widest() + 2 * CELL_PAD * self.cap, h)
        self.cell_h = h
        self.cells = [
            label(self._shape_style(Rectangle(width=w, height=h)), f"cell[{i}]")
            for i in range(len(plan.values))
        ]
        self.row = VGroup(*self.cells).arrange(RIGHT, buff=0)
        self.texts = [self._value(v).move_to(c) for v, c in zip(plan.values, self.cells)]
        self.indices = [
            caption_text(str(i), theme).next_to(c, DOWN, buff=INDEX_GAP * self.cap)
            for i, c in enumerate(self.cells)
        ]
        # Future values, built now so the fit accounts for them.
        self.future = {
            i: self._value(_s(op.value)).move_to(self.cells[op.at])
            for i, op in enumerate(ops) if isinstance(op, SetValue)
        }
        self.shown.add(self.row, *self.texts, *self.indices)
        self.layout.add(self.shown, *self.future.values())

    def final(self) -> VGroup:
        texts = list(self.texts)
        for i, op in enumerate(self.ops):
            if isinstance(op, Swap):
                a, b = op.at
                texts[a], texts[b] = texts[b], texts[a]
            elif isinstance(op, SetValue):
                texts[op.at] = self.future[i]
        for text, cell in zip(texts, self.cells):
            text.move_to(cell)
        return VGroup(self.row, *texts, *self.indices)

    def animations(self, i, op):
        if isinstance(op, Highlight):
            return self.refocus([self.cells[k] for k in op.at])
        if isinstance(op, Swap):
            a, b = op.at
            ta, tb = self.texts[a], self.texts[b]
            self.texts[a], self.texts[b] = tb, ta
            return self.refocus([self.cells[a], self.cells[b]]) + [
                CyclicReplace(ta, tb, path_arc=self._arc(a, b))
            ]
        old, self.texts[op.at] = self.texts[op.at], self.future[i]
        return self.refocus([self.cells[op.at]]) + [FadeOut(old), FadeIn(self.future[i])]

    def _arc(self, a: int, b: int) -> float:
        """Arc angle whose peak height is a fixed fraction of a cell.

        A fixed angle would loop high across a long row and leave the frame
        mid-flight -- which the geometric probe cannot see (its documented
        LIMITATION). The sagitta of chord d at angle t is (d/2)tan(t/4).
        """
        d = abs(b - a) * self.cells[0].width
        return 4 * math.atan(2 * SWAP_LIFT * self.cell_h / d)


class _StackBoard(_Board):
    def __init__(self, theme, plan, ops):
        super().__init__(theme, plan, ops)
        h = self.cap * (1 + 2 * CELL_PAD)
        w = max(self._widest() + 2 * CELL_PAD * self.cap, h)
        self.cell_h = h
        # Invisible slots fix where each depth sits; never added to the scene.
        slots = VGroup(*[Rectangle(width=w, height=h) for _ in range(max(plan.max_depth, 1))])
        slots.arrange(UP, buff=0)
        self.slots = list(slots)
        gap = STACK_WALL_GAP * self.cap
        bl = slots.get_corner(DOWN + LEFT) + (LEFT + DOWN) * gap
        br = slots.get_corner(DOWN + RIGHT) + (RIGHT + DOWN) * gap
        rise = UP * (slots.get_top()[1] - bl[1])
        container = label(VGroup(
            Line(bl + rise, bl), Line(bl, br), Line(br, br + rise),
        ).set_stroke(theme.palette.muted, width=STROKE), "stack container")
        self.container = container
        self.stack = [self._cell(v, k) for k, v in enumerate(plan.values)]
        self.future: dict[int, VMobject] = {}
        depth = len(self.stack)
        for i, op in enumerate(ops):
            if isinstance(op, Push):
                self.future[i] = self._cell(_s(op.value), depth)
                depth += 1
            elif isinstance(op, Pop):
                depth -= 1
        self.shown.add(container, *self.stack)
        self.layout.add(slots, self.shown, *self.future.values())

    def final(self) -> VGroup:
        stack = list(self.stack)
        for i, op in enumerate(self.ops):
            if isinstance(op, Push):
                stack.append(self.future[i])
            elif isinstance(op, Pop):
                stack.pop()
        return VGroup(self.container, *stack)

    def _cell(self, s: str, depth: int) -> VGroup:
        box = self._shape_style(Rectangle(width=self.slots[0].width, height=self.cell_h))
        box.move_to(self.slots[depth])
        return label(VGroup(box, self._value(s).move_to(box)), f"stack[{depth}]")

    def animations(self, i, op):
        if isinstance(op, Highlight):
            return self.refocus([self.stack[k][0] for k in op.at])
        if isinstance(op, Push):
            cell = self.future[i]
            self.stack.append(cell)
            return self.enter_focused(cell[0]) + [FadeIn(cell, shift=DOWN * self.cell_h / 2)]
        cell = self.stack.pop()
        self.focused = [m for m in self.focused if m is not cell[0]]
        return self.refocus([]) + [FadeOut(cell, shift=UP * self.cell_h / 2)]


class _NodeBoard(_Board):
    """Shared by tree and graph: round nodes joined by straight edges."""

    def _radius(self) -> float:
        return max(self._widest(), self.cap) / 2 + CELL_PAD * self.cap

    def _make_node(self, s: str) -> VGroup:
        circle = self._shape_style(Circle(radius=self.r))
        return label(VGroup(circle, self._value(s).move_to(circle)), f"node {s!r}")

    def _edge(self, a: VGroup, b: VGroup) -> Line:
        # buff=r stops the line at each circle's rim.
        return Line(a.get_center(), b.get_center(), buff=self.r).set_stroke(
            self.theme.palette.muted, width=STROKE
        )

    def _traverse(self, op: Traverse) -> list[Animation]:
        u, v = (_s(r) for r in op.edge)
        edge = self.edges[frozenset((u, v))]
        return self.refocus([self.nodes[v][0]]) + [
            edge.animate.set_stroke(self.theme.palette.accent)
        ]

    def _highlight(self, op: Highlight) -> list[Animation]:
        return self.refocus([self.nodes[_s(r)][0] for r in op.at])


class _TreeBoard(_NodeBoard):
    def __init__(self, theme, plan, ops):
        super().__init__(theme, plan, ops)
        self.r = self._radius()
        # Full-slot layout: every possible position on the deepest level gets
        # its own column, so no later insert can collide with an existing node.
        pitch = 2 * self.r + NODE_GAP * self.cap
        width = 2 ** (plan.levels - 1) * pitch
        drop = 2 * self.r + LEVEL_GAP * self.cap

        self.nodes: dict[str, VGroup] = {}
        self.edges: dict[frozenset[str], Line] = {}
        pos: dict[int, VGroup] = {}
        for slot in sorted(plan.slots):
            node = self._make_node(plan.slots[slot])
            if slot:
                parent = pos[(slot - 1) // 2]
                side = LEFT if slot % 2 else RIGHT
                offset = width / 2 ** (_depth(slot) + 1)
                node.move_to(parent.get_center() + side * offset + DOWN * drop)
                key = frozenset((plan.slots[(slot - 1) // 2], plan.slots[slot]))
                self.edges[key] = self._edge(parent, node)
            pos[slot] = node
            self.nodes[plan.slots[slot]] = node

        # An inserted node arrives together with the edge up to its parent.
        self.future = {
            i: VGroup(self.edges[frozenset((_s(op.parent), _s(op.value)))],
                      self.nodes[_s(op.value)])
            for i, op in enumerate(ops) if isinstance(op, Insert)
        }
        arriving = {m for g in self.future.values() for m in g}
        self.shown.add(*[e for e in self.edges.values() if e not in arriving],
                       *[n for n in self.nodes.values() if n not in arriving])
        self.layout.add(self.shown, *self.future.values())

    def final(self) -> VGroup:
        # Trees only grow: after the last insert every node and edge is shown.
        return VGroup(*self.edges.values(), *self.nodes.values())

    def animations(self, i, op):
        if isinstance(op, Highlight):
            return self._highlight(op)
        if isinstance(op, Traverse):
            return self._traverse(op)
        return self.enter_focused(self.future[i][1][0]) + [FadeIn(self.future[i])]


class _GraphBoard(_NodeBoard):
    def __init__(self, theme, plan, ops):
        super().__init__(theme, plan, ops)
        self.r = self._radius()
        n = len(plan.nodes)
        # Nodes on an ellipse, first node at the top, clockwise. Radius chosen
        # so neighbours clear each other; every chord of a convex ring misses
        # the other nodes, so edges never pass through a node.
        pitch = 2 * self.r + GRAPH_GAP * self.cap
        ry = pitch / (2 * math.sin(math.pi / n)) if n > 1 else 0.0
        rx = GRAPH_ASPECT * ry
        offsets = [
            RIGHT * rx * math.sin(2 * math.pi * k / n) + UP * ry * math.cos(2 * math.pi * k / n)
            for k in range(n)
        ]
        nodes = [self._make_node(s) for s in plan.nodes]
        for node, off in zip(nodes[1:], offsets[1:]):
            node.move_to(nodes[0].get_center() + off - offsets[0])
        self.nodes = dict(zip(plan.nodes, nodes))
        self.edges = {
            frozenset((u, v)): self._edge(self.nodes[u], self.nodes[v])
            for u, v in plan.edges
        }
        self.shown.add(*self.edges.values(), *nodes)
        self.layout.add(self.shown)

    def final(self) -> VGroup:
        # Operations on a graph only change emphasis, never its contents.
        return VGroup(*self.edges.values(), *self.nodes.values())

    def animations(self, i, op):
        if isinstance(op, Highlight):
            return self._highlight(op)
        return self._traverse(op)


BOARDS = {
    "array": _ArrayBoard,
    "stack": _StackBoard,
    "tree": _TreeBoard,
    "graph": _GraphBoard,
}


# --- the component ------------------------------------------------------------


@register
class DataStructureViz(Component):
    name = "DataStructureViz"
    Params = DataStructureVizParams

    def regions(self) -> set[Region]:
        return {Region.STAGE}

    def min_seconds(self) -> float:
        """Sum of per-step legibility minimums (semantic rung, SCENE_SPEC.md §8)."""
        return INTRO_MIN + sum(OP_MIN[op.op] for op in self.params.operations) + HOLD_MIN

    def latex_strings(self) -> list[str]:
        # Values render as monospace Text; nothing goes through LaTeX.
        return []

    def board(self, theme: Theme) -> _Board:
        """The fitted structure. Pure in (params, theme) -- the carry-in seam."""
        p: DataStructureVizParams = self.params
        board = BOARDS[p.kind](theme, _plan(p), list(p.operations))
        label(board.layout, f"{p.kind}")
        fit_to_region(board.layout, Region.STAGE)
        return board

    def build(self, scene: ChalkdustScene) -> None:
        p: DataStructureVizParams = self.params
        board = self.board(scene.theme)
        label(board.shown, p.kind)

        # Every step plays at its minimum scaled by how much longer the
        # narration is than min_seconds(), up to ANIM_STRETCH; narration
        # beyond that becomes a pause after each step, in proportion to it.
        # scene.budget() turns the weights into whole frames summing exactly
        # to the beat (D-002).
        mins = [INTRO_MIN] + [OP_MIN[op.op] for op in p.operations]
        stretch = scene.beat_duration / self.min_seconds()
        play = min(stretch, ANIM_STRETCH)
        pause = stretch - play
        weights: list[float] = []
        for m in mins:
            weights += [m * play, m * pause] if pause > 0 else [m * play]
        weights.append(HOLD_MIN * stretch)
        if scene.beat_frames < len(weights):
            # budget() would still hand every segment a frame, making the
            # clip longer than its narration. Refuse instead.
            raise LayoutError(
                f"{len(weights)} timed steps cannot fit in {scene.beat_frames} "
                "frames; the narration is far too short for this many "
                "operations. Split the beat.",
                kind="overflow",
            )
        times = iter(scene.budget(*weights))

        scene.play(FadeIn(board.shown), run_time=next(times))
        scene.settle(f"{p.kind} shown")
        if pause > 0:
            scene.wait(next(times))
        for i, op in enumerate(p.operations):
            scene.play(*board.animations(i, op), run_time=next(times))
            scene.settle(f"operations[{i}] {op.op}")
            if pause > 0:
                scene.wait(next(times))
        scene.wait(next(times))

    @classmethod
    def examples(cls):
        return [
            # Selection sort, first pass.
            {"kind": "array", "initial": [29, 10, 14, 37, 13],
             "operations": [{"op": "highlight", "at": [0, 1]},
                            {"op": "swap", "at": [0, 1]},
                            {"op": "highlight", "at": [2]},
                            {"op": "set", "at": 4, "value": 99}]},
            {"kind": "stack", "initial": ["main"],
             "operations": [{"op": "push", "value": "fib(3)"},
                            {"op": "push", "value": "fib(2)"},
                            {"op": "pop"},
                            {"op": "highlight", "at": [1]}]},
            # BST insert of 6 under 3, then the search path for 6.
            {"kind": "tree", "initial": [8, 3, 10, 1, None, None, 14],
             "operations": [{"op": "insert", "parent": 3, "side": "right", "value": 6},
                            {"op": "traverse", "edge": [8, 3]},
                            {"op": "traverse", "edge": [3, 6]}]},
            # Ring order A, B, D, E, C: no two edges cross.
            {"kind": "graph",
             "initial": {"nodes": ["A", "B", "D", "E", "C"],
                         "edges": [["A", "B"], ["A", "C"], ["B", "D"],
                                   ["C", "D"], ["D", "E"]]},
             "operations": [{"op": "highlight", "at": ["A"]},
                            {"op": "traverse", "edge": ["A", "B"]},
                            {"op": "traverse", "edge": ["A", "C"]},
                            {"op": "traverse", "edge": ["B", "D"]}]},
        ]

    @classmethod
    def stress(cls):
        many_ops = [{"op": "swap", "at": [0, 11]}, {"op": "highlight", "at": [5]}] * 6
        long_word = "identifier_" + "x" * 49  # 60 chars, no break points
        return [
            # (a) every kind at its cap with values ~3x realistic length.
            {"kind": "array", "initial": [f"value_{i:03d}" for i in range(12)],
             "operations": many_ops},
            {"kind": "array", "initial": list(range(100, 112)), "operations": many_ops},
            {"kind": "stack", "initial": [f"frame_{i}()" for i in range(8)],
             "operations": [{"op": "pop"}, {"op": "push", "value": "recurse()"}] * 6},
            {"kind": "tree", "initial": list(range(10, 41))},
            {"kind": "graph",
             "initial": {"nodes": [f"city_{i}" for i in range(10)],
                         "edges": [[f"city_{i}", f"city_{(i + k) % 10}"]
                                   for i in range(10) for k in (1, 3)]}},
            # (b) one unwrappable 60-character token.
            {"kind": "array", "initial": [long_word]},
            {"kind": "graph", "initial": {"nodes": ["https://example.com/" + "a" * 40, "B"],
                                          "edges": [["https://example.com/" + "a" * 40, "B"]]}},
            # (c) empty and minimal.
            {"kind": "stack", "initial": []},
            {"kind": "array", "initial": [0]},
            {"kind": "tree", "initial": ["root"]},
            {"kind": "graph", "initial": {"nodes": ["only"]}},
            # Deepest legal tree: a single spine down the left edge.
            {"kind": "tree", "initial": [1],
             "operations": [{"op": "insert", "parent": 1, "side": "left", "value": 2},
                            {"op": "insert", "parent": 2, "side": "left", "value": 3},
                            {"op": "insert", "parent": 3, "side": "left", "value": 4},
                            {"op": "insert", "parent": 4, "side": "left", "value": 5}]},
        ]


@artifact_builder(DataStructureViz.name)
def _artifact(params: DataStructureVizParams, theme: Theme) -> Mobject:
    """The structure as the beat leaves it, for a later beat's carry-in
    (SCENE_SPEC.md §6): b02 DataStructureViz registers "bucket_array", b03
    carries it in. Fitted exactly as the producing beat fitted it, so it is
    the same size the viewer last saw; CarryIn places and dims it."""
    final = DataStructureViz(params).board(theme).final()
    return label(final, f"{params.kind} (final state)")
