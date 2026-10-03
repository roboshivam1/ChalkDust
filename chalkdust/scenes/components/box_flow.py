"""BoxFlow: a system / pipeline diagram -- labelled boxes joined by arrows.

The model names the boxes and says which feeds which; it never says where
anything goes. Layout is a small layered (Sugiyama-style) pass, left to right:

  1. Break cycles by reversing DFS back edges, so the graph can be layered.
     A reversed edge is still DRAWN in its true direction -- it just routes
     through the layered structure backwards, arrowhead on its real target.
  2. Longest-path layering: every node sits one column right of its furthest
     predecessor.
  3. An edge spanning several columns gets an empty waypoint slot in each
     column it crosses, stacked like a node, so it threads between boxes
     instead of through them.
  4. Barycentre sweeps order each column to reduce crossings.

Why edges can never cross a label: every box in a column shares the column's
width, so the only places a segment can be are the gaps BETWEEN columns (which
hold no boxes) and the reserved waypoint slots WITHIN a column (which hold no
boxes either). The guarantee is structural, not checked after the fact.

Every ordering decision keys on declaration index, never on set or dict
iteration of the input, so one spec always lays out the same way (D-004).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import (
    Create,
    FadeIn,
    Polygon,
    RoundedRectangle,
    ShowPassingFlash,
    VGroup,
    VMobject,
    smooth,
)
from pydantic import Field, field_validator, model_validator

from chalkdust.core.models import Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import (
    Component,
    ComponentParams,
    label,
    register,
    wrap,
)
from chalkdust.scenes.regions import fit_to_region, region_rect
from chalkdust.scenes.theme import Theme, body_text

# Density caps that fire at schema validation, before anything is built. Ten
# boxes is already a busy slide; past that the right fix is splitting the
# system across beats, and the error should say so rather than name a font
# size. Content inside the caps that still cannot fit refuses via
# fit_to_region's overflow.
MAX_NODES = 10
MAX_EDGES = 16

LABEL_WRAP = 16  # characters per line inside a box

# Geometry at natural size, in Manim units. fit_to_region scales the whole
# diagram uniformly, so these are proportions as much as sizes.
PAD_X = 0.3          # label to box side
PAD_Y = 0.22         # label to box top/bottom
CORNER = 0.12
GAP_X = 1.0          # between columns -- where every edge segment lives
GAP_Y = 0.45         # between items stacked in one column
SLOT_H = 0.25        # waypoint slot reserved for an edge passing a column
LOOP_H = 0.6         # headroom reserved above a box that has a self-loop
LOOP_R = 0.22        # self-loop arc radius
PORT_GAP = 0.2       # preferred spacing of several edges on one box side
PORT_MARGIN = 0.12   # keep ports off the rounded corners
TIP_LEN = 0.2
TIP_W = 0.18
STROKE = 3
FLOW_STROKE = 6

# Per-step legibility floors for the semantic rung (SCENE_SPEC.md §8 rung 2):
# below these a viewer cannot register what appeared before the next thing
# arrives.
NODES_MIN_SECONDS = 0.6
EDGES_MIN_SECONDS = 0.5
EXTRA_MIN_SECONDS = 0.6
FLOW_MIN_SECONDS = 0.5
HOLD_MIN_SECONDS = 1.0

# Relative weights for scene.budget(). Boxes get more time than the arrows
# that lead to them -- the label is what has to be read.
NODES_WEIGHT = 2.0
EDGES_WEIGHT = 1.0
EXTRA_WEIGHT = 1.5
FLOW_WEIGHT = 1.0
HOLD_WEIGHT = 3.0


class BoxFlowNode(ComponentParams):
    # `id` is only for edges to refer to; `label` is what is shown.
    id: str
    label: str

    @field_validator("id", "label")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class BoxFlowEdge(ComponentParams):
    source: str
    target: str


class BoxFlowParams(ComponentParams):
    nodes: list[BoxFlowNode] = Field(min_length=1, max_length=MAX_NODES)
    edges: list[BoxFlowEdge] = Field(default_factory=list, max_length=MAX_EDGES)
    # After the diagram is built, send a pulse along the arrows layer by
    # layer, so the viewer sees the direction data actually moves.
    animate_flow: bool = False

    @model_validator(mode="after")
    def _graph_sane(self) -> BoxFlowParams:
        ids = [n.id for n in self.nodes]
        seen: list[str] = []
        for i in ids:
            if i in seen:
                raise ValueError(f"duplicate node id {i!r}")
            seen.append(i)
        pairs: list[tuple[str, str]] = []
        for k, e in enumerate(self.edges):
            for end in (e.source, e.target):
                if end not in ids:
                    raise ValueError(
                        f"edges[{k}] refers to unknown node {end!r}; "
                        f"nodes are {ids}"
                    )
            # Two identical arrows would draw on top of each other and say
            # nothing the first did not.
            if (e.source, e.target) in pairs:
                raise ValueError(
                    f"edges[{k}] duplicates {e.source!r} -> {e.target!r}"
                )
            pairs.append((e.source, e.target))
        return self


# --- graph layout (pure; no Manim) -----------------------------------------


@dataclass(frozen=True)
class _Graph:
    """The layered structure, before any geometry.

    Items are node indices 0..n-1 followed by waypoint slots n.. ; `columns`
    lists item ids per column in final top-to-bottom order.
    """

    n_nodes: int
    layer: tuple[int, ...]                 # per node
    columns: tuple[tuple[int, ...], ...]   # per column, item ids
    item_column: tuple[int, ...]           # per item
    # Per edge: the chain of items it threads, lower column first; empty for
    # a self-loop.
    chains: tuple[tuple[int, ...], ...]
    reversed: tuple[bool, ...]             # per edge: drawn right-to-left
    self_loop: tuple[bool, ...]            # per edge
    target: tuple[int, ...]                # per edge: node the arrow points at

    @property
    def n_columns(self) -> int:
        return len(self.columns)


def _layout_graph(p: BoxFlowParams) -> _Graph:
    index = {node.id: i for i, node in enumerate(p.nodes)}
    n = len(p.nodes)
    ends = [(index[e.source], index[e.target]) for e in p.edges]
    self_loop = [s == t for s, t in ends]

    # 1. Cycle breaking: DFS in declaration order; an edge into a node still
    #    on the stack closes a cycle and is reversed for layering only.
    out: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k, (s, t) in enumerate(ends):
        if not self_loop[k]:
            out[s].append((k, t))
    state = [0] * n  # 0 unvisited, 1 on stack, 2 done
    rev = [False] * len(ends)

    def visit(u: int) -> None:
        state[u] = 1
        for k, v in out[u]:
            if state[v] == 1:
                rev[k] = True
            elif state[v] == 0:
                visit(v)
        state[u] = 2

    for u in range(n):
        if state[u] == 0:
            visit(u)

    # The acyclic edge set, as (lower, upper) node pairs.
    dag = [
        (t, s) if rev[k] else (s, t)
        for k, (s, t) in enumerate(ends)
    ]

    # 2. Longest-path layering over a topological order (Kahn, ties by index).
    indeg = [0] * n
    succ: list[list[int]] = [[] for _ in range(n)]
    for k, (a, b) in enumerate(dag):
        if not self_loop[k]:
            succ[a].append(b)
            indeg[b] += 1
    layer = [0] * n
    ready = [u for u in range(n) if indeg[u] == 0]
    while ready:
        u = ready.pop(0)
        for v in succ[u]:
            layer[v] = max(layer[v], layer[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    n_cols = max(layer) + 1

    # 3. Waypoint slots for edges that span more than one column.
    item_column = list(layer)
    chains: list[tuple[int, ...]] = []
    for k, (a, b) in enumerate(dag):
        if self_loop[k]:
            chains.append(())
            continue
        chain = [a]
        for c in range(layer[a] + 1, layer[b]):
            item_column.append(c)
            chain.append(len(item_column) - 1)
        chain.append(b)
        chains.append(tuple(chain))

    columns: list[list[int]] = [[] for _ in range(n_cols)]
    for item, c in enumerate(item_column):
        columns[c].append(item)

    # 4. Crossing reduction: barycentre of neighbours in the adjacent column.
    up: list[list[int]] = [[] for _ in item_column]    # neighbours to the left
    down: list[list[int]] = [[] for _ in item_column]  # neighbours to the right
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            down[a].append(b)
            up[b].append(a)

    def reorder(c: int, ref: int, nbrs: list[list[int]]) -> None:
        pos = {item: i for i, item in enumerate(columns[ref])}
        here = {item: i for i, item in enumerate(columns[c])}

        def key(item: int) -> tuple[float, int]:
            ns = nbrs[item]
            bary = sum(pos[x] for x in ns) / len(ns) if ns else here[item]
            return (bary, here[item])

        columns[c].sort(key=key)

    for _ in range(3):
        for c in range(1, n_cols):
            reorder(c, c - 1, up)
        for c in range(n_cols - 2, -1, -1):
            reorder(c, c + 1, down)
    for c in range(1, n_cols):
        reorder(c, c - 1, up)

    return _Graph(
        n_nodes=n,
        layer=tuple(layer),
        columns=tuple(tuple(col) for col in columns),
        item_column=tuple(item_column),
        chains=tuple(chains),
        reversed=tuple(rev),
        self_loop=tuple(self_loop),
        target=tuple(t for _, t in ends),
    )


@dataclass(frozen=True)
class _Step:
    kind: str     # nodes | edges | extra | flow | hold
    column: int   # which column it concerns; -1 for extra / back-flow / hold
    weight: float
    min_seconds: float


def _steps(g: _Graph, animate_flow: bool) -> list[_Step]:
    """The beat's animation, as an ordered list of budgeted steps.

    Shared by build() and min_seconds() so the semantic rung and the renderer
    can never disagree about what the beat contains.
    """
    steps = [_Step("nodes", 0, NODES_WEIGHT, NODES_MIN_SECONDS)]
    forward_into = [False] * g.n_columns
    has_extra = False
    for k, chain in enumerate(g.chains):
        if g.self_loop[k] or g.reversed[k]:
            has_extra = True
        else:
            forward_into[g.item_column[chain[-1]]] = True
    for c in range(1, g.n_columns):
        if forward_into[c]:
            steps.append(_Step("edges", c, EDGES_WEIGHT, EDGES_MIN_SECONDS))
        steps.append(_Step("nodes", c, NODES_WEIGHT, NODES_MIN_SECONDS))
    if has_extra:
        steps.append(_Step("extra", -1, EXTRA_WEIGHT, EXTRA_MIN_SECONDS))
    if animate_flow and g.chains:
        for c in range(g.n_columns):
            if c == 0 or forward_into[c]:
                steps.append(_Step("flow", c, FLOW_WEIGHT, FLOW_MIN_SECONDS))
        if has_extra:
            steps.append(_Step("flow", -1, FLOW_WEIGHT, FLOW_MIN_SECONDS))
    steps.append(_Step("hold", -1, HOLD_WEIGHT, HOLD_MIN_SECONDS))
    return steps


# --- drawing helpers --------------------------------------------------------


def _arrow(points: list[np.ndarray], theme: Theme) -> VGroup:
    """A polyline ending in a filled tip at points[-1].

    Built by hand rather than with Arrow: Arrow is one straight segment, and
    TipableVMobject.add_tip re-fits the WHOLE path to make room for the tip,
    which would nudge waypoints off their reserved slots.
    """
    end, prev = points[-1], points[-2]
    d = (end - prev) / np.linalg.norm(end - prev)
    base = end - d * TIP_LEN
    perp = np.array([-d[1], d[0], 0.0])
    path = VMobject(stroke_color=theme.palette.muted, stroke_width=STROKE)
    path.set_points_as_corners([*points[:-1], base])
    tip = Polygon(end, base + perp * TIP_W / 2, base - perp * TIP_W / 2,
                  stroke_width=0, fill_color=theme.palette.muted,
                  fill_opacity=1)
    return VGroup(path, tip)


def _loop(box: VMobject, theme: Theme) -> VGroup:
    """A self-loop: an arc over the box's top, landing back on it.

    Drawn in the LOOP_H headroom the column reserved above this box.
    """
    top, cx = box.get_top()[1], box.get_center()[0]
    centre = np.array([cx, top + TIP_LEN + 0.08, 0.0])
    arc = [centre + LOOP_R * np.array([np.cos(a), np.sin(a), 0.0])
           for a in np.linspace(0, np.pi, 16)]
    return _arrow([np.array([cx + LOOP_R, top, 0.0]), *arc,
                   np.array([cx - LOOP_R, top, 0.0])], theme)


def _flash(path: VMobject, theme: Theme, first_half: bool) -> ShowPassingFlash:
    """An accent pulse along a copy of `path`.

    Within one flow step the pulse runs along the arrows in the first half and
    around the boxes they reach in the second, so it reads as arriving.
    """
    copy = path.copy().set_stroke(theme.palette.accent, width=FLOW_STROKE)
    if first_half:
        def rate(t: float) -> float:
            return smooth(min(1.0, 2 * t))
    else:
        def rate(t: float) -> float:
            return smooth(max(0.0, 2 * t - 1))
    return ShowPassingFlash(copy, time_width=0.6, rate_func=rate)


@register
class BoxFlow(Component):
    name = "BoxFlow"
    Params = BoxFlowParams

    def regions(self) -> set[Region]:
        # The diagram is wide by nature; it takes the whole stage.
        return {Region.STAGE}

    def build(self, scene: ChalkdustScene) -> None:
        p: BoxFlowParams = self.params
        theme = scene.theme
        g = _layout_graph(p)
        n = g.n_nodes

        wrapped = [wrap(node.label, LABEL_WRAP) for node in p.nodes]
        texts = [body_text(s, theme) for s in wrapped]
        # A box's height comes from its line count, not its ink: measured
        # against a block with full ascent and descent ("Hg"), so "Lexer" and
        # "Parser" get the same box even though one has a descender.
        ref_h: dict[int, float] = {}
        for s in wrapped:
            lines = s.count("\n") + 1
            if lines not in ref_h:
                ref_h[lines] = body_text("\n".join(["Hg"] * lines), theme).height
        box_h = [ref_h[s.count("\n") + 1] + 2 * PAD_Y for s in wrapped]
        has_loop = [False] * n
        for k, loop in enumerate(g.self_loop):
            if loop:
                has_loop[g.target[k]] = True

        # --- geometry, relative to the stage rect's centre ------------------
        # Every box in a column takes the column's width; see the module
        # docstring for why that is what keeps edges off the labels.
        col_w = [max(texts[i].width for i in col if i < n) + 2 * PAD_X
                 for col in g.columns]

        def item_h(item: int) -> float:
            if item >= n:
                return SLOT_H
            return box_h[item] + (LOOP_H if has_loop[item] else 0.0)

        col_x = [0.0]
        for c in range(1, g.n_columns):
            col_x.append(col_x[-1] + col_w[c - 1] / 2 + GAP_X + col_w[c] / 2)
        mid = (col_x[0] - col_w[0] / 2 + col_x[-1] + col_w[-1] / 2) / 2

        # Each column is stacked top to bottom and centred vertically.
        item_y = [0.0] * len(g.item_column)
        for col in g.columns:
            top = (sum(item_h(i) for i in col) + GAP_Y * (len(col) - 1)) / 2
            for item in col:
                if item < n:
                    # A loop's headroom sits above the box, not around it.
                    loop = LOOP_H if has_loop[item] else 0.0
                    item_y[item] = top - loop - box_h[item] / 2
                else:
                    item_y[item] = top - SLOT_H / 2
                top -= item_h(item) + GAP_Y

        anchor = region_rect(Region.STAGE).center

        def at(x: float, y: float) -> np.ndarray:
            return anchor + np.array([x - mid, y, 0.0])

        def left(c: int) -> float:
            return col_x[c] - col_w[c] / 2

        def right(c: int) -> float:
            return col_x[c] + col_w[c] / 2

        nodes: list[VGroup] = []
        for i, node in enumerate(p.nodes):
            c = g.item_column[i]
            box = RoundedRectangle(width=col_w[c], height=box_h[i],
                                   corner_radius=CORNER,
                                   stroke_color=theme.palette.muted,
                                   stroke_width=STROKE)
            box.move_to(at(col_x[c], item_y[i]))
            texts[i].move_to(box.get_center())
            nodes.append(label(VGroup(box, texts[i]), f"node[{node.id}]"))

        # --- ports: spread several edge ends along one box side -------------
        # Sorted by the y of the ADJACENT item in the chain (not the far end),
        # so first and last segments fan out without crossing. Ties break by
        # edge index identically on both ends, so a two-way pair runs parallel.
        sides: dict[tuple[int, str], list[tuple[float, int]]] = {}
        for k, chain in enumerate(g.chains):
            if chain:
                sides.setdefault((chain[0], "R"), []).append(
                    (-item_y[chain[1]], k))
                sides.setdefault((chain[-1], "L"), []).append(
                    (-item_y[chain[-2]], k))
        port: dict[tuple[int, str, int], float] = {}
        for (i, side), ends in sides.items():
            ends.sort()
            m = len(ends)
            gap = (min(PORT_GAP, (box_h[i] - 2 * PORT_MARGIN) / (m - 1))
                   if m > 1 else 0.0)
            for j, (_, k) in enumerate(ends):
                port[(i, side, k)] = item_y[i] + ((m - 1) / 2 - j) * gap

        edges: list[VGroup] = []
        for k, chain in enumerate(g.chains):
            if not chain:
                edge = _loop(nodes[g.target[k]][0], theme)
            else:
                a, b = chain[0], chain[-1]
                pts = [at(right(g.item_column[a]), port[(a, "R", k)])]
                for slot in chain[1:-1]:
                    c = g.item_column[slot]
                    pts += [at(left(c), item_y[slot]),
                            at(right(c), item_y[slot])]
                pts.append(at(left(g.item_column[b]), port[(b, "L", k)]))
                if g.reversed[k]:
                    pts.reverse()
                edge = _arrow(pts, theme)
            edges.append(label(edge, f"edge[{k}]"))

        diagram = label(VGroup(*nodes, *edges), "BoxFlow")
        fit_to_region(diagram, Region.STAGE)
        scene.exclusive(*nodes)

        # --- animation ------------------------------------------------------
        def in_column(c: int) -> list[int]:
            return [i for i in range(n) if g.item_column[i] == c]

        def edges_for(column: int) -> list[int]:
            # column -1 means the edges drawn after the layers: back edges
            # and self-loops.
            if column == -1:
                return [k for k in range(len(g.chains))
                        if g.self_loop[k] or g.reversed[k]]
            return [k for k, ch in enumerate(g.chains)
                    if ch and not g.reversed[k]
                    and g.item_column[ch[-1]] == column]

        def draw(ks: list[int]) -> list:
            # One introducer per labelled edge group, so the scene holds the
            # edge as a single named mobject; lag_ratio=1 draws the line
            # first and the tip after it.
            return [Create(edges[k], lag_ratio=1.0) for k in ks]

        def flow(column: int) -> list:
            if column == 0:
                return [_flash(nodes[i][0], theme, first_half=False)
                        for i in in_column(0)]
            ks = edges_for(column)
            reached = sorted({g.target[k] for k in ks})
            return ([_flash(edges[k][0], theme, first_half=True) for k in ks]
                    + [_flash(nodes[i][0], theme, first_half=False)
                       for i in reached])

        steps = _steps(g, p.animate_flow)
        times = scene.budget(*[s.weight for s in steps])
        for step, t in zip(steps, times):
            if step.kind == "nodes":
                scene.play(*[FadeIn(nodes[i]) for i in in_column(step.column)],
                           run_time=t)
            elif step.kind == "edges":
                scene.play(*draw(edges_for(step.column)), run_time=t)
            elif step.kind == "extra":
                scene.play(*draw(edges_for(-1)), run_time=t)
            elif step.kind == "flow":
                scene.play(*flow(step.column), run_time=t)
            else:  # hold
                scene.settle("diagram built")
                scene.wait(t)

    # --- semantic-rung hooks (SCENE_SPEC.md §8 rung 2) ----------------------

    def min_seconds(self) -> float:
        """Shortest total animation at which every box and arrow can still be
        registered: the sum of the per-step floors."""
        g = _layout_graph(self.params)
        return sum(s.min_seconds for s in _steps(g, self.params.animate_flow))

    def latex_strings(self) -> list[str]:
        """BoxFlow sets every label as plain text; nothing goes to LaTeX."""
        return []

    # --- fixtures -----------------------------------------------------------

    @classmethod
    def examples(cls):
        return [
            # A straight pipeline.
            {"nodes": [{"id": "src", "label": "Source code"},
                       {"id": "lex", "label": "Lexer"},
                       {"id": "parse", "label": "Parser"},
                       {"id": "gen", "label": "Code generator"}],
             "edges": [{"source": "src", "target": "lex"},
                       {"source": "lex", "target": "parse"},
                       {"source": "parse", "target": "gen"}]},
            # Fan-out and fan-in, with a skip edge across a column.
            {"nodes": [{"id": "client", "label": "Client"},
                       {"id": "lb", "label": "Load balancer"},
                       {"id": "app1", "label": "App server 1"},
                       {"id": "app2", "label": "App server 2"},
                       {"id": "cache", "label": "Cache"},
                       {"id": "db", "label": "Database"}],
             "edges": [{"source": "client", "target": "lb"},
                       {"source": "lb", "target": "app1"},
                       {"source": "lb", "target": "app2"},
                       {"source": "app1", "target": "cache"},
                       {"source": "app2", "target": "cache"},
                       {"source": "cache", "target": "db"},
                       {"source": "app1", "target": "db"}],
             "animate_flow": True},
            # A training loop: a cycle back to the start, plus a self-loop.
            {"nodes": [{"id": "data", "label": "Training data"},
                       {"id": "model", "label": "Model"},
                       {"id": "loss", "label": "Loss"},
                       {"id": "opt", "label": "Optimizer"}],
             "edges": [{"source": "data", "target": "model"},
                       {"source": "model", "target": "loss"},
                       {"source": "loss", "target": "opt"},
                       {"source": "opt", "target": "model"},
                       {"source": "opt", "target": "opt"}],
             "animate_flow": True},
        ]

    @classmethod
    def stress(cls):
        long_id = "AbstractSingletonProxyFactoryBeanConfigurationResolverImpl"
        url = "https://example.com/api/v2/very/long/endpoint/path/no/spaces"
        # Both caps, branching in two dimensions, with a back edge spanning
        # the whole diagram.
        lattice = [{"source": f"n{s}", "target": f"n{t}"}
                   for s, t in [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4),
                                (2, 5), (3, 5), (3, 6), (4, 7), (5, 7),
                                (5, 8), (6, 8), (7, 9), (8, 9), (9, 0),
                                (6, 9)]]
        return [
            # At the caps with short labels: fits.
            {"nodes": [{"id": f"n{i}", "label": f"Stage {i}"}
                       for i in range(MAX_NODES)],
             "edges": lattice, "animate_flow": True},
            # At the caps with labels three times a realistic length: refuses.
            {"nodes": [{"id": f"n{i}", "label": f"Service component number "
                                                f"{i} with a long descriptive "
                                                f"name"}
                       for i in range(MAX_NODES)],
             "edges": lattice, "animate_flow": True},
            # Long forward and back edges, each spanning three columns, so
            # waypoint routing runs under the registry-wide layout tests.
            {"nodes": [{"id": c, "label": c.upper()} for c in "abcde"],
             "edges": [{"source": "a", "target": "b"},
                       {"source": "b", "target": "c"},
                       {"source": "c", "target": "d"},
                       {"source": "d", "target": "e"},
                       {"source": "a", "target": "d"},
                       {"source": "e", "target": "b"}],
             "animate_flow": True},
            # Unwrappable tokens.
            {"nodes": [{"id": "svc", "label": long_id},
                       {"id": "api", "label": url}],
             "edges": [{"source": "svc", "target": "api"}]},
            # Minimal: one box, nothing flowing.
            {"nodes": [{"id": "x", "label": "x"}], "animate_flow": True},
        ]
