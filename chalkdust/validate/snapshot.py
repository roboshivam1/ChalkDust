"""Component snapshots (SCENE_SPEC.md §11 rule 6): fail loudly when Manim shifts.

A snapshot records what a component builds for each of its examples(), probed
exactly as the geometric validator probes -- animations snapped to their end
state, no frame encoded. A carry-in consumer's case is built with its fixture
artifacts on screen (continuity.fixture_beat), so the carried target is part
of what is recorded.

Text metrics depend on which fonts are installed: the theme's Archivo / Inter
/ JetBrains Mono fall back to Arial / Courier New on a box without them, and
to Helvetica Neue / Menlo on a Mac. Components measure text and decide from
the measurement -- where a line breaks, which tick labels fit, whether a lens
or a frame shows the zoom, which half of STAGE a narrow column lands in -- so
the exact tree a component builds is itself font-dependent, not only its
coordinates. A snapshot therefore has two halves:

  structure       Compared on every machine. A pure projection (project())
                  of the exact record onto the facts no text measurement can
                  change: the timeline (each play's animation classes and run
                  time, each wait, each settle point's label) and, at every
                  settle point, the tree of on-screen mobjects with
                    - type and label of every node, colour and opacity of
                      every leaf, TeX sources verbatim, and each text leaf's
                      characters with whitespace removed (wrap() turns a
                      space into a newline, or breaks inside a word wider
                      than a line, and a repair rescales its width);
                    - every node and node boundary as built -- labelled
                      items (a given, a unit's label, a value label), table
                      cells, tick labels on a number line -- except inside a
                      line-wrap container (LAYOUT_CHOICES "lines": a
                      paragraph laid out in measured lines, each line a group
                      of text and TeX pieces). Such a container keeps its own
                      type and label and records only the *flow* it draws:
                      its runs in order, text merged across the pieces and
                      lines that split it, with colour; TeX verbatim. Where a
                      line breaks -- how many line groups, their labels,
                      which words each holds -- is measured; what is written,
                      in what order and colour, is not;
                    - the coarse regions (title_bar, stage, lower_third)
                      containing each top-level mobject. Membership of
                      stage_left / stage_right is a width test a narrow
                      column passes or fails by font;
                    - for each other decision in LAYOUT_CHOICES, only its
                      name in place of what it decided: which of a node's
                      labels survive (the rest of its children stay), or
                      which node is built (and so how it is animated).

  per fingerprint  Compared, per font fingerprint (platform + resolved
                  fonts), only on a machine whose fingerprint matches; skipped
                  elsewhere with the reason stated. Two maps keyed by
                  fingerprint:
                    layout    the exact timeline: every node's text with its
                              line breaks, every line group, every label a
                              layout choice kept, all regions, the label of
                              each animation's target. Compared exactly.
                    geometry  bounding box and effective font size of every
                              node in that tree, DFS order, compared within
                              GEOMETRY_TOL.
                  project(layout[fp]) equals the structure for every recorded
                  fingerprint; a regenerate that breaks that drops the stale
                  fingerprint with a note.

No projection survives a font so wide that a component refuses its example
(a LayoutError): that is a refusal the layout tests report, not drift.

Pixels are deliberately not snapshotted. They vary with fonts and rasteriser
noise, so a pixel test would be flaky and then ignored, which is worse than
none.

Regenerate only on purpose, with a reason; the reason is kept in the file:

    python -m chalkdust.validate.snapshot --reason "why" [--component NAME ...]

`--hide-font NAME` (repeatable) records as if NAME were not installed, so one
box can keep the fallback fingerprint's baseline up to date: hiding Archivo,
Inter and JetBrains Mono resolves the theme exactly as a box without them does
(theme.resolve_fonts' fallback chain), giving the fallback fingerprint.

A component that declares `snapshot_exempt` (RawScene: generated code, no
fixed visual) has no snapshot; regenerating one is refused.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

from manim import MarkupText, Mobject, SingleStringMathTex, Text, VMobject, config
from manim.mobject.mobject import _AnimationBuilder
from manim.mobject.svg.svg_mobject import VMobjectFromSVGPath

from chalkdust.continuity import CarryIn, beat_component, fixture_beat
from chalkdust.core.models import Region
from chalkdust.core.version import MANIM_VERSION
from chalkdust.scenes import theme as theme_mod
from chalkdust.scenes.components import Component, get_component, registered_names
from chalkdust.scenes.regions import bbox, region_rect
from chalkdust.scenes.theme import get_theme, resolve_fonts
from chalkdust.validate.geometric import LayoutProbe

DEFAULT_DIR = Path("tests") / "snapshots"
THEME = "default"
DURATION = 10.0      # seconds; any fixed value works, it only scales run times

GEOMETRY_TOL = 0.01  # Manim units; real layout shifts are tenths, noise is 1e-6
FONT_SIZE_TOL = 0.1
REGION_TOL = 0.05    # a bbox touching a region edge must not flip across fonts

# Regions whose membership is a placement decision, not a width test: a
# component puts a mobject in the title bar, on the stage or in the lower
# third. Whether it also fits inside one half of STAGE depends on how wide its
# text measures (seen: EquationDerivation's narrow lines and a FreeBodyDiagram
# label flip with the font), so the halves are font-dependent.
COARSE_REGIONS = (Region.TITLE_BAR.value, Region.STAGE.value, Region.LOWER_THIRD.value)

class Choice(NamedTuple):
    """A decision a component takes by measuring text (see LAYOUT_CHOICES)."""

    decides: str      # "lines", "labels" or "node"
    name: str         # what the structure records in place of the outcome
    labels: str = ""  # "labels" only: full-match pattern of the chosen children


# Decisions a component takes by measuring text, so their outcome is
# font-dependent. component -> {full-match pattern on a node's label: Choice}:
#   "lines"   the node is a line-wrap container: a paragraph laid out in
#             measured lines. The structure keeps its type and label and
#             records the flow it draws (see _flow) in place of its lines.
#   "labels"  which of the node's children whose label matches
#             `Choice.labels` survive. The structure keeps every other child
#             as built, and records {"name", "kept"} in place of the chosen
#             ones, "kept" being the type and paint of what survives, per
#             label stem (font-invariant: see GraphPlot below).
#   "node"    which node is built at all. The structure records the node as
#             {"layout_choice": name}, and an animation of it as
#             "layout_choice:<name>" (its class follows the node: a lens
#             grows by Transform, a frame is drawn by Create).
# The exact node, lines, labels and animation are kept per fingerprint.
# Anything not listed here is structure: a decision that turns out to depend
# on the font fails test_structure_unchanged on the other machine, loudly,
# rather than being absorbed by a rule broader than the decision.
LAYOUT_CHOICES: dict[str, dict[str, Choice]] = {
    # _Flow.lines() fills each line on measured widths, and _split_long_words
    # hard-breaks a word at a character count derived from the measured
    # _char_width: how many lines the statement, each given and the find
    # value take, and which words each line holds, depends on the font (seen:
    # example 0's "kinetic" moves down a line in Inter; example 2 re-splits).
    # The labelled items themselves -- the statement, given[i], find value --
    # and their order are the spec's, and stay structure. The statement is
    # revealed one play per line, so its line count also sets how many plays
    # the timeline has; it was the same (3, 1 and 2 lines) under every font
    # tried, so the timeline stays structure, and a font that re-counts the
    # statement's lines fails test_structure_unchanged rather than passing
    # unnoticed.
    "ProblemStatement": {
        "statement": Choice("lines", "statement lines"),
        r"given\[\d+\]": Choice("lines", "given lines"),
        "find value": Choice("lines", "find lines"),
    },
    # _pick_labels keeps a tick label only where its measured box clears the
    # curves, the other axis and its neighbours (seen: example 1's axes keep
    # 9 tick labels in Arial, 8 in Inter, 7 in Verdana). The axis lines and
    # one tick mark per tick are drawn whatever the font, and stay structure,
    # as does the type and colour of the labels each axis keeps (every axis
    # keeps at least MIN_AXIS_LABELS, all drawn the same way: tick_label
    # chooses LaTeX per axis, from its range). If too few labels fit an
    # interior axis they move to the plot edge, each with a tick mark of its
    # own; no example does that under any font tried (Arial, Inter, Verdana,
    # Times New Roman, Georgia, Segoe UI), so the tick-mark count stays
    # structure until one is seen to.
    "GraphPlot": {"axes": Choice("labels", "tick labels the axes keep",
                                 r"[xy]tick\[-?\d+\]")},
    # _settled_lens magnifies the focus in place, or beside the target in the
    # room STAGE leaves, or -- with no room for MIN_ZOOM -- frames it instead:
    # lens or frame depends on how wide the focus text measures (seen:
    # example 0 is a lens in Arial and Inter, a frame in Verdana).
    "ZoomHighlight": {"zoom lens": Choice("node", "lens or frame"),
                      "focus frame": Choice("node", "lens or frame")},
}

_WS = re.compile(r"\s+")


# --- capture ----------------------------------------------------------------


def snapshot_exempt(name: str) -> str | None:
    """The component's stated reason for having no snapshot, or None."""
    return get_component(name).snapshot_exempt or None


def snapshotted_names() -> list[str]:
    """Every registered component that must ship a snapshot."""
    return [n for n in registered_names() if not snapshot_exempt(n)]


def fingerprint() -> str:
    """What this machine's text geometry depends on."""
    t = resolve_fonts(get_theme(THEME), warn=False).type
    return f"{sys.platform}|{t.heading_font}|{t.body_font}|{t.mono_font}"


@contextmanager
def fonts_hidden(names: Iterable[str]) -> Iterator[None]:
    """Resolve fonts as if `names` were not installed.

    theme.resolve_fonts reads the installed families from
    theme._installed_fonts(); answering without `names` makes every scene
    built in the block resolve the theme exactly as a machine lacking them
    does, so its snapshot is that machine's (fingerprint() included). A name
    that is not installed is refused: hiding it would change nothing, and a
    baseline recorded "without" a font this box never had would mislabel it.
    """
    hide = frozenset(names)
    if not hide:
        yield
        return
    real = theme_mod._installed_fonts
    available = real()
    absent = sorted(hide - available)
    if absent:
        raise ValueError(f"not installed, so cannot be hidden: {absent}")
    theme_mod._installed_fonts = lambda: available - hide  # type: ignore[assignment]
    # With no installed fallback left for a role, resolve_fonts keeps the
    # theme's own name and lets Pango substitute: the fingerprint would then
    # name a font this box does not draw with (hiding Arial on a box that
    # has only the fallbacks would record "Archivo|Inter").
    unbacked = sorted(set(fingerprint().split("|")[1:]) - (available - hide))
    if unbacked:
        theme_mod._installed_fonts = real
        raise ValueError(f"hiding {sorted(hide)} leaves no installed fallback for "
                         f"{unbacked}, so the fingerprint would be mislabelled")
    try:
        yield
    finally:
        theme_mod._installed_fonts = real


def _target(animation: Any) -> str | None:
    """Label of the mobject an animation (or `.animate` builder) acts on."""
    return getattr(getattr(animation, "mobject", None), "_chalk_label", None)


class SnapshotProbe(LayoutProbe):
    """A LayoutProbe that records the timeline and every settle state."""

    def __init__(self, component: Component | CarryIn, **kwargs) -> None:
        super().__init__(component, **kwargs)
        self.timeline: list[dict[str, Any]] = []
        self.geometry: list[list[dict[str, Any]]] = []

    def play(self, *animations, **kwargs) -> None:  # type: ignore[override]
        own = max((getattr(a, "run_time", 1.0) for a in animations), default=1.0)
        self.timeline.append({
            "op": "play",
            # `.animate` builders are recorded as such: preparing one here
            # would build a second animation from the same builder.
            "animations": ["animate" if isinstance(a, _AnimationBuilder)
                           else type(a).__name__ for a in animations],
            "targets": [_target(a) for a in animations],
            "run_time": _r(kwargs.get("run_time", own)),
        })
        super().play(*animations, **kwargs)

    def wait(self, duration: float = 1.0, *args, **kwargs) -> None:  # type: ignore[override]
        self.timeline.append({"op": "wait", "duration": _r(duration)})
        super().wait(duration, *args, **kwargs)

    def settle(self, label: str = "settle point") -> None:
        tree, geometry = [], []
        for mob in self.mobjects:
            node = _exact(mob, geometry)
            box = bbox(mob)
            node["regions"] = [r.value for r in Region
                               if region_rect(r).contains(box, tol=REGION_TOL)]
            tree.append(node)
        self.timeline.append({"op": "settle", "label": label, "mobjects": tree})
        self.geometry.append(geometry)
        super().settle(label)


def _r(x: float, places: int = 3) -> float:
    return round(float(x), places)


def _exact(mob: Mobject, geometry: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact description of `mob` as built here; appends its geometry, DFS order."""
    box = bbox(mob)
    tag = getattr(mob, "_chalk_font_size", None)
    geometry.append({
        "bbox": [_r(box.x), _r(box.y), _r(box.width), _r(box.height)],
        "font_size": None if tag is None else _r(tag, 2),
    })

    node: dict[str, Any] = {"type": type(mob).__name__}
    if hasattr(mob, "_chalk_label"):
        node["label"] = mob._chalk_label  # type: ignore[attr-defined]
    if isinstance(mob, (Text, MarkupText)):
        node["text"] = mob.original_text
    elif isinstance(mob, SingleStringMathTex):
        node["tex"] = mob.tex_string

    # Stop at text and at runs of bare glyph outlines: their count can change
    # with the font (ligatures), so it is not structure.
    glyphs = all(isinstance(c, VMobjectFromSVGPath) for c in mob.submobjects)
    if isinstance(mob, (Text, MarkupText, SingleStringMathTex)) or glyphs:
        node.update(_paint(mob))
    else:
        node["children"] = [_exact(c, geometry) for c in mob.submobjects]
    return node


def _paint(mob: Mobject) -> dict[str, Any]:
    """Colour and opacity, read from the first family member with points."""
    for m in mob.get_family():
        if isinstance(m, VMobject) and len(m.points):
            colour = m.get_fill_color() or m.get_stroke_color()
            return {
                "color": colour.to_hex() if colour is not None else None,
                "fill_opacity": _r(m.get_fill_opacity()),
                "stroke_opacity": _r(m.get_stroke_opacity()),
            }
    return {}


# --- projection: the font-independent half -------------------------------------

_PAINT = ("color", "fill_opacity", "stroke_opacity")
_STEM = re.compile(r"\[[^\]]*\]$")


def _choice(choices: dict[str, Choice], label: str | None) -> Choice | None:
    if label is None:
        return None
    return next((c for pattern, c in choices.items() if re.fullmatch(pattern, label)),
                None)


def _paint_of(node: dict[str, Any]) -> dict[str, Any]:
    return {k: node[k] for k in _PAINT if k in node}


def _flow(node: dict[str, Any]) -> list[dict[str, Any]]:
    """The runs a line-wrap container draws, independent of where it broke.

    Leaves in drawing order; text has its whitespace removed (a break may
    fall between words or, for a word wider than a line, inside one) and
    adjacent text leaves of the same type and paint merge into one run --
    a run of prose is split into one piece per line it spans. TeX is never
    broken by layout, so each source stays its own run, verbatim.
    """
    runs: list[dict[str, Any]] = []

    def leaves(n: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if "children" in n:
            for c in n["children"]:
                yield from leaves(c)
        else:
            yield n

    for leaf in leaves(node):
        paint = _paint_of(leaf)
        if "tex" in leaf:
            runs.append({"type": leaf["type"], "tex": leaf["tex"], **paint})
            continue
        if "text" not in leaf:
            runs.append({"type": leaf["type"], **paint})
            continue
        text = _WS.sub("", leaf["text"])
        if not text:
            continue
        last = runs[-1] if runs else None
        if (last is not None and "text" in last and last["type"] == leaf["type"]
                and _paint_of(last) == paint):
            last["text"] += text
        else:
            runs.append({"type": leaf["type"], "text": text, **paint})
    return runs


def _kept(chosen: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Type and paint of the children a "labels" choice kept, per label stem
    (xtick[3] -> xtick), each distinct kind once."""
    kinds: dict[str, dict[str, dict[str, Any]]] = {}
    for c in chosen:
        kind = {"type": c["type"], **_paint_of(c)}
        kinds.setdefault(_STEM.sub("", c["label"]), {})[
            json.dumps(kind, sort_keys=True)] = kind
    return {stem: [k for _, k in sorted(by.items())] for stem, by in sorted(kinds.items())}


def _project_node(node: dict[str, Any], choices: dict[str, Choice]) -> dict[str, Any]:
    choice = _choice(choices, node.get("label"))
    if choice and choice.decides == "node":
        return {"layout_choice": choice.name}
    out = {k: v for k, v in node.items()
           if k not in ("children", "text", "tex", "regions")}
    if choice and choice.decides == "lines":
        # Paint lives on the runs; the container keeps type and label.
        for k in _PAINT:
            out.pop(k, None)
        out["flow"] = _flow(node)
        return out
    if "text" in node:
        out["text"] = _WS.sub("", node["text"])
    if "tex" in node:
        out["tex"] = node["tex"]
    if "children" in node:
        children = node["children"]
        if choice and choice.decides == "labels":
            def chosen(c: dict[str, Any]) -> bool:
                return re.fullmatch(choice.labels, c.get("label") or "") is not None
            out["layout_choice"] = {"name": choice.name,
                                    "kept": _kept([c for c in children if chosen(c)])}
            children = [c for c in children if not chosen(c)]
        out["children"] = [_project_node(c, choices) for c in children]
    return out


def project(name: str, timeline: list[dict[str, Any]]) -> dict[str, Any]:
    """The structure: what `name`'s exact timeline records that no text
    measurement can change (see the module docstring)."""
    choices = LAYOUT_CHOICES.get(name, {})

    def built(target: str | None) -> str | None:
        choice = _choice(choices, target)
        return choice.name if choice and choice.decides == "node" else None

    out = []
    for op in timeline:
        if op["op"] == "play":
            targets = op.get("targets") or [None] * len(op["animations"])
            out.append({
                "op": "play",
                "animations": [f"layout_choice:{built(t)}" if built(t) else a
                               for a, t in zip(op["animations"], targets)],
                "run_time": op["run_time"],
            })
        elif op["op"] == "settle":
            mobjects = []
            for node in op["mobjects"]:
                p = _project_node(node, choices)
                p["regions"] = [r for r in node.get("regions", []) if r in COARSE_REGIONS]
                mobjects.append(p)
            out.append({"op": "settle", "label": op["label"], "mobjects": mobjects})
        else:
            out.append(dict(op))
    return {"timeline": out}


def capture_all(name: str, params: dict[str, Any]) -> tuple[dict, list, list]:
    """(structure, exact timeline, geometry) for one component instance --
    built as the pipeline builds its beat (continuity.fixture_beat): a
    carry-in consumer's case with its fixture artifacts on screen first, so
    the snapshot is the frame it draws and records the carried target with
    it."""
    spec, recipes = fixture_beat(name, params)
    probe = SnapshotProbe(beat_component(spec, recipes), theme=THEME,
                          duration=DURATION, strict=False)
    probe.construct()
    return project(name, probe.timeline), probe.timeline, probe.geometry


def capture(name: str, params: dict[str, Any]) -> tuple[dict, list]:
    """(structure, geometry) for one component instance; see capture_all."""
    structure, _, geometry = capture_all(name, params)
    return structure, geometry


# --- compare ----------------------------------------------------------------


def diff_structure(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Exact comparison, reported as path: expected -> actual."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual), key=str):
            out += diff_structure(expected.get(key), actual.get(key), f"{path}.{key}")
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: {len(expected)} items -> {len(actual)}"]
        out = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            out += diff_structure(e, a, f"{path}[{i}]")
        return out
    return [] if expected == actual else [f"{path}: {expected!r} -> {actual!r}"]


def diff_geometry(expected: list, actual: list) -> list[str]:
    """Same shape required; numbers compared within tolerance."""
    if len(expected) != len(actual):
        return [f"settle points: {len(expected)} -> {len(actual)}"]
    out = []
    for s, (exp_nodes, act_nodes) in enumerate(zip(expected, actual)):
        if len(exp_nodes) != len(act_nodes):
            out.append(f"settle[{s}]: {len(exp_nodes)} nodes -> {len(act_nodes)}")
            continue
        for n, (e, a) in enumerate(zip(exp_nodes, act_nodes)):
            if any(abs(x - y) > GEOMETRY_TOL for x, y in zip(e["bbox"], a["bbox"])):
                out.append(f"settle[{s}] node[{n}] bbox: {e['bbox']} -> {a['bbox']}")
            ef, af = e["font_size"], a["font_size"]
            if (ef is None) != (af is None) or (
                    ef is not None and abs(ef - af) > FONT_SIZE_TOL):
                out.append(f"settle[{s}] node[{n}] font_size: {ef} -> {af}")
    return out


# --- files ------------------------------------------------------------------


def snapshot_path(name: str, directory: Path = DEFAULT_DIR) -> Path:
    return Path(directory) / f"{name}.json"


def load(name: str, directory: Path = DEFAULT_DIR) -> dict | None:
    path = snapshot_path(name, directory)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def recorded_layout(case: dict[str, Any]) -> dict[str, list]:
    """A recorded case's exact timeline, per fingerprint.

    Files written before the structure/layout split kept the exact timeline
    as "structure" (one shared tree: every recorded fingerprint had to match
    it) and only geometry per fingerprint; that reads losslessly as the same
    exact timeline for each of those fingerprints.
    """
    if "layout" in case:
        return case["layout"]
    return {fp: case["structure"]["timeline"] for fp in case.get("geometry", {})}


def regenerate(name: str, reason: str, directory: Path = DEFAULT_DIR) -> list[str]:
    """Rewrite one component's snapshot from this machine. Returns notes.

    Other fingerprints' layout and geometry are kept for a case whose params
    are unchanged and whose exact timeline still projects to this machine's
    structure, and dropped (with a note) otherwise -- they describe a build
    that no longer exists. A reason identical to the last one recorded is
    not repeated: recording the same change under a second font set (see
    --hide-font) is one regeneration.
    """
    if snapshot_exempt(name):
        raise ValueError(f"{name} is exempt from snapshots: {snapshot_exempt(name)}")
    old = load(name, directory) or {"cases": [], "reasons": []}
    old_cases = old["cases"]
    fp = fingerprint()
    notes, cases = [], []
    for i, params in enumerate(get_component(name).examples()):
        structure, timeline, geometry = capture_all(name, params)
        prior = old_cases[i] if i < len(old_cases) else None
        layout_kept: dict[str, Any] = {}
        geometry_kept: dict[str, Any] = {}
        others = sorted(set(prior["geometry"]) - {fp}) if prior else []
        if prior and prior["params"] == params:
            layouts = recorded_layout(prior)
            for other in others:
                if other in layouts and project(name, layouts[other]) == structure:
                    layout_kept[other] = layouts[other]
                    geometry_kept[other] = prior["geometry"][other]
                else:
                    notes.append(f"{name} case {i}: structure changed; dropped the "
                                 f"layout and geometry for {other}")
        elif others:
            notes.append(f"{name} case {i}: params changed; dropped the layout and "
                         f"geometry for {others}")
        cases.append({
            "params": params,
            "structure": structure,
            "layout": dict(sorted({**layout_kept, fp: timeline}.items())),
            "geometry": dict(sorted({**geometry_kept, fp: geometry}.items())),
        })

    reasons = list(old["reasons"])
    if not reasons or reasons[-1] != reason:
        reasons.append(reason)
    data = {
        "component": name,
        "manim_version": MANIM_VERSION,
        "reasons": reasons,
        "cases": cases,
    }
    path = snapshot_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF endings regardless of platform, so a regeneration diffs cleanly.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m chalkdust.validate.snapshot",
        description="Regenerate component snapshots. Only with a stated reason.")
    parser.add_argument("--reason", required=True,
                        help="why the snapshot is changing; recorded in the file")
    parser.add_argument("--component", action="append",
                        help="component name; repeatable (default: all)")
    parser.add_argument("--hide-font", action="append", default=[], metavar="FAMILY",
                        help="record as if this installed font family were absent "
                             "(repeatable); hiding the theme fonts records the "
                             "fallback fingerprint")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)
    if not args.reason.strip():
        parser.error("--reason must say why")

    exempt = [n for n in args.component or [] if snapshot_exempt(n)]
    if exempt:
        parser.error(f"exempt from snapshots: {exempt}")

    absent = sorted(set(args.hide_font) - theme_mod._installed_fonts())
    if absent:
        parser.error(f"not installed, so cannot be hidden: {absent}")

    config.verbosity = "WARNING"
    with fonts_hidden(args.hide_font):
        for name in args.component or snapshotted_names():
            for note in regenerate(name, args.reason.strip(), args.dir):
                print(note)
            print(f"wrote {snapshot_path(name, args.dir)} [{fingerprint()}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
