"""Component snapshots (SCENE_SPEC.md §11 rule 6): fail loudly when Manim shifts.

A snapshot records what a component builds for each of its examples() -- and,
for a component that acts on a carried artifact, each carried_examples() case,
built with its artifacts on screen as the render builds it -- probed exactly as
the geometric validator probes -- animations snapped to their end
state, no frame encoded. It has two halves:

  structure  Font-independent, compared on every machine. The timeline (each
             play's animation classes and run time, each wait, each settle
             point) and, at every settle point, the tree of on-screen
             mobjects: type, label, text or TeX source, colour, opacity, and
             which layout regions contain each top-level mobject.

  geometry   Font-dependent. Bounding box and effective font size of every node
             in that tree. Text metrics depend on which fonts are installed --
             the theme's Archivo / Inter / JetBrains Mono fall back to Arial /
             Courier New on a box without them, and to Helvetica Neue / Menlo
             on a Mac -- so geometry is stored per fingerprint (platform +
             resolved fonts) and compared, within GEOMETRY_TOL, only on a
             machine with a matching fingerprint. Elsewhere that half is
             skipped with the reason stated.

Pixels are deliberately not snapshotted. They vary with fonts and rasteriser
noise, so a pixel test would be flaky and then ignored, which is worse than
none.

Regenerate only on purpose, with a reason; the reason is kept in the file:

    python -m chalkdust.validate.snapshot --reason "why" [--component NAME ...]

A component that declares `snapshot_exempt` (RawScene: generated code, no
fixed visual) has no snapshot; regenerating one is refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from manim import MarkupText, Mobject, SingleStringMathTex, Text, VMobject, config
from manim.mobject.mobject import _AnimationBuilder
from manim.mobject.svg.svg_mobject import VMobjectFromSVGPath

from chalkdust.continuity import ArtifactRecipe, CarryIn
from chalkdust.core.models import Region
from chalkdust.core.version import MANIM_VERSION
from chalkdust.scenes.components import Component, get_component, registered_names
from chalkdust.scenes.regions import bbox, region_rect
from chalkdust.scenes.theme import get_theme, resolve_fonts
from chalkdust.validate.fixtures import FixtureCase, fixture_cases, lent_builders
from chalkdust.validate.geometric import LayoutProbe

DEFAULT_DIR = Path("tests") / "snapshots"
THEME = "default"
DURATION = 10.0      # seconds; any fixed value works, it only scales run times

GEOMETRY_TOL = 0.01  # Manim units; real layout shifts are tenths, noise is 1e-6
FONT_SIZE_TOL = 0.1
REGION_TOL = 0.05    # a bbox touching a region edge must not flip across fonts


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


class SnapshotProbe(LayoutProbe):
    """A LayoutProbe that records the timeline and every settle state."""

    def __init__(self, component: Component, **kwargs) -> None:
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
            "run_time": _r(kwargs.get("run_time", own)),
        })
        super().play(*animations, **kwargs)

    def wait(self, duration: float = 1.0, *args, **kwargs) -> None:  # type: ignore[override]
        self.timeline.append({"op": "wait", "duration": _r(duration)})
        super().wait(duration, *args, **kwargs)

    def settle(self, label: str = "settle point") -> None:
        structure, geometry = [], []
        for mob in self.mobjects:
            node = _structure(mob, geometry)
            box = bbox(mob)
            node["regions"] = [r.value for r in Region
                               if region_rect(r).contains(box, tol=REGION_TOL)]
            structure.append(node)
        self.timeline.append({"op": "settle", "label": label, "mobjects": structure})
        self.geometry.append(geometry)
        super().settle(label)


def _r(x: float, places: int = 3) -> float:
    return round(float(x), places)


def _structure(mob: Mobject, geometry: list[dict[str, Any]]) -> dict[str, Any]:
    """Font-independent description of `mob`; appends its geometry, DFS order."""
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
        node["children"] = [_structure(c, geometry) for c in mob.submobjects]
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


def capture(name: str, params: dict[str, Any],
            carry_in: tuple[ArtifactRecipe, ...] = ()) -> tuple[dict, list]:
    """(structure, geometry) for one component instance, built with the
    artifacts it carries in (if any) on screen first, as the render does."""
    component = get_component(name)(params)
    probe = SnapshotProbe(CarryIn(component, carry_in) if carry_in else component,
                          theme=THEME, duration=DURATION, strict=False)
    with lent_builders(name):
        probe.construct()
    return {"timeline": probe.timeline}, probe.geometry


def snapshot_cases(name: str) -> list[FixtureCase]:
    """The cases a snapshot records: examples(), then carried_examples()."""
    return fixture_cases(name, "examples")


def case_key(case: FixtureCase) -> dict[str, Any]:
    """What identifies a recorded case. `carry_in` is recorded only for a
    carried case, so snapshots of plain components are unchanged by it."""
    key: dict[str, Any] = {"params": case.params}
    if case.carry_in:
        key["carry_in"] = case.carry_in_json()
    return key


def recorded_key(case: dict[str, Any]) -> dict[str, Any]:
    """case_key() of a case as stored in a snapshot file."""
    return {k: case[k] for k in ("params", "carry_in") if k in case}


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


def regenerate(name: str, reason: str, directory: Path = DEFAULT_DIR) -> list[str]:
    """Rewrite one component's snapshot from this machine. Returns notes.

    Other machines' geometry is kept for a case whose params and structure
    are unchanged, and dropped (with a note) where they changed -- it
    describes a layout that no longer exists.
    """
    if snapshot_exempt(name):
        raise ValueError(f"{name} is exempt from snapshots: {snapshot_exempt(name)}")
    old = load(name, directory) or {"cases": [], "reasons": []}
    old_cases = old["cases"]
    fp = fingerprint()
    notes, cases = [], []
    for i, case in enumerate(snapshot_cases(name)):
        structure, geometry = capture(name, case.params, case.carry_in)
        key = case_key(case)
        prior = old_cases[i] if i < len(old_cases) else None
        kept: dict[str, Any] = {}
        if prior and recorded_key(prior) == key and prior["structure"] == structure:
            kept = {k: v for k, v in prior["geometry"].items() if k != fp}
        elif prior and set(prior["geometry"]) - {fp}:
            notes.append(f"{name} case {i}: structure changed; dropped geometry for "
                         f"{sorted(set(prior['geometry']) - {fp})}")
        cases.append({**key, "structure": structure,
                      "geometry": {**kept, fp: geometry}})

    data = {
        "component": name,
        "manim_version": MANIM_VERSION,
        "reasons": old["reasons"] + [reason],
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
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)
    if not args.reason.strip():
        parser.error("--reason must say why")

    exempt = [n for n in args.component or [] if snapshot_exempt(n)]
    if exempt:
        parser.error(f"exempt from snapshots: {exempt}")

    config.verbosity = "WARNING"
    for name in args.component or snapshotted_names():
        for note in regenerate(name, args.reason.strip(), args.dir):
            print(note)
        print(f"wrote {snapshot_path(name, args.dir)} [{fingerprint()}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
