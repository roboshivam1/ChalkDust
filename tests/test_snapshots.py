"""Snapshot tests (SCENE_SPEC.md §11 rule 6).

Walks the registry like test_layout.py: every component's examples() must
match its recorded snapshot in tests/snapshots/. A component without one
fails -- shipping a snapshot is part of shipping a component (§5), so a
missing file is a missing deliverable, not a reason to skip. Component
branches written before this harness existed carry none: merging one
requires recording its snapshot in the same step, from the repo root:

    python -m chalkdust.validate.snapshot --reason "<Name> merged" --component <Name>

The structure is compared on every machine; the exact layout and geometry
only where the font fingerprint matches (see chalkdust/validate/snapshot.py
for which facts are which, and why).
"""

from __future__ import annotations

import copy
import json
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from chalkdust.scenes import theme as theme_mod
from chalkdust.scenes.components import get_component, registered_names
from chalkdust.validate.snapshot import (
    capture,
    capture_all,
    diff_geometry,
    diff_structure,
    fingerprint,
    fonts_hidden,
    load,
    main,
    project,
    recorded_layout,
    regenerate,
    snapshot_exempt,
    snapshot_path,
    snapshotted_names,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _regen(name: str) -> str:
    return (f'python -m chalkdust.validate.snapshot --reason "<why>" '
            f'--component {name}')


@lru_cache(maxsize=None)
def _current(name: str) -> list[tuple[dict, list, list]]:
    return [capture_all(name, params) for params in get_component(name).examples()]


def _recorded(name: str) -> dict:
    snap = load(name, SNAPSHOT_DIR)
    if snap is None:
        pytest.fail(f"{name} has no snapshot; shipping one is part of shipping "
                    f"the component (SCENE_SPEC.md §5). Record it: {_regen(name)}")
    recorded = [case["params"] for case in snap["cases"]]
    if recorded != get_component(name).examples():
        pytest.fail(f"{name}.examples() changed since its snapshot was recorded. "
                    f"If intended: {_regen(name)}")
    return snap


def _skip_without_baseline(name: str, snap: dict) -> str:
    fp = fingerprint()
    if any(fp not in case["geometry"] or fp not in recorded_layout(case)
           for case in snap["cases"]):
        recorded = sorted({k for case in snap["cases"] for k in case["geometry"]})
        pytest.skip(
            f"no layout/geometry baseline for this machine ({fp}); recorded for "
            f"{recorded}. Text metrics depend on the installed fonts, so the "
            f"exact layout and geometry are compared only where they match. "
            f"Add one: {_regen(name)}")
    return fp


@pytest.mark.parametrize("name", registered_names())
def test_component_is_snapshotted_or_formally_exempt(name):
    # A snapshot with no cases compares nothing and passes forever; that is
    # how RawScene.json sat vacuous. Every component either records at least
    # one case, or declares why it cannot (and then ships no snapshot file
    # that would claim coverage it does not give).
    reason = snapshot_exempt(name)
    if reason:
        assert reason.strip(), f"{name}: an exemption must state its reason"
        assert get_component(name).examples() == [], (
            f"{name} is exempt but declares examples(); snapshot them instead")
        assert load(name, SNAPSHOT_DIR) is None, (
            f"{name} is exempt but tests/snapshots/{name}.json exists")
    else:
        assert _recorded(name)["cases"], (
            f"{name}'s snapshot records no cases, so it checks nothing; give "
            f"{name} examples() and record it: {_regen(name)}")


@pytest.mark.parametrize("name", snapshotted_names())
class TestLibrary:
    def test_structure_unchanged(self, name):
        snap = _recorded(name)
        for i, (case, (structure, _, _)) in enumerate(zip(snap["cases"], _current(name))):
            diffs = diff_structure(case["structure"], structure)
            assert not diffs, (
                f"{name} example {i} built differently from its snapshot. If "
                f"Manim or the component changed on purpose: {_regen(name)}\n"
                + "\n".join(diffs[:20]))

    def test_layout_unchanged(self, name):
        # The exact tree -- line breaks, the tick labels kept, all regions --
        # on a machine whose fonts match the recording.
        snap = _recorded(name)
        fp = _skip_without_baseline(name, snap)
        for i, (case, (_, timeline, _)) in enumerate(zip(snap["cases"], _current(name))):
            diffs = diff_structure(recorded_layout(case)[fp], timeline)
            assert not diffs, (
                f"{name} example {i} lays out differently from its snapshot "
                f"under {fp}. If Manim or the component changed on purpose: "
                f"{_regen(name)}\n" + "\n".join(diffs[:20]))

    def test_geometry_unchanged(self, name):
        snap = _recorded(name)
        fp = _skip_without_baseline(name, snap)
        for i, (case, (_, _, geometry)) in enumerate(zip(snap["cases"], _current(name))):
            diffs = diff_geometry(case["geometry"][fp], geometry)
            assert not diffs, (
                f"{name} example {i} lays out differently from its snapshot. If "
                f"Manim or the component changed on purpose: {_regen(name)}\n"
                + "\n".join(diffs[:20]))

    def test_every_recorded_font_set_agrees_on_the_structure(self, name):
        # Pure data, no build: each fingerprint's exact layout must project
        # to the one shared structure, and pair node for node with its
        # geometry. A fingerprint whose layout projected differently would
        # make test_structure_unchanged fail on that machine.
        snap = _recorded(name)
        for i, case in enumerate(snap["cases"]):
            layouts = recorded_layout(case)
            assert set(layouts) == set(case["geometry"]), (
                f"{name} example {i}: layout and geometry recorded for different "
                f"fingerprints: {sorted(layouts)} vs {sorted(case['geometry'])}")
            for fp, timeline in layouts.items():
                diffs = diff_structure(case["structure"], project(name, timeline))
                assert not diffs, (
                    f"{name} example {i}: the {fp} layout projects to a different "
                    f"structure. Regenerate: {_regen(name)}\n" + "\n".join(diffs[:20]))
                settles = [op for op in timeline if op["op"] == "settle"]
                assert [_count(op["mobjects"]) for op in settles] == [
                    len(nodes) for nodes in case["geometry"][fp]], (
                    f"{name} example {i}: {fp} geometry does not pair with its layout")


def _count(nodes: list[dict]) -> int:
    return sum(1 + _count(n.get("children", [])) for n in nodes)


@pytest.fixture(scope="module")
def title():
    return capture("TitleCard", get_component("TitleCard").examples()[1])


def _text(s: str, color: str = "#E6EDF3", label: str | None = None) -> dict:
    node = {"type": "Text", "text": s, "color": color,
            "fill_opacity": 1.0, "stroke_opacity": 1.0}
    if label:
        node["label"] = label
    return node


def _tex(s: str, color: str = "#E6EDF3") -> dict:
    return {"type": "MathTex", "tex": s, "color": color,
            "fill_opacity": 1.0, "stroke_opacity": 1.0}


def _line(i: int, *runs: dict) -> dict:
    return {"type": "VGroup", "label": f"statement[{i}]", "children": list(runs)}


def _settle(*mobjects: dict, regions=("stage",)) -> list[dict]:
    return [{"op": "settle", "label": "shown",
             "mobjects": [{**m, "regions": list(regions)} for m in mobjects]}]


def _paragraph(*lines: dict) -> dict:
    return {"type": "VGroup", "label": "statement", "children": list(lines)}


class TestHarness:
    """The harness must catch drift, tolerate noise, and only regenerate on
    purpose."""

    def test_missing_snapshot_fails_naming_the_record_command(self, monkeypatch):
        # A component merged without its snapshot must fail -- not skip -- and
        # say exactly how to record one, or nobody ever will.
        monkeypatch.setattr(sys.modules[__name__], "load", lambda name, d: None)
        with pytest.raises(pytest.fail.Exception) as failed:
            _recorded("TitleCard")
        assert ("python -m chalkdust.validate.snapshot --reason"
                in str(failed.value))
        assert "--component TitleCard" in str(failed.value)

    def test_layout_drift_is_caught(self, title):
        _, geometry = title
        moved = copy.deepcopy(geometry)
        moved[0][0]["bbox"][0] += 0.1
        assert diff_geometry(geometry, moved)

    def test_float_noise_is_tolerated(self, title):
        _, geometry = title
        nudged = copy.deepcopy(geometry)
        nudged[0][0]["bbox"][0] += 0.001
        assert diff_geometry(geometry, nudged) == []

    def test_font_size_drift_is_caught(self, title):
        _, geometry = title
        shrunk = copy.deepcopy(geometry)
        node = next(n for n in shrunk[0] if n["font_size"] is not None)
        node["font_size"] -= 2
        assert diff_geometry(geometry, shrunk)

    def test_timing_change_is_caught(self, title):
        structure, _ = title
        changed = copy.deepcopy(structure)
        changed["timeline"][0]["run_time"] += 0.5
        assert diff_structure(structure, changed)

    def test_exempt_component_cannot_be_regenerated(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--reason", "x", "--component", "RawScene", "--dir", str(tmp_path)])
        assert not snapshot_path("RawScene", tmp_path).exists()

    @pytest.mark.parametrize("argv", [[], ["--reason", "   "]])
    def test_regenerating_requires_a_reason(self, tmp_path, argv):
        with pytest.raises(SystemExit):
            main(argv + ["--component", "TitleCard", "--dir", str(tmp_path)])
        assert not snapshot_path("TitleCard", tmp_path).exists()

    def _with_other_machine(self, tmp_path, mutate_structure: bool) -> Path:
        regenerate("TitleCard", "first", tmp_path)
        path = snapshot_path("TitleCard", tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        case = data["cases"][0]
        case["geometry"]["other|A|B|C"] = case["geometry"][fingerprint()]
        case["layout"]["other|A|B|C"] = copy.deepcopy(case["layout"][fingerprint()])
        if mutate_structure:
            case["layout"]["other|A|B|C"][0]["run_time"] = 99.0
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_regenerate_keeps_other_machines_when_layout_is_unchanged(self, tmp_path):
        path = self._with_other_machine(tmp_path, mutate_structure=False)
        assert regenerate("TitleCard", "second", tmp_path) == []
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "other|A|B|C" in data["cases"][0]["geometry"]
        assert "other|A|B|C" in data["cases"][0]["layout"]
        assert data["reasons"] == ["first", "second"]

    def test_regenerate_drops_other_machines_when_layout_changed(self, tmp_path):
        path = self._with_other_machine(tmp_path, mutate_structure=True)
        notes = regenerate("TitleCard", "second", tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "other|A|B|C" not in data["cases"][0]["geometry"]
        assert "other|A|B|C" not in data["cases"][0]["layout"]
        assert notes and "structure changed" in notes[0]

    def test_regenerate_keeps_a_machine_whose_line_breaks_differ(self, tmp_path):
        # Another font breaks the same text elsewhere: its exact layout
        # differs but projects to the same structure, so its baseline stays.
        regenerate("TitleCard", "first", tmp_path)
        path = snapshot_path("TitleCard", tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        case = data["cases"][0]
        other = copy.deepcopy(case["layout"][fingerprint()])
        leaf = next(n for op in other if op["op"] == "settle"
                    for n in _leaves(op["mobjects"]) if " " in n.get("text", ""))
        leaf["text"] = leaf["text"].replace(" ", "\n", 1)
        case["layout"]["other|A|B|C"] = other
        case["geometry"]["other|A|B|C"] = case["geometry"][fingerprint()]
        path.write_text(json.dumps(data), encoding="utf-8")
        assert regenerate("TitleCard", "second", tmp_path) == []
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["cases"][0]["layout"]["other|A|B|C"] == other

    def test_regenerate_reads_a_pre_split_file(self, tmp_path):
        # Before the split a file held the exact tree as "structure" and bare
        # geometry per fingerprint. Regenerating one keeps another machine's
        # baseline (its exact tree still projects to the structure).
        regenerate("TitleCard", "first", tmp_path)
        path = snapshot_path("TitleCard", tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        fp = fingerprint()
        for case in data["cases"]:
            exact = case.pop("layout")[fp]
            for op in exact:
                op.pop("targets", None)
            case["structure"] = {"timeline": exact}
            case["geometry"] = {"other|A|B|C": case["geometry"][fp]}
        path.write_text(json.dumps(data), encoding="utf-8")
        assert regenerate("TitleCard", "second", tmp_path) == []
        data = json.loads(path.read_text(encoding="utf-8"))
        assert all(set(c["geometry"]) == set(c["layout"]) == {"other|A|B|C", fp}
                   for c in data["cases"])

    def test_same_reason_twice_is_recorded_once(self, tmp_path):
        # Recording one change under a second font set (--hide-font) is one
        # regeneration, not two.
        regenerate("TitleCard", "fonts", tmp_path)
        regenerate("TitleCard", "fonts", tmp_path)
        assert load("TitleCard", tmp_path)["reasons"] == ["fonts"]


def _leaves(nodes: list[dict]):
    for n in nodes:
        if "children" in n:
            yield from _leaves(n["children"])
        else:
            yield n


class TestProjection:
    """The structure keeps what the spec decides and drops what measuring
    text decides -- each dropped fact below is a difference seen between
    Arial, Inter/Archivo and wider or narrower fonts, and each kept one was
    the same under all of them."""

    WRAPPED = _settle(_paragraph(
        _line(0, _text("A block of mass"), _tex("m"), _text("is released from rest")),
        _line(1, _text("at the top of a rough plane inclined at"), _tex(r"\theta"))))

    def test_rewrapping_a_paragraph_keeps_the_structure(self):
        # ProblemStatement under Inter: "kinetic" moved to the next line; under
        # Verdana a line group was added. Neither is structure.
        rewrapped = _settle(_paragraph(
            _line(0, _text("A block of mass"), _tex("m")),
            _line(1, _text("is released from rest at the top")),
            _line(2, _text("of a rough plane inclined"), _text("at"), _tex(r"\theta"))))
        assert project("ProblemStatement", self.WRAPPED) == project(
            "ProblemStatement", rewrapped)

    def test_a_word_broken_mid_word_keeps_the_structure(self):
        # A word wider than a line is hard-broken where the measured width
        # falls (ProblemStatement._split_long_words, wrap()).
        one = _settle(_text("Supercalifragilistic word", label="callout"))
        two = _settle(_text("Supercalifrag\nilistic word", label="callout"))
        assert project("Callout", one) == project("Callout", two)

    @pytest.mark.parametrize("edit", [
        lambda p: p[0]["mobjects"][0]["children"][0]["children"][0].update(text="A block of masses"),
        lambda p: p[0]["mobjects"][0]["children"][0]["children"][1].update(tex="M"),
        lambda p: p[0]["mobjects"][0]["children"][1]["children"][0].update(color="#58A6FF"),
        lambda p: p[0]["mobjects"][0].update(label="statement 2"),
        lambda p: p[0]["mobjects"][0]["children"][0]["children"].reverse(),
    ], ids=["text", "tex", "colour", "label", "order"])
    def test_what_is_written_is_still_structure(self, edit):
        changed = copy.deepcopy(self.WRAPPED)
        edit(changed)
        assert diff_structure(project("ProblemStatement", self.WRAPPED),
                              project("ProblemStatement", changed))

    def test_half_of_stage_is_not_structure_but_the_region_is(self):
        # A narrow column falls inside STAGE_LEFT in one font and not another
        # (EquationDerivation, FreeBodyDiagram under Times New Roman).
        node = _text("x", label="eq")
        stage = project("EquationDerivation", _settle(node, regions=("stage",)))
        left = project("EquationDerivation",
                       _settle(node, regions=("stage", "stage_left")))
        lower = project("EquationDerivation", _settle(node, regions=("lower_third",)))
        assert stage == left
        assert stage != lower

    LINE = {"type": "Line", "color": "#8B949E", "fill_opacity": 0.0, "stroke_opacity": 1.0}
    TICKS = [_text(str(k), "#8B949E", f"xtick[{k}]") for k in (-2, -1, 1, 2)]
    AXES = {"type": "VGroup", "label": "axes", "children": [LINE, LINE, LINE, *TICKS]}

    def test_which_tick_labels_survive_is_not_structure(self):
        # GraphPlot example 1 keeps 9 tick labels in Arial and 8 in Inter.
        dropped = {**self.AXES, "children": [self.LINE] * 3 + self.TICKS[1:]}
        assert project("GraphPlot", _settle(self.AXES)) == project(
            "GraphPlot", _settle(dropped))
        # ...but only GraphPlot's axes make that choice; elsewhere it is drift.
        assert project("NumberLineWalk", _settle(self.AXES)) != project(
            "NumberLineWalk", _settle(dropped))

    @pytest.mark.parametrize("edit", [
        lambda ch: ch.pop(0),
        lambda ch: ch.pop(2),
        lambda ch: ch.__setitem__(0, {**ch[0], "color": "#FF0000"}),
        lambda ch: ch.__setitem__(slice(3, None), [{**t, "color": "#FF0000"} for t in ch[3:]]),
        lambda ch: ch.__setitem__(slice(3, None), [_tex(t["text"], "#8B949E") | {
            "label": t["label"]} for t in ch[3:]]),
    ], ids=["axis line deleted", "tick mark deleted", "axis line recoloured",
            "tick labels recoloured", "tick labels typeset differently"])
    def test_axis_lines_and_tick_marks_are_still_structure(self, edit):
        # The axes, and one tick mark per tick, are drawn whatever the font:
        # only which labels survive is the layout's choice. Losing or
        # recolouring a line, or the labels changing kind, is drift.
        changed = copy.deepcopy(self.AXES)
        edit(changed["children"])
        assert diff_structure(project("GraphPlot", _settle(self.AXES)),
                              project("GraphPlot", _settle(changed)))

    GIVEN = {"type": "VGroup", "label": "given", "children": [
        _text("GIVEN", "#8B949E"),
        {"type": "VGroup", "label": "given[0]", "children": [
            {"type": "VGroup", "children": [_tex("m = 2"), _text("on a calibrated")]},
            {"type": "VGroup", "children": [_text("balance")]}]},
        {"type": "VGroup", "label": "given[1]", "children": [
            {"type": "VGroup", "children": [_tex(r"\theta = 30^\circ")]}]}]}

    def test_given_items_are_structure_their_lines_are_not(self):
        # Each given is laid out in measured lines (a font may break "on a
        # calibrated balance" elsewhere), but how many givens there are, and
        # what each holds, is the spec's.
        rewrapped = copy.deepcopy(self.GIVEN)
        rewrapped["children"][1]["children"] = [
            {"type": "VGroup", "children": [_tex("m = 2"), _text("on a")]},
            {"type": "VGroup", "children": [_text("calibrated balance")]}]
        assert project("ProblemStatement", _settle(self.GIVEN)) == project(
            "ProblemStatement", _settle(rewrapped))
        merged = copy.deepcopy(self.GIVEN)
        first, second = merged["children"][1:]
        first["children"] += second["children"]
        merged["children"].remove(second)
        assert diff_structure(project("ProblemStatement", _settle(self.GIVEN)),
                              project("ProblemStatement", _settle(merged)))

    def test_lens_or_frame_is_not_structure_but_its_timing_is(self):
        # ZoomHighlight example 0 is a lens in Arial and Inter, a frame in
        # Verdana: a different node, drawn by a different animation.
        lens = {"type": "Group", "label": "zoom lens", "children": [_text("load factor")]}
        frame = {"type": "SurroundingRectangle", "label": "focus frame",
                 "color": "#58A6FF", "fill_opacity": 0.0, "stroke_opacity": 1.0}

        def timeline(node, cls, run_time=1.5):
            return [{"op": "play", "animations": [cls], "targets": [node["label"]],
                     "run_time": run_time}, *_settle(node)]

        assert project("ZoomHighlight", timeline(lens, "Transform")) == project(
            "ZoomHighlight", timeline(frame, "Create"))
        assert project("ZoomHighlight", timeline(lens, "Transform")) != project(
            "ZoomHighlight", timeline(frame, "Create", run_time=2.0))

    @pytest.mark.parametrize("name, i", [
        ("ProblemStatement", 0), ("ProblemStatement", 2), ("GraphPlot", 1)])
    def test_real_layouts_differ_by_font_but_not_the_structure(self, name, i):
        # Recorded under Arial and under the theme fonts, ProblemStatement
        # examples 0 and 2 break their lines in different places and GraphPlot
        # example 1 keeps a different set of tick labels. Skipped until both
        # are recorded.
        case = _recorded(name)["cases"][i]
        layouts = recorded_layout(case)
        fallback, theme = "win32|Arial|Arial|Courier New", "win32|Archivo|Inter|JetBrains Mono"
        if not {fallback, theme} <= set(layouts):
            pytest.skip(f"needs {fallback} and {theme} recorded; have {sorted(layouts)}")
        assert layouts[fallback] != layouts[theme]
        assert project(name, layouts[fallback]) == project(
            name, layouts[theme]) == case["structure"]

    # Font-invariant facts the structure must keep, each found identical
    # under Arial, Inter/Archivo, Verdana, Times New Roman and Georgia/Segoe
    # UI: an edit of the real recorded layout that a regression could make
    # must change the structure, so every machine catches it.
    @pytest.mark.parametrize("name, i, edit", [
        ("GraphPlot", 0, lambda t: _labelled(t, "axes")["children"].__setitem__(
            slice(None), [c for c in _labelled(t, "axes")["children"]
                          if c["type"] != "Line"])),
        ("GraphPlot", 0, lambda t: [c.update(color="#FF0000")
                                    for c in _labelled(t, "axes")["children"][:2]]),
        ("GraphPlot", 0, lambda t: _labelled(t, "axes")["children"].pop(5)),
        ("UnitBreakdown", 0, lambda t: _merge(_labelled(t, "UnitBreakdown")["children"], 1, 2)),
        ("ProblemStatement", 0, lambda t: _merge(_labelled(t, "given")["children"], 1, 2)),
        ("ProblemStatement", 0, lambda t: _labelled(t, "find value").pop("label")),
        ("DataStructureViz", 0, lambda t: _merge(
            next(n for n in _settled(t) if _child(n, "value '29'"))["children"], 0, 1)),
        ("StepTrace", 0, lambda t: _merge(_labelled(t, "header")["children"], 0, 1)),
        ("StepTrace", 0, lambda t: _labelled(t, "frame[1]")["children"].insert(
            0, _labelled(t, "frame[0]")["children"].pop())),
        ("NumberLineWalk", 0, lambda t: _merge(_labelled(t, "tick labels")["children"], 0, 1)),
        ("SplitCompare", 0, lambda t: _merge(
            _labelled(t, "left side")["children"][1]["children"], 0, 1)),
    ], ids=["GraphPlot axis lines and tick marks deleted", "GraphPlot axis lines recoloured",
            "GraphPlot one tick mark deleted", "UnitBreakdown labels merged",
            "ProblemStatement givens merged", "ProblemStatement find value ungrouped",
            "DataStructureViz value labels merged", "StepTrace header cells merged",
            "StepTrace cell moved between frames", "NumberLineWalk tick labels merged",
            "SplitCompare lines merged"])
    def test_a_font_invariant_regression_changes_the_structure(self, name, i, edit):
        case = _recorded(name)["cases"][i]
        for fp, layout in recorded_layout(case).items():
            assert diff_structure(case["structure"], project(name, layout)) == []
            changed = copy.deepcopy(layout)
            edit(changed)
            assert diff_structure(case["structure"], project(name, changed)), (
                f"{name} example {i} under {fp}: the edit is invisible to the structure")


def _settled(timeline: list[dict]) -> list[dict]:
    """Every node at the last settle point, depth first."""
    def walk(nodes):
        for n in nodes:
            yield n
            yield from walk(n.get("children", []))
    return list(walk([op for op in timeline if op["op"] == "settle"][-1]["mobjects"]))


def _labelled(timeline: list[dict], label: str) -> dict:
    return next(n for n in _settled(timeline) if n.get("label") == label)


def _child(node: dict, label: str) -> bool:
    return any(c.get("label") == label for c in node.get("children", []))


def _merge(children: list[dict], a: int, b: int) -> None:
    """Fold children[b] into children[a], as a regression that drew two items
    as one would: one node holding both texts, or both nodes' children."""
    into, gone = children[a], children.pop(b)
    if "children" in into:
        into["children"] += gone["children"]
    else:
        into["text"] += " " + gone["text"]


class TestHiddenFonts:
    def test_hiding_the_resolved_fonts_changes_the_fingerprint_and_restores(self, monkeypatch):
        # --hide-font records the baseline of a machine without those fonts:
        # the fingerprint must follow, and nothing may leak past the block.
        # Pinned on a controlled font set -- the theme fonts plus the Windows
        # fallbacks -- not on whatever this box has installed: on a box with
        # only the fallbacks, hiding what it resolves (Arial, Courier New)
        # leaves no role a font and is rightly refused (see
        # test_hiding_the_last_fallback_is_refused).
        installed = frozenset({"Archivo", "Inter", "JetBrains Mono", "Arial", "Courier New"})
        monkeypatch.setattr(theme_mod, "_installed_fonts", lambda: installed)
        before = fingerprint()
        assert before == f"{sys.platform}|Archivo|Inter|JetBrains Mono"
        _, heading, body, mono = before.split("|")
        with fonts_hidden({heading, body, mono}):
            hidden = fingerprint()
        assert hidden != before
        assert not {heading, body, mono} & set(hidden.split("|")[1:])
        assert hidden == f"{sys.platform}|Arial|Arial|Courier New"
        assert fingerprint() == before
        assert theme_mod._installed_fonts() == installed

    def test_hiding_a_font_that_is_not_installed_is_refused(self, tmp_path):
        missing = "No Such Font Family 9f3c"
        assert missing not in theme_mod._installed_fonts()
        with pytest.raises(ValueError, match="cannot be hidden"):
            with fonts_hidden([missing]):
                pass
        with pytest.raises(SystemExit):
            main(["--reason", "x", "--component", "TitleCard", "--hide-font", missing,
                  "--dir", str(tmp_path)])
        assert not snapshot_path("TitleCard", tmp_path).exists()

    def test_hiding_the_last_fallback_is_refused(self, monkeypatch):
        # On a box with only the fallbacks, hiding Arial leaves heading and
        # body with nothing installed: resolve_fonts would keep the theme's
        # names, and the baseline would be filed under "Archivo|Inter" on a
        # box that does not have them.
        fallbacks = frozenset({"Arial", "Courier New"})
        monkeypatch.setattr(theme_mod, "_installed_fonts", lambda: fallbacks)
        with pytest.raises(ValueError, match="no installed fallback"):
            with fonts_hidden({"Arial"}):
                pass
        assert theme_mod._installed_fonts() == fallbacks
