"""Snapshot tests (SCENE_SPEC.md §11 rule 6).

Walks the registry like test_layout.py: every component's examples() must
match its recorded snapshot in tests/snapshots/. A component without one
fails -- shipping a snapshot is part of shipping a component (§5).
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

import pytest

from chalkdust.scenes.components import get_component, registered_names
from chalkdust.validate.snapshot import (
    capture,
    diff_geometry,
    diff_structure,
    fingerprint,
    load,
    main,
    regenerate,
    snapshot_path,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _regen(name: str) -> str:
    return (f'python -m chalkdust.validate.snapshot --reason "<why>" '
            f'--component {name}')


@lru_cache(maxsize=None)
def _current(name: str) -> list[tuple[dict, list]]:
    return [capture(name, params) for params in get_component(name).examples()]


def _recorded(name: str) -> dict:
    snap = load(name, SNAPSHOT_DIR)
    if snap is None:
        pytest.fail(f"{name} has no snapshot. Record one: {_regen(name)}")
    recorded = [case["params"] for case in snap["cases"]]
    if recorded != get_component(name).examples():
        pytest.fail(f"{name}.examples() changed since its snapshot was recorded. "
                    f"If intended: {_regen(name)}")
    return snap


@pytest.mark.parametrize("name", registered_names())
class TestLibrary:
    def test_structure_unchanged(self, name):
        snap = _recorded(name)
        for i, (case, (structure, _)) in enumerate(zip(snap["cases"], _current(name))):
            diffs = diff_structure(case["structure"], structure)
            assert not diffs, (
                f"{name} example {i} built differently from its snapshot. If "
                f"Manim or the component changed on purpose: {_regen(name)}\n"
                + "\n".join(diffs[:20]))

    def test_geometry_unchanged(self, name):
        snap = _recorded(name)
        fp = fingerprint()
        if any(fp not in case["geometry"] for case in snap["cases"]):
            recorded = sorted({k for case in snap["cases"] for k in case["geometry"]})
            pytest.skip(
                f"no geometry baseline for this machine ({fp}); recorded for "
                f"{recorded}. Text metrics depend on the installed fonts, so "
                f"geometry is compared only where they match. Add one: {_regen(name)}")
        for i, (case, (_, geometry)) in enumerate(zip(snap["cases"], _current(name))):
            diffs = diff_geometry(case["geometry"][fp], geometry)
            assert not diffs, (
                f"{name} example {i} lays out differently from its snapshot. If "
                f"Manim or the component changed on purpose: {_regen(name)}\n"
                + "\n".join(diffs[:20]))


@pytest.fixture(scope="module")
def title():
    return capture("TitleCard", get_component("TitleCard").examples()[1])


class TestHarness:
    """The harness must catch drift, tolerate noise, and only regenerate on
    purpose."""

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
        if mutate_structure:
            case["structure"]["timeline"][0]["run_time"] = 99.0
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_regenerate_keeps_other_machines_when_layout_is_unchanged(self, tmp_path):
        path = self._with_other_machine(tmp_path, mutate_structure=False)
        assert regenerate("TitleCard", "second", tmp_path) == []
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "other|A|B|C" in data["cases"][0]["geometry"]
        assert data["reasons"] == ["first", "second"]

    def test_regenerate_drops_other_machines_when_layout_changed(self, tmp_path):
        path = self._with_other_machine(tmp_path, mutate_structure=True)
        notes = regenerate("TitleCard", "second", tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "other|A|B|C" not in data["cases"][0]["geometry"]
        assert notes and "structure changed" in notes[0]
