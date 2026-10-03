"""Snapshot tests (SCENE_SPEC.md §11 rule 6).

Walks the registry like test_layout.py: every component's examples() (and
carried_examples(), built with their carried artifacts) must match its
recorded snapshot in tests/snapshots/. A component without one
fails -- shipping a snapshot is part of shipping a component (§5), so a
missing file is a missing deliverable, not a reason to skip. Component
branches written before this harness existed carry none: merging one
requires recording its snapshot in the same step, from the repo root:

    python -m chalkdust.validate.snapshot --reason "<Name> merged" --component <Name>
"""

from __future__ import annotations

import copy
import json
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from chalkdust.scenes.components import get_component, registered_names
from chalkdust.validate.snapshot import (
    capture,
    case_key,
    diff_geometry,
    diff_structure,
    fingerprint,
    load,
    main,
    recorded_key,
    regenerate,
    snapshot_cases,
    snapshot_exempt,
    snapshot_path,
    snapshotted_names,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _regen(name: str) -> str:
    return (f'python -m chalkdust.validate.snapshot --reason "<why>" '
            f'--component {name}')


@lru_cache(maxsize=None)
def _current(name: str) -> list[tuple[dict, list]]:
    return [capture(name, case.params, case.carry_in) for case in snapshot_cases(name)]


def _recorded(name: str) -> dict:
    snap = load(name, SNAPSHOT_DIR)
    if snap is None:
        pytest.fail(f"{name} has no snapshot; shipping one is part of shipping "
                    f"the component (SCENE_SPEC.md §5). Record it: {_regen(name)}")
    recorded = [recorded_key(case) for case in snap["cases"]]
    if recorded != [case_key(case) for case in snapshot_cases(name)]:
        pytest.fail(f"{name}.examples() changed since its snapshot was recorded. "
                    f"If intended: {_regen(name)}")
    return snap


@pytest.mark.parametrize("name", registered_names())
def test_component_is_snapshotted_or_formally_exempt(name):
    # A snapshot with no cases compares nothing and passes forever; that is
    # how RawScene.json sat vacuous. Every component either records at least
    # one case, or declares why it cannot (and then ships no snapshot file
    # that would claim coverage it does not give).
    reason = snapshot_exempt(name)
    if reason:
        assert reason.strip(), f"{name}: an exemption must state its reason"
        assert snapshot_cases(name) == [], (
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

    def test_carried_case_is_recorded_with_its_artifact(self, tmp_path):
        # A component acting on a carried artifact records each case with the
        # recipe it carries, built with that artifact on screen first.
        regenerate("ZoomHighlight", "carried", tmp_path)
        data = json.loads(snapshot_path("ZoomHighlight", tmp_path).read_text(encoding="utf-8"))
        case = data["cases"][0]
        assert case["carry_in"][0]["producer"] == "BulletReveal"
        settle = next(op for op in case["structure"]["timeline"] if op["op"] == "settle")
        labels = [m.get("label") for m in settle["mobjects"]]
        assert "carried[chain_causes]" in labels

    def test_plain_case_records_no_carry_in(self, tmp_path):
        # Snapshots of components that build from params alone are unchanged.
        regenerate("TitleCard", "plain", tmp_path)
        data = json.loads(snapshot_path("TitleCard", tmp_path).read_text(encoding="utf-8"))
        assert all("carry_in" not in case for case in data["cases"])
