"""CLI surface: verbs, exit codes, and what the operator sees."""

from __future__ import annotations

import pytest

from chalkdust import cli, pipeline
from chalkdust.speech.base import TTSError
from test_pipeline import EXAMPLE, example_spec, fake_tts, write_spec  # noqa: F401


def test_validate_example(capsys):
    assert cli.main(["validate", str(EXAMPLE)]) == cli.EXIT_OK
    assert "ok (4 beats)" in capsys.readouterr().out


def _bad_component(spec):
    spec["beats"][1]["component"] = "NotAComponent"


def _dense_title(spec):
    spec["beats"][0]["params"]["title"] = "Supercalifragilisticexpialidocious " * 14


def _broken(exc):
    def raise_(*args, **kwargs):
        raise exc
    return raise_


@pytest.mark.parametrize("edit,backend,patch,code,label", [
    (_bad_component, "fake", None, cli.EXIT_SPEC_INVALID, "spec invalid"),
    (_dense_title, "fake", None, cli.EXIT_LAYOUT_REFUSED, "layout refused"),
    (None, "no_such_backend", None, cli.EXIT_SPEECH_FAILED, "speech failed"),
    (None, "fake", ("render_beat", RuntimeError("boom")),
     cli.EXIT_RENDER_FAILED, "render failed"),
    (None, "fake", ("assemble", TTSError("ffmpeg failed")),
     cli.EXIT_ASSEMBLY_FAILED, "assembly failed"),
], ids=["spec", "layout", "speech", "render", "assembly"])
def test_known_failures_have_distinct_exit_codes(
        edit, backend, patch, code, label, tmp_path, monkeypatch, capsys, fake_tts):
    spec = example_spec()
    if edit:
        edit(spec)
    if patch:
        monkeypatch.setattr(pipeline, patch[0], _broken(patch[1]))
    path = write_spec(tmp_path / "spec.json", spec, backend=backend)

    rc = cli.main(["render", str(path), "--cache-dir", str(tmp_path / "cache"),
                   "--work-dir", str(tmp_path / "work")])

    assert rc == code
    assert capsys.readouterr().err.startswith(f"chalkdust: {label}: ")


def test_render_reports_cache_per_beat_and_stays_quiet(
        tmp_path, monkeypatch, capfd, fake_tts):
    # One beat keeps this fast; test_pipeline covers the full example.
    spec = example_spec()
    spec["beats"] = spec["beats"][:1]
    monkeypatch.chdir(tmp_path)
    path = write_spec(tmp_path / "spec.json", spec)

    assert cli.main(["render", str(path)]) == cli.EXIT_OK
    first = capfd.readouterr()
    assert "b01  speech synth   render rebuilt" in first.out
    assert "beats: 0 cached, 1 rebuilt" in first.out
    # Defaults land under cwd: out/, .cache/, work/ -- and never media/.
    assert (tmp_path / "out" / "cs-binary-search-intro-draft.mp4").exists()
    assert not (tmp_path / "media").exists()
    # Manim's INFO logging ("File ready at ...", "Animation 0 : ...") is off.
    assert "INFO" not in first.out + first.err

    assert cli.main(["render", str(path), "--out", "again.mp4"]) == cli.EXIT_OK
    second = capfd.readouterr().out
    assert "b01  speech cached  render cached " in second
    assert "beats: 1 cached, 0 rebuilt" in second
    assert (tmp_path / "again.mp4").exists()
