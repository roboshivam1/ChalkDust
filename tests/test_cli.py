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


def _overlong_narration(spec):
    spec["beats"][1]["narration"] = " ".join(["word"] * 70) + "."


def _broken(exc):
    def raise_(*args, **kwargs):
        raise exc
    return raise_


@pytest.mark.parametrize("edit,backend,patch,code,label", [
    (_bad_component, "fake", None, cli.EXIT_SPEC_INVALID, "spec invalid"),
    (_overlong_narration, "fake", None, cli.EXIT_SEMANTIC_REFUSED, "semantic refused"),
    (_dense_title, "fake", None, cli.EXIT_LAYOUT_REFUSED, "layout refused"),
    (None, "no_such_backend", None, cli.EXIT_SPEECH_FAILED, "speech failed"),
    (None, "fake", ("render_beat", RuntimeError("boom")),
     cli.EXIT_RENDER_FAILED, "render failed"),
    (None, "fake", ("assemble", TTSError("ffmpeg failed")),
     cli.EXIT_ASSEMBLY_FAILED, "assembly failed"),
], ids=["spec", "semantic", "layout", "speech", "render", "assembly"])
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


def test_render_reports_a_degraded_raw_scene_beat(
        tmp_path, monkeypatch, capfd, fake_tts):
    # A RawScene that falls back to BulletReveal (D-010) must say so on its
    # line and in the summary, not pass as an ordinary rebuilt beat.
    spec = example_spec()
    spec["beats"] = [{**spec["beats"][1], "id": "b01", "component": "RawScene",
                      "params": {"rationale": "needs the OS", "code": "import os"}}]
    monkeypatch.chdir(tmp_path)
    path = write_spec(tmp_path / "spec.json", spec)

    assert cli.main(["render", str(path)]) == cli.EXIT_OK
    out = capfd.readouterr().out
    assert "b01  speech synth   render rebuilt" in out
    assert "RawScene -> DEGRADED to BulletReveal (reason in " in out
    assert "raw_scene_usage.jsonl" in out
    assert "beats: 0 cached, 1 rebuilt, 1 degraded (b01)" in out


@pytest.mark.parametrize("encode,code,message", [
    # Windows PowerShell 5.1's Out-File and '>' write UTF-16 LE with a BOM.
    (lambda text: text.replace("\n", "\r\n").encode("utf-16"),
     cli.EXIT_SPEC_INVALID, "is UTF-16 text; specs must be UTF-8"),
    # Stray bytes that are no encoding of anything.
    (lambda text: text.encode("utf-8").replace(b"cs-binary", b"\x80\x81", 1),
     cli.EXIT_SPEC_INVALID, "is not UTF-8 text (byte 0x80 at offset"),
    # A UTF-8 BOM (Notepad, Set-Content -Encoding utf8) is still UTF-8.
    (lambda text: text.encode("utf-8-sig"), cli.EXIT_OK, None),
], ids=["utf16", "stray_bytes", "utf8_bom"])
def test_spec_encoding_is_utf8_or_a_readable_refusal(
        encode, code, message, tmp_path, capsys):
    spec = example_spec()
    spec["beats"] = spec["beats"][:1]
    text = write_spec(tmp_path / "plain.json", spec).read_text(encoding="utf-8")
    path = tmp_path / "spec.json"
    path.write_bytes(encode(text))

    rc = cli.main(["validate", str(path), "--work-dir", str(tmp_path / "work")])

    err = capsys.readouterr().err
    assert rc == code
    if message:
        assert err.startswith("chalkdust: spec invalid: ")
        assert message in err
    assert "Traceback" not in err


@pytest.mark.parametrize("argv", [
    ["validate", "--work-dir", "{file}"],
    ["render", "--work-dir", "{file}", "--cache-dir", "{tmp}/cache"],
    ["render", "--cache-dir", "{file}", "--work-dir", "{tmp}/work"],
], ids=["validate_work_dir", "render_work_dir", "render_cache_dir"])
def test_a_dir_option_naming_a_file_is_refused_readably(
        argv, tmp_path, capsys, fake_tts):
    spec = example_spec()
    spec["beats"] = spec["beats"][:1]
    path = write_spec(tmp_path / "spec.json", spec)
    afile = tmp_path / "afile"
    afile.write_text("not a directory", encoding="utf-8")
    argv = [a.format(file=afile, tmp=tmp_path) for a in argv]

    rc = cli.main([argv[0], str(path), *argv[1:]])

    err = capsys.readouterr().err
    assert rc == cli.EXIT_DIRECTORY_UNUSABLE
    assert err.startswith("chalkdust: directory unusable: --")
    assert f"{afile}: it exists and is a file, not a directory" in err
    assert fake_tts.calls == []  # refused before any speech


def test_a_spec_with_no_beats_is_refused_at_validation(tmp_path, capsys):
    spec = {**example_spec(), "beats": []}
    path = write_spec(tmp_path / "spec.json", spec)

    rc = cli.main(["validate", str(path), "--work-dir", str(tmp_path / "work")])

    out, err = capsys.readouterr()
    assert rc == cli.EXIT_SPEC_INVALID
    assert "ok (0 beats)" not in out
    assert err.startswith("chalkdust: spec invalid: ")
    assert "beats: a video needs at least one beat" in err


EXAMPLES = sorted(EXAMPLE.parent.glob("*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_every_example_validates_and_leaves_the_voice_to_the_platform(
        path, tmp_path, capsys):
    # An example names no TTS backend or voice: tts.resolve_voice picks the
    # platform's default backend and that backend's default voice, so the
    # same file renders on macOS and Windows alike.
    spec = pipeline.load_spec(path)
    assert not {"backend", "voice_id"} & spec.voice.model_fields_set
    assert cli.main(["validate", str(path), "--work-dir", str(tmp_path / "work")]) \
        == cli.EXIT_OK
    assert f"ok ({len(spec.beats)} beats)" in capsys.readouterr().out


def test_an_example_meets_the_phase_0_exit_shape():
    # ROADMAP.md Phase 0 exit: a hand-authored 5-beat spec, built from the
    # Phase 0 components.
    phase0 = {"TitleCard", "BulletReveal", "EquationDerivation"}
    shapes = [(len(s.beats), {b.component for b in s.beats})
              for s in map(pipeline.load_spec, EXAMPLES)]
    assert (5, phase0) in shapes, shapes


def _hostile_video_ids(root):
    # Each would have named a directory outside the work dir, which render
    # deletes and recreates for assembly (f1 rb3 verifier repros).
    return {"relative": "../../victim/precious",
            "absolute": (root / "victim" / "precious").as_posix(),
            "illegal": "a:b?c*"}


@pytest.mark.parametrize("kind", ["relative", "absolute", "illegal"])
def test_a_video_id_that_is_not_a_slug_is_refused_and_touches_nothing(
        kind, tmp_path, capsys, fake_tts):
    victim = tmp_path / "victim" / "precious-draft"
    victim.mkdir(parents=True)
    (victim / "keep.txt").write_text("not the pipeline's", encoding="utf-8")
    spec = example_spec()
    spec["video_id"] = _hostile_video_ids(tmp_path)[kind]
    path = write_spec(tmp_path / "spec.json", spec)

    # --out given, so only the assembly dir could escape: the relative id
    # resolves to work/assemble/../../victim/precious-draft, i.e. `victim`.
    rc = cli.main(["render", str(path), "--cache-dir", str(tmp_path / "cache"),
                   "--work-dir", str(tmp_path / "work"),
                   "--out", str(tmp_path / "video.mp4")])

    assert rc == cli.EXIT_SPEC_INVALID
    assert "video_id: " in capsys.readouterr().err
    assert (victim / "keep.txt").exists()
    assert fake_tts.calls == []


def test_a_control_character_in_text_is_refused_every_time(tmp_path, capsys):
    # A NUL in a title: Pango refused it as a 'colour' build_error, and the
    # text cache that refusal left behind let a second validate pass and
    # render the title blank. Rung 1 now refuses it, run after run.
    spec = example_spec()
    spec["beats"][0]["params"]["title"] = "Bin" + chr(0) + "ary"
    path = write_spec(tmp_path / "spec.json", spec)
    argv = ["validate", str(path), "--work-dir", str(tmp_path / "work")]

    assert [cli.main(argv), cli.main(argv)] == [cli.EXIT_SPEC_INVALID] * 2
    assert "b01.params.title: control character U+0000 at offset 3" in \
        capsys.readouterr().err


@pytest.mark.parametrize("narration", [".", "...", " -- !? "])
def test_narration_with_nothing_to_speak_is_refused_at_rung_1(
        narration, tmp_path, capsys, fake_tts):
    # Register N-9: "." passed validation, the voice spoke nothing, and the
    # render died at speech on ffprobe's "N/A". A narration with no letter or
    # digit is refused before any speech, naming the beat and the field.
    spec = example_spec()
    spec["beats"][1]["narration"] = narration
    path = write_spec(tmp_path / "spec.json", spec)

    rc = cli.main(["render", str(path), "--cache-dir", str(tmp_path / "cache"),
                   "--work-dir", str(tmp_path / "work")])

    err = capsys.readouterr().err
    assert rc == cli.EXIT_SPEC_INVALID
    assert err.startswith("chalkdust: spec invalid: ")
    assert "beats.1.narration: Value error, narration has nothing to speak" in err
    assert fake_tts.calls == []
