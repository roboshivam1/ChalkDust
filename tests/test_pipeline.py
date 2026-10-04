"""Pipeline stage order, failure typing, and incremental re-render.

Speech goes through a test double registered in tts.BACKENDS -- the same
selection the pipeline uses for a real voice -- so these tests need no TTS
install and run identically on every OS.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from chalkdust import pipeline
from chalkdust.core.models import Quality, VoiceConfig
from chalkdust.render import worker
from chalkdust.scenes.theme import DEFAULT, THEMES
from chalkdust.speech import tts
from chalkdust.speech.base import TTSError, probe_duration, run

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "binary_search.json"


class FakeTTS:
    """A tone whose length follows the character count of the text.

    A tone, not silence: normalize_audio trims edge silence, so a silent file
    would measure ~0s. Length follows characters, not words, so any narration
    edit moves the duration and therefore the beat's render key.

    ffmpeg writes to the exact path tts.py hands over -- named with this
    backend's `raw_format`, as the TTSBackend protocol requires -- and picks
    the container from its extension.
    """

    name = "fake"
    raw_format = "wav"
    default_voice = "tone"

    def __init__(self, seconds_per_char: float = 0.02) -> None:
        self.seconds_per_char = seconds_per_char
        self.calls: list[str] = []

    def synthesize(self, text: str, voice: VoiceConfig, out_path: Path) -> None:
        self.calls.append(text)
        seconds = max(1.0, self.seconds_per_char * len(text))
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", f"sine=frequency=220:duration={seconds:.3f}", str(out_path)])


def write_spec(path: Path, spec: dict, backend: str = "fake") -> Path:
    """Write `spec` with its voice pointed at `backend`."""
    spec = {**spec, "voice": {"backend": backend, "voice_id": "tone"}}
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def example_spec() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


@pytest.fixture
def fake_tts(monkeypatch) -> FakeTTS:
    fake = FakeTTS()
    monkeypatch.setitem(tts.BACKENDS, fake.name, fake)
    return fake


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """The example spec rendered once at draft, shared by the checks below.
    Runs from an empty cwd so stray Manim output would be visible."""
    root = tmp_path_factory.mktemp("rendered")
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(tts.BACKENDS, "fake", FakeTTS())
        mp.chdir(root)
        spec_path = write_spec(root / "spec.json", example_spec())
        result = pipeline.render(spec_path, Quality.DRAFT, root / "video.mp4",
                                 cache_dir=root / "cache", work_dir=root / "work")
    return root, spec_path, result


def _streams(path: Path) -> list[str]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return proc.stdout.split()


class TestRender:
    def test_example_produces_mp4_matching_narration(self, rendered):
        root, _, result = rendered
        assert result.output == root / "video.mp4"
        assert sorted(_streams(result.output)) == ["audio", "video"]
        # Video is authoritative and never shorter than the narration. Manim
        # rounds every play() up to whole frames, so at 15fps a beat with
        # several reveals runs a few frames past its audio (b03: +0.26s).
        narration = sum(b.duration for b in result.beats)
        total = probe_duration(result.output)
        assert narration <= total <= narration + 0.35 * len(result.beats)

    def test_first_run_rebuilds_every_beat(self, rendered):
        _, _, result = rendered
        assert [b.beat_id for b in result.beats] == ["b01", "b02", "b03", "b04"]
        assert result.rebuilt == ["b01", "b02", "b03", "b04"]

    def test_scratch_is_contained_and_cleaned(self, rendered):
        root, _, _ = rendered
        assert not (root / "media").exists(), "Manim wrote into cwd"
        # Partial movie files are removed once each beat is in the cache, and
        # assembly intermediates once the MP4 exists. The worker owns the
        # Manim scratch layout: per-beat video dirs under work/manim/videos.
        videos = root / "work" / "manim" / "videos"
        assert videos.is_dir()
        assert not list(videos.glob("beat_*"))
        assert not [p for p in videos.rglob("*") if p.is_file()]
        assert not any((root / "work" / "assemble").iterdir())

    def test_narration_edit_rebuilds_only_that_beat(self, rendered, fake_tts):
        root, spec_path, _ = rendered
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["beats"][2]["narration"] += " Twenty steps, for a million names."
        spec_path.write_text(json.dumps(spec), encoding="utf-8")

        result = pipeline.render(spec_path, Quality.DRAFT, root / "edited.mp4",
                                 cache_dir=root / "cache", work_dir=root / "work")

        assert result.rebuilt == ["b03"]
        assert [b.beat_id for b in result.beats if not b.speech_cached] == ["b03"]
        assert len(fake_tts.calls) == 1


class TestValidation:
    def test_spec_errors_are_collected_before_anything_runs(self, tmp_path):
        spec = example_spec()
        spec["theme"] = "neon"
        spec["beats"][0]["component"] = "NotAComponent"
        spec["beats"][1]["params"]["colour"] = "red"
        path = write_spec(tmp_path / "spec.json", spec)

        with pytest.raises(pipeline.SpecInvalid) as exc:
            pipeline.validate(path, work_dir=tmp_path / "work")
        message = str(exc.value)
        assert "unknown theme 'neon'" in message
        assert "b01.component: unknown component 'NotAComponent'" in message
        assert "colour: Extra inputs are not permitted" in message

    def test_layout_refusal_happens_before_speech(self, tmp_path, fake_tts):
        spec = example_spec()
        spec["beats"][0]["params"]["title"] = "Supercalifragilisticexpialidocious " * 14
        path = write_spec(tmp_path / "spec.json", spec)

        with pytest.raises(pipeline.LayoutRefused, match=r"b01: 1 finding\(s\)\n  \[overflow\]"):
            pipeline.render(path, cache_dir=tmp_path / "cache",
                            work_dir=tmp_path / "work")
        assert fake_tts.calls == []


    def test_semantic_refusal_happens_before_layout_and_speech(
            self, tmp_path, fake_tts):
        # Rung 2 (SCENE_SPEC.md §8): 70 words passes the schema's 80-word
        # bound but runs ~26s at the estimated pace -- more than one beat may
        # carry. No layout fixes that, so it is refused before rung 3.
        spec = example_spec()
        spec["beats"][1]["narration"] = " ".join(["word"] * 70) + "."
        path = write_spec(tmp_path / "spec.json", spec)

        with pytest.raises(pipeline.SemanticRefused,
                           match=r"b02: 1 finding\(s\)\n  \[duration\]"):
            pipeline.render(path, cache_dir=tmp_path / "cache",
                            work_dir=tmp_path / "work")
        assert fake_tts.calls == []


RAW_CODE = """
from manim import *

class Sweep(Scene):
    def construct(self):
        sq = Square()
        self.play(Create(sq), run_time=1)
        self.play(sq.animate.rotate(PI / 4), run_time=1)
"""


class TestRawSceneRouting:
    """Validation routes a RawScene beat as the worker does: never through
    the host-scene probe, where RawScene.build() raises (SCENE_SPEC.md §7)."""

    def test_raw_scene_beat_is_not_probed_in_a_host_scene(self, tmp_path, capsys):
        spec = example_spec()
        spec["beats"][1] = {**spec["beats"][1], "component": "RawScene",
                            "params": {"rationale": "a rotation the library lacks",
                                       "code": RAW_CODE}}
        path = write_spec(tmp_path / "spec.json", spec)

        assert len(pipeline.validate(path, work_dir=tmp_path / "work").beats) == 4
        assert "b02  RawScene: layout is asserted in its own render" in \
            capsys.readouterr().out

    def test_raw_scene_certain_to_degrade_is_checked_as_its_fallback(
            self, tmp_path, capsys):
        spec = example_spec()
        spec["beats"][1] = {**spec["beats"][1], "component": "RawScene",
                            "params": {"rationale": "needs the OS", "code": "import os"}}
        path = write_spec(tmp_path / "spec.json", spec)

        checked = pipeline.checked_beats(pipeline.load_spec(path))
        assert checked[1].spec.component == "BulletReveal" and checked[1].geometric
        pipeline.validate(path, work_dir=tmp_path / "work")
        assert "b02  RawScene will degrade to BulletReveal (forbidden_import:" in \
            capsys.readouterr().out


class TestFailureTyping:
    """Failures are typed by the stage they happen in, not by exception class."""

    def test_speech_backend_failure(self, tmp_path):
        path = write_spec(tmp_path / "spec.json", example_spec(),
                          backend="no_such_backend")
        with pytest.raises(pipeline.SpeechFailed, match="b01: unknown TTS backend"):
            pipeline.render(path, cache_dir=tmp_path / "cache",
                            work_dir=tmp_path / "work")

    def test_render_failure(self, tmp_path, fake_tts, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("manim exploded")

        monkeypatch.setattr(pipeline, "render_beat", broken)
        path = write_spec(tmp_path / "spec.json", example_spec())
        with pytest.raises(pipeline.RenderFailed, match="b01 \\(TitleCard\\): manim exploded"):
            pipeline.render(path, cache_dir=tmp_path / "cache",
                            work_dir=tmp_path / "work")

    def test_ffmpeg_failure_in_assembly_is_not_speech_failure(
            self, tmp_path, fake_tts, monkeypatch):
        # assemble.py raises TTSError (it shares speech.base.run), which must
        # still surface as an assembly failure.
        def broken(*args, **kwargs):
            raise TTSError("ffmpeg failed: concat")

        monkeypatch.setattr(pipeline, "render_beat", lambda beat, *a: beat)
        monkeypatch.setattr(pipeline, "assemble", broken)
        path = write_spec(tmp_path / "spec.json", example_spec())
        with pytest.raises(pipeline.AssemblyFailed, match="ffmpeg failed: concat"):
            pipeline.render(path, cache_dir=tmp_path / "cache",
                            work_dir=tmp_path / "work")


def _one_beat_spec(component: str, params: dict, **beat) -> dict:
    return {"video_id": "one-beat", "beats": [
        {"id": "b01", "component": component, "params": params,
         "narration": "A short line of narration for a one beat video.", **beat}]}


class TestValidateLadder:
    """Rungs 1-3 in order, rung 3 with the render's own mechanical repair
    (SCENE_SPEC.md §8, §9 step 1), all before speech."""

    def test_mechanically_repairable_beat_passes_validation(
            self, tmp_path, off_edge_beat, capsys):
        # _OffEdge (conftest) draws its text 8 units right of centre: the
        # geometric probe says out_of_bounds, and a nudge fixes it. The
        # worker renders it repaired, so validation must not refuse it.
        path = write_spec(tmp_path / "spec.json", _one_beat_spec(
            off_edge_beat.spec.component, {"text": "Off the edge", "dx": 8.0}))

        assert len(pipeline.validate(path, work_dir=tmp_path / "work").beats) == 1
        assert "b01  repaired mechanically (out_of_bounds): mobject #0 " \
            "nudged (" in capsys.readouterr().out

    def test_refusal_carries_typed_findings_per_rung(self, tmp_path, fake_tts):
        spec = example_spec()
        spec["beats"][0]["params"]["title"] = "Supercalifragilisticexpialidocious " * 14
        path = write_spec(tmp_path / "spec.json", spec)

        with pytest.raises(pipeline.LayoutRefused) as exc:
            pipeline.render(path, cache_dir=tmp_path / "cache",
                            work_dir=tmp_path / "work")
        assert [(r.beat_id, sorted(r.kinds())) for r in exc.value.reports] == \
            [("b01", ["overflow"])]
        assert str(exc.value).startswith(
            "rung 3 (geometric, after mechanical repair) refused 1 beat(s): b01\n")
        assert fake_tts.calls == []


def _short_path(path: Path) -> str:
    """The Windows 8.3 form of an existing `path` (C:/Users/LOKAVY~1/...)."""
    import ctypes

    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, len(buf))
    assert n, f"GetShortPathNameW failed for {path}"
    return buf.value


@pytest.mark.skipif(sys.platform != "win32", reason="8.3 short names are Windows-only")
def test_dirs_handed_to_manim_are_long_paths(tmp_path, fake_tts, monkeypatch):
    # Register N-3: TeX breaks on the '~' of an 8.3 name, so a work dir given
    # in short form must reach Manim -- validation, repair probe and render
    # alike -- with every short name expanded.
    long_dir = tmp_path / "a long work directory"
    long_dir.mkdir()
    short = _short_path(long_dir)
    if "~" not in short:
        pytest.skip(f"this volume generates no 8.3 names ({short})")

    handed: list[str] = []

    def spy(real):
        def tempconfig(settings, *args, **kwargs):
            handed.extend(str(settings[k]) for k in ("media_dir", "video_dir")
                          if k in settings)
            return real(settings, *args, **kwargs)
        return tempconfig

    monkeypatch.setattr(pipeline, "tempconfig", spy(pipeline.tempconfig))
    monkeypatch.setattr(worker, "tempconfig", spy(worker.tempconfig))
    path = write_spec(tmp_path / "spec.json",
                      _one_beat_spec("TitleCard", {"title": "Long paths only"}))
    pipeline.render(path, Quality.DRAFT, tmp_path / "video.mp4",
                    cache_dir=tmp_path / "cache", work_dir=Path(short))

    # pipeline's scratch (validate + render), the worker's repair probe, and
    # the worker's render (media_dir + video_dir).
    assert len(handed) >= 5
    assert [d for d in handed if "~" in d] == []
    assert all(Path(d).is_relative_to(Path(os.path.realpath(long_dir))) for d in handed)


class TestCarryInEndToEnd:
    """Register N-4: a spec file whose b02 carries in the artifact b01
    registers goes through validate, speech and render, and b02's key holds
    b01's construction params (SCENE_SPEC.md §6, D-004). The carry_in_video
    fixture lends TitleCard an artifact builder and registers _Highlight,
    whose build() raises unless the carried artifact is on screen -- so a
    rendered b02 proves the render was built with its recipes."""

    def _render(self, root: Path, spec: dict, out: str) -> pipeline.RunResult:
        path = write_spec(root / "spec.json", spec)
        return pipeline.render(path, Quality.DRAFT, root / out,
                               cache_dir=root / "cache", work_dir=root / "work")

    def test_editing_the_producer_rerenders_the_carrying_beat(
            self, tmp_path, carry_in_video, fake_tts):
        spec = carry_in_video("Bucket array").model_dump(mode="json")
        first = self._render(tmp_path, spec, "first.mp4")
        assert first.rebuilt == ["b01", "b02"]
        assert sorted(_streams(first.output)) == ["audio", "video"]

        # b01's narration only: b01 re-times (FakeTTS: 0.02 s a character,
        # 1 s floor) and re-renders, but its params -- what b02 rebuilds the
        # artifact from -- are unchanged, so b02 stays cached.
        spec["beats"][0]["narration"] += " It holds every key the table has seen so far."
        assert self._render(tmp_path, spec, "narration.mp4").rebuilt == ["b01"]

        # b01's params: the carried artifact changes, so b02 must re-render
        # although nothing in b02's own spec or audio changed.
        spec["beats"][0]["params"]["title"] = "Bucket list"
        edited = self._render(tmp_path, spec, "params.mp4")
        assert edited.rebuilt == ["b01", "b02"]
        assert all(b.speech_cached for b in edited.beats)


class TestRenderKeyTerms:
    """What the pipeline hands the worker reaches the beat's render key
    (D-004, D-006): the spec's theme and the quality tier are key terms; the
    work dir is scratch and is not -- moving it must not re-render."""

    SPEC = _one_beat_spec("TitleCard", {"title": "Key terms"})

    def _render(self, root: Path, spec: dict, quality=Quality.DRAFT,
                work: str = "work") -> pipeline.RunResult:
        path = write_spec(root / "spec.json", spec)
        return pipeline.render(path, quality, root / f"{quality.value}.mp4",
                               cache_dir=root / "cache", work_dir=root / work)

    def _clips(self, root: Path) -> int:
        return len(list((root / "cache" / "beats").glob("*.mp4")))

    def test_two_tiers_two_keys(self, tmp_path, fake_tts):
        assert self._render(tmp_path, self.SPEC, Quality.DRAFT).rebuilt == ["b01"]
        assert self._render(tmp_path, self.SPEC, Quality.FINAL).rebuilt == ["b01"]
        assert self._clips(tmp_path) == 2

    def test_two_themes_two_keys(self, tmp_path, fake_tts, monkeypatch):
        paper = replace(DEFAULT, name="paper",
                        palette=replace(DEFAULT.palette, bg="#FAF7F0", fg="#1F2328"))
        monkeypatch.setitem(THEMES, "paper", paper)
        assert self._render(tmp_path, self.SPEC).rebuilt == ["b01"]
        assert self._render(tmp_path, {**self.SPEC, "theme": "paper"}).rebuilt == ["b01"]
        assert self._clips(tmp_path) == 2

    def test_work_dir_is_not_a_key_term(self, tmp_path, fake_tts):
        assert self._render(tmp_path, self.SPEC, work="work_a").rebuilt == ["b01"]
        assert self._render(tmp_path, self.SPEC, work="work_b").rebuilt == []
        assert self._clips(tmp_path) == 1
