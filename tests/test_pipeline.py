"""Pipeline stage order, failure typing, and incremental re-render.

Speech goes through a test double registered in tts.BACKENDS -- the same
selection the pipeline uses for a real voice -- so these tests need no TTS
install and run identically on every OS.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from chalkdust import pipeline
from chalkdust.core.models import Quality, VoiceConfig
from chalkdust.speech import tts
from chalkdust.speech.base import TTSError, probe_duration, run

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "binary_search.json"


class FakeTTS:
    """A tone whose length follows the character count of the text.

    A tone, not silence: normalize_audio trims edge silence, so a silent file
    would measure ~0s. Length follows characters, not words, so any narration
    edit moves the duration and therefore the beat's render key.

    ffmpeg writes to the exact path tts.py hands over (.aiff today) and picks
    the container from its extension.
    """

    name = "fake"

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
        # assembly intermediates once the MP4 exists.
        assert not any((root / "work" / "manim" / "beats").iterdir())
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
