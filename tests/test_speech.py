"""Speech stage: backend raw formats, per-platform voice resolution, and the
real backends on the machines that have them.

Real-synthesis tests are gated on the platform (or the kokoro extra) that the
backend needs; everything else runs everywhere.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from chalkdust.core.cache import Cache, tts_key
from chalkdust.core.models import VoiceConfig
from chalkdust.speech import tts
from chalkdust.speech.backends.kokoro import Kokoro
from chalkdust.speech.backends.windows_sapi import sapi_rate
from chalkdust.speech.base import CHANNELS, SAMPLE_RATE, TTSError, run

HAS_KOKORO = importlib.util.find_spec("kokoro") is not None
TEXT = "Two different keys can land in the same bucket. That is a collision."

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="SAPI exists only on Windows")
macos_only = pytest.mark.skipif(sys.platform != "darwin", reason="`say` exists only on macOS")
needs_kokoro = pytest.mark.skipif(
    not HAS_KOKORO, reason='needs the kokoro extra: pip install -e ".[kokoro]" (Python <3.13)'
)


def _stream(path: Path) -> list[str]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip().split(",")


def _assert_canonical(path: Path) -> None:
    assert _stream(path) == ["pcm_s16le", str(SAMPLE_RATE), str(CHANNELS)]


# --- raw format (the stage normalises whatever the backend declares) ----------


class FlacBackend:
    """Stands in for an engine whose native output is not WAV or AIFF."""

    name = "fake_flac"
    raw_format = "flac"
    default_voice = "none"

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def synthesize(self, text: str, voice: VoiceConfig, out_path: Path) -> None:
        self.paths.append(out_path)
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=1.5:sample_rate=44100", str(out_path)])


def test_stage_names_raw_file_by_backend_format(tmp_path, monkeypatch):
    fake = FlacBackend()
    monkeypatch.setitem(tts.BACKENDS, fake.name, fake)

    path, duration = tts.synthesize(TEXT, VoiceConfig(backend=fake.name), Cache(tmp_path))

    assert fake.paths[0].suffix == ".flac"
    assert not fake.paths[0].exists()  # raw intermediate cleaned up
    _assert_canonical(path)
    assert duration == pytest.approx(1.5, abs=0.05)


def test_backends_declare_their_native_format():
    assert {n: b.raw_format for n, b in tts.BACKENDS.items()} == {
        "macos_say": "aiff", "windows_sapi": "wav", "kokoro": "wav",
    }


# --- resolution (docs/VOICE.md) -----------------------------------------------


def test_unnamed_backend_is_say_on_macos_with_unchanged_cache_key(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    resolved = tts.resolve_voice(VoiceConfig())
    assert (resolved.backend, resolved.voice_id) == ("macos_say", "Daniel")
    # Audio cached on a Mac before resolution existed must still hit.
    assert tts_key(TEXT, resolved) == tts_key(TEXT, VoiceConfig())


def test_unnamed_backend_is_sapi_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    resolved = tts.resolve_voice(VoiceConfig(rate=1.2))
    assert resolved == VoiceConfig(
        backend="windows_sapi", voice_id="Microsoft David Desktop", rate=1.2
    )


def test_one_spec_keys_differently_per_platform(monkeypatch):
    voice = VoiceConfig.model_validate({})  # what a spec with no voice block yields
    keys = set()
    for platform in ("darwin", "win32"):
        monkeypatch.setattr(sys, "platform", platform)
        keys.add(tts_key(TEXT, tts.resolve_voice(voice)))
    assert len(keys) == 2


def test_named_backend_without_voice_gets_that_backends_default(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    resolved = tts.resolve_voice(VoiceConfig(backend="kokoro"))
    assert (resolved.backend, resolved.voice_id) == ("kokoro", "af_heart")


def test_named_voice_is_kept(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    resolved = tts.resolve_voice(VoiceConfig(voice_id="Microsoft Zira Desktop"))
    assert (resolved.backend, resolved.voice_id) == ("windows_sapi", "Microsoft Zira Desktop")


def test_unnamed_backend_on_other_platforms_raises(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(TTSError, match="no default TTS backend"):
        tts.resolve_voice(VoiceConfig())


# --- Windows SAPI ---------------------------------------------------------------


def test_sapi_rate_maps_multiplier_to_log_steps():
    assert [sapi_rate(r) for r in (1 / 3, 1.0, 3 ** 0.5, 3.0)] == [-10, 0, 5, 10]
    with pytest.raises(TTSError, match="outside what SAPI can speak"):
        sapi_rate(4.0)


@windows_only
def test_sapi_synthesizes_measured_canonical_audio(tmp_path):
    path, duration = tts.synthesize(TEXT, VoiceConfig(), Cache(tmp_path))
    _assert_canonical(path)
    assert 2.0 < duration < 8.0


@windows_only
def test_sapi_text_cannot_break_out_of_the_script(tmp_path):
    # Straight and curly quotes, a variable, a subexpression: all spoken, not run.
    text = "It's a ‘test’ “quoted” $env:PATH $(exit 7) '; exit 1"
    _, duration = tts.synthesize(text, VoiceConfig(), Cache(tmp_path))
    assert duration > 1.0


@windows_only
def test_sapi_long_narration_is_not_limited_by_the_command_line(tmp_path):
    # 9000 characters overflowed Windows' 32767-character command line when the
    # text travelled there (raw FileNotFoundError, WinError 206); stdin has no cap.
    text = ("collision " * 900)[:9000]
    _, duration = tts.synthesize(text, VoiceConfig(rate=3.0), Cache(tmp_path))
    assert duration > 100.0


@windows_only
def test_sapi_unknown_voice_raises_plain_message(tmp_path):
    voice = VoiceConfig(backend="windows_sapi", voice_id="Daniel")
    with pytest.raises(TTSError, match="No matching voice is installed") as exc:
        tts.synthesize(TEXT, voice, Cache(tmp_path))
    assert "CLIXML" not in str(exc.value)


# --- macOS say -----------------------------------------------------------------


@macos_only
def test_say_synthesizes_measured_canonical_audio(tmp_path):
    path, duration = tts.synthesize(TEXT, VoiceConfig(), Cache(tmp_path))
    _assert_canonical(path)
    assert 2.0 < duration < 8.0


# --- Kokoro ----------------------------------------------------------------------


@pytest.mark.skipif(HAS_KOKORO, reason="kokoro extra is installed")
def test_kokoro_without_extra_says_how_to_install(tmp_path):
    with pytest.raises(TTSError, match=r"pip install -e \"\.\[kokoro\]\""):
        # A real Kokoro voice: called directly, the backend sees the model's
        # unresolved voice_id default ('Daniel'), which it now rejects first.
        voice = VoiceConfig(backend="kokoro", voice_id="af_heart")
        Kokoro().synthesize(TEXT, voice, tmp_path / "k.wav")


@pytest.mark.parametrize("voice_id", ["", "Microsoft David Desktop"])
def test_kokoro_voice_without_a_language_letter_raises_plain_message(tmp_path, voice_id):
    # Checked before the pipeline loads: '' used to raise a raw IndexError, and
    # a letter outside KPipeline's LANG_CODES a raw AssertionError inside it.
    voice = VoiceConfig(backend="kokoro", voice_id=voice_id)
    with pytest.raises(TTSError, match="must start with a Kokoro language letter"):
        Kokoro().synthesize(TEXT, voice, tmp_path / "k.wav")


@needs_kokoro
def test_kokoro_synthesizes_measured_canonical_audio(tmp_path):
    path, duration = tts.synthesize(TEXT, VoiceConfig(backend="kokoro"), Cache(tmp_path))
    _assert_canonical(path)
    assert 2.0 < duration < 8.0


# --- Python 3.13 -----------------------------------------------------------------


def test_speech_stage_needs_no_audioop():
    # audioop left the stdlib in 3.13; the stage must not drag it (or pydub,
    # its main importer) in. Fresh interpreter, so nothing else pre-loaded it.
    code = ("import sys, chalkdust.speech.tts; "
            "print(sorted({'audioop', 'pydub'} & set(sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]"
