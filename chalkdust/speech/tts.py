"""Speech stage: narration text -> measured audio.

Runs before rendering. Every beat's animation timing derives from the duration
measured here (D-002), so this stage must complete before a render key can even
be constructed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from chalkdust.core.cache import Cache, tts_key
from chalkdust.core.models import Beat, Video, VoiceConfig
from chalkdust.speech.backends.kokoro import Kokoro
from chalkdust.speech.backends.macos_say import MacOSSay
from chalkdust.speech.backends.windows_sapi import WindowsSAPI
from chalkdust.speech.base import TTSBackend, TTSError, normalize_audio, probe_duration

BACKENDS: dict[str, TTSBackend] = {
    b.name: b for b in (MacOSSay(), WindowsSAPI(), Kokoro())  # type: ignore[list-item]
}

# The zero-install development voice of each platform, used when a spec names
# no backend -- so one spec runs unchanged on macOS and on Windows. A spec that
# must sound the same everywhere names its backend (docs/VOICE.md).
PLATFORM_BACKENDS = {
    "darwin": "macos_say",
    "win32": "windows_sapi",
}


def get_backend(name: str) -> TTSBackend:
    if name not in BACKENDS:
        raise TTSError(f"unknown TTS backend {name!r}; have: {sorted(BACKENDS)}")
    return BACKENDS[name]


def resolve_voice(voice: VoiceConfig) -> VoiceConfig:
    """Fill in what the spec left unsaid, from the platform and the backend.

    Only fields the spec actually set count as said (pydantic's
    model_fields_set); VoiceConfig's own defaults stand in for "unset". An
    unset backend becomes this platform's; an unset voice_id becomes that
    backend's default_voice, since voice ids mean nothing across backends.

    The resolved config, not the spec's, is what gets hashed: the same spec
    produces different audio on macOS and Windows, so it must not share a
    cache key across them (D-004). On macOS the result is identical to the
    old defaults, so existing cached audio stays valid there.
    """
    said = voice.model_fields_set
    backend = voice.backend
    if "backend" not in said:
        if sys.platform not in PLATFORM_BACKENDS:
            raise TTSError(
                f"no default TTS backend on {sys.platform!r}; name one in the "
                f"spec's voice.backend (have: {sorted(BACKENDS)})"
            )
        backend = PLATFORM_BACKENDS[sys.platform]
    voice_id = voice.voice_id if "voice_id" in said else get_backend(backend).default_voice
    return VoiceConfig(backend=backend, voice_id=voice_id, rate=voice.rate)


def synthesize(text: str, voice: VoiceConfig, cache: Cache) -> tuple[Path, float]:
    """Return (wav path, duration). Cached on (text, resolved voice).

    On a cache hit this is a stat call and an ffprobe -- cheap enough that
    re-running the whole stage during iteration costs nothing.
    """
    voice = resolve_voice(voice)
    key = tts_key(text, voice)
    slot = cache.slot("tts", key, ".wav")

    if not slot.exists:
        backend = get_backend(voice.backend)
        # The raw file carries the backend's own format as its extension so
        # ffmpeg can read it back. Sibling of slot.tmp so it lands in the
        # same directory.
        raw = slot.tmp.with_name(f".raw-{key}.{backend.raw_format}")
        try:
            backend.synthesize(text, voice, raw)
            # Normalise into the temp path, then commit atomically -- a crash
            # mid-convert must not leave a partial file the cache would trust.
            normalize_audio(raw, slot.tmp)
        finally:
            raw.unlink(missing_ok=True)
        slot.commit()

    return slot.path, probe_duration(slot.path)


def synthesize_beat(beat: Beat, voice: VoiceConfig, cache: Cache) -> Beat:
    beat.audio_path, beat.duration = synthesize(beat.spec.narration, voice, cache)
    return beat


def synthesize_video(video: Video, cache: Cache, verbose: bool = True) -> Video:
    """Fill in audio_path and duration for every beat."""
    voice = resolve_voice(video.spec.voice)
    for beat in video.beats:
        was_cached = cache.slot("tts", tts_key(beat.spec.narration, voice), ".wav").exists
        synthesize_beat(beat, voice, cache)
        if verbose:
            mark = "cached" if was_cached else "synth "
            print(f"  [{mark}] {beat.id}  {beat.duration:5.2f}s  "
                  f"{beat.spec.narration[:52]}")
    if verbose:
        print(f"  total narration: {video.total_duration:.1f}s")
    return video
