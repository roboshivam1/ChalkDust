"""Speech stage: narration text -> measured audio.

Runs before rendering. Every beat's animation timing derives from the duration
measured here (D-002), so this stage must complete before a render key can even
be constructed.
"""

from __future__ import annotations

from pathlib import Path

from chalkdust.core.cache import Cache, tts_key
from chalkdust.core.models import Beat, Video, VoiceConfig
from chalkdust.speech.backends.kokoro import Kokoro
from chalkdust.speech.backends.macos_say import MacOSSay
from chalkdust.speech.base import TTSBackend, TTSError, normalize_audio, probe_duration

BACKENDS: dict[str, TTSBackend] = {
    b.name: b for b in (MacOSSay(), Kokoro())  # type: ignore[list-item]
}


def get_backend(name: str) -> TTSBackend:
    if name not in BACKENDS:
        raise TTSError(f"unknown TTS backend {name!r}; have: {sorted(BACKENDS)}")
    return BACKENDS[name]


def synthesize(text: str, voice: VoiceConfig, cache: Cache) -> tuple[Path, float]:
    """Return (wav path, duration). Cached on (text, voice).

    On a cache hit this is a stat call and an ffprobe -- cheap enough that
    re-running the whole stage during iteration costs nothing.
    """
    key = tts_key(text, voice)
    slot = cache.slot("tts", key, ".wav")

    if not slot.exists:
        backend = get_backend(voice.backend)
        # `say` emits AIFF; the extension must reflect that so ffmpeg can
        # read it back. Sibling of slot.tmp so it lands in the same directory.
        raw = slot.tmp.with_name(f".raw-{key}.aiff")
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
    for beat in video.beats:
        was_cached = cache.slot(
            "tts", tts_key(beat.spec.narration, video.spec.voice), ".wav"
        ).exists
        synthesize_beat(beat, video.spec.voice, cache)
        if verbose:
            mark = "cached" if was_cached else "synth "
            print(f"  [{mark}] {beat.id}  {beat.duration:5.2f}s  "
                  f"{beat.spec.narration[:52]}")
    if verbose:
        print(f"  total narration: {video.total_duration:.1f}s")
    return video
