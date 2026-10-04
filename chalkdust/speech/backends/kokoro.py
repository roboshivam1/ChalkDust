"""Kokoro backend -- self-hosted neural TTS, the intended production default
(ARCHITECTURE.md §10: cost at fleet scale).

Optional: install with  pip install -e ".[kokoro]". The import is deferred to
first use so the base install, and every other backend, work without torch.
Written against kokoro 0.9.4 (PyPI, and hexgrad/kokoro kokoro/pipeline.py):

    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    for result in pipeline(text, voice='af_heart', speed=1.0):
        result.audio   # torch.FloatTensor, 24kHz mono

Model weights and voice packs download from Hugging Face on first use and are
cached by huggingface_hub after that.

Voices are named <lang><gender>_<name>; the first letter is the pipeline's
lang_code ('a' American English, 'b' British English, ...). Full list:
https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
"""

from __future__ import annotations

import wave
from pathlib import Path

from chalkdust.core.models import VoiceConfig
from chalkdust.speech.base import TTSError

REPO_ID = "hexgrad/Kokoro-82M"
# Kokoro's output rate, fixed by the model. Coincides with our canonical rate
# today, but normalize_audio owns that conversion either way.
KOKORO_RATE = 24_000
# The language letters KPipeline accepts as lang_code (kokoro 0.9.4,
# pipeline.py LANG_CODES). It asserts on anything else, so we check first.
LANG_CODES = frozenset("abefhijpz")


def lang_code(voice_id: str) -> str:
    """The pipeline language a voice belongs to: its first letter."""
    if not voice_id or voice_id[0] not in LANG_CODES:
        raise TTSError(
            f"kokoro voice_id {voice_id!r} must start with a Kokoro language letter "
            f"({', '.join(sorted(LANG_CODES))}), e.g. 'af_heart'; see "
            f"https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md"
        )
    return voice_id[0]


class Kokoro:
    name = "kokoro"
    raw_format = "wav"
    default_voice = "af_heart"  # graded A in VOICES.md, the best of the set

    def __init__(self) -> None:
        # One pipeline per language. Building one loads the model, which is
        # far too slow to repeat for every beat.
        self._pipelines: dict[str, object] = {}

    def _pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            try:
                from kokoro import KPipeline
            except ImportError as e:
                raise TTSError(
                    f"Kokoro backend needs the optional extra: "
                    f"pip install -e \".[kokoro]\" ({e})"
                ) from e
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id=REPO_ID)
        return self._pipelines[lang_code]

    def synthesize(self, text: str, voice: VoiceConfig, out_path: Path) -> None:
        pipeline = self._pipeline(lang_code(voice.voice_id))

        import numpy as np  # a kokoro dependency; deferred with it

        # The pipeline yields one result per chunk of text it decided to split
        # off; a beat's narration is one take, so stitch them back together.
        chunks = [r.audio.numpy() for r in pipeline(text, voice=voice.voice_id,
                                                    speed=voice.rate)]
        if not chunks:
            raise TTSError(f"kokoro produced no audio for {text[:40]!r}")
        pcm = (np.clip(np.concatenate(chunks), -1.0, 1.0) * 32767).astype("<i2")

        with wave.open(str(out_path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(KOKORO_RATE)
            f.writeframes(pcm.tobytes())
