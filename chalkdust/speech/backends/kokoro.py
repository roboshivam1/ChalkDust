"""Kokoro backend -- not yet wired.

Intentionally a stub. Rather than guess at the API surface, this should be
filled in to match the working Kokoro integration already in use elsewhere,
so both projects share one calling convention.

To implement: synthesise `text` with `voice.voice_id`, write a WAV to
`out_path`. Format does not matter -- normalize_audio converts it.
"""

from __future__ import annotations

from pathlib import Path

from chalkdust.core.models import VoiceConfig
from chalkdust.speech.base import TTSError


class Kokoro:
    name = "kokoro"

    def synthesize(self, text: str, voice: VoiceConfig, out_path: Path) -> None:
        raise TTSError(
            "Kokoro backend not implemented. Use backend='macos_say' for now, "
            "or fill in chalkdust/speech/backends/kokoro.py."
        )
