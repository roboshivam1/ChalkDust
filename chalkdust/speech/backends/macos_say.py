"""macOS `say` backend.

Zero install, decent quality, useful for development. Not a production voice --
it exists so the pipeline can run end to end on real measured durations before
a model backend is wired in.

List voices with:  say -v '?'
"""

from __future__ import annotations

from pathlib import Path

from chalkdust.core.models import VoiceConfig
from chalkdust.speech.base import require, run

# `say` takes rate in words per minute; our VoiceConfig.rate is a multiplier,
# so we scale from a natural baseline.
BASE_WPM = 175


class MacOSSay:
    name = "macos_say"

    def synthesize(self, text: str, voice: VoiceConfig, out_path: Path) -> None:
        require("say")
        # `say` cannot emit WAV, so the caller hands us an .aiff path and
        # normalize_audio converts it afterwards.
        run([
            "say",
            "-v", voice.voice_id,
            "-r", str(int(BASE_WPM * voice.rate)),
            "-o", str(out_path),
            text,  # passed as a list arg, so no shell quoting concerns
        ])
