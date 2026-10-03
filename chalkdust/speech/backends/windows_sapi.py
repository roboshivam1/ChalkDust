"""Windows SAPI backend, via System.Speech in Windows PowerShell.

The Windows counterpart of macos_say: zero install, decent quality, useful for
development. Not a production voice -- it exists so the pipeline can run end to
end on real measured durations on a Windows machine.

List voices with:
  powershell -c "Add-Type -A System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices().VoiceInfo.Name"
"""

from __future__ import annotations

import base64
import math
from pathlib import Path

from chalkdust.core.models import VoiceConfig
from chalkdust.speech.base import TTSError, require, run

# SAPI rate is an integer step in [-10, 10] on a log scale: +10 is three times
# the voice's natural speed, -10 a third of it (measured: 7.19s at 0, 2.36s at
# +10, 21.57s at -10). Our VoiceConfig.rate is a multiplier, so we convert,
# accepting the 3**0.1 (~12%) quantisation of SAPI's integer steps.
RATE_STEPS = 10
RATE_SPAN = 3.0

# Strings reach PowerShell as base64 literals and are decoded there. Quoting
# text into a script by escaping is fragile -- PowerShell treats the curly
# quotes ‘ ’ as string delimiters too -- and base64 cannot break out.
#
# The trap writes failures as one plain line straight to stderr. Left to
# itself, PowerShell run with -EncodedCommand serialises errors as CLIXML,
# which run() would surface as unreadable markup.
SCRIPT = """
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
trap {{ [Console]::Error.WriteLine($_.Exception.Message); exit 1 }}
function arg($b) {{ [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($b)) }}
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{
    $s.SelectVoice((arg '{voice}'))
    $s.Rate = {rate}
    $s.SetOutputToWaveFile((arg '{out}'))
    $s.Speak((arg '{text}'))
}} finally {{
    $s.Dispose()
}}
"""


def sapi_rate(rate: float) -> int:
    """Map a speed multiplier to SAPI's integer rate step."""
    if not 1 / RATE_SPAN <= rate <= RATE_SPAN:
        raise TTSError(
            f"rate {rate} is outside what SAPI can speak "
            f"({1 / RATE_SPAN:.2f}x to {RATE_SPAN:.0f}x)"
        )
    return round(RATE_STEPS * math.log(rate, RATE_SPAN))


def _b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


class WindowsSAPI:
    name = "windows_sapi"
    raw_format = "wav"  # SetOutputToWaveFile: 22.05kHz mono PCM for the stock voices
    default_voice = "Microsoft David Desktop"

    def synthesize(self, text: str, voice: VoiceConfig, out_path: Path) -> None:
        # Windows PowerShell 5.1, not pwsh: System.Speech ships with .NET
        # Framework and is absent from the .NET that PowerShell 7 runs on.
        require("powershell")
        script = SCRIPT.format(
            voice=_b64(voice.voice_id),
            rate=sapi_rate(voice.rate),
            out=_b64(str(out_path)),
            text=_b64(text),
        )
        # -EncodedCommand takes base64 of UTF-16LE, so the script itself
        # needs no command-line quoting either.
        run([
            "powershell", "-NoProfile", "-NonInteractive",
            "-EncodedCommand", base64.b64encode(script.encode("utf-16-le")).decode("ascii"),
        ])
