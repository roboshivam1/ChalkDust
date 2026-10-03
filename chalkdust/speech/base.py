"""TTS backend interface and shared audio utilities.

Backends produce audio at a path we give them. Everything else -- caching,
format normalisation, duration measurement -- happens once, here, so backends
stay small and interchangeable.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

from chalkdust.core.models import VoiceConfig

# One canonical intermediate format. Speech is mono and band-limited, so 24kHz
# is transparent and keeps files small. Fixing this here means the assembly
# stage never has to reconcile mismatched sample rates.
SAMPLE_RATE = 24_000
CHANNELS = 1
CODEC = "pcm_s16le"


class TTSError(RuntimeError):
    pass


class TTSBackend(Protocol):
    """Synthesise `text` to `out_path`, in the backend's own raw format.

    Each engine has a native output (`say` writes AIFF, SAPI and Kokoro WAV),
    so the backend declares it as `raw_format` -- the file extension the stage
    gives `out_path`, which is how ffmpeg recognises the file when
    normalize_audio converts it to the canonical format afterwards in tts.py.

    `default_voice` is used when a spec names no voice_id: voice ids are
    meaningful only to the backend that owns them (docs/VOICE.md).
    """

    name: str
    raw_format: str
    default_voice: str

    def synthesize(self, text: str, voice: VoiceConfig, out_path: Path) -> None: ...


def require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise TTSError(f"{binary!r} not found on PATH")
    return path


def run(cmd: list[str], input: str | None = None) -> None:
    """Run a command, raising with captured stderr on failure.

    ffmpeg writes everything to stderr, so a bare CalledProcessError tells you
    nothing useful. This surfaces the actual message. `input`, if given, is
    written to the command's stdin.

    A command that cannot even start (on Windows, e.g. a command line over the
    32767-character limit, WinError 206) raises TTSError too, not a raw OSError.
    """
    try:
        proc = subprocess.run(cmd, input=input, capture_output=True, text=True)
    except OSError as e:
        raise TTSError(f"{cmd[0]} could not be started: {e}") from e
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-6:]
        raise TTSError(f"{cmd[0]} failed:\n" + "\n".join(tail))


def probe_duration(path: Path) -> float:
    """Exact duration in seconds, via ffprobe.

    This number drives every animation run time in the beat (D-002), so it must
    come from the file itself -- never from a words-per-minute estimate.

    A file with no audio in it is a speech failure, said in words: ffprobe
    reports its duration as "N/A" (a voice given nothing it can pronounce,
    e.g. SAPI on the narration ".", writes a header-only WAV), and a
    zero-length clip would give the beat no frames to animate in.
    """
    require("ffprobe")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    reported = proc.stdout.strip()
    if proc.returncode != 0 or not reported:
        raise TTSError(f"could not probe duration of {path}")
    try:
        seconds = float(reported)
    except ValueError:
        seconds = math.nan
    if not math.isfinite(seconds) or seconds <= 0:
        raise TTSError(
            f"the synthesised audio {path} has no measurable duration (ffprobe "
            f"reports {reported!r}): the voice produced no sound. Check that "
            "the narration has words the voice can speak.")
    return seconds


def normalize_audio(src: Path, dst: Path, trim_silence: bool = True) -> None:
    """Convert to the canonical format, optionally trimming edge silence.

    Trimming matters for sync: most TTS backends pad a little silence at each
    end, and that padding lands between beats as dead air once the video is
    concatenated.

    The double `areverse` is the standard trick -- ffmpeg's silenceremove only
    trims from the start, so we reverse, trim again, and reverse back.

    Note we do NOT loudness-normalise here. Per-beat loudnorm would flatten the
    narration's dynamics; that belongs at assembly, on the finished mix.
    """
    require("ffmpeg")
    filters = []
    if trim_silence:
        detect = "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.05"
        filters += [detect, "areverse", detect, "areverse"]

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-c:a", CODEC, str(dst)]
    run(cmd)
