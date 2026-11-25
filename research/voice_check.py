"""Measure narration audio, since nobody on the build side can listen to it.

    python -m research.voice_check samples --out DIR        # one clip per voice
    python -m research.voice_check wav FILE.wav --text "..."  # check one file

What is measured per clip (all from the samples, no opinion involved):
  duration     seconds of audio
  max_gap      longest internal silence (below -40 dBFS for 0.15 s or more)
  peak_dbfs    loudest sample; clipped = samples at full scale
  wer          word error rate of a local speech-to-text pass (faster-whisper
               small.en on CPU) against the script: a clip the recogniser
               cannot read back is a clip a student will struggle with too
  hyp          what the recogniser heard

Numbers in the script are compared as words ("three" == "3") so a recogniser
writing digits does not count as an error.

Run with the Kokoro venv (Python 3.12): .lab/venv-k312.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

# Kokoro English voices worth hearing (hexgrad/Kokoro-82M VOICES.md grades:
# af_heart A, af_bella A-, the rest B- to C+). a = American, b = British.
CANDIDATES = ["af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah",
              "am_michael", "am_fenrir", "am_puck", "bf_emma", "bf_isabella", "bm_george",
              "bm_fable"]

SENTENCE = ("The derivative of x squared plus three x is two x plus three. So at x equals "
            "negative one and a half the tangent line is flat, and the area under x squared "
            "from zero to two is eight thirds, about two point six seven.")

_NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
        "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}


def words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower().replace("-", " ")):
        out.append(_NUM.get(w, w))
    return out


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return round(d[len(h)] / max(1, len(r)), 3)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as f:
        sr, n, ch = f.getframerate(), f.getnframes(), f.getnchannels()
        pcm = np.frombuffer(f.readframes(n), dtype="<i2").astype(float) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return pcm, sr


def measure(path: Path) -> dict:
    pcm, sr = read_wav(path)
    win = int(0.02 * sr)
    frames = pcm[: len(pcm) // win * win].reshape(-1, win)
    rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)
    quiet = rms_db < -40
    # Internal gaps only: ignore leading and trailing silence.
    loud = np.nonzero(~quiet)[0]
    gap, best = 0, 0
    if loud.size:
        for q in quiet[loud[0]:loud[-1] + 1]:
            gap = gap + 1 if q else 0
            best = max(best, gap)
    peak = float(np.abs(pcm).max())
    return {"duration": round(len(pcm) / sr, 2), "max_gap": round(best * 0.02, 2),
            "peak_dbfs": round(20 * np.log10(peak + 1e-12), 1),
            "clipped": int((np.abs(pcm) >= 0.999).sum())}


_ASR = None


def transcribe(path: Path) -> str:
    global _ASR
    if _ASR is None:
        from faster_whisper import WhisperModel
        _ASR = WhisperModel("small.en", device="cpu", compute_type="int8")
    segs, _ = _ASR.transcribe(str(path), language="en", beam_size=5)
    return " ".join(s.text.strip() for s in segs)


def check(path: Path, text: str) -> dict:
    m = measure(path)
    hyp = transcribe(path)
    return {**m, "wer": wer(text, hyp), "hyp": hyp}


def samples(out: Path, voices: list[str], text: str) -> list[dict]:
    from chalkdust.core.cache import Cache
    from chalkdust.core.models import VoiceConfig
    from chalkdust.speech.tts import synthesize_timed

    out.mkdir(parents=True, exist_ok=True)
    cache = Cache(out / ".cache")
    rows = []
    for v in voices:
        t0 = time.time()
        path, dur, _ = synthesize_timed(text, VoiceConfig(backend="kokoro", voice_id=v), cache)
        synth = round(time.time() - t0, 2)
        dst = out / f"{v}.wav"
        dst.write_bytes(path.read_bytes())
        # An MP3 copy too, so the clips open anywhere.
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(dst), "-b:a", "128k",
                        str(out / f"{v}.mp3")], check=True)
        row = {"voice": v, "synth_s": synth, **check(dst, text)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "voices.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("samples")
    s.add_argument("--out", required=True)
    s.add_argument("--voices", default=",".join(CANDIDATES))
    s.add_argument("--text", default=SENTENCE)
    w = sub.add_parser("wav")
    w.add_argument("file")
    w.add_argument("--text", required=True)
    a = ap.parse_args()
    if a.cmd == "samples":
        samples(Path(a.out), a.voices.split(","), a.text)
    else:
        print(json.dumps(check(Path(a.file), a.text)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
