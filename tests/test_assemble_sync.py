"""Every beat's narration starts on the beat's first frame in the finished
video, and the audio ends with its last frame.

Audio is master (D-002) and each beat is ceil(audio * fps) frames, so a cut is
the one place sync can be lost: if a beat's audio comes out of the mux and
concat any longer or shorter than its video, every later beat's narration
slides against its animation, and the error grows with the beat count.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from chalkdust.core.models import Video, VideoSpec
from chalkdust.render.assemble import assemble

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "binary_search.json"
FPS = 15  # draft tier
SR = 48000  # what the finished file's audio is decoded at here
BURST = 0.1  # seconds of noise that marks the start of each beat's narration
# Narration lengths at a TTS-like 24 kHz. None is a whole number of AAC frames
# (1024 samples at 48 kHz) or of video frames, as real narration never is.
NARRATION = [2.0307, 1.7113, 2.4671, 1.2949]
TOLERANCE_MS = 1.0  # far below one frame (67 ms here); exact assembly is ~0


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def _pcm(path: Path, by_timestamp: bool = False) -> np.ndarray:
    """Mono samples at SR from the file's first audio stream.

    by_timestamp: laid out by the container's timestamps, as a player that
    honours them plays it (sample i is at i / SR s, a gap is silence, an
    overlap is dropped). Otherwise every decoded sample back to back, as
    ffmpeg's default transcode and players that ignore timestamps do. A
    stream whose timestamps and samples agree reads the same either way.
    """
    layout = (["-af", f"aresample={SR}:async=1:min_hard_comp=0.001:first_pts=0"]
              if by_timestamp else ["-ar", str(SR)])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", *layout,
         "-ac", "1", "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def _probe(path: Path, stream: str, entries: str) -> list:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-of", "json", "-select_streams", stream,
         "-show_entries", entries, str(path)],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)[entries.split("=")[0] + "s"]


def _onset(audio: np.ndarray, burst: np.ndarray, expected: int) -> int:
    """Sample where `burst` best matches `audio`, searched within 0.3 s of
    `expected` (cross-correlation, so loudness normalisation does not move it)."""
    lo = max(0, expected - int(0.3 * SR))
    seg = audio[lo:expected + int(0.3 * SR) + len(burst)]
    corr = np.correlate(seg, burst, mode="valid")
    return lo + int(np.argmax(corr))


@pytest.fixture(scope="module")
def assembled(tmp_path_factory):
    """Four synthetic beats assembled once: (video, frames per beat, final MP4).
    Each clip is ceil(narration * fps) frames, as Manim renders it; each
    narration starts with its own noise burst and is silent after it."""
    tmp = tmp_path_factory.mktemp("assembled")
    spec = VideoSpec.model_validate(json.loads(EXAMPLE.read_text(encoding="utf-8")))
    video = Video.from_spec(spec)
    assert len(video.beats) == len(NARRATION)

    frames = []
    for i, (beat, seconds) in enumerate(zip(video.beats, NARRATION)):
        n = math.ceil(seconds * FPS)
        frames.append(n)
        clip, wav = tmp / f"clip_{beat.id}.mp4", tmp / f"{beat.id}.wav"
        _ffmpeg("-f", "lavfi", "-i", f"testsrc=size=160x90:rate={FPS}",
                "-frames:v", str(n), "-pix_fmt", "yuv420p", str(clip))
        _ffmpeg("-f", "lavfi", "-i",
                f"anoisesrc=duration={BURST}:sample_rate=24000:amplitude=0.5:seed={i + 1}",
                "-af", f"apad=whole_len={round(seconds * 24000)}", str(wav))
        beat.render_path, beat.audio_path = clip, wav

    out = tmp / "final.mp4"
    assemble(video, out, tmp / "assemble", verbose=False)
    return video, frames, out


@pytest.mark.parametrize("by_timestamp", [True, False],
                         ids=["by-timestamp", "back-to-back"])
def test_narration_starts_on_each_beats_first_frame(assembled, by_timestamp):
    video, frames, out = assembled
    pts = [float(f["best_effort_timestamp_time"])
           for f in _probe(out, "v:0", "frame=best_effort_timestamp_time")]
    assert len(pts) == sum(frames)
    audio_start = 0.0 if by_timestamp else float(
        _probe(out, "a:0", "stream=start_time")[0]["start_time"])
    audio = _pcm(out, by_timestamp)

    first_frame, expected, drift = 0, 0, []
    for beat, n in zip(video.beats, frames):
        burst = _pcm(beat.audio_path)[:int(BURST * SR)]
        at = _onset(audio, burst, expected)
        drift.append(round((audio_start + at / SR - pts[first_frame]) * 1000, 2))
        first_frame += n
        expected = at + round(n / FPS * SR)

    assert all(abs(d) <= TOLERANCE_MS for d in drift), (
        f"narration minus first frame, ms, per beat: {drift}")


def test_audio_ends_with_the_last_frame(assembled):
    _, frames, out = assembled
    video_s = float(_probe(out, "v:0", "stream=duration")[0]["duration"])
    audio_s = float(_probe(out, "a:0", "stream=duration")[0]["duration"])
    assert video_s == pytest.approx(sum(frames) / FPS, abs=1e-3)
    assert abs(audio_s - video_s) * 1000 <= TOLERANCE_MS, (
        f"audio {audio_s:.6f} s against video {video_s:.6f} s")
