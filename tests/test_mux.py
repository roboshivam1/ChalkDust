"""Muxing a beat never clips its narration (D-002: audio is master)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from chalkdust.render.assemble import mux_beat

FPS = 15  # draft tier: one lost frame is 67 ms of narration


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def _stream_duration(path: Path, stream: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream,
         "-show_entries", "stream=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def test_mux_never_clips_audio(tmp_path):
    # A clip several frames shorter than its narration: what Manim's
    # per-animation frame quantisation produces on an animation-heavy beat.
    # The gap is well above one AAC frame (~21 ms), so encoder padding alone
    # cannot make a clipping mux pass.
    video, audio, out = tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "m.mp4"
    _ffmpeg("-f", "lavfi", "-i", f"testsrc=size=320x180:rate={FPS}:duration=3.6",
            "-pix_fmt", "yuv420p", str(video))
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=24000:duration=3.879",
            str(audio))

    mux_beat(video, audio, out)

    assert _stream_duration(out, "a:0") >= 3.879 - 0.03, "narration was clipped"
    # The picture must last as long as the sound, not cut to black under it.
    assert _stream_duration(out, "v:0") >= 3.879 - 1 / FPS
