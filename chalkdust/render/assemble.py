"""Assembly: silent beat renders + cached audio -> one finished video.

Manim produces video only; TTS produces audio only. Keeping them separate is
what makes the cache work per-stage (D-004) -- but it means every beat must be
muxed before the video can be watched.

Three steps:
  1. mux   -- pair each beat's video with its audio
  2. concat -- join beats into one file
  3. master -- loudness-normalise the finished mix

The muxed beats and the joined file carry uncompressed audio, each beat's cut
to exactly its video's length; only master encodes AAC. AAC codes whole
1024-sample frames: a beat encoded to AAC on its own decodes with its last
frame's padding still on (and the first beat with its encoder delay). Joining
those left the audio's samples and timestamps disagreeing at every cut: played
back to back, each later beat's narration started up to 21 ms further behind
its first frame, cut after cut (binary_search at final: +5 ms at the first
beat, +49 ms at the fourth).
"""

from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

from chalkdust.core.models import Video
from chalkdust.scenes.base import frames_covering
from chalkdust.speech.base import probe_duration, require, run

# YouTube's target. Normalising louder just gets turned back down on playback,
# and costs headroom.
TARGET_LUFS = -14.0
TARGET_LRA = 11.0
TARGET_PEAK = -1.5

# Every intermediate's video uses VCODEC, so concat can use the fast demuxer
# path instead of re-encoding. A mismatch here produces a file that plays
# locally and breaks on upload.
SAMPLE_RATE = 48000
VCODEC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18"]
ACODEC = ["-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE), "-ac", "2"]
# A muxed beat's audio: uncompressed, so it is exactly as many samples long as
# the beat's video and joins the next beat sample-exactly (module docstring).
PCM = ["-c:a", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "2"]


def probe_frame_rate(path: Path) -> Fraction:
    """A video's frame rate, exactly (e.g. 60/1), via ffprobe."""
    require("ffprobe")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return Fraction(proc.stdout.strip())


def mux_beat(video_path: Path, audio_path: Path, out_path: Path) -> None:
    """Combine one beat's silent video with its narration.

    Audio is master (D-002): the narration always plays in full. Manim
    quantises every animation to whole frames, so the clip usually lands a
    little SHORT of the audio (measured: 3.879 s audio, 3.867 s video), and the
    shortfall grows with the number of animations. Capping at the video's
    length clipped the end of the sentence.

    So the result runs to whichever input is longer, rounded up to whole
    frames. A short video holds its last frame -- the beat's settled end
    state -- and the audio is padded with silence to exactly the video's
    length in samples, so the next beat's narration starts on its first frame.
    `out_path` must name a container that takes PCM audio (assemble uses .mov).
    """
    require("ffmpeg")
    fps = probe_frame_rate(video_path)
    # A clip is whole frames, so its probed duration rounds to its frame count;
    # audio that runs past the clip's last frame needs every frame it reaches.
    frames = max(round(probe_duration(video_path) * fps),
                 frames_covering(probe_duration(audio_path), float(fps)))
    samples = round(frames * SAMPLE_RATE / fps)

    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(audio_path),
        # Both pads are unbounded; the frame and sample counts cut them.
        "-vf", "tpad=stop_mode=clone:stop=-1",
        "-frames:v", str(frames),
        "-af", f"aresample={SAMPLE_RATE},apad,atrim=end_sample={samples}",
        *VCODEC, *PCM,
        "-map", "0:v:0", "-map", "1:a:0",
        str(out_path),
    ])


def concat(parts: list[Path], out_path: Path, work_dir: Path) -> None:
    """Join muxed beats in order.

    Uses the concat demuxer with re-encoding. The stream-copy path is faster
    but requires byte-identical encoder settings across inputs; re-encoding
    once is cheap insurance against a subtle mismatch producing a corrupt file.
    """
    require("ffmpeg")
    listing = work_dir / "concat.txt"
    # Paths must be absolute -- the demuxer resolves them relative to the list
    # file, which is a common source of confusing "no such file" errors.
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))

    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        *VCODEC, *PCM,  # still uncompressed: master encodes the AAC, once
        str(out_path),
    ])


def master(src: Path, dst: Path) -> None:
    """Loudness-normalise the finished mix.

    Deliberately done here and not per beat: normalising each beat separately
    would pull quiet beats up to match loud ones and flatten the narration's
    natural dynamics. Loudness is a property of the whole video.

    This is single-pass loudnorm -- less precise than two-pass, but well within
    tolerance for speech and half the processing time.

    loudnorm stamps its output on a 100 ms grid: its last frames leave a gap
    in the timestamps (a player plays it as silence, so the narration after
    it ran late) and run past the end. asetpts restamps the samples back to
    back, and the trim cuts the mix to `src`'s length, which is its video's
    (concat), so the audio ends with the last frame.
    """
    require("ffmpeg")
    samples = round(probe_duration(src) * SAMPLE_RATE)
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-af", f"loudnorm=I={TARGET_LUFS}:LRA={TARGET_LRA}:TP={TARGET_PEAK},"
               f"aresample={SAMPLE_RATE},asetpts=N/SR/TB,atrim=end_sample={samples}",
        "-c:v", "copy",  # video untouched; only the audio filter runs
        *ACODEC,
        str(dst),
    ])


def assemble(video: Video, out_path: Path, work_dir: Path,
             verbose: bool = True) -> Path:
    """Full assembly. Requires every beat to have audio_path and render_path."""
    work_dir.mkdir(parents=True, exist_ok=True)

    missing = [b.id for b in video.beats if not (b.audio_path and b.render_path)]
    if missing:
        raise ValueError(
            f"beats {missing} lack audio or video; run the speech and render "
            "stages first"
        )

    muxed = []
    for beat in video.beats:
        part = work_dir / f"muxed_{beat.id}.mov"
        mux_beat(beat.render_path, beat.audio_path, part)  # type: ignore[arg-type]
        muxed.append(part)
        if verbose:
            print(f"  muxed {beat.id}  {probe_duration(part):5.2f}s")

    joined = work_dir / "joined.mov"
    concat(muxed, joined, work_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master(joined, out_path)

    video.output_path = out_path
    if verbose:
        print(f"  final: {out_path}  {probe_duration(out_path):.2f}s")
    return out_path
