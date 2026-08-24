"""Assembly: silent beat renders + cached audio -> one finished video.

Manim produces video only; TTS produces audio only. Keeping them separate is
what makes the cache work per-stage (D-004) -- but it means every beat must be
muxed before the video can be watched.

Three steps:
  1. mux   -- pair each beat's video with its audio
  2. concat -- join beats into one file
  3. master -- loudness-normalise the finished mix
"""

from __future__ import annotations

from pathlib import Path

from chalkdust.core.models import Video
from chalkdust.speech.base import probe_duration, require, run

# YouTube's target. Normalising louder just gets turned back down on playback,
# and costs headroom.
TARGET_LUFS = -14.0
TARGET_LRA = 11.0
TARGET_PEAK = -1.5

# Every intermediate uses these, so concat can use the fast demuxer path
# instead of re-encoding. A mismatch here produces a file that plays locally
# and breaks on upload.
VCODEC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18"]
ACODEC = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


def mux_beat(video_path: Path, audio_path: Path, out_path: Path) -> None:
    """Combine one beat's silent video with its narration.

    Video duration is authoritative. Manim quantises to whole frames, so the
    video is typically a few milliseconds longer than the audio -- we pad the
    audio with silence rather than letting ffmpeg truncate the video to match.
    """
    require("ffmpeg")
    v_dur = probe_duration(video_path)

    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(audio_path),
        # apad adds silence; -t caps the result at the video's exact length.
        "-af", "apad",
        "-t", f"{v_dur:.6f}",
        *VCODEC, *ACODEC,
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
        *VCODEC, *ACODEC,
        str(out_path),
    ])


def master(src: Path, dst: Path) -> None:
    """Loudness-normalise the finished mix.

    Deliberately done here and not per beat: normalising each beat separately
    would pull quiet beats up to match loud ones and flatten the narration's
    natural dynamics. Loudness is a property of the whole video.

    This is single-pass loudnorm -- less precise than two-pass, but well within
    tolerance for speech and half the processing time.
    """
    require("ffmpeg")
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-af", f"loudnorm=I={TARGET_LUFS}:LRA={TARGET_LRA}:TP={TARGET_PEAK}",
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
        part = work_dir / f"muxed_{beat.id}.mp4"
        mux_beat(beat.render_path, beat.audio_path, part)  # type: ignore[arg-type]
        muxed.append(part)
        if verbose:
            print(f"  muxed {beat.id}  {probe_duration(part):5.2f}s")

    joined = work_dir / "joined.mp4"
    concat(muxed, joined, work_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master(joined, out_path)

    video.output_path = out_path
    if verbose:
        print(f"  final: {out_path}  {probe_duration(out_path):.2f}s")
    return out_path
