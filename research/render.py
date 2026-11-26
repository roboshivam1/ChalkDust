"""Render a run's spec to MP4 with the real ChalkDust CLI, plus a contact sheet.

    python -m research.render .lab/runs/<run>/<model>/<method>/<topic>_s0 --name gemma12b-incline --quality final

Writes .lab/videos/<name>-<quality>.mp4 and <name>-contact.png (one frame
every few seconds, tiled) so a person -- or the lead -- can check the frames
before a video goes on the watch list. Uses the repo's own `chalkdust render`
(the harness venv's editable install), MiKTeX first on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from research.score import ensure_miktex_first

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / ".lab" / "venv-harness" / "Scripts" / "chalkdust.exe"
OUT = REPO / ".lab" / "videos"


def render(run_dir: Path, name: str, quality: str = "draft", jobs: int | None = None) -> dict:
    ensure_miktex_first()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = run_dir / "spec.json"
    mp4 = OUT / f"{name}-{quality}.mp4"
    work = REPO / ".lab" / "render-work" / name
    t0 = time.time()
    cmd = [str(CLI), "render", str(spec), "--quality", quality, "--out", str(mp4), "--work-dir", str(work),
           "--cache-dir", str(REPO / ".lab" / "render-cache")]
    if jobs:
        cmd += ["-j", str(jobs)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=os.environ)
    secs = time.time() - t0
    res = {"spec": str(spec), "mp4": str(mp4), "quality": quality, "rc": p.returncode, "render_seconds": secs,
           "tail": (p.stdout + p.stderr)[-800:]}
    if p.returncode == 0 and mp4.exists():
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                    "default=nw=1:nk=1", str(mp4)], capture_output=True, text=True).stdout.strip() or 0)
        res["duration_s"] = dur
        step = max(dur / 16, 1.0)
        sheet = OUT / f"{name}-contact.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp4), "-vf",
                        f"fps=1/{step:.2f},scale=480:-1,tile=4x4", "-frames:v", "1", str(sheet)])
        res["contact"] = str(sheet)
    (OUT / f"{name}-{quality}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--quality", default="draft", choices=["draft", "final"])
    ap.add_argument("-j", "--jobs", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(render(Path(a.run_dir), a.name, a.quality, a.jobs), indent=2))
