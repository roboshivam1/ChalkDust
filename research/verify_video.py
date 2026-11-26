"""Check a rendered ChalkDust video by measurement, on the encoded MP4.

    HF_HUB_OFFLINE=1 PYTHONPATH=. .lab/venv-k312/Scripts/python research/verify_video.py         SPEC.json VIDEO.mp4 CACHE_DIR OUT_DIR

CACHE_DIR is the render's --cache-dir (the Kokoro word timings are read from
it). Run 3's acceptance gate for an animation: every phase join 0 px over the
threshold in the encoded 1080p video, on a render whose first moments are cued.

Per beat: audio length vs clip length; cue times (from the cached Kokoro word
timings) with the picture change measured just before/after each cue; Kokoro
word starts vs faster-whisper word starts on the same WAV. Per seam: the last
frame of beat k vs the first frame of beat k+1 in the final file, against the
typical frame-to-frame change inside a hold (encoder noise). Writes
verify.json and seam/cue/busiest PNGs into OUT_DIR.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import chalkdust.scenes.components  # noqa: F401
from chalkdust.core.cache import Cache
from chalkdust.core.models import VideoSpec
from chalkdust.scenes.components.animation import cue_times_for
from chalkdust.scenes.components.base import make_component
from chalkdust.speech.tts import resolve_voice, synthesize_timed
from research.voice_check import pcm16k

spec_p, video_p, cache_p, out = sys.argv[1:5]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
spec = VideoSpec.model_validate(json.loads(Path(spec_p).read_text(encoding="utf-8")))
cache = Cache(Path(cache_p))
voice = resolve_voice(spec.voice)

probe = json.loads(subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
     "stream=r_frame_rate,nb_read_frames,width,height", "-of", "json", video_p],
    capture_output=True, text=True, check=True).stdout)["streams"][0]
num, den = map(int, probe["r_frame_rate"].split("/"))
fps = num / den
W, H = 640, 360
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video_p, "-vf", f"scale={W}:{H}",
                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True,
                     check=True).stdout
frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
n = len(frames)


def diff(i, j):
    d = np.abs(frames[i].astype(int) - frames[j].astype(int)).max(axis=2)
    return {"px_over_24": int((d > 24).sum()), "max": int(d.max())}


def full_frame(i, name):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video_p, "-vf",
                    f"select=eq(n\\,{i})", "-vframes", "1", str(out / name)], check=True)


consec = [diff(i, i + 1)["px_over_24"] for i in range(n - 1)]
noise = float(np.percentile(consec, 50))

from faster_whisper import WhisperModel  # noqa: E402

asr = WhisperModel("small.en", device="cpu", compute_type="int8")

report = {"fps": fps, "frames": n, "median_consecutive_px_over_24": noise, "beats": [],
          "seams": []}
start = 0
for b in spec.beats:
    wav, dur, words = synthesize_timed(b.narration, voice, cache)
    nframes = int(np.ceil(dur * fps - 1e-9))
    comp = make_component(b.component, b.params)
    cues = cue_times_for(comp, b.narration, words, dur)
    segs, _ = asr.transcribe(pcm16k(wav), language="en", word_timestamps=True)
    heard = [w for s in segs for w in s.words]
    # Pair Kokoro words with whisper words by normalised text, in order.
    norm = lambda s: "".join(c for c in s.lower() if c.isalnum())  # noqa: E731
    hk = [(norm(w.word), w.start) for w in heard]
    offs, j = [], 0
    for w in words:
        k = norm(w.word)
        for jj in range(j, min(j + 4, len(hk))):
            if hk[jj][0] == k:
                offs.append(w.start - hk[jj][1])
                j = jj + 1
                break
    cue_rows = []
    for m, t in cues.items():
        f = start + int(round(t * fps))
        before, after = max(start, f - 2), min(start + nframes - 1, f + 6)
        cue_rows.append({"moment": m, "t": round(t, 2), "frame": f,
                         "change_2_before_to_cue": diff(before, f)["px_over_24"],
                         "change_cue_to_6_after": diff(f, after)["px_over_24"]})
        full_frame(after, f"{b.id}_{m}_after.png")
    report["beats"].append({
        "id": b.id, "audio_s": round(dur, 3), "clip_frames": nframes,
        "clip_s": round(nframes / fps, 3), "kokoro_words": len(words),
        "whisper_words": len(heard), "paired": len(offs),
        "kokoro_minus_whisper_start_s": {
            "median": round(float(np.median(offs)), 3) if offs else None,
            "p90_abs": round(float(np.percentile(np.abs(offs), 90)), 3) if offs else None},
        "cues": cue_rows})
    if start:
        report["seams"].append({"before": start - 1, "after": start,
                                **diff(start - 1, start)})
        Image.fromarray(np.concatenate([frames[start - 1], frames[start]], axis=1)).save(
            out / f"seam_{b.id}.png")
    start += nframes
report["total_frames_expected"] = start
# Busiest frame: most changed pixels vs the previous frame.
busy = int(np.argmax(consec)) + 1
report["busiest_frame"] = busy
full_frame(busy, "busiest.png")
(out / "verify.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps(report, indent=1))
