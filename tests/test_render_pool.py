"""The render pool (ARCHITECTURE.md §4): beats rendered across processes give
the same clips, keys and video as the sequential render, and a beat failing in
a worker process fails the run with the same typed error and exit code.

Speech uses test_pipeline's FakeTTS. It runs in this process (speech is not
pooled), so registering it here is enough; the worker processes only render.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from chalkdust import cli, pipeline
from chalkdust.core.models import Quality
from chalkdust.render import worker
from chalkdust.render.pool import default_jobs
from chalkdust.scenes.theme import LatexToolchainError
from chalkdust.speech import tts
from chalkdust.speech.base import probe_duration
from test_pipeline import FakeTTS, example_spec, fake_tts, write_spec  # noqa: F401


def _tiny_spec() -> dict:
    """Three beats, two distinct: b03 is b01 under another id, so it has b01's
    render key -- sequentially the second of them is a cache hit, and the pool
    must neither render it twice nor race two processes on one cache slot."""
    spec = example_spec()
    b01, b02 = spec["beats"][:2]
    spec["beats"] = [b01, b02, {**b01, "id": "b03"}]
    return spec


def _frames(path: Path) -> int:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True)
    return int(proc.stdout.strip())


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    """The tiny spec rendered sequentially (--jobs 1) and pooled (--jobs 3),
    each into its own cache and work dir."""
    root = tmp_path_factory.mktemp("pool")
    runs = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(tts.BACKENDS, "fake", FakeTTS())
        spec_path = write_spec(root / "spec.json", _tiny_spec())
        for name, jobs in (("sequential", 1), ("pooled", 3)):
            result = pipeline.render(spec_path, Quality.DRAFT, root / f"{name}.mp4",
                                     cache_dir=root / name / "cache",
                                     work_dir=root / name / "work", jobs=jobs)
            runs[name] = (root / name, result)
    return spec_path, runs


def test_pooled_render_matches_sequential(both, capfd):
    _, runs = both
    (seq_root, seq), (pool_root, pool) = runs["sequential"], runs["pooled"]

    # Same beats, same measured durations, same cache verdicts (b03 is a hit
    # on b01's clip either way).
    assert [(b.beat_id, b.duration, b.render_cached) for b in pool.beats] == \
        [(b.beat_id, b.duration, b.render_cached) for b in seq.beats]
    assert pool.rebuilt == seq.rebuilt == ["b01", "b02"]

    # Same render keys, and clips of the same length, frame for frame.
    seq_clips = {p.name: p for p in (seq_root / "cache" / "beats").glob("*.mp4")}
    pool_clips = {p.name: p for p in (pool_root / "cache" / "beats").glob("*.mp4")}
    assert sorted(pool_clips) == sorted(seq_clips) and len(seq_clips) == 2
    for name, clip in seq_clips.items():
        assert probe_duration(pool_clips[name]) == probe_duration(clip)
        assert _frames(pool_clips[name]) == _frames(clip)
    assert probe_duration(pool.output) == probe_duration(seq.output)

    # Workers left nothing behind: no partials, no private caches, no temp
    # clips in the cache.
    manim = pool_root / "work" / "manim"
    assert not [p for p in (manim / "videos").rglob("*") if p.is_file()]
    assert not (manim / "pool").exists()
    assert not list((pool_root / "cache" / "beats").glob(".tmp-*"))


def test_cache_hits_never_enter_the_pool(both, fake_tts, capfd):
    spec_path, runs = both
    root, _ = runs["pooled"]
    capfd.readouterr()
    again = pipeline.render(spec_path, Quality.DRAFT, root / "again.mp4",
                            cache_dir=root / "cache", work_dir=root / "work", jobs=3)
    assert again.rebuilt == []
    assert "across" not in capfd.readouterr().out  # no pool was started


def _fail_b02(beat, *args, **kwargs):
    """Stands in for worker.render_beat inside the worker processes: b02 fails,
    every other beat renders for real. Module level, so it pickles by
    reference and the workers import it from here."""
    if beat.id == "b02":
        raise RuntimeError("manim exploded in a worker")
    return worker.render_beat(beat, *args, **kwargs)


def _hang_b02(beat, *args, **kwargs):
    if beat.id == "b02":
        raise LatexToolchainError("TeX is slow or hung")
    return worker.render_beat(beat, *args, **kwargs)


@pytest.mark.parametrize("stand_in,code,label,message", [
    (_fail_b02, cli.EXIT_RENDER_FAILED, "render failed",
     "b02 (BulletReveal): manim exploded in a worker"),
    (_hang_b02, cli.EXIT_TOOLCHAIN_FAILED, "toolchain failed",
     "b02 (BulletReveal): TeX is slow or hung"),
], ids=["render", "toolchain"])
def test_failing_beat_in_the_pool_is_typed_as_sequentially(
        stand_in, code, label, message, tmp_path, monkeypatch, capfd, fake_tts):
    monkeypatch.setattr(pipeline, "render_beat", stand_in)
    spec = example_spec()
    spec["beats"] = spec["beats"][:3]
    path = write_spec(tmp_path / "spec.json", spec)

    rc = cli.main(["render", str(path), "--jobs", "3",
                   "--cache-dir", str(tmp_path / "cache"),
                   "--work-dir", str(tmp_path / "work")])

    out, err = capfd.readouterr()
    assert "rendering 3 beat(s) across 3 processes" in out  # really pooled
    assert rc == code
    assert err.startswith(f"chalkdust: {label}: {message}")
    # The beats already running when b02 failed finished and are cached, so a
    # re-run after the fix renders only b02.
    assert len(list((tmp_path / "cache" / "beats").glob("*.mp4"))) == 2
    assert not list((tmp_path / "cache" / "beats").glob(".tmp-*"))


def test_default_jobs_is_one_per_pending_beat_at_most_the_cpus():
    assert default_jobs(0, cpus=24) == 1
    assert default_jobs(1, cpus=24) == 1
    assert default_jobs(5, cpus=24) == 5
    assert default_jobs(40, cpus=24) == 24
    assert default_jobs(5, cpus=1) == 1
