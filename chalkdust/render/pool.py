"""Render beats across a process pool (ARCHITECTURE.md §4).

Beats are independent by construction (D-005, SCENE_SPEC.md §2): a beat's clip
depends only on its own spec, its measured duration and the recipes of what it
carries in. So the render stage can hand each beat to its own process and
concatenate afterwards; nothing about assembly changes.

What crosses the process boundary, and what does not:

  - Manim's `config` is a process global. Threads would share it; processes
    each get their own. Every worker process sets its own config for each
    beat it renders, inside `scratch()` (pipeline.manim_scratch: media dir,
    verbosity) and the worker's own tempconfig (tier, per-beat video dir).
    Nothing in this process's config reaches a worker, and nothing a worker
    sets comes back.
  - Workers are started with "spawn" on every OS, not "fork": a forked child
    would inherit this process's Manim config, open cairo/pyav state and any
    threads mid-flight. Spawn gives every platform the same, clean start.
  - Manim's own text and LaTeX caches (`{media_dir}/texts`, `{media_dir}/Tex`)
    are NOT shared between processes. Manim writes them in place, not
    atomically: measured here, four processes building the same Text at once
    left empty SVGs in `texts/`, which every later run -- sequential too --
    then fails to parse ("cannot be shaped at all"). So each worker process
    gets its own `text_dir` and `tex_dir` under `{media_dir}/pool/`, seeded
    once at process start from the shared ones (warmed by validation and by
    the caller's cache check, which builds every beat at its render tier),
    and removed when the pool is done. Everything else under the media dir is
    already safe to share: each beat renders into its own video dir, named
    by its render key (worker._render), and theme.check_latex writes its
    records atomically, compiling in per-process scratch dirs.

What stays in the caller (pipeline.render_stage_pooled): measuring which
beats are already cached -- a hit never enters the pool -- and choosing what
may be pooled at all. Two processes must never write one cache slot (or one
Manim video dir, named by the same key), so the pool is handed only beats
whose slot is known before they render, one beat per slot: library beats,
deduplicated by key. A RawScene beat's slot is not known up front (one that
degrades commits its fallback under a key that exists only inside the
render), so the caller renders those itself, sequentially, after the pool.
Each pooled beat commits to the cache exactly as a sequential render does:
the worker writes `slot.tmp`, commits, and removes its partials.

Errors are not pickled across the boundary as exceptions: a typed exception
whose constructor takes more than a message does not survive unpickling, and
a failure that cannot be reported is worse than the failure. The worker
returns the failure's kind ("toolchain" for theme.LatexToolchainError,
"render" for anything else), its message and its traceback; the caller turns
that into the same typed error a sequential render raises.

A worker process that dies (a crash in native code, out of memory, killed
from outside) reports nothing. The executor then fails every unfinished beat
with BrokenProcessPool and stops the other workers, so which beat killed it
cannot be told from the results: the dead one and the ones stopped with it
look the same. Each worker therefore marks a beat as started before rendering
it, and render_pool raises WorkerDied naming every beat that had started and
not finished -- the crashed one is among them -- rather than blaming one.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import shutil
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

from manim import tempconfig

from chalkdust.continuity import ArtifactRecipe
from chalkdust.core.models import Beat
from chalkdust.scenes.theme import LatexToolchainError

# Called as render(beat, recipes=recipes): worker.render_beat with every
# argument the beats share (theme, ctx, cache, work dir) bound by keyword in a
# functools.partial. Must be picklable by reference -- a module-level
# function, or a partial of one -- to reach a worker process: see can_ship.
RenderFn = Callable[..., object]
ScratchFn = Callable[[], AbstractContextManager[object]]

# ProcessPoolExecutor's limit on Windows (WaitForMultipleObjects).
MAX_WORKERS_WINDOWS = 61

# Manim's per-content caches each worker process keeps to itself (see the
# module docstring): config key -> subdirectory of the shared media dir.
PRIVATE_CACHES = {"text_dir": "texts", "tex_dir": "Tex"}

# Set once in each worker process by _start_worker: its own text_dir and
# tex_dir. A per-process global, never shared: it lives in the worker.
_own_dirs: dict[str, str] = {}

# Where render_one marks a beat as started (one empty file per beat index),
# so the caller can say which beats were in flight when a worker died. Set
# per worker process by _start_worker; in {media_dir}/pool/, which the caller
# reads and removes.
STARTED = "started"
_started_dir: list[Path] = []


def available_cpus() -> int:
    """CPUs this process may run on (affinity-aware where Python knows how)."""
    count = getattr(os, "process_cpu_count", os.cpu_count)()
    return max(1, count or 1)


# At draft (480p15), the fewest beats to render worth starting a pool for.
# Measured on a quiet 24-core Windows machine (cold caches, 2 runs each; see
# default_jobs): with 2 beats the pool lost ~1 s every time, with 3 it was a
# wash, with 4 it lost 0.7 s on binary_search and won ~1 s on others, and from
# 5 it won every time (1.5-2 s at 5, 7 s at 9-10).
DRAFT_MIN_POOLED = 5


def default_jobs(pending: int, cpus: int | None = None, draft: bool = False) -> int:
    """How many processes to render `pending` beats with, when not told.

    One per beat that needs rendering, never more than the CPUs available:
    a Manim render is one busy core (cairo rasterises single-threaded) plus
    the encoder, so beats beyond the core count only queue. Never more than
    `pending` either: an idle worker still pays its spawn and Manim's import.
    At most one pending beat means no pool at all, and so does a `draft`
    render of fewer than DRAFT_MIN_POOLED beats.

    Measured on a 24-core Windows machine (8 P + 16 E cores), quiet (no other
    render running), cold render and Manim caches, speech held constant, two
    runs each. Render stage, --jobs 1 -> this default:
      - 1080p60: 1.57x for 4 beats (25.8 s -> 16.5 s), 1.67x for 5, 1.8-2.0x
        for 9-10 (e.g. 124.6 s -> 64.1 s); whole command 1.27-1.47x. Well
        short of the beat count: the longest beat bounds it, and concurrent
        renders contend (not profiled further).
      - 480p15: a beat renders in one to two seconds, so the pool's fixed
        cost (spawning workers, each importing Manim) is the whole story for
        small specs: 2 beats +0.9-1.1 s slower, 3 beats -0.2..+0.2 s, 4 beats
        +0.7 s (binary_search) or -1.0 s (others), 5 beats -1.5..-2.0 s, 9-10
        beats -6.6..-7.7 s (1.5-1.6x). Hence DRAFT_MIN_POOLED.
    `--jobs N` is taken as given; `--jobs 1` keeps the sequential path.
    """
    cpus = available_cpus() if cpus is None else max(1, cpus)
    if draft and pending < DRAFT_MIN_POOLED:
        return 1
    limit = MAX_WORKERS_WINDOWS if os.name == "nt" else cpus
    return max(1, min(pending, cpus, limit))


def can_ship(fn: object) -> bool:
    """Whether `fn` can be sent to a worker process (pickled by reference).

    A module-level function, or a functools.partial of one over picklable
    arguments, can. A closure or lambda cannot -- that is what a test double
    patched over pipeline.render_beat usually is -- and then the caller
    renders in this process instead, through the same function.
    """
    try:
        pickle.dumps(fn)
    except (pickle.PicklingError, AttributeError, TypeError):
        return False
    return True


@dataclass(frozen=True)
class BeatDone:
    """What a worker process sends back for one beat. Plain data only."""

    index: int
    render_path: Path | None = None
    degraded: bool = False
    failure: str | None = None  # None, "render" or "toolchain"
    message: str = ""
    trace: str = ""


class WorkerDied(Exception):
    """A worker process died, so the pool broke and no beat it was running
    reported back. Not attributed to one beat: `in_flight` holds the indices
    of every beat that had started and not finished (the dead process was
    rendering one of them; the rest were stopped with it), in beat order, and
    `unfinished` every beat the pool did not finish, started or not."""

    def __init__(self, in_flight: Sequence[int], unfinished: Sequence[int],
                 message: str) -> None:
        super().__init__(message)
        self.in_flight = tuple(in_flight)
        self.unfinished = tuple(unfinished)
        self.message = message


class BeatFailed(Exception):
    """A pooled beat failed. `kind` is "render" or "toolchain"; the caller
    raises its own typed error from it (pipeline.RenderFailed/ToolchainFailed),
    with the same message a sequential render gives."""

    def __init__(self, index: int, kind: str, message: str, trace: str = "") -> None:
        super().__init__(message)
        self.index = index
        self.kind = kind
        self.message = message
        self.trace = trace


def _start_worker(media_dir: Path, pool_dir: Path) -> None:
    """Process initializer: give this process its own text and LaTeX caches,
    seeded from the shared ones. Empty files are skipped -- an empty SVG is
    exactly what a torn write leaves, and Manim would trust it."""
    own = pool_dir / f"worker-{os.getpid()}"
    for key, name in PRIVATE_CACHES.items():
        target = own / name
        target.mkdir(parents=True, exist_ok=True)
        source = media_dir / name
        if source.is_dir():
            for f in source.iterdir():
                if f.is_file() and f.stat().st_size > 0:
                    shutil.copyfile(f, target / f.name)
        _own_dirs[key] = str(target)
    started = pool_dir / STARTED
    started.mkdir(parents=True, exist_ok=True)
    _started_dir[:] = [started]


def _nothing() -> None:
    """The job render_pool submits last, only for its wake-up (see there)."""


def render_one(index: int, beat: Beat, recipes: Sequence[ArtifactRecipe],
               render: RenderFn, scratch: ScratchFn) -> BeatDone:
    """Runs in a worker process: render one beat under this process's own
    Manim config -- the caller's scratch settings, then this process's own
    text and LaTeX dirs, which the worker's media_dir does not override (they
    are absolute) -- and report the result as plain data. The beat is marked
    as started first, so a process that dies mid-render is traced to the
    beats that were running (see WorkerDied)."""
    for started in _started_dir:
        (started / str(index)).touch()
    try:
        with scratch(), tempconfig(dict(_own_dirs)):
            render(beat, recipes=recipes)
    except LatexToolchainError as exc:
        return BeatDone(index, failure="toolchain", message=str(exc),
                        trace=traceback.format_exc())
    except Exception as exc:  # reported, not raised: see the module docstring
        return BeatDone(index, failure="render", message=str(exc),
                        trace=traceback.format_exc())
    return BeatDone(index, beat.render_path, beat.degraded)


def render_pool(work: Sequence[tuple[int, Beat, Sequence[ArtifactRecipe]]],
                render: RenderFn, scratch: ScratchFn, media_dir: Path,
                workers: int, on_done: Callable[[int], None]) -> None:
    """Render every (index, beat, recipes) in `work` across `workers` spawned
    processes, setting each beat's render_path and degraded as the worker did.
    `media_dir` is the shared Manim media dir the worker renders under; each
    process's private caches live in `{media_dir}/pool/` while it runs.

    `on_done(index)` is called in this process once that beat's result is in,
    in completion order; the caller reports in beat order. On a failure, beats
    not yet started are cancelled, those already running are let finish (their
    clips commit to the cache, so a re-run reuses them), and BeatFailed is
    raised for the failed beat with the lowest index -- the one a sequential
    render would have stopped at, among those that ran.

    If a worker process died, its beat sent no report and the pool stopped
    every other running beat with it, so no single beat can be blamed:
    WorkerDied is raised naming every beat in flight. A beat that reported its
    own failure is still raised as BeatFailed first -- that attribution is
    certain.
    """
    beats = {index: beat for index, beat, _ in work}
    failures: list[BeatFailed] = []
    died: list[int] = []  # beats lost to a dead worker (BrokenProcessPool)
    broken = ""
    context = multiprocessing.get_context("spawn")
    workers = max(1, min(workers, len(work)))
    if os.name == "nt":
        workers = min(workers, MAX_WORKERS_WINDOWS)
    pool_dir = media_dir / "pool"
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                   initializer=_start_worker,
                                   initargs=(media_dir, pool_dir))
    try:
        running: dict[Future[BeatDone], int] = {
            executor.submit(render_one, index, beat, tuple(recipes), render, scratch): index
            for index, beat, recipes in work
        }
        # ProcessPoolExecutor.submit (CPython 3.13) wakes the executor's
        # manager thread *before* it spawns the worker the job may need, so
        # the manager can go back to waiting on the processes it knew of,
        # without the newest one. If that one then dies, nothing notices until
        # another beat reports: measured here, a worker that died at once went
        # unseen for two minutes while the others rendered. One more submit,
        # once every worker exists, wakes the manager to watch them all. The
        # job does nothing and its result is not waited on.
        executor.submit(_nothing)
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                index = running.pop(future)
                try:
                    done = future.result()
                except BrokenProcessPool as exc:
                    # A process died (crash, out of memory, killed): no report
                    # came back, and every unfinished beat fails the same way,
                    # the dead process's and the ones stopped with it alike.
                    died.append(index)
                    broken = str(exc)
                    continue
                except Exception as exc:  # e.g. the job could not be sent
                    done = BeatDone(index, failure="render",
                                    message=f"{type(exc).__name__}: {exc}",
                                    trace=traceback.format_exc())
                if done.failure:
                    if not failures:
                        # Stop starting new beats; let running ones finish.
                        for pending in running:
                            pending.cancel()
                    failures.append(BeatFailed(index, done.failure, done.message,
                                               done.trace))
                    continue
                beats[index].render_path = done.render_path
                beats[index].degraded = done.degraded
                on_done(index)
            running = {f: i for f, i in running.items() if not f.cancelled()}
        if died and not failures:
            raise _worker_died(pool_dir / STARTED, sorted(died), broken)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(pool_dir, ignore_errors=True)
    if failures:
        raise min(failures, key=lambda f: f.index)


def _worker_died(started_dir: Path, unfinished: list[int], reason: str) -> WorkerDied:
    """The pool broke: name the beats that had started and not finished -- the
    one whose process died is among them -- rather than any single beat."""
    marked = ({int(p.name) for p in started_dir.iterdir() if p.name.isdigit()}
              if started_dir.is_dir() else set())
    in_flight = [i for i in unfinished if i in marked]
    return WorkerDied(in_flight, unfinished, reason)
