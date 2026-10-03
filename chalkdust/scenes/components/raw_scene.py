"""RawScene: the escape hatch (SCENE_SPEC.md §7).

Some beats need a visual the library does not have. RawScene carries Manim
code -- the one place generated code runs -- under four constraints:

  1. Constrained imports. A static AST pass rejects any import outside
     ALLOWED_MODULES, any reference to a system module name (as a name, an
     attribute, or an import re-exported by an allowed package), and the
     builtins and dunder attributes that reach around an import. At runtime
     the code's namespace gets builtins with those names removed and an
     `__import__` that enforces the same allowlist, including on the names
     a `from` import binds. This keeps an LLM from reaching for
     the filesystem, shell, or network; it is NOT a security boundary against
     a hostile author. The subprocess and the timeout bound the damage.
  2. A subprocess with a timeout. The code renders out of process, so a hang
     or a crash costs one beat, never the pipeline.
  3. The same layout assertions as every component: safe area and legibility
     after every play() and at the end of the scene.
  4. Any failure DEGRADES to BulletReveal with the narration as content
     (D-010). A degraded beat is always better than a failed video.

Every use is appended to a JSONL usage log, `<work_dir>/raw_scene_usage.jsonl`
(USAGE_LOG_NAME): one line per render_raw_beat call, with the rationale, the
outcome, and for a degradation its reason. That log is the roadmap for the
component library -- RawScene firing repeatedly for similar visuals means a
missing component (ROADMAP.md Phase 3).

Timing (D-002): generated code cannot know the beat's audio duration, so its
clip is fitted afterwards -- held on its last frame if it runs short, sped up
uniformly if it runs long.

Because the code runs out of process, a RawScene beat is rendered through
`render_raw_beat`, not by building into a host ChalkdustScene; `build()`
raises to make a wrong route loud.
"""

from __future__ import annotations

import ast
import builtins
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from manim import Scene, ValueTracker, tempconfig
from pydantic import Field

import chalkdust
from chalkdust.core.cache import Cache, content_hash
from chalkdust.core.models import Beat, BeatSpec, BuildContext, Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import Component, ComponentParams, register
from chalkdust.scenes.regions import LayoutError, assert_in_safe_area, assert_legible
from chalkdust.scenes.theme import get_theme, resolve_fonts

# Generous for a draft render of one beat; a hang is the only thing that
# should ever reach it.
TIMEOUT_S = 120.0

USAGE_LOG_NAME = "raw_scene_usage.jsonl"

# Root modules generated code may import. Everything a Manim scene needs.
ALLOWED_MODULES = frozenset({"manim", "numpy", "math"})

# Module names rejected even without an import statement, in case one is
# reachable through an allowed module's namespace.
FORBIDDEN_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "builtins", "ctypes", "multiprocessing", "threading", "signal", "io",
    "tempfile", "pickle", "marshal", "urllib", "http", "requests",
})

# Builtins that execute strings, touch files, or reach around the import
# guard. Rejected statically and absent from the runtime namespace.
FORBIDDEN_NAMES = frozenset({
    "__import__", "eval", "exec", "compile", "open", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "help",
})

# Name the generated code runs as; lets the child find the scene it defined.
_USER_MODULE = "__rawscene__"


class RawSceneError(Exception):
    """A RawScene that cannot be used. `kind` is the degradation reason:

      syntax           -- the code does not parse
      forbidden_import -- imports or names a module outside the allowlist
      forbidden_name   -- uses an escape builtin or a dunder attribute
      no_scene         -- does not define exactly one Scene subclass
      timeout          -- did not finish within the time limit
      crash            -- raised, or the subprocess died
      layout           -- failed a layout assertion

    Plus one routing error that is not a degradation reason:

      out_of_process   -- build() called from a host scene
    """

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class RawSceneParams(ComponentParams):
    # Required and non-empty: the usage log is only a roadmap if every entry
    # says what the library was missing.
    rationale: str = Field(min_length=1)
    code: str = Field(min_length=1)


@register
class RawScene(Component):
    name = "RawScene"
    Params = RawSceneParams

    def regions(self) -> set[Region]:
        # Generated code may draw anywhere inside the safe area.
        return {Region.TITLE_BAR, Region.STAGE, Region.LOWER_THIRD}

    def build(self, scene: ChalkdustScene) -> None:
        raise RawSceneError(
            "RawScene renders out of process; route the beat through "
            "raw_scene.render_raw_beat, not a host ChalkdustScene",
            kind="out_of_process",
        )


# --- static check -----------------------------------------------------------


def check_code(code: str) -> None:
    """Reject code statically. Raises RawSceneError; returns None if clean."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise RawSceneError(f"line {exc.lineno}: {exc.msg}", kind="syntax") from None

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node)
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_MODULES:
                raise RawSceneError(f"line {node.lineno}: reference to module "
                                    f"{node.id!r}", kind="forbidden_import")
            if node.id in FORBIDDEN_NAMES:
                raise RawSceneError(f"line {node.lineno}: use of {node.id!r}",
                                    kind="forbidden_name")
        elif isinstance(node, ast.Attribute):
            # `f.os.system(...)` after `import manim.utils.file_ops as f`: a
            # system module reached as an attribute of an allowed one. The
            # runtime import guard never sees attribute access, so this is
            # the only check covering that route.
            if node.attr in FORBIDDEN_MODULES:
                raise RawSceneError(f"line {node.lineno}: attribute reaching "
                                    f"module {node.attr!r}", kind="forbidden_import")
            if node.attr.startswith("__"):
                raise RawSceneError(f"line {node.lineno}: dunder attribute "
                                    f"{node.attr!r}", kind="forbidden_name")


def _check_import(node: ast.Import | ast.ImportFrom) -> None:
    """The allowlist, plus the routes around it: a system module re-exported
    by an allowed one (`from manim.utils.file_ops import os`), and a star
    import that would bind such re-exports wholesale."""
    if isinstance(node, ast.ImportFrom) and node.level:
        raise RawSceneError(f"line {node.lineno}: relative import",
                            kind="forbidden_import")
    if isinstance(node, ast.Import):
        dotted, names = [alias.name for alias in node.names], []
    else:
        dotted, names = [node.module or ""], [alias.name for alias in node.names]

    for path in dotted:
        root = path.split(".")[0]
        if root not in ALLOWED_MODULES:
            raise RawSceneError(f"line {node.lineno}: import of {root!r}; allowed: "
                                f"{sorted(ALLOWED_MODULES)}", kind="forbidden_import")
    for part in [p for path in dotted for p in path.split(".")[1:]] + names:
        if part in FORBIDDEN_MODULES:
            raise RawSceneError(f"line {node.lineno}: import of module {part!r} "
                                "through an allowed package", kind="forbidden_import")
    # `from manim import *` is how every scene starts; a star from a
    # submodule binds whatever that module happened to import.
    if names == ["*"] and dotted[0] not in ALLOWED_MODULES:
        raise RawSceneError(f"line {node.lineno}: star import from submodule "
                            f"{dotted[0]!r}", kind="forbidden_import")


# --- child process ----------------------------------------------------------
# Everything below up to the parent section runs inside the subprocess.


def _guarded_import(name: str, globals: Any = None, locals: Any = None,
                    fromlist: Any = (), level: int = 0) -> Any:
    """`__import__` for generated code only. Manim's own lazy imports go
    through the real builtins of Manim's modules and are unaffected.

    `from X import y` arrives here with y in `fromlist`, so a module y that
    an allowed X merely re-exports (`from manim.utils.file_ops import os`) is
    refused. `import X.Y as f` then `f.os` is attribute access, which never
    comes through here; check_code's attribute rule covers that route.
    """
    root = name.split(".")[0]
    if level or root not in ALLOWED_MODULES:
        raise RawSceneError(f"runtime import of {name!r}; allowed: "
                            f"{sorted(ALLOWED_MODULES)}", kind="forbidden_import")
    module = builtins.__import__(name, globals, locals, fromlist, level)
    for item in fromlist or ():
        if item == "*":
            # A star import cannot be filtered after the fact. From the
            # top-level packages it is expected (and binds only harmless
            # modules like typing); from a submodule it is refused outright.
            if name not in ALLOWED_MODULES:
                raise RawSceneError(f"runtime star import from submodule {name!r}",
                                    kind="forbidden_import")
            continue
        value = getattr(module, item, None)
        if isinstance(value, ModuleType) and \
                value.__name__.split(".")[0] not in ALLOWED_MODULES:
            raise RawSceneError(f"runtime import of module {value.__name__!r} "
                                f"via {name!r}", kind="forbidden_import")
    return module


def _user_namespace() -> dict[str, Any]:
    """Globals for generated code: builtins minus the escape hatches, with the
    allowlisting import. Functions the code defines keep these globals, so
    imports inside construct() are guarded too."""
    safe = {k: v for k, v in vars(builtins).items() if k not in FORBIDDEN_NAMES}
    safe["__import__"] = _guarded_import
    return {"__name__": _USER_MODULE, "__builtins__": safe}


def _settle(scene: Scene, label: str) -> None:
    """The settle-point assertions ChalkdustScene applies to components.

    ValueTrackers are skipped: they are invisible mobjects that store their
    value as a point, so a tracker at 20 would read as off-screen.
    """
    for mob in scene.mobjects:
        if isinstance(mob, ValueTracker):
            continue
        name = type(mob).__name__
        assert_in_safe_area(mob, label=f"{label}: {name}")
        assert_legible(mob, label=f"{label}: {name}")


def _checked(user_scene: type[Scene]) -> type[Scene]:
    """The generated scene, with layout assertions after every play() and at
    the end. Subclassing keeps the generated code itself untouched."""

    def play(self, *args, **kwargs):
        user_scene.play(self, *args, **kwargs)
        _settle(self, f"after play {self.renderer.num_plays}")

    def construct(self):
        user_scene.construct(self)
        _settle(self, "end of scene")

    return type("CheckedRawScene", (user_scene,), {"play": play, "construct": construct})


def _execute(job: dict[str, Any]) -> Path:
    """Run the generated code and render it. Returns the produced movie."""
    ns = _user_namespace()
    exec(compile(job["code"], "<RawScene>", "exec"), ns)  # noqa: S102 -- the point
    scenes = [v for v in ns.values()
              if isinstance(v, type) and issubclass(v, Scene)
              and v.__module__ == _USER_MODULE]
    if len(scenes) != 1:
        raise RawSceneError(f"expected one Scene subclass, found {len(scenes)}",
                            kind="no_scene")

    settings = {
        "pixel_width": job["pixel_width"],
        "pixel_height": job["pixel_height"],
        "frame_rate": job["frame_rate"],
        "background_color": job["background"],
        "media_dir": job["media_dir"],
        "output_file": "raw",
        "disable_caching": True,
        "verbosity": "WARNING",
        "progress_bar": "none",
    }
    with tempconfig(settings):
        scene = _checked(scenes[0])()
        scene.render()
        movie = Path(scene.renderer.file_writer.movie_file_path)
    if not movie.exists():
        raise RawSceneError("scene produced no movie (did it play anything?)",
                            kind="crash")
    return movie


def _child_main(job_path: str) -> None:
    """Subprocess entry point. Always writes a result file it can; a missing
    result file means the process died, which the parent reports as crash."""
    job = json.loads(Path(job_path).read_text())
    try:
        result = {"ok": True, "movie": str(_execute(job))}
    except RawSceneError as exc:
        result = {"ok": False, "kind": exc.kind, "message": str(exc)}
    except LayoutError as exc:
        result = {"ok": False, "kind": "layout", "message": f"[{exc.kind}] {exc}"}
    except Exception as exc:  # generated code may raise anything
        result = {"ok": False, "kind": "crash",
                  "message": f"{type(exc).__name__}: {exc}"}
    Path(job["result"]).write_text(json.dumps(result))


# --- parent process ---------------------------------------------------------

_CHILD_BOOT = ("import sys; from chalkdust.scenes.components.raw_scene "
               "import _child_main; _child_main(sys.argv[1])")


@dataclass(frozen=True)
class RawOutcome:
    ok: bool
    kind: str | None = None  # RawSceneError kind when not ok
    message: str = ""


def run_raw_scene(params: RawSceneParams, duration: float, quality: dict[str, int],
                  theme: str, out_path: Path, work_dir: Path,
                  timeout: float = TIMEOUT_S) -> RawOutcome:
    """Render generated code to `out_path`, fitted to `duration`. Never raises
    for anything the code does; failures come back as an outcome.

    `quality` is a worker.TIERS entry as a dict: pixel_width, pixel_height,
    frame_rate.
    """
    try:
        check_code(params.code)
    except RawSceneError as exc:
        return RawOutcome(False, exc.kind, str(exc))

    work_dir.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix="rawscene-", dir=work_dir))
    try:
        job = {
            "code": params.code,
            **quality,
            "background": get_theme(theme).palette.bg,
            "media_dir": str(job_dir / "media"),
            "result": str(job_dir / "result.json"),
        }
        job_path = job_dir / "job.json"
        job_path.write_text(json.dumps(job))

        # The child must import the same chalkdust as this process, whatever
        # its working directory or editable install points at.
        env = dict(os.environ)
        pkg_root = str(Path(chalkdust.__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (pkg_root, env.get("PYTHONPATH")) if p)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _CHILD_BOOT, str(job_path)],
                cwd=job_dir, env=env, capture_output=True, text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return RawOutcome(False, "timeout", f"exceeded {timeout:g}s")

        result_path = Path(job["result"])
        if not result_path.exists():
            tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
            return RawOutcome(False, "crash", f"exit {proc.returncode}: {tail}")
        result = json.loads(result_path.read_text())
        if not result["ok"]:
            return RawOutcome(False, result["kind"], result["message"])

        _fit_to_duration(Path(result["movie"]), out_path, duration,
                         quality["frame_rate"])
        return RawOutcome(True)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def _fit_to_duration(src: Path, dst: Path, duration: float, fps: int) -> None:
    """Make the clip exactly `duration` long (D-002): hold the last frame if
    short, speed up uniformly if long."""
    from chalkdust.render.assemble import VCODEC
    from chalkdust.speech.base import probe_duration, require, run

    require("ffmpeg")
    natural = probe_duration(src)
    if natural <= duration:
        vf = f"tpad=stop_mode=clone:stop_duration={duration - natural:.6f}"
    else:
        # Scale frame START times so the source's last frame starts exactly
        # one frame before the end: the clip must finish on the settled state
        # the layout checks passed. Scaling by duration/natural can push that
        # frame past the cut.
        frame = 1 / fps
        vf = f"setpts=PTS*{(duration - frame) / (natural - frame):.6f}"
    # The fps filter, not -r: output-side -r resampling dropped the final
    # frame of a sped-up clip (seen at 3 s -> 1.8 s, 15 fps).
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vf", f"{vf},fps={fps}", "-t", f"{duration:.6f}", "-an",
         *VCODEC, str(dst)])


def degrade_spec(spec: BeatSpec) -> BeatSpec:
    """The BulletReveal that replaces a failed RawScene: the narration, one
    sentence per bullet, overflow folded into the last of six. Continuity
    fields are dropped -- BulletReveal neither uses nor rebuilds artifacts."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", spec.narration) if s]
    if len(sentences) > 6:
        sentences = sentences[:5] + [" ".join(sentences[5:])]
    return BeatSpec(id=spec.id, narration=spec.narration, component="BulletReveal",
                    params={"items": sentences}, transition=spec.transition)


def _log_use(log_path: Path, spec: BeatSpec, params: RawSceneParams,
             outcome: RawOutcome, cached: bool) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "beat_id": spec.id,
        "rationale": params.rationale,
        "code_hash": content_hash(params.code),
        "outcome": "rendered" if outcome.ok else "degraded",
        "reason": outcome.kind,
        "detail": outcome.message[:500],
        "cached": cached,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def render_raw_beat(beat: Beat, theme: str, ctx: BuildContext, cache: Cache,
                    work_dir: Path, timeout: float = TIMEOUT_S) -> Path:
    """render_beat for a RawScene beat: render out of process, or degrade.

    A successful render is cached under the beat's normal render key. A
    failure is not cached -- a timeout or crash may be environmental -- so it
    re-runs next time; the degraded BulletReveal is cached under its own key.
    Sets beat.degraded on fallback, and logs every call.
    """
    from chalkdust.render.worker import TIERS, render_beat, render_key  # worker imports us
    from chalkdust.validate.repair import RepairPlan

    if beat.duration is None:
        raise ValueError(f"{beat.id} has no duration; the speech stage must run first")
    params = RawSceneParams.model_validate(beat.spec.params)

    # The worker's key: font-resolved theme, tier settings, repair plan. A
    # RawScene is never mechanically repaired (it renders out of process), so
    # its plan is the empty one.
    key = render_key(beat, resolve_fonts(get_theme(theme)), ctx, RepairPlan())
    slot = cache.slot("beats", key, ".mp4")
    cached = slot.exists
    if cached:
        outcome = RawOutcome(True)
    else:
        outcome = run_raw_scene(params, beat.duration, asdict(TIERS[ctx.quality]),
                                theme, slot.tmp, work_dir, timeout)
        if outcome.ok:
            slot.commit()
    _log_use(work_dir / USAGE_LOG_NAME, beat.spec, params, outcome, cached)

    if outcome.ok:
        beat.render_path = slot.path
        return slot.path

    fallback = Beat(spec=degrade_spec(beat.spec), duration=beat.duration)
    path = render_beat(fallback, theme, ctx, cache, work_dir)
    beat.render_path = path
    beat.degraded = True
    return path
