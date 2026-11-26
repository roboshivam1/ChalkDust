"""RawScene: the escape hatch (SCENE_SPEC.md §7).

Some beats need a visual the library does not have. RawScene carries Manim
code -- the one place generated code runs -- under four constraints:

  1. Constrained imports: an ALLOWLIST, enforced twice.
     Statically (check_code), every import must be one of the vetted paths
     in raw_scene_allowlist (`from manim import <name>|*`, math, a numpy
     subset), every name the code reads must be one it binds itself, a safe
     builtin, or an allowlisted name, and private/dunder attributes plus a
     short list of attributes that reach files, processes or interpreter
     frames are rejected.
     At runtime the code executes in a namespace BUILT from the allowlist --
     copies of the vetted values in plain namespace objects, never a module
     object -- with a restricted __builtins__ that has no __import__, open,
     exec, eval, compile, getattr and the like. Import statements are
     rewritten into lookups in that vetted table, so there is no import
     machinery for the code to reach. Either layer alone refuses the known
     escapes (manim.utils.commands.capture, manim.utils.file_ops.*).
     This raises the bar well above an LLM's reach; it is still not an OS
     boundary against a hostile author. The subprocess and the timeout bound
     the damage.
  2. A subprocess with a timeout. The code renders out of process, so a hang
     or a crash costs one beat, never the pipeline.
  3. The same layout assertions as every component: safe area and legibility
     after every play() and at the end of the scene.
  4. Any failure DEGRADES to BulletReveal with the narration as content
     (D-010). A degraded beat is always better than a failed video.

Every use is appended to a JSONL usage log, USAGE_LOG_NAME in the work_dir
handed to render_raw_beat. The pipeline hands it the Manim dir, so on disk the
log is `<--work-dir>/manim/raw_scene_usage.jsonl` (`work/manim/...` by
default): one line per render_raw_beat call, with the rationale, the outcome,
and for a degradation its reason. That log is the roadmap for the
component library -- RawScene firing repeatedly for similar visuals means a
missing component (ROADMAP.md Phase 3).

Timing (D-002): generated code cannot know the beat's audio duration, so its
clip is fitted afterwards -- held on its last frame if it runs short, sped up
uniformly if it runs long.

Because the code runs out of process, a RawScene beat is rendered through
`render_raw_beat`, not by building into a host ChalkdustScene; `build()`
raises to make a wrong route loud. The render worker routes RawScene beats
here before any repair or scene build.
"""

from __future__ import annotations

import ast
import builtins
import json
import os
import re
import shutil
import string
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import manim
from manim import Scene, ValueTracker, tempconfig
from pydantic import Field, ValidationError

import chalkdust
from chalkdust.core.cache import Cache, content_hash
from chalkdust.core.models import Beat, BeatSpec, BuildContext, Region
from chalkdust.scenes.base import ChalkdustScene
from chalkdust.scenes.components.base import Component, ComponentParams, register
from chalkdust.scenes.components.raw_scene_allowlist import (
    MANIM,
    importable_names,
    vetted_modules,
)
from chalkdust.scenes.regions import LayoutError, assert_in_safe_area, assert_legible
from chalkdust.scenes.theme import get_theme, resolve_fonts, restricted_tex_env

# Generous for a draft render of one beat; a hang is the only thing that
# should ever reach it.
TIMEOUT_S = 120.0

USAGE_LOG_NAME = "raw_scene_usage.jsonl"

# Builtins generated code may use: constructors, iteration, arithmetic,
# exceptions. An allowlist, so a new escape builtin in a future Python is
# absent by default. Missing on purpose: __import__, open, exec, eval,
# compile, getattr/setattr/delattr/hasattr, globals/locals/vars/dir, type,
# input, breakpoint, help, exit/quit, memoryview.
SAFE_BUILTINS = frozenset("""
abs all any bin bool callable chr classmethod complex dict divmod enumerate
filter float format frozenset hash hex id int isinstance issubclass iter len
list map max min next object oct ord pow print property range repr reversed
round set slice sorted staticmethod str sum super tuple zip
True False None NotImplemented Ellipsis
ArithmeticError AssertionError AttributeError Exception IndexError KeyError
LookupError NotImplementedError OverflowError RuntimeError StopIteration
TypeError ValueError ZeroDivisionError
""".split())

# Attributes rejected by the static check even on allowlisted objects:
# interpreter frames and code (the route from any generator or traceback back
# to a module's globals), and methods that write files, open viewers, start a
# render or swap the LaTeX compiler command. Private and dunder attributes are
# rejected wholesale on top of these.
FORBIDDEN_ATTRIBUTES = frozenset("""
gi_frame gi_code gi_yieldfrom cr_frame cr_code cr_await ag_frame ag_code
ag_await f_back f_builtins f_code f_globals f_locals tb_frame tb_next
func_globals func_code
file_writer render embed interactive_embed show get_image save_image save
tofile dump ctypes tex_template tex_compiler
""".split())

# System modules that must not appear even as an attribute name: nothing
# allowlisted exposes them, so their name in an attribute chain is an attempt.
SYSTEM_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "builtins", "ctypes", "multiprocessing", "threading", "signal", "io",
    "tempfile", "pickle", "marshal", "urllib", "http", "requests", "commands",
    "file_ops",
})

# Names that refer to modules, for the error kind only: reading one of these
# unbound is an attempt to reach a module (forbidden_import), not just an
# unknown name (forbidden_name).
_MODULE_NAMES = (frozenset(sys.stdlib_module_names) | SYSTEM_MODULES
                 | {"manim", "numpy", "utils"}
                 | {k for k, v in vars(manim).items() if isinstance(v, ModuleType)})

# Name the generated code runs as; lets the child find the scene it defined.
_USER_MODULE = "__rawscene__"

# Global the rewritten import statements call. A dunder, so the static check
# refuses it in the code itself; it only ever returns vetted values anyway.
_IMPORT_HOOK = "__rawscene_import__"


class RawSceneError(Exception):
    """A RawScene that cannot be used. `kind` is the degradation reason:

      params           -- params do not validate (no rationale, no code)
      carry_in         -- the beat carries artifacts in; out-of-process code
                          cannot draw them
      syntax           -- the code does not parse
      forbidden_import -- imports or names a module or name outside the
                          allowlist
      forbidden_name   -- uses a name outside the allowlist, an escape
                          builtin, or a private/dunder/forbidden attribute
      no_scene         -- does not define exactly one Scene subclass
      timeout          -- did not finish within the time limit
      crash            -- raised, the subprocess died, or its result was
                          unreadable
      layout           -- failed a layout assertion
      fit              -- the clip could not be fitted to the beat's duration

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

    # Exempt from the component snapshot harness (SCENE_SPEC.md §11 rule 6),
    # read by tests/test_snapshots.py. A snapshot pins what a component builds
    # from fixed examples; RawScene builds nothing of its own -- each beat
    # carries different generated code, rendered out of process. What Manim
    # can shift underneath it is the API surface that code is given, and that
    # is pinned instead by the allowlist drift guard (vetted_modules() raises
    # on a missing name; tests/test_raw_scene.py TestAllowlist).
    snapshot_exempt = ("renders generated code out of process; no fixed visual "
                       "to snapshot. Drift guard: raw_scene_allowlist.")

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
    """Reject code statically. Raises RawSceneError; returns None if clean.

    Every import must be a vetted path and name; every name read must be bound
    by the code itself, a safe builtin, or on the allowlist. Binding is
    collected for the whole module, not per scope -- a name bound in one
    function and read unbound in another passes here and fails at runtime
    with NameError, because the runtime namespace holds only the allowlist.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise RawSceneError(f"line {exc.lineno}: {exc.msg}", kind="syntax") from None

    visible = _bound_names(tree) | SAFE_BUILTINS | MANIM
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node)
        elif isinstance(node, ast.Name):
            _check_name(node, visible)
        elif isinstance(node, ast.Attribute):
            _check_attribute(node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            _check_format_fields(node)


def _bound_names(tree: ast.AST) -> set[str]:
    """Every name the code binds anywhere: assignments, defs, arguments,
    imports, except/match targets. Star imports bind their module's names."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        elif isinstance(node, ast.Import):
            bound.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    if not node.level and node.module in vetted_modules():
                        bound.update(importable_names(node.module))
                else:
                    bound.add(a.asname or a.name)
    return bound


def _check_import(node: ast.Import | ast.ImportFrom) -> None:
    vetted = vetted_modules()
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name not in vetted:
                raise RawSceneError(
                    f"line {node.lineno}: import of {alias.name!r}; importable: "
                    f"{sorted(vetted)}", kind="forbidden_import")
            root, *rest = alias.name.split(".")
            # `import numpy.linalg` binds `numpy`; the dotted path must then
            # be reachable through the vetted root.
            if not alias.asname and rest and not _reachable(vetted, root, rest):
                raise RawSceneError(
                    f"line {node.lineno}: import {alias.name} without `as`; use "
                    f"`from {alias.name} import ...`", kind="forbidden_import")
        return

    if node.level:
        raise RawSceneError(f"line {node.lineno}: relative import",
                            kind="forbidden_import")
    if node.module == "__future__":
        return
    if node.module not in vetted:
        raise RawSceneError(f"line {node.lineno}: import from {node.module!r}; "
                            f"importable: {sorted(vetted)}", kind="forbidden_import")
    allowed = importable_names(node.module)
    for alias in node.names:
        if alias.name != "*" and alias.name not in allowed:
            raise RawSceneError(f"line {node.lineno}: {alias.name!r} is not on the "
                                f"RawScene allowlist for {node.module!r}",
                                kind="forbidden_import")


def _reachable(vetted: dict[str, Any], root: str, rest: list[str]) -> bool:
    obj = vetted.get(root)
    for part in rest:
        obj = vars(obj).get(part) if obj is not None else None
    return obj is not None


def _check_name(node: ast.Name, visible: set[str]) -> None:
    if node.id.startswith("__"):
        raise RawSceneError(f"line {node.lineno}: dunder name {node.id!r}",
                            kind="forbidden_name")
    if isinstance(node.ctx, ast.Load) and node.id not in visible:
        kind = "forbidden_import" if node.id in _MODULE_NAMES else "forbidden_name"
        raise RawSceneError(f"line {node.lineno}: {node.id!r} is not on the RawScene "
                            "allowlist and is not defined by the code", kind=kind)


def _check_attribute(node: ast.Attribute) -> None:
    if node.attr in SYSTEM_MODULES:
        raise RawSceneError(f"line {node.lineno}: attribute reaching module "
                            f"{node.attr!r}", kind="forbidden_import")
    if node.attr.startswith("_"):
        raise RawSceneError(f"line {node.lineno}: private or dunder attribute "
                            f"{node.attr!r}", kind="forbidden_name")
    if node.attr in FORBIDDEN_ATTRIBUTES:
        raise RawSceneError(f"line {node.lineno}: attribute {node.attr!r} reaches "
                            "files, processes or interpreter internals",
                            kind="forbidden_name")


def _check_format_fields(node: ast.Constant) -> None:
    """`"{0.__class__}".format(x)` reads attributes the AST never shows. A
    replacement field that walks into a private attribute is refused in any
    string constant (LaTeX braces like `{x}` never contain `._`)."""
    try:
        fields = [f for _, f, _, _ in string.Formatter().parse(node.value) if f]
    except ValueError:
        return  # not a format string
    for field in fields:
        if "._" in field or field.startswith("_"):
            raise RawSceneError(f"line {node.lineno}: format field {field!r} reads a "
                                "private attribute", kind="forbidden_name")


# --- child process ----------------------------------------------------------
# Everything below up to the parent section runs inside the subprocess.


class _RewriteImports(ast.NodeTransformer):
    """Turn every import statement into a call to the vetted-import hook.

    The runtime namespace has no __import__, so an import statement left in
    place could not run; rewritten, it can only ever bind a vetted value.
    `from __future__` lines are compile-time directives and are dropped.
    """

    def visit_Import(self, node: ast.Import) -> list[ast.stmt]:
        return [
            ast.copy_location(ast.Assign(
                targets=[ast.Name(alias.asname or alias.name.split(".")[0], ast.Store())],
                value=_hook_call(alias.name, None, root=alias.asname is None),
            ), node)
            for alias in node.names
        ]

    def visit_ImportFrom(self, node: ast.ImportFrom) -> list[ast.stmt]:
        if node.module == "__future__" and not node.level:
            return [ast.copy_location(ast.Pass(), node)]
        path = "." * node.level + (node.module or "")
        out: list[ast.stmt] = []
        for alias in node.names:
            call = _hook_call(path, alias.name)
            stmt: ast.stmt = (ast.Expr(call) if alias.name == "*" else ast.Assign(
                targets=[ast.Name(alias.asname or alias.name, ast.Store())], value=call))
            out.append(ast.copy_location(stmt, node))
        return out


def _hook_call(path: str, name: str | None, root: bool = False) -> ast.Call:
    return ast.Call(func=ast.Name(_IMPORT_HOOK, ast.Load()),
                    args=[ast.Constant(path), ast.Constant(name), ast.Constant(root)],
                    keywords=[])


def _safe_builtins() -> dict[str, Any]:
    safe = {k: getattr(builtins, k) for k in SAFE_BUILTINS}
    # The class statement calls this implicitly; code cannot name it (dunder).
    safe["__build_class__"] = builtins.__build_class__
    return safe


def _user_namespace() -> dict[str, Any]:
    """Globals for generated code, built from the allowlist: every
    `from manim import *` name pre-bound (np and rate_functions are vetted
    namespaces, not modules), the import hook, and the restricted builtins.
    Functions the code defines keep these globals, so code Manim calls back
    later (updaters, construct) runs under them too."""
    vetted = vetted_modules()
    ns: dict[str, Any] = dict(vars(vetted["manim"]))

    def vetted_import(path: str, name: str | None, root: bool = False) -> Any:
        if path not in vetted:
            raise RawSceneError(f"runtime import of {path!r}; importable: "
                                f"{sorted(vetted)}", kind="forbidden_import")
        module = vetted[path]
        if name is None:
            return vetted[path.split(".")[0]] if root else module
        if name == "*":
            ns.update(vars(module))
            return None
        if name not in vars(module):
            raise RawSceneError(f"runtime import of {name!r} from {path!r}: not on "
                                "the RawScene allowlist", kind="forbidden_import")
        return vars(module)[name]

    ns.update({"__name__": _USER_MODULE, "__builtins__": _safe_builtins(),
               _IMPORT_HOOK: vetted_import})
    return ns


def _exec_user(code: str, ns: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run generated code in the allowlist namespace. Independent of
    check_code: this is the runtime half of the allowlist."""
    ns = _user_namespace() if ns is None else ns
    tree = ast.fix_missing_locations(_RewriteImports().visit(ast.parse(code)))
    exec(compile(tree, "<RawScene>", "exec"), ns)  # noqa: S102 -- the point
    return ns


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
    check_code(job["code"])  # again, in the process that runs it
    ns = _exec_user(job["code"])
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
    """Render generated code to `out_path`, fitted to `duration`. Never
    raises: every failure -- the code's, the subprocess's, reading its
    result, fitting the clip -- comes back as an outcome with its reason.

    `quality` is a worker.TIERS entry as a dict: pixel_width, pixel_height,
    frame_rate.
    """
    try:
        check_code(params.code)
    except RawSceneError as exc:
        return RawOutcome(False, exc.kind, str(exc))

    job_dir: Path | None = None
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        job_dir = Path(tempfile.mkdtemp(prefix="rawscene-", dir=work_dir))
        return _run_job(params, duration, quality, theme, out_path, job_dir, timeout)
    except Exception as exc:  # environment: disk, spawn, a missing tool
        return RawOutcome(False, "crash", f"{type(exc).__name__}: {exc}")
    finally:
        if job_dir is not None:
            shutil.rmtree(job_dir, ignore_errors=True)


def _run_job(params: RawSceneParams, duration: float, quality: dict[str, int],
             theme: str, out_path: Path, job_dir: Path, timeout: float) -> RawOutcome:
    job = {
        "code": params.code,
        **quality,
        "background": get_theme(theme).palette.bg,
        "media_dir": str(job_dir / "media"),
        "result": str(job_dir / "result.json"),
    }
    job_path = job_dir / "job.json"
    job_path.write_text(json.dumps(job))

    # The child must import the same chalkdust as this process, whatever its
    # working directory or editable install points at.
    env = dict(os.environ)
    pkg_root = str(Path(chalkdust.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (pkg_root, env.get("PYTHONPATH")) if p)
    # RawScene's allowlist lets code build Tex/MathTex, whose LaTeX could read
    # host files into the frame (`\input{<path>}`). That compile runs in this
    # child, in job_dir (cwd below, an isolated scratch dir), so force the TeX
    # file-access restriction on here too -- the same control theme._compile
    # uses -- not just a name denylist. On MiKTeX that closes only the braced
    # `\input{}`/`\include{}`; a plain `Tex(r"\input <abs path>")` still
    # renders the file (measured). See theme.TEX_FILE_ACCESS_ENV.
    env = restricted_tex_env(env)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_BOOT, str(job_path)],
            cwd=job_dir, env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return RawOutcome(False, "timeout", f"exceeded {timeout:g}s")

    result_path = Path(job["result"])
    if not result_path.exists():
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-6:])
        return RawOutcome(False, "crash", f"exit {proc.returncode}: {tail}")
    try:
        result = json.loads(result_path.read_text())
        if not result["ok"]:
            return RawOutcome(False, str(result["kind"]), str(result["message"]))
        movie = Path(result["movie"])
    except (ValueError, KeyError, TypeError) as exc:
        return RawOutcome(False, "crash",
                          f"unreadable child result: {type(exc).__name__}: {exc}")

    try:
        _fit_to_duration(movie, out_path, duration, quality["frame_rate"])
    except Exception as exc:  # ffprobe/ffmpeg missing or failing
        return RawOutcome(False, "fit", f"{type(exc).__name__}: {exc}")
    return RawOutcome(True)


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
    fields are dropped: the fallback carries nothing in, and a later beat
    carrying in what the RawScene beat registers is refused at resolution
    (RawScene has no artifact builder), whatever renders in its place."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", spec.narration) if s]
    if len(sentences) > 6:
        sentences = sentences[:5] + [" ".join(sentences[5:])]
    return BeatSpec(id=spec.id, narration=spec.narration, component="BulletReveal",
                    params={"items": sentences}, transition=spec.transition)


def _log_use(log_path: Path, spec: BeatSpec, outcome: RawOutcome,
             cached: bool) -> None:
    # From the raw params, so a beat whose params do not validate is logged too.
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "beat_id": spec.id,
        "rationale": str(spec.params.get("rationale", "")),
        "code_hash": content_hash(str(spec.params.get("code", ""))),
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
    Sets beat.degraded on fallback, and logs every call. Nothing about the
    RawScene itself raises out of here; only the fallback render can.
    """
    from chalkdust.render.worker import TIERS, render_beat, render_key  # worker imports us
    from chalkdust.validate.repair import RepairPlan

    if beat.duration is None:
        raise ValueError(f"{beat.id} has no duration; the speech stage must run first")

    cached = False
    try:
        params = RawSceneParams.model_validate(beat.spec.params)
    except ValidationError as exc:
        outcome = RawOutcome(False, "params", str(exc))
    else:
        if beat.spec.carry_in:
            # Carried artifacts are rebuilt inside a host ChalkdustScene; the
            # generated code runs in another process and could never see them.
            # Degrading is the loud, logged answer. Whether the semantic rung
            # should refuse such a spec earlier is the operator's call.
            outcome = RawOutcome(False, "carry_in", (
                f"{beat.id} carries in {beat.spec.carry_in}; RawScene renders out "
                "of process and cannot draw carried artifacts"))
        else:
            # The worker's key: font-resolved theme, tier settings, repair
            # plan. A RawScene is never mechanically repaired (it renders out
            # of process), so its plan is the empty one.
            key = render_key(beat, resolve_fonts(get_theme(theme)), ctx, RepairPlan())
            slot = cache.slot("beats", key, ".mp4")
            cached = slot.exists
            if cached:
                outcome = RawOutcome(True)
            else:
                outcome = run_raw_scene(params, beat.duration,
                                        asdict(TIERS[ctx.quality]), theme,
                                        slot.tmp, work_dir, timeout)
                if outcome.ok:
                    try:
                        slot.commit()
                    except OSError as exc:
                        outcome = RawOutcome(False, "crash",
                                             f"{type(exc).__name__}: {exc}")
    _log_use(work_dir / USAGE_LOG_NAME, beat.spec, outcome, cached)

    if outcome.ok:
        beat.render_path = slot.path
        beat.degraded = False
        return slot.path

    fallback = Beat(spec=degrade_spec(beat.spec), duration=beat.duration)
    path = render_beat(fallback, theme, ctx, cache, work_dir)
    beat.render_path = path
    beat.degraded = True
    return path
