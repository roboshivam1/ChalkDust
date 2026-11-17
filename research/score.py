"""Score generated beats and specs with ChalkDust's own validator, per beat,
without stopping at the first failure.

The pipeline's check_semantics / check_layout raise on the first bad report,
which is right for a render and wrong for an experiment: best-of-N needs a
score for every candidate. So this module calls the same rung functions the
pipeline uses, one beat at a time:

  rung 1  BeatSpec + the component's Params model + control characters
          (pipeline.load_spec's checks, per beat)
  rung 2  validate_semantic (duration, capacity, LaTeX compiles, regions)
  rung 3  repair_beat (geometric probe with bounded mechanical repair)

Findings are bucketed. `toolchain` (TeX timed out) is the machine, and
`build_error` is a library bug triggered by the input -- neither is counted as
the model producing an invalid beat; both are reported separately and the
beat is retried once. The headline "spec valid" number comes from the real
CLI (`chalkdust validate`), run on assembled specs (validate_spec_cli).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import ValidationError

from chalkdust.core.models import BeatSpec
from chalkdust.pipeline import PROBE_SECONDS, _control_characters, manim_scratch
from chalkdust.render.worker import long_path
from chalkdust.scenes.components.base import get_component
from chalkdust.validate.repair import repair_beat
from chalkdust.validate.semantic import validate_semantic

REPO = Path(__file__).resolve().parents[1]
HARNESS_KINDS = {"toolchain", "build_error"}
MIKTEX = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64"


def ensure_miktex_first() -> None:
    """LaTeX needs MiKTeX's latex/dvisvgm first on PATH (CLAUDE.md)."""
    p = os.environ.get("PATH", "")
    if MIKTEX.exists() and not p.lower().startswith(str(MIKTEX).lower()):
        os.environ["PATH"] = str(MIKTEX) + os.pathsep + p


# --- mechanical LaTeX-escape repair ------------------------------------------
# A model writing "\theta" inside a JSON string emits the JSON escape \t (a TAB)
# followed by "heta"; "\frac" becomes form-feed + "rac". Grammar-constrained
# decoding allows these escapes, rung 1 refuses the control character. This
# deterministic fix restores the backslash. It is counted, so the report can
# show how often each model needs it.
_ESC = {"\t": "\\t", "\b": "\\b", "\f": "\\f", "\r": "\\r", "\v": "\\v"}
_LATEX_AFTER_NEWLINE = re.compile(r"\n(?=(abla|u\b|eq|ot|ewline|leq|geq|e\b|i\b|orm|ewcommand))")
_NO_REPAIR_KEYS = {"source", "narration"}  # CodeWalk source keeps real newlines


def repair_escapes(value, key: str | None = None) -> tuple[object, int]:
    if isinstance(value, str):
        if key in _NO_REPAIR_KEYS:
            return value, 0
        n = 0
        out = []
        for ch in value:
            if ch in _ESC:
                out.append(_ESC[ch]); n += 1
            else:
                out.append(ch)
        s = "".join(out)
        s, k = _LATEX_AFTER_NEWLINE.subn(r"\\n", s)
        return s, n + k
    if isinstance(value, dict):
        tot = 0
        new = {}
        for k, v in value.items():
            new[k], c = repair_escapes(v, k)
            tot += c
        return new, tot
    if isinstance(value, list):
        tot = 0
        new = []
        for v in value:
            nv, c = repair_escapes(v, key)
            new.append(nv); tot += c
        return new, tot
    return value, 0


# --- per-beat scoring ----------------------------------------------------------

# --- hard quality flags the validator does not check ---------------------------
# Found in the first rendered video (2026-11-17): the validator passed a beat
# whose narration contained "$v = 20\,\mathrm{m/s}$" (the voice reads it
# aloud) and a ProblemStatement whose GIVEN items were bare LaTeX without $...$
# (drawn as literal backslashes). Both are model failures the render shows.
_LATEX_CMD = re.compile(r"\\[A-Za-z]+|\$|\^\{|_\{")
# Fields that are prose (drawn as text; inline maths only inside $...$).
_PROSE_FIELDS = {"text", "given", "find", "items", "heading", "title", "subtitle", "kicker",
                 "body", "verdict", "claim", "justification", "label", "note", "callout"}
# Fields whose whole value is math-mode LaTeX.
_MATH_FIELDS = {"steps", "work", "math", "value", "units", "unit"}


def _outside_dollars(s: str) -> str:
    return re.sub(r"\$[^$]*\$", " ", s)


# Narration is spoken: the voice reads '^' as "caret" and drops or misreads
# Unicode super/subscripts and the radical sign. The beat prompt forbids '^';
# review found 66 clean E1b beats with '^' and 78 with superscripts.
_NARR_SYMBOL = re.compile(r"\^|[²³¹⁰-₟√]")


def _math_strings(beat: dict) -> set[str]:
    """The strings the component itself compiles as LaTeX (its own
    latex_strings() hook, the one the semantic rung uses). A field the
    component typesets as maths is not prose, whatever it is called:
    FreeBodyDiagram's force `label` is LaTeX by design ('mg \\sin\\theta')."""
    try:
        import chalkdust.scenes.components  # noqa: F401  (fills the registry)
        from chalkdust.scenes.components.base import make_component
        return {s.strip() for s in make_component(beat.get("component"), beat.get("params", {})).latex_strings()}
    except Exception:
        return set()


def quality_flags(beat: dict) -> list[str]:
    flags = []
    narr = str(beat.get("narration", ""))
    if _LATEX_CMD.search(narr):
        flags.append("narration_latex")
    if _NARR_SYMBOL.search(narr):
        flags.append("narration_symbol")

    # Only ProblemStatement typesets inline $...$ in prose; everywhere else a
    # '$' or a backslash command in a prose field is drawn literally.
    inline_ok = beat.get("component") == "ProblemStatement"
    math = _math_strings(beat)

    def walk(v, key):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, k)
        elif isinstance(v, list):
            for x in v:
                walk(x, key)
        elif isinstance(v, str) and key in _PROSE_FIELDS and v.strip() not in math:
            rest = _outside_dollars(v) if inline_ok else v
            if re.search(r"\\[A-Za-z]+|\\[,;!]|\^\{|_\{", rest):
                flags.append(f"prose_latex:{key}")
            if "$" in v and (not inline_ok or v.count("$") % 2):
                flags.append(f"stray_dollar:{key}")
    walk(beat.get("params", {}), None)
    return sorted(set(flags))


@dataclass
class BeatScore:
    beat_id: str
    component: str
    level: int                      # 0 schema fail, 1 schema ok, 2 semantic ok, 3 layout ok
    rung1_errors: list[str] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)   # rung 2/3 findings {rung, kind, message}
    harness_fault: bool = False
    repaired_mechanically: bool = False
    escape_repairs: int = 0
    flags: list[str] = field(default_factory=list)   # hard quality flags (quality_flags)
    text_chars: int = 0
    narration_words: int = 0
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """Passes the validator (rungs 1-3) -- what `chalkdust validate` checks."""
        return self.level == 3

    @property
    def clean(self) -> bool:
        """Passes the validator AND the hard quality flags."""
        return self.level == 3 and not self.flags

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        d["clean"] = self.clean
        return d


def text_chars(params) -> int:
    """Visible text volume of a beat's params (anti-gaming floor: a beat that
    shows nothing always passes layout)."""
    if isinstance(params, str):
        return len(params.strip())
    if isinstance(params, dict):
        return sum(text_chars(v) for k, v in params.items() if k not in {"op", "kind", "language", "reveal", "emphasis", "side", "work_format"})
    if isinstance(params, list):
        return sum(text_chars(v) for v in params)
    if isinstance(params, (int, float)) and not isinstance(params, bool):
        return len(str(params))
    return 0


def score_beat(beat: dict, beat_id: str, work_dir: Path, theme: str = "default",
               fix_escapes: bool = True) -> BeatScore:
    """Rungs 1-3 on one carry-in-free beat. Never raises."""
    t0 = time.perf_counter()
    component = str(beat.get("component", ""))
    params = beat.get("params", {})
    narration = beat.get("narration", "")
    reps = 0
    if fix_escapes:
        params, reps = repair_escapes(params)
    sc = BeatScore(beat_id, component, 0, escape_repairs=reps,
                   flags=quality_flags({**beat, "params": params}),
                   text_chars=text_chars(params),
                   narration_words=len(str(narration).split()))
    # rung 1
    errs: list[str] = []
    spec = None
    try:
        spec = BeatSpec(id=beat_id, narration=narration, component=component,
                        params=params if isinstance(params, dict) else {})
    except ValidationError as exc:
        errs += [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
    except Exception as exc:
        errs.append(f"{type(exc).__name__}: {exc}")
    if spec is not None:
        errs += _control_characters(spec.narration, f"{beat_id}.narration")
        errs += _control_characters(spec.params, f"{beat_id}.params")
        try:
            get_component(component).Params.model_validate(spec.params)
        except KeyError as exc:
            errs.append(f"component: {exc.args[0]}")
        except ValidationError as exc:
            errs += [f"params.{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        except Exception as exc:
            errs.append(f"params: {type(exc).__name__}: {exc}")
    if errs or spec is None:
        sc.rung1_errors = errs
        sc.seconds = time.perf_counter() - t0
        return sc
    sc.level = 1
    ensure_miktex_first()
    work_dir = long_path(Path(work_dir))
    (work_dir / "manim").mkdir(parents=True, exist_ok=True)
    try:
        with manim_scratch(work_dir):
            rep2 = validate_semantic(spec, (), media_dir=work_dir / "manim", theme=theme)
            sc.findings += [{"rung": 2, "kind": f.kind, "message": f.message[:400]} for f in rep2.findings]
            if rep2.ok:
                sc.level = 2
                res = repair_beat(spec, theme, PROBE_SECONDS)
                sc.findings += [{"rung": 3, "kind": f.kind, "message": f.message[:400]}
                                for f in res.report.findings]
                if res.ok:
                    sc.level = 3
                    sc.repaired_mechanically = res.repaired
    except Exception as exc:  # the scorer itself broke: a harness fault
        sc.findings.append({"rung": 0, "kind": "build_error",
                            "message": f"scorer: {type(exc).__name__}: {exc}"[:400]})
    sc.harness_fault = any(f["kind"] in HARNESS_KINDS for f in sc.findings)
    sc.seconds = time.perf_counter() - t0
    return sc


_POOL = None


def _score_job(args):
    beat, beat_id, work_dir = args
    return score_beat(beat, beat_id, Path(work_dir))


def score_many(jobs: list[tuple[dict, str, Path]], workers: int = 4) -> list[BeatScore]:
    """Score several candidates at once in a persistent spawned process pool.
    Each job has its own work dir, so Manim's text/Tex caches never collide
    (CLAUDE.md: shared caches tore SVGs). One job runs in-process."""
    global _POOL
    if len(jobs) <= 1 or workers <= 1:
        return [score_beat(b, i, Path(w)) for b, i, w in jobs]
    if _POOL is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ensure_miktex_first()
        _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    try:
        return list(_POOL.map(_score_job, [(b, i, str(w)) for b, i, w in jobs]))
    except Exception:  # a dead pool is a harness fault: fall back in-process
        _POOL = None
        return [score_beat(b, i, Path(w)) for b, i, w in jobs]


def selection_key(sc: BeatScore, floor_chars: int = 12) -> tuple:
    """Best-of-N ordering: validity first, then a content floor (a beat must
    show something), then fewer findings. Higher is better."""
    floor_ok = sc.text_chars >= floor_chars or sc.component == "TitleCard"
    return (sc.level, -len(sc.flags), int(floor_ok), -len(sc.findings) - len(sc.rung1_errors),
            -sc.escape_repairs)


# --- whole spec, through the real CLI -------------------------------------------

def validate_spec_cli(spec_path: Path, work_dir: Path, timeout: float = 900) -> dict:
    """`chalkdust validate` on an assembled spec: the headline validity number."""
    ensure_miktex_first()
    exe = REPO / ".lab" / "venv-harness" / "Scripts" / "chalkdust.exe"
    t0 = time.perf_counter()
    try:
        p = subprocess.run([str(exe), "validate", str(spec_path), "--work-dir", str(work_dir)],
                           capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                           errors="replace", env=os.environ.copy())
        out = (p.stdout + "\n" + p.stderr).strip()
        # drop MiKTeX update-check noise
        out = "\n".join(l for l in out.splitlines() if "major issue" not in l and "MiKTeX" not in l)
        return {"exit": p.returncode, "ok": p.returncode == 0, "seconds": time.perf_counter() - t0,
                "output": out[-3000:]}
    except subprocess.TimeoutExpired:
        return {"exit": None, "ok": False, "seconds": time.perf_counter() - t0, "output": "timeout"}


def assemble_spec(video_id: str, beats: list[dict], voice: dict | None = None) -> dict:
    """A VideoSpec dict from scored beats (ids b01..)."""
    out = []
    for i, b in enumerate(beats, 1):
        params, _ = repair_escapes(b.get("params", {}))
        out.append({"id": f"b{i:02d}", "narration": b.get("narration", ""),
                    "component": b.get("component"), "params": params,
                    "transition": "cross_fade" if i > 1 else "cut"})
    spec = {"video_id": video_id, "channel": "research", "theme": "default", "beats": out}
    if voice:
        spec["voice"] = voice
    return spec


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-").lower()
    return (s or "video")[:90]
