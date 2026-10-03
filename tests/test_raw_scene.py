"""RawScene escape hatch (SCENE_SPEC.md §7).

Each degradation path has its own test. The subprocess tests really spawn a
child and (for the success path) really render at draft quality -- the
behaviour under test is what happens out of process.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from chalkdust.core.cache import Cache
from chalkdust.core.models import Beat, BeatSpec, BuildContext, Quality
from chalkdust.render.worker import QUALITY_FLAGS
from chalkdust.scenes.components.raw_scene import (
    USAGE_LOG_NAME,
    RawScene,
    RawSceneError,
    RawSceneParams,
    _fit_to_duration,
    _user_namespace,
    check_code,
    degrade_spec,
    render_raw_beat,
    run_raw_scene,
)
from chalkdust.speech.base import probe_duration
from chalkdust.validate.geometric import validate_beat

DRAFT = QUALITY_FLAGS[Quality.DRAFT]

GOOD = """
from manim import *

class Good(Scene):
    def construct(self):
        sq = Square()
        self.play(Create(sq), run_time=1)
        self.play(sq.animate.rotate(PI / 4), run_time=1)
"""

# Over the 80-word cap's neighbourhood, several sentences: the worst
# realistic narration a degraded beat has to carry.
LONG_NARRATION = (
    "Two different keys can land in the same bucket. That is a collision. "
    "The table keeps both, chained in a list. Lookups now walk that list. "
    "With a weak hash, most keys pile into a few buckets. Then every lookup "
    "walks a long chain, and constant time quietly becomes linear time. "
    "That is the failure mode we are going to fix."
)


def _params(code: str) -> RawSceneParams:
    return RawSceneParams(rationale="test visual", code=code)


def _run(code: str, tmp_path, timeout: float = 60.0):
    return run_raw_scene(_params(code), 3.0, DRAFT, "default",
                         tmp_path / "out.mp4", tmp_path / "work", timeout=timeout)


def _scene(body: str) -> str:
    return ("from manim import *\n\nclass S(Scene):\n    def construct(self):\n"
            + "".join(f"        {line}\n" for line in body.splitlines()))


def _kind(code: str) -> str:
    with pytest.raises(RawSceneError) as exc_info:
        check_code(code)
    return exc_info.value.kind


class TestStaticCheck:
    def test_allowed_imports_pass(self):
        check_code("from manim import *\nimport numpy as np\nimport math\n")

    def test_syntax(self):
        assert _kind("class Broken(Scene:\n    pass") == "syntax"

    @pytest.mark.parametrize("code", [
        "import os", "import subprocess", "from sys import argv",
        "import socket", "import os.path", "from . import x",
    ])
    def test_forbidden_import(self, code):
        assert _kind(code) == "forbidden_import"

    def test_module_name_without_import(self):
        # Reachable through an allowed module's namespace is still rejected.
        assert _kind("from manim import *\nos.system('x')") == "forbidden_import"

    @pytest.mark.parametrize("code", [
        # A system module re-exported by an allowed package.
        "from manim.utils.file_ops import os",
        "from manim.utils.file_ops import os as o\no.getcwd()",
        # ...or reached as an attribute of one.
        "import manim.utils.file_ops as f\nf.os.system('x')",
        "from manim.utils import file_ops\nfile_ops.shutil.rmtree('x')",
        # A star from a submodule binds whatever it imported.
        "from manim.utils.file_ops import *",
    ])
    def test_system_module_through_allowed_package(self, code):
        assert _kind(code) == "forbidden_import"

    def test_allowed_submodules_pass(self):
        check_code("import numpy.linalg\n"
                   "from manim.utils.rate_functions import smooth\n"
                   "x = numpy.linalg.norm([3, 4])\n")

    @pytest.mark.parametrize("code", [
        "eval('1')", "exec('x=1')", "open('f')", "__import__('os')",
        "getattr(x, 'y')", "().__class__.__bases__",
    ])
    def test_forbidden_name(self, code):
        assert _kind(code) == "forbidden_name"


class TestRuntimeGuard:
    """The namespace generated code runs in, independent of the AST pass."""

    def test_forbidden_import_at_runtime(self):
        with pytest.raises(RawSceneError) as exc_info:
            exec("import os", _user_namespace())
        assert exc_info.value.kind == "forbidden_import"

    def test_allowed_import_at_runtime(self):
        ns = _user_namespace()
        exec("import math\nfrom manim import Circle", ns)
        assert ns["math"].pi > 3 and ns["Circle"] is not None

    def test_escape_builtins_absent(self):
        with pytest.raises(NameError):
            exec("open('x')", _user_namespace())

    @pytest.mark.parametrize("code", [
        "from manim.utils.file_ops import os as o; o.getcwd()",
        "from manim.utils.file_ops import *",
    ])
    def test_system_module_through_allowed_package_at_runtime(self, code):
        with pytest.raises(RawSceneError) as exc_info:
            exec(code, _user_namespace())
        assert exc_info.value.kind == "forbidden_import"

    def test_star_from_top_level_package_at_runtime(self):
        # `from manim import *` binds a few module objects (typing, np);
        # the fromlist check must not break the line every scene starts with.
        ns = _user_namespace()
        exec("from manim import *\nfrom numpy import *", ns)
        assert ns["Circle"] is not None


class TestDegradationPaths:
    """Every failure comes back as an outcome with its reason; none raises."""

    def test_syntax_degrades(self, tmp_path):
        out = _run("class Broken(Scene:", tmp_path)
        assert (out.ok, out.kind) == (False, "syntax")

    def test_forbidden_import_degrades(self, tmp_path):
        out = _run("import subprocess\n" + GOOD, tmp_path)
        assert (out.ok, out.kind) == (False, "forbidden_import")

    def test_no_scene_degrades(self, tmp_path):
        out = _run("from manim import *\nx = Square()", tmp_path)
        assert (out.ok, out.kind) == (False, "no_scene")

    def test_timeout_degrades(self, tmp_path):
        out = _run(_scene("while True:\n    pass"), tmp_path, timeout=8.0)
        assert (out.ok, out.kind) == (False, "timeout")

    def test_crash_degrades(self, tmp_path):
        out = _run(_scene("raise ValueError('generated code bug')"), tmp_path)
        assert (out.ok, out.kind) == (False, "crash")
        assert "generated code bug" in out.message

    def test_layout_degrades(self, tmp_path):
        out = _run(_scene("t = Text('off screen').shift(RIGHT * 9)\n"
                          "self.play(FadeIn(t))"), tmp_path)
        assert (out.ok, out.kind) == (False, "layout")
        assert "out_of_bounds" in out.message


class TestDegradedBeat:
    def test_degrade_spec_carries_narration(self):
        spec = BeatSpec(id="b07", narration=LONG_NARRATION, component="RawScene",
                        params={"rationale": "r", "code": "x"})
        degraded = degrade_spec(spec)
        assert degraded.component == "BulletReveal"
        assert 1 <= len(degraded.params["items"]) <= 6
        assert " ".join(degraded.params["items"]) == LONG_NARRATION
        # Worst realistic narration still lays out clean as bullets.
        assert validate_beat(degraded).ok

    def test_render_falls_back_and_logs(self, tmp_path):
        spec = BeatSpec(id="b07", narration="A point traces the curve.",
                        component="RawScene",
                        params={"rationale": "needs os", "code": "import os"})
        beat = Beat(spec=spec, duration=2.0)
        path = render_raw_beat(beat, "default", BuildContext(),
                               Cache(tmp_path / "cache"), tmp_path)
        assert beat.degraded and path.exists() and beat.render_path == path
        entry = json.loads((tmp_path / USAGE_LOG_NAME).read_text().splitlines()[-1])
        assert entry["outcome"] == "degraded"
        assert entry["reason"] == "forbidden_import"
        assert entry["rationale"] == "needs os"


class TestSuccess:
    def test_renders_fitted_to_duration_and_logs(self, tmp_path):
        spec = BeatSpec(id="b07", narration="A square turns.", component="RawScene",
                        params={"rationale": "rotation demo", "code": GOOD})
        beat = Beat(spec=spec, duration=3.5)  # natural run is 2s: held to fit
        cache = Cache(tmp_path / "cache")
        path = render_raw_beat(beat, "default", BuildContext(), cache, tmp_path)
        assert not beat.degraded and path.exists()
        # Whole frames at 15 fps: within one frame of the audio duration.
        assert abs(probe_duration(path) - 3.5) <= 1 / 15 + 1e-6

        # Second call is a cache hit, and still logged as a use.
        render_raw_beat(Beat(spec=spec, duration=3.5), "default", BuildContext(),
                        cache, tmp_path)
        lines = [json.loads(x) for x in
                 (tmp_path / USAGE_LOG_NAME).read_text().splitlines()]
        assert [(e["outcome"], e["cached"]) for e in lines] == \
            [("rendered", False), ("rendered", True)]


class TestFitToDuration:
    def test_long_run_sped_up_to_budget(self, tmp_path):
        # A 3 s, 45-frame source (the real 2 s-of-play clip shape) into a
        # 1.8 s budget. Every frame is red except the LAST, which is blue:
        # truncation would also hit the duration, and a speed-up that drops
        # the final frame would too -- only the right speed-up ends on blue,
        # i.e. on the settled state the layout checks passed.
        src, dst = tmp_path / "src.mp4", tmp_path / "dst.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=red:s=64x64:r=15",
             "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=15",
             "-filter_complex", "[0:v]trim=end_frame=44[a];[1:v]trim=end_frame=1[b];"
                                "[a][b]concat=n=2:v=1",
             "-pix_fmt", "yuv420p", str(src)], check=True)
        assert probe_duration(src) == pytest.approx(3.0)
        _fit_to_duration(src, dst, 1.8, 15)

        assert abs(probe_duration(dst) - 1.8) <= 1 / 15 + 1e-6
        frames = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(dst),
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, capture_output=True).stdout
        last = frames[-64 * 64 * 3:]
        r, _, b = last[3 * (32 * 64 + 32):][:3]  # its centre pixel
        assert b > 200 and r < 50


class TestComponent:
    def test_rationale_required(self):
        with pytest.raises(Exception):
            RawScene({"rationale": "", "code": GOOD})

    def test_rejects_unknown_param(self):
        with pytest.raises(Exception):
            RawScene({"rationale": "r", "code": GOOD, "x_range": [0, 1]})

    def test_build_in_host_scene_is_refused(self):
        spec = BeatSpec(id="b01", narration="n", component="RawScene",
                        params={"rationale": "r", "code": GOOD})
        assert validate_beat(spec).kinds() == {"build_error"}
