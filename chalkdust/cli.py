"""Command line entry point.

Phase 0 verbs work on a hand-written spec file:

  chalkdust validate <spec>                      rungs 1-3, nothing rendered
  chalkdust render <spec> --quality draft|final  spec -> MP4

These sit beside the verbs still to come, and keep their meanings: `make`
(topic -> video, Phase 2) and the manual overrides in OPERATIONS.md §8
(`script --edit`, `render --beats ... --force`, `publish --skip-gate`). The
`render` here is the same verb those overrides extend.

Each known failure gets its own exit code, so a calling script can tell a spec
to fix from a render to retry. 2 is left to argparse for usage errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chalkdust import pipeline
from chalkdust.core.models import Quality

EXIT_OK = 0
EXIT_SPEC_INVALID = 3
EXIT_LAYOUT_REFUSED = 4
EXIT_SPEECH_FAILED = 5
EXIT_RENDER_FAILED = 6
EXIT_ASSEMBLY_FAILED = 7
# Added after 3-7 were in use, so it keeps them stable rather than sitting
# in ladder order between spec (rung 1) and layout (rung 3).
EXIT_SEMANTIC_REFUSED = 8
# A --work-dir or --cache-dir that cannot be a directory, or a work dir
# whose full path has a '~' LaTeX cannot compile under: fix the command
# line, not the spec.
EXIT_DIRECTORY_UNUSABLE = 9

EXIT_CODES: dict[type[pipeline.PipelineError], tuple[int, str]] = {
    pipeline.SpecInvalid: (EXIT_SPEC_INVALID, "spec invalid"),
    pipeline.SemanticRefused: (EXIT_SEMANTIC_REFUSED, "semantic refused"),
    pipeline.LayoutRefused: (EXIT_LAYOUT_REFUSED, "layout refused"),
    pipeline.SpeechFailed: (EXIT_SPEECH_FAILED, "speech failed"),
    pipeline.RenderFailed: (EXIT_RENDER_FAILED, "render failed"),
    pipeline.AssemblyFailed: (EXIT_ASSEMBLY_FAILED, "assembly failed"),
    pipeline.DirectoryUnusable: (EXIT_DIRECTORY_UNUSABLE, "directory unusable"),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chalkdust",
        description="Declarative spec -> Manim explainer video with synced voiceover.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("spec", type=Path, help="path to a video spec (JSON)")
    common.add_argument("--work-dir", type=Path, default=pipeline.DEFAULT_WORK_DIR,
                        help="scratch for Manim and assembly (default: %(default)s)")
    common.add_argument("-v", "--verbose", action="store_true",
                        help="show Manim's own logging and progress bars")

    verbs = parser.add_subparsers(dest="verb", required=True)
    verbs.add_parser("validate", parents=[common],
                     help="check schema, content and layout without speech or render")

    render = verbs.add_parser("render", parents=[common],
                              help="render a spec to a finished MP4")
    render.add_argument("--quality", choices=[q.value for q in Quality],
                        default=Quality.DRAFT.value,
                        help="draft = 480p15, final = 1080p60 (default: %(default)s)")
    render.add_argument("--out", type=Path, default=None,
                        help="output MP4 (default: out/<video_id>-<quality>.mp4)")
    render.add_argument("--cache-dir", type=Path, default=pipeline.DEFAULT_CACHE_DIR,
                        help="content-addressed cache (default: %(default)s)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.verb == "validate":
            spec = pipeline.validate(args.spec, args.work_dir, args.verbose)
            print(f"{args.spec}: ok ({len(spec.beats)} beats)")
        else:
            quality = Quality(args.quality)
            print(f"rendering {args.spec} at {quality.value}")
            result = pipeline.render(args.spec, quality, args.out, args.cache_dir,
                                     args.work_dir, args.verbose)
            print(f"wrote {result.output}")
    except pipeline.PipelineError as exc:
        code, what = EXIT_CODES[type(exc)]
        print(f"chalkdust: {what}: {exc}", file=sys.stderr)
        return code
    return EXIT_OK
