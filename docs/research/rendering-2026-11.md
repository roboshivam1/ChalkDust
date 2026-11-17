# Rendering stack research, November 2026

Collected 17 November 2026 for the desktop-app direction in `chalkdust-research-2026-10.md`. That file's rendering track was never run; this closes it. Every claim about a tool, version, licence, date or benchmark cites a page fetched on 17 Nov 2026 (Sources at the end) or a named local file on the lab machine. Anything not confirmed that way is marked **[unverified]**. Nothing was installed, rendered or benchmarked for this note. The measurements still needed are listed in section 8, for when someone is back at the machine.

This is research, not a decision. D-001 to D-012 stand until the owners decide.

---

## 0. Outcome: recommended stack

| Layer | Recommendation | Why, in one line |
|---|---|---|
| Engine | **Stay on Manim CE (MIT), Cairo renderer.** Plan a pin bump to 0.22.x, which makes the video encoder a config setting (§5.1). Do not adopt ManimGX now. Re-evaluate it when the triggers in §2.7 are met. | ManimGX 0.1.1 crashes on Intel integrated GPUs under Windows: Iris Xe in #50, UHD 600 in #78. A related shader bug on UHD 620 under Linux/Mesa is #97. It documents a CPU fallback only for Linux. It replaces Pango text and LaTeX, and it belongs to a competitor. |
| Maths | **Move from MiKTeX to Typst through Manim CE's own `MathTypst`**, which already ships in the pinned 0.21.0. Install the `typst` Python package (Apache-2.0, about 30 MB wheel). Keep MiKTeX as a fallback until a benchmark and a coverage test on the real specs pass (§8). | This removes the LaTeX install, compiles inside the Python process instead of spawning TeX, and imports through `SVGMobject` as VMobjects, as `MathTex` does. |
| Maths syntax the LLM writes | **Decide by measurement:** either keep LaTeX in specs and convert it with mitex (Apache-2.0) vendored offline, or switch the spec to Typst maths. Chemistry goes through typsium (MIT) for `\ce`-style input. | Small models know LaTeX far better. mitex does not support LaTeX packages, so `\ce` via mitex is very likely out (inferred, §4.3). |
| Encoding | **Do not ship the PyPI PyAV wheel or a GPL ffmpeg.exe as they are.** Ship an LGPL-only FFmpeg build and encode with the OS H.264 encoder (`h264_mf` on Windows, `h264_videotoolbox` on macOS). Manim CE 0.22.x's `[video_encoder] codec` setting makes the switch a config change. Fallback: buy an x264 commercial licence. Get counsel on H.264 patents before launch. | The current stack already loads GPL x264 into ChalkDust's own process: Manim CE 0.21.0 hard-codes `libx264`, and the PyAV 19.0.1 wheel bundles `libx264-165.dll` and x265 (§5.1). The GPL problem is not specific to ManimGX. |
| Speed tiers | **Keep 480p15 for validation. Make the student-laptop final 1080p30, with a 720p30 "fast" option. Keep 1080p60 for the cloud and fast machines.** | Measured on the lab ROG: a 3-minute video at 1080p60 renders cold in 123 to 149 s on 24 CPU cores **[corrected 2026-11-17: the pool ran 9 or 10 worker processes, one per beat, not 24. How many cores they kept busy is unmeasured; see §6.1]**. Class-C laptops have a fraction of that (§6). 1080p30 halves the frames. |

**Corrections to the October notes**

- **Manim CE licence.** It is MIT, verified. The installed wheel carries two MIT files: `Copyright (c) 2018 3Blue1Brown LLC` and `Copyright (c) 2024, the Manim Community Developers`.
- **ManimGX launch date.** The first PyPI releases (0.1.0 and 0.1.1) are dated 7 Oct 2026, and the repo was created 5 Oct 2026. The notes said "about 10 Oct".
- **ManimGX speed.** The "90x" is real arithmetic on Academa's own benchmark, but one scene (orbit) is 76% of Manim CE's total time. Per-scene ratios run from 6.3x to 179x (§2.3).
- **LaTeX-free maths.** Manim CE gained native `Typst` and `MathTypst` in 0.21.0 (10 Aug 2026). LaTeX-free maths no longer needs ManimGX.
- **ffmpeg legal page.** It has now been fetched and quoted (§5.2). It says to make sure your program uses no GPL libraries, "notably libx264".

---

## 1. Manim Community Edition and ManimGL: licences, versions, status

### Manim CE

- **Licence: MIT.** The repo's `LICENSE.community` reads:
  - "Permission is hereby granted, free of charge, to any person obtaining a copy of this software ... to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software"
  - The only condition: "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software."
  - PyPI lists the SPDX expression `MIT` with dual copyright (3Blue1Brown LLC and the Manim Community Developers).
  - Locally, `manim-0.21.0.dist-info/METADATA` says `License-Expression: MIT` and ships both `LICENSE` (3Blue1Brown LLC, 2018) and `LICENSE.community`.
- **Versions.** The latest is **0.22.1 (27 Oct 2026)**, after 0.22.0 (8 Oct 2026) and 0.21.0 (10 Aug 2026). ChalkDust pins **0.21.0** (`pyproject.toml`). Requires-Python is >=3.11. The extras are `gui`, `jupyterlab` and `typst`.
- **What 0.21.0 added** (already in the pin):
  - "first-class support for rendering text and markup via Typst (optional dependency)", as `Typst` and `MathTypst`
  - "Speed up Cairo renderer by 2.2x on animation-heavy scenes" (pixel-identical)
  - Parallel encoding of partial movie files (`max_inflight_encoders`): a 56% reduction on an encoding-heavy benchmark and "roughly 8–11%" on larger real renders
- **What 0.22.0 changed** (not in the pin):
  - Breaking: output is chosen by a single `format` setting. `[ffmpeg]` config became `[media]`. `--flush_cache` became `manim cache clear`. A new `Manager` changes custom renderer and file-writer signatures.
  - New ManimPango 0.7.0 that fixes kerning at very small font sizes. (The local venv already has manimpango 0.7.0.)
  - "Preserved the layout of `MathTypst` capture groups" (PR #4960), which matters if ChalkDust colours sub-parts of Typst maths. **[corrected 2026-11-17: the changelog's wording is "Preserve the layout of `MathTypst` capture groups by @behackl in PR #4960"; the substance is unchanged.]**
  - Several OpenGL renderer fixes.
  - In the v0.22.1 source, the cached-segment video encoder is a config setting (`[video_encoder] codec`, default `auto` = `libx264` for MP4). That is the hook for a non-GPL encoder (§5.1). **[corrected 2026-11-17: this arrived in 0.22.0, not 0.22.1. The 0.22.0 changelog says "Video encoding is now configurable, too", with the new `[video_encoder]` section and the `--video-codec`, `--pixel-format` and repeatable `--encoder-option KEY=VALUE` CLI options. Per the v0.22.0 release notes, Manim's own cache keys now include the encoder settings, so existing cached segments re-render once. 0.22.1 is a small patch release with no encoder changes.]**
- **Status.** Active, with releases about every 1 to 2 months in 2026. A move to 0.22.x is an ordinary pin bump under SCENE_SPEC.md §10, but the config renames touch `render/worker.py`'s `tempconfig` use **[unverified: not checked line by line]**.

### ManimGL (3b1b/manim)

- **Licence: MIT.** The README says "This project falls under the MIT license."
- **Version: 1.7.2, released 13 Dec 2024.** Nothing has been released on PyPI since. Its classifiers list Python 3.7 to 3.10.
- **Status.** It is 3Blue1Brown's own engine and is installed as `manimgl`. It renders through OpenGL and still needs a LaTeX distribution: ManimGX's benchmark runner lists "a LaTeX distribution, for ManimCE and ManimGL". It is not a candidate for ChalkDust. Its API differs from CE, its last release is almost two years old, and it gains nothing on the LaTeX problem.

---

## 2. ManimGX (academa-labs/manimgx)

### 2.1 Claim check

| Claim in the Oct notes | Verdict | Evidence |
|---|---|---|
| Repo academa-labs/manimgx, launched about 10 Oct 2026 | **Close.** Repo created 5 Oct 2026; PyPI 0.1.0 and 0.1.1 both dated 7 Oct 2026. No GitHub release has been published; the Releases section is empty. | GitHub API, PyPI |
| Manim CE-compatible API | **Partly.** The README says "compatible with Manim CE's API" and "ManimGX targets the API of Manim Community Edition." Its own guide lists many differences (§2.4). In the earlier Show HN thread, the submitter (sinaatalay) claimed "We are fully compatible with Manim CE", while another reply said "we decided to make part of its Python interface compatible with ManimCE". | README, docs, HN |
| Rust/wgpu GPU core | **Yes.** The Rust renderer "uses wgpu for native GPU rendering and WebGPU in the browser." | docs home |
| About 90x faster than CE on an M4 Pro, own benchmark | **Yes, as arithmetic, with large caveats** (§2.3): "90.6× faster than ManimCE", "36.2 seconds of video, rendered in 3.08 seconds", measured 7 Oct 2026 on an Apple M4 Pro, 12 cores, 24 GB. | benchmarks/ **[corrected 2026-11-17: the two quoted sentences are in the repo README, not on the `benchmarks/` page. The benchmarks page has the table (90.6× in its "Relative to ManimGX" row, total 3.078 s) and the hardware and date.]** |
| Typst maths, LaTeX via mitex | **Yes.** "Typst typesets text and math, with mitex translating LaTeX input to Typst." In the docs: "LaTeX packages and `\def` don't [work]." | README, docs |
| No LaTeX or ffmpeg install | **Yes.** "Write LaTeX or Typst. No LaTeX installation needed." The pip package says "Fonts, typesetting and video encoding are included." | README |
| Python 3.13+ | **Yes.** `Requires-Python >=3.13`. Wheels exist for Windows x86-64, macOS x86-64/arm64, Linux x86-64/arm64, and wasm32. ChalkDust's venv is Python 3.13.12. | PyPI |
| `manimgx inspect` layout checks | **Yes.** It checks for "a mobject that the frame cuts off", overlapping text, "a line through a text, or a fill over it" and "text too small to read". It exits 1 on layout problems. "Checks cover the sampled moments, not every frame of the video." | rendering docs |
| Code MIT, native wheels GPL-3.0-or-later because of bundled x264 | **Yes.** "ManimGX's own code is MIT-licensed." "Native wheels include x264 and are GPL-3.0-or-later as a whole." Bundled FFmpeg code is LGPL-2.1-or-later. The LICENSE file is plain MIT, "Copyright (c) 2026 Academa, Inc." | README, LICENSE |
| Owned by competitor Academa | **Yes.** It is "an open-source project by Academa". PyPI lists the author as "Academa Team (team@academa.ai)". | docs, PyPI |

A note on the PyPI licence expression. It is a long SPDX list that includes `GPL-2.0-or-later` and `LGPL-2.1-or-later`, which cover the bundled native code. Its `GPL-3.0-or-later WITH Font-exception-2.0` entry is a font licence term, not the x264 term. The source for the x264 claim is the README sentence quoted above.

### 2.2 Maturity, as of 17 Nov 2026

- **Size and age.** 2,784 stars, about six weeks old.
- **Releases.** Only 0.1.0 and 0.1.1, both 7 Oct. The changelog's "Unreleased" section has more fixes than either release.
- **Activity on `main`.** The newest commit is 24 Oct 2026 ("Number a line's ticks with the decimal places they need"). Two 22 Oct commits are titled "Take #61 off main for review" and "Take #62 to #66 off main for review". Pushes continue on other branches (`pushed_at` 16 Nov 2026), including `fix/windows-direct3d` and `fix/font-family-names`. Development is active but pull requests are waiting for review. **[corrected 2026-11-17: the 16 Nov `pushed_at` (22:56:47Z) matches dependabot PR #98, opened two seconds later. The head of `fix/windows-direct3d` (59a0719) was committed on 22 Oct and backs the still-open PR #67. Of the branch heads checked (`main`, `fix/windows-direct3d`, `studio-ready`, `present-in-window`, `player-failure`, `time/one-evaluator`), the newest human commit is from 24 Oct (`main` and `time/one-evaluator`). `ci/source-store`, `docs/hosting-unit-path` and the other `fix/*` branches were not checked. So "active" is supported up to late October and not after it, on what was checked. Open dependabot and community PRs (#95, #96, #98) are waiting.]**
- **Issues.** 19 open issues on the issues page; the API's `open_issues_count` of 38 includes pull requests. Only 1 closed issue was visible. Maintainers have answered almost none of the bug reports fetched:
  - #78 had no maintainer comment.
  - #50 had only a 👍 reaction.
  - #74 had a community reply.
  - #60 got "We'll do this! Thanks." from sinaatalay, who also submitted the Show HN.
- **Contributors.** Seven distinct accounts appear in recent commits (sinaatalay on nearly all of them), including a `claude` co-author account and dependabot. **[corrected 2026-11-17: the 24 newest commits on `main` (GitHub API) have five author logins: sinaatalay (15), Don-Yin (2), japbcoelho (3), mertkayacs (3) and dependabot[bot] (1). So sinaatalay wrote most of them, not "nearly all". The `claude` co-author account is [unverified]: co-author trailers were not checked.]**

### 2.3 The benchmark, scene by scene

From the repo's `benchmarks/` page: 1920×1080, 60 fps, H.264 `yuv420p` MP4, timed from process launch to finished MP4, fresh processes, Manim CE's cache off, runs one after another. Versions: ManimGX 0.1.0, ManimGL 1.7.2, ManimCE 0.21.0, Blender 5.2.2 LTS.

| Scene (video length) | ManimGX (s) | ManimGL (s) | ManimCE (s) | CE ÷ GX |
|---|---|---|---|---|
| orbit (10 s) | 1.184 | 6.463 | 211.809 | 179x |
| matrix (7.2 s) | 0.360 | 8.287 | 7.140 | 19.8x |
| pendulums (13 s) | 0.756 | 16.812 | 4.745 | 6.3x |
| hierarchy (3 s) | 0.334 | 1.789 | 16.169 | 48x |
| linked_rings (3 s) | 0.443 | 1.933 | 38.962 | 88x |
| **Total (36.2 s)** | **3.078** | **35.284** | **278.826** | **90.6x** |

How to read it:

- **One scene dominates.** orbit is 211.8 s of Manim CE's 278.8 s total (76%).
- **Fewer frames from ManimGX.** On matrix and pendulums, Manim CE beat ManimGL, and ManimGX's lead is 6x to 20x. ManimGX also "encodes static frames once as longer frames", so its matrix and pendulums videos have fewer frames (243 and 721) than the other engines' output. That flatters ManimGX. **[corrected 2026-11-17: that quote is not on the page. Its words are "Where the picture holds still, ManimGX encodes it once, as a longer frame": 243 frames over 433 frames of time (matrix) and 721 over 781 (pendulums). The page adds that ManimGX's video "also ends with one extra frame".]**
- **Uneven run counts.** ManimGX and ManimGL times are means of three runs; Manim CE and Blender were run once.
- **Same encoder settings.** ManimGX, ManimGL and Manim CE all encode with x264 `ultrafast`, CRF 23.
- **Unknowns.** The page does not say which Manim CE renderer was used (Cairo or OpenGL), whether MathTex compilation fell inside the timed window, or whether Manim CE used parallel encoding.
- **No independent benchmark found.** Searches turned up only the vendor's numbers and articles repeating them. None of the ChalkDust components has been run on ManimGX.

### 2.4 Compatibility for ChalkDust's components

ManimGX's "Coming from Manim CE" guide lists these differences:

- Module paths differ ("Manim CE's modules are not there"), and NumPy is not in the package.
- Quality flags become `-r` / `--fps`.
- `Tex`/`MathTex` convert to Typst; "LaTeX packages and `\def` don't convert". Double braces make a part "wherever they are".
- A tall 9:16 frame is 8 units wide, where Manim CE's is 4.5.
- Updaters "get the exact time since their last call", and dt-updaters "run on a clock of their own".
- "`.animate` applies its calls when its animation starts."
- "Animations of one mobject that start at the same time combine: their movements add."
- `MovingCameraScene` is `Scene`. `ApplyMethod`, `ScaleInPlace` and `FadeToColor` are replaced by `.animate`. Getters such as `get_width()` become properties.

Text is not Pango. ManimGX "sets text and formulas itself, with Typst and fonts that come with it". The default font is New Computer Modern. The text docs do not mention `MarkupText`.

Bug reports that touch ChalkDust's needs (all open when fetched):

| Issue | What it says | Why it matters to ChalkDust |
|---|---|---|
| #50 (20 Oct) | Render crashes on Windows with **Intel Iris Xe**: access violation in `igvk64.dll` (the Intel Vulkan driver). It reproduces with a minimal script, on 0.1.1. | Iris Xe is a typical class-C student laptop GPU. |
| #78 (24 Oct) | Same crash on Intel UHD 600, Windows 11. Forcing `WGPU_BACKEND=dx12` gave one good render, and later attempts still failed. "No CPU rendering fallback option was found in the CLI help." | No way out on weak Windows machines. |
| #97 (12 Nov) | On **Linux**, Intel UHD 620 with Mesa 25.2.8 (ANV Vulkan): a vector-area shader test fails (`0 != 0.4921875`), and regression frames differ ("74 frames look different from its reference"). It is on `main`, not a crash, with no maintainer reply. | Wrong output on an Intel iGPU, through a different driver path. |
| #74 (23 Oct) | `MathTex` with `\frac` plus `tex_to_color_map` crashes with `TypstError: unknown command: \fr`. | EquationDerivation-style colouring. |
| #60 (22 Oct) | Request for a native LaTeX backend because the mitex path fails on valid Manim CE LaTeX (e.g. `M\"obius`). | LaTeX coverage gap. |
| #33 (17 Oct) | `Text(font=...)` silently falls back to New Computer Modern for one of a font's family names, "and the layout check reports no problems". | ChalkDust's theme fonts and its `unrenderable_text` guard. |
| #32 (17 Oct) | `always_redraw` raises RecursionError when a shape becomes an empty VMobject. | Updater-driven components. |
| #37 (17 Oct) | `inspect` under-reports text size in 1080x1920 frames. | Layout checks. |

On Windows, a "Draw with Direct3D 12 on Windows" commit (c217a81) is linked from #50. It landed on `main` on 22 Oct, alongside the "take ... off main for review" commits, and a `fix/windows-direct3d` branch exists. It is in no PyPI release. Whether it is on `main` now is **[unverified]**. **[corrected 2026-11-17: it is not on `main`. It arrived as PR #62, which was merged at 15:40:48Z on 22 Oct without review. At 18:49Z the same day, commit e49a64a ("Take #62 to #66 off main for review") restored main to a7be82b's tree, which removes the `wgpu::Backends::DX12` line from `rust/engine/src/render.rs`. The change went back up as PR #67, "Draw with Direct3D 12 on Windows", which is still open and unmerged. `main`'s head (1fb4dd7, 24 Oct) comes after the revert. Separately, #78 reports that forcing `WGPU_BACKEND=dx12` still failed after one good render, so D3D12 alone may not fix the Intel crashes.]**

On CPU fallback, the docs say only that "Linux wheels bundle Mesa's lavapipe, for Vulkan rendering on the CPU when no suitable GPU is available." Nothing is documented for Windows or macOS.

### 2.5 What switching would touch in ChalkDust

- **The Manim pin and every cache key that carries it** (`core/version.py`, SCENE_SPEC.md §10).
- **The font-resolved theme and the snapshot font fingerprints.** Text would no longer go through Pango, so `theme.check_renderable` and every `tests/snapshots/*.json` layout half would need re-deriving.
- **The `invalid_latex` rung.** `semantic.check_latex_source` compiles raw LaTeX with TeX today. Under mitex, "valid" changes meaning.
- **Timing.** "Animations of one mobject that start at the same time combine" and the updater-clock changes could shift `budget()`-driven segments, which the frame-exact tests (`tests/test_budget_frames.py`) would catch.
- **Overlap with the validator.** `manimgx inspect` does what ChalkDust's geometric probe already does, at sampled moments only.

### 2.6 The GPL wheel in a paid closed-source app

- The wheel is "GPL-3.0-or-later as a whole". Importing it into ChalkDust's Python process means the two run in one address space.
- The GPL FAQ says: "If modules are designed to run linked together in a shared address space, that almost surely means combining them into one program." On dynamic linking: "Linking a GPL covered work statically or dynamically with other modules is making a combined work based on the GPL covered work."
- So shipping ManimGX's native wheel inside the closed app would, on the FSF's reading, put ChalkDust's own code under GPL obligations.
- The way out would be a ManimGX build without x264, so the wheel is MIT plus LGPL FFmpeg. That build would encode with a non-GPL encoder and needs a source build of their Rust core **[unverified that it builds cleanly without x264]**.
- **The same problem already exists in today's Manim CE stack** (§5.1). It is not a reason to prefer Manim CE over ManimGX. It is a reason to fix encoding either way.

### 2.7 Verdict and triggers to re-evaluate

Not now. Re-evaluate ManimGX if all of these hold:

1. A release (not just `main`) fixes the Intel-iGPU crashes on Windows (#50, #78), and the Intel shader bug (#97) is closed. Retest on an Iris Xe laptop. (As of 17 Nov 2026 the candidate fix, PR #67, is open and unmerged: see the correction in §2.4.)
2. A CPU fallback is documented for Windows and macOS.
3. A native LaTeX backend lands (#60), or mitex covers the LaTeX in ChalkDust's examples and stress specs.
4. ChalkDust's 20 components and their stress cases run with no layout or timing change beyond what a re-snapshot explains.

Being owned by a direct competitor adds a supply-chain risk. The code is MIT, so a fork stays possible.

---

## 3. Other engines for a desktop app

| Engine | Licence (quoted) | Status | Speed | Fit for ChalkDust |
|---|---|---|---|---|
| **Manim CE, Cairo** (current) | MIT (§1) | 0.22.1, 27 Oct 2026 | CPU rasteriser. Measured on the ROG in §6. | The baseline. All 20 components, the validator and the snapshots are built on it. |
| **Manim CE, OpenGL renderer** (`--renderer=opengl`) | MIT | Under-documented. The official FAQ says the docs "do not contain the relevant base classes like `OpenGLMobject` and `OpenGLVMobject`". 0.22.0 fixed several OpenGL code paths, including ones "skipped due to a string comparison bug". **[corrected 2026-11-17: that is not the changelog's wording. It reads "Fix OpenGL-specific code paths that were skipped because renderer comparisons against the string `"opengl"` never matched" (PR #4929).]** | Not measured here **[unverified]**. | Weak. It is a parallel mobject hierarchy (`OpenGLMobject`), so components, the layout probe and snapshots would need re-validation, and it needs a working OpenGL context on class-C machines. |
| **ManimGL** | MIT | 1.7.2, Dec 2024 | 11.5x slower than ManimGX on the vendor benchmark, but faster than CE on 3 of 5 scenes. | No. Different API, needs LaTeX, stale. |
| **ManimGX** | MIT code; wheels GPL-3.0-or-later | 0.1.1, 7 Oct 2026 | Fastest on paper (§2.3) | Not now (§2.7). |
| **Motion Canvas** | MIT | npm `@motion-canvas/core` 3.17.2 published 14 Dec 2024; a 3.18.0-alpha.0 from 16 Feb 2025 is the last GitHub release | Renders in the browser to an image sequence (PNG/JPEG/WebP). Video needs an external ffmpeg step. | No. A TypeScript rewrite of every component, no Python, and no maths documented on the rendering page. |
| **Revideo** | MIT ("MIT — see LICENSE") | npm `@revideo/core` 0.11.0, 10 Jul 2026. The repo page now shows `midrender/revideo`. | Headless browser; "runs anywhere Node and a headless browser run" | No, for the same rewrite cost. It would also bundle Chromium in the app. |
| **Remotion** | Source-available, **not** open source (below) | npm `remotion` 4.0.544, 16 Nov 2026 | Headless browser | No. The licence and pricing do not fit a paid desktop app made by a company of more than 3 people (below). |

**Remotion licence, quoted** (`LICENSE.md` on `main`):

- Free licence eligibility: "an individual", "a for-profit organization with up to 3 employees", "a non-profit or not-for-profit organization", or someone "evaluating whether Remotion is a good fit".
- Free licence grant: "to use the software non-commercially or commercially for the purpose of creating videos and images".
- Disallowed: "It is not allowed to copy or modify Remotion code for the purpose of selling, renting, licensing, relicensing, or sublicensing your own derivate of Remotion."
- Anyone else needs a Company License.
- Pricing (pricing page): "Remotion for Automators: $0.01 per render, charged in increments of 1000 renders ($10), with a minimum of $100 per month". It lists "video editors, prompt-to-video apps" as Automator use cases. **[corrected 2026-11-17: the pricing page has a second Company License tier that this note omitted: "Remotion for Creators", at "$25 per month per seat". Its full use-case list is "video editors, prompt-to-video apps, and embedding the Remotion Player". The Free License also covers non-profits and anyone "evaluating whether Remotion is a good fit, and are not yet using it in a commercial way".]**
- FAQ: "Remotion is source-available software, but it is not open source software". "For client-side rendering, there is telemetry enabled by default which cannot be disabled", which would not suit an offline student app.
- The licence header says "In Remotion 5.0, the license will slightly change" (PR #3750). **[unverified: the change was not read.]**

For the four checks: (a) commercial use is allowed only under the free tier (3 people or fewer) or a paid Company License. (b) Shipping a modified Remotion inside a paid app is limited by the "derivate of Remotion" clause. (c) Rendered videos are the purpose of the grant and carry no stated restriction. (d) Per-render fees, a monthly minimum and telemetry.

---

## 4. Maths without a LaTeX install

ChalkDust's validator compiles every LaTeX string. The brief puts a cold run at about 46 s for a 10-beat spec with MiKTeX. A renderer that runs inside the Python process removes the TeX process launches and the MiKTeX install, and makes best-of-N cheaper.

### 4.1 The options

| Tool | Licence | Exam-maths coverage | Output Manim can import | Speed | Package size |
|---|---|---|---|---|---|
| **Typst via Manim CE `MathTypst`** (typst-py) | Typst compiler: Apache-2.0; typst-py: "released under the Apache-2.0 license" | Typst maths has `frac` (and `a/b`), `mat` (matrices), `vec` (column vectors), `cases`, `accent`, and `&` alignment. Integrals appear in the installed Manim docstring example `Typst(r"$ integral_a^b f(x) dif x $")`. Vector arrows over symbols are **[unverified]**. Input is **Typst syntax, not LaTeX.** | **SVG to `SVGMobject` to VMobjects, built in.** Locally, `typst_file_writing.py` calls `typst.compile(..., format="svg")` and caches by content hash in Manim's `tex_dir`. | In-process Rust, no subprocess (local source). Per-expression time not measured **[unverified]**. | `typst` 0.15.2 (5 Nov 2026) wheel: 29.6 MB (Windows x64), 30.4 MB (macOS arm64) |
| **mitex** (LaTeX to Typst, used inside Typst) | Apache-2.0 | Supports `\newcommand`, `\color`/`\textcolor`, "aligned, matrix, and cases". "Packages: not supported yet" (listed as planned). **[corrected 2026-11-17: no such quote exists on the Typst Universe page or the GitHub README. Both list "Package support, which means that you can change set of commands by telling MiTeX to use a list of packages" as an unchecked item under "Features to Implement". The conclusion stands: packages are not supported yet.]** `\def` is not mentioned. | Typst content, then the Typst route above | Its README's own, "not strict" figure: about 32.5k OI Wiki equations converted in about 0.085 s natively (x86) and 2.28 s in WASM, against about 109 s for texmath. Conversion only, not typesetting. | Typst package 0.2.7 (7 Apr 2026): 112 kB; WASM about 185 KB |
| **tylax** (LaTeX to Typst, Python) | Apache-2.0 | "High-performance bidirectional LaTeX <-> Typst converter". Coverage not listed. | Typst source text, then the Typst route | No figures | 0.3.8 (14 Sep 2026); about 1.5 MB abi3 wheels |
| **typsium** (Typst chemistry) | MIT | A `ce` function with reactions, arrow styles (equilibrium and others), charges, oxidation numbers, states (g)/(l)/(s)/(aq), isotopes | Typst content | n/a | Typst package 0.3.2 (15 Jun 2026); minimum Typst 0.13.1 |
| **whalogen** (Typst chemistry, "inspired by mhchem") | Apache-2.0 | `ce("HCl + H2O -> H3O+ + Cl-")` in maths mode | Typst content | n/a | 0.3.0, last updated 10 Feb 2025 (less active) |
| **MathJax** (Node, offline SVG) | Apache-2.0 | The fullest LaTeX coverage here. The mhchem extension "implements the `\ce` and `\pu` chemical equation macros". | SVG via `tex2svg` in Node, serialised with `adaptor.serializeXML` | Not measured **[unverified]** | `mathjax` 4.1.3 (3 Jul 2026), 20 MB unpacked. **Needs a Node runtime bundled in the app** (size [unverified]). |
| **KaTeX** | MIT | Strong LaTeX subset | **No SVG output.** `output` is `html`, `mathml` or `htmlAndMathml`, so a browser is needed to lay it out. | n/a | `katex` 0.19.1 (23 Oct 2026), 4 MB unpacked |
| **ziamath** (pure Python) | MIT | LaTeX "provided by latex2mathml". Not "a comprehensive implementation" of MathML. Matrices, environments and `\ce` are not documented. | SVG "drawn entirely with `<path>` elements" | Not stated | 0.13 (6 Feb 2026), Beta. Bundles the STIX Two Math font. Size [unverified]. |
| **latex2mathml** | MIT | LaTeX to MathML only; it does not draw anything | Needs a MathML renderer, e.g. ziamath | n/a | 3.81.1 (7 Sep 2026) |
| **matplotlib mathtext** | matplotlib licence **[unverified: not fetched]** | "a lightweight TeX expression parser and layout engine". Fractions, radicals, `\left`/`\right`, `\vec`/`\hat` are documented. **No `\begin{...}` environments are documented, so no matrices.** | Glyph paths via matplotlib (an adapter would be needed) | n/a | Large dependency |

### 4.2 The route that fits ChalkDust: Manim CE's own `MathTypst`

What the installed 0.21.0 source shows (`manim/mobject/text/typst_mobject.py`, `manim/utils/typst_file_writing.py`):

- **Compilation.** `Typst` is "compiled to SVG via the `typst` Python package (a self-contained Rust binary extension — no system-level install required) and then imported through `SVGMobject`". This is the same import path `MathTex` uses, so `fit_to_region`, `settle()` and the layout probe see ordinary VMobjects.
- **Caching and the render pool.** It writes `{hash}.typ` and `{hash}.svg` into `config.get_dir("tex_dir")` and reuses an existing SVG. ChalkDust's render pool already gives each worker its own `Tex/` dir (CLAUDE.md, "Render pool"), so the torn-SVG protection carries over.
- **Sub-part selection.** `MathTypst` supports `{{ ... : label }}` groups and `select(label)`, the counterpart of `MathTex` substrings for colouring.
- **Sizing.** Compilation is at 10 pt "so that the post-import scaling based on Manim's font_size property matches TeX / MathTex more closely". A stroke-width correction (`_TYPST_SVG_STROKE_WIDTH_SCALE = 0.5`) makes fraction bars match TeX weight.
- **Gaps to close:**
  1. `typst` is not installed in the venv; `pip install "manim[typst]"` adds it (`typst>=0.14`).
  2. The wrapper passes only `font_paths` to `typst.compile`. typst-py supports `package_path` / `package_cache_path` for "a local package directory", but Manim does not expose them. Using `@preview/mitex` or `@preview/typsium` offline therefore needs either a pre-filled Typst package cache shipped with the app or a small ChalkDust-side wrapper that calls typst-py directly. The first works **[unverified]**: it depends on typst-py's default cache location.
  3. The `MathTypst` capture-group layout fix (PR #4960) is in 0.22.0, not 0.21.0.
  4. The validator needs a Typst equivalent of `check_latex_source`. The exception typst-py raises, and how much of the error it carries, are **[unverified]**. For comparison, ManimGX's own engine surfaced a mitex failure as `TypstError: unknown command: \fr` (#74), and its docs say an unknown command's "error names the command". A new refusal kind (or a renamed `invalid_latex`) needs an owner decision under the "one kind, one place that raises it" rule.
  5. Fonts: Typst output looks like New Computer Modern by default. Matching the theme means passing `font_paths` and `#set text(font: ...)` in `typst_preamble`, and putting the resolved maths font into the render key.

### 4.3 LaTeX in the spec or Typst in the spec?

- **Keep LaTeX, convert with mitex.**
  - What stays the same: the LLM prompt, the training data and any distilled-teacher outputs.
  - Cost: mitex has no package support, so `\ce` (mhchem) almost certainly fails **[inferred from "packages not supported"; not tested]** (that is a paraphrase, not a quote; see the §4.1 correction for the source wording). Exotic macros fail too, as ManimGX's #60 shows for `\"`.
  - Chemistry would need its own field (typsium `ce`) or a pre-pass.
- **Switch to Typst maths.** It is shorter and has no backslash escaping inside JSON strings. Small local models will have seen far less Typst than LaTeX, so error rates may rise **[unverified, measure it]**. The grammar-constrained decoder could carry a Typst-maths grammar.
- **Recommendation.** Measure both on the existing example specs (§8) before choosing. Either way, every string goes through the compile check, which is now cheap enough to run inside best-of-N.

### 4.4 Packaging comparison

- **Typst route.** About 30 MB of wheel per platform (typst-py), plus about 0.1 MB each for mitex and typsium, plus any theme fonts. No other runtime. **[corrected 2026-11-17: Typst Universe lists the mitex 0.2.7 archive at 112 kB and the typsium 0.3.2 archive at 16.4 kB, so typsium is well under 0.1 MB. The total is unchanged in practice.]**
- **MathJax route.** MathJax (20 MB unpacked) plus a bundled Node runtime **[size unverified]**.
- **MiKTeX.** Installer size not fetched **[unverified]**. The current CLAUDE.md shows a "first compile downloads packages", which an offline student install cannot rely on.

---

## 5. Video encoding licences for a paid desktop app

### 5.1 What ChalkDust ships today (local evidence)

- **Manim CE 0.21.0 hard-codes x264 for MP4 output.**
  - `manim/scene/scene_file_writer.py` line 656 sets `partial_movie_file_codec = "libx264"` (with `crf 23`, `yuv420p`), both in the installed copy and in the upstream file at tag v0.21.0.
  - The only alternatives in that function are `libvpx-vp9` when `movie_file_extension == ".webm"` and `qtrle` for transparent output.
  - Encoding happens inside the Python process through PyAV.
- **Manim CE 0.22.x makes the encoder configurable.**
  - The v0.22.1 source has a `[video_encoder]` config section (`codec = auto`, `pixel_format = auto`, plus `[video_encoder.options]`).
  - `auto` still resolves to `libx264` / `yuv420p` for MP4 (`_config/video_encoder.py`, `_default_profile`).
  - Any other name is accepted if PyAV has it as a video encoder and the pixel format is one the encoder supports (`_validate_profile`).
  - So `h264_mf` or `h264_videotoolbox` would be a config value, not a patch. Untested **[unverified: e.g. `h264_mf` may need `nv12`]**.
- **PyAV 19.0.1 bundles GPL encoders.**
  - **[corrected 2026-11-17: 19.0.1 (3 Oct 2026) is what the venv has installed, but it is not the latest. PyPI's latest is 19.0.2 (21 Oct 2026). The upstream build script on `main` still adds `--enable-libx265` unconditionally and `--enable-libx264` on all but 32-bit ARM, so 19.0.2 wheels very likely bundle the same encoders. Its wheel contents were not inspected: [unverified].]**
  - Its dist-info says `License-Expression: BSD-3-Clause` (the binding's source).
  - `av.libs/` ships `libx264-165-*.dll` and `libx265-*.dll`, plus SVT-AV1, libvpx, dav1d and others.
  - The FFmpeg configure line embedded in `avcodec-63-*.dll` includes `--enable-version3 ... --enable-libx265 --enable-libx264 ... --enable-mediafoundation ... --enable-nvenc ... --enable-amf --enable-libvpl`, with no `--enable-gpl`.
  - The same DLL's licence string reads "LGPL version 3 or later".
  - Its encoder list includes `h264_mf`, `hevc_mf`, `av1_mf`, `h264_nvenc`, `h264_qsv`, `h264_amf`, `h264_d3d12va`, `libsvtav1` and `libx264`.
  - The upstream build script (`pyav-ffmpeg/scripts/build-ffmpeg.py`) matches: it appends `--enable-libx265`, and `--enable-libx264` on all but 32-bit ARM, without `--enable-gpl`.
- **The PyAV maintainer's position, from issue #2270** (closed, then locked 28 Oct 2026):
  - "The license is for the source code, not the binary wheels."
  - The wheels are "'LGPLv2.1 or later + Commercial exceptions' OR 'GPLv2 or later'".
  - On applying the x264/x265 commercial exception: "it's implied — either by your actions (i.e. not sharing full source code, like the GPL requires) or you explicitly state the license you're choosing."
  - Also: "The x265 org negotiates those on a case by case basis."
- **Why that position is not enough to rely on.** x264's own page says it "is released under the terms of the GNU GPL" (v2) and "is also available under a commercial license", via `x264licensing@videolan.org`. A commercial licence is a contract you sign, not something implied by not shipping source. **Treat the PyPI PyAV wheel as GPL-encumbered for a closed app.** (This is a reading of the sources, not legal advice.)
- **The CLI on the lab machine is a GPL build.** ChalkDust shells out to `ffmpeg` 8.1 "full_build-www.gyan.dev", whose configuration starts with `--enable-gpl --enable-version3`.

### 5.2 What ffmpeg.org/legal.html says (fetched, quoted)

- "FFmpeg is licensed under the GNU Lesser General Public License (LGPL) version 2.1 or later. However, FFmpeg incorporates several optional parts and optimizations that are covered by the GNU General Public License (GPL) version 2 or later. If those parts get used the GPL applies to all of FFmpeg."
- "FFmpeg is not available under any other licensing terms, especially not proprietary/commercial ones, not even in exchange for payment."
- LGPL compliance checklist, in short:
  - Compile without `--enable-gpl` and `--enable-nonfree`.
  - Link dynamically (DLLs).
  - Distribute the exact corresponding source, with a `changes.diff` and the configure line, hosted on the same server as the binary.
  - Put the notice "This software uses code of FFmpeg licensed under the LGPLv2.1 and its source can be downloaded here" on every download page.
  - Mention FFmpeg in the about box and the EULA. "Remove any prohibition of reverse engineering from your EULA."
  - Do not rename the DLLs to obfuscated names.
  - "Make sure your program is not using any GPL libraries (notably libx264)."
- Patents: "We do not know, we are not lawyers". For a commercial product: "You might have a problem here ... once you start trying to make money from patented technologies, the owners of the patents will come after their licensing fees. Notably, MPEG LA is vigilant and diligent about collecting for MPEG-related technologies."

### 5.3 GPL, the CLI, and the videos themselves

- **A separate ffmpeg.exe called by command line.** GPL FAQ: "pipes, sockets and command-line arguments are communication mechanisms normally used between two separate programs", and an aggregate with non-free programs is permitted. Shipping a GPL ffmpeg.exe next to a closed app is therefore defensible, but you still owe that binary's GPL source offer, and the patent question is unchanged.
- **The same library inside your Python process.** As in §2.6, "almost surely" one program.
- **The rendered videos.** GPL FAQ: "copyright law does not give you any say in the use of the output people make from their data using your program". So students' videos, and any videos ChalkDust uses as training data, are not GPL-encumbered by the encoder.

### 5.4 Patents (H.264, HEVC) and patent-free codecs

- **H.264 (Via LA's AVC/H.264 programme)**, fees page:
  - "Decoder, encoder or product consisting of one decoder and one encoder = "unit"."
  - Units 1 to 100,000 per year cost $0.00 ("Available to one legal entity in an affiliated group"), then $0.20 each to 5,000,000, then $0.10, with a cap of "$9.75M per year in 2017 and after".
  - The same tiers apply to "Products incorporated into a personal computer's operating system at the time of sale".
  - Via LA's FAQ says "It is up to each user to determine its own licensing needs." **[corrected 2026-11-17: the sentence is on the AVC/H.264 licensing page cited here, not in an FAQ that was fetched. In full it reads "It is up to each user to determine its own licensing needs and how it wishes to meet them."]**
  - Whether ChalkDust needs its own AVC licence when it calls the OS's encoder rather than shipping one is **not established by these sources**. Ask counsel. A zero-fee tier still usually means signing a licence and reporting units **[unverified]**.
- **OpenH264 (Cisco).**
  - Source licence: the "Two-Clause BSD license".
  - Royalty coverage: "Cisco is only covering the licensing fees for its own binary module", and products "must download it at the time the product or project is installed".
  - Self-compiled builds are not covered: "the team is responsible for paying all applicable license fees".
  - So it works only with an install-time download of Cisco's binary. FFmpeg's `libopenh264` wrapper exists; its quality against the OS encoders is untested **[unverified]**.
- **HEVC.**
  - Via LA's HEVC/VVC page says the programme was acquired by Access Advance on 15 Dec 2025 and is now run by VCL Advance.
  - Terms: $0.00 for "the first 1 to 100,000 units" (one Legal Entity per affiliated group), then $0.30 a unit in R1 regions (US, Canada, Europe, Japan, South Korea and others) and $0.20 elsewhere, with a cap of $30,000,000 per enterprise. It says the licence "extends to devices implementing the technology". How software-only products are treated is not stated.
  - Other HEVC pools were not checked **[unverified]**.
  - x265's `COPYING` is GPL v2 and adds: "This program is also available under a commercial proprietary license. For more information, contact us at license @ x265.com."
  - Avoid HEVC. It gives no benefit for this content and adds a second pool plus GPL x265.
- **AV1.**
  - AOMedia Patent License 1.0 grants "a non-sublicensable, perpetual, worldwide, non-exclusive, no-charge, royalty-free ... patent license ... to make, use, sell, offer for sale, import or distribute any Implementation". Defensive termination applies if you sue claiming an Implementation infringes.
  - SVT-AV1 (already in PyAV's build) is under the "BSD 3-Clause Clear License".
  - libaom's `LICENSE` is a two-clause BSD-style text, "Copyright (c) 2016, Alliance for Open Media". It says "Redistributions of source code must retain the above copyright notice", with the same for binaries in their documentation.
  - Playback of AV1 MP4s on students' default players (Windows needs an AV1 extension; older Macs lack hardware AV1 decode) is **[unverified]**. Too risky as the only output.
- **VP9 (libvpx).**
  - Google's WebM patent grant is "perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable", "to make, have made, use, offer to sell, sell, import, transfer". It terminates if you sue.
  - Manim CE already encodes `.webm` partials with `libvpx-vp9` with no code change (`movie_file_extension`).
  - libvpx's `LICENSE` is a BSD 3-clause text. The third clause is non-endorsement: "Neither the name of Google, nor the WebM Project, nor the names ...".

### 5.5 OS encoders

- **Windows.**
  - Microsoft's Media Foundation H.264 encoder (`Mfh264enc.dll`) supports Baseline, Main and High ("High profile (requires Windows 8)"); minimum client is Windows 7.
  - "If a certified hardware encoder is present, it will generally be used instead of the inbox system encoder."
  - FFmpeg exposes it as `h264_mf`, which is present in both PyAV's Windows build and the local ffmpeg 8.1.
  - Windows N editions lacking Media Foundation is **[unverified]**.
- **macOS.** pyav-ffmpeg's Darwin build adds `--enable-videotoolbox` (build script). Apple's VideoToolbox page could not be read **[unverified: availability details]**.
- **Quality.** Hardware encoders at matched bitrate against x264 CRF 23 on flat 3b1b-style frames are untested **[unverified]**. Measure (§8).

### 5.6 Encoding options for the paid app

| Option | Licence position | Engineering cost | Notes |
|---|---|---|---|
| **A. LGPL FFmpeg + PyAV built without x264/x265; encode via `h264_mf` / `h264_videotoolbox`** (recommended) | LGPL only. Follow the §5.2 checklist. | Build and ship your own pyav-ffmpeg variant: the upstream script has no LGPL switch. Then pick the encoder: in Manim 0.22.x through `[video_encoder] codec`; in 0.21.0 by patching or subclassing the file writer. Alternative: render `.webm` (VP9) partials, which Manim supports today, and do the final H.264 encode in ChalkDust's mux stage. | An LGPL-only fork exists (JScheumann/pyav-ffmpeg-lgpl), but it decodes VP8/VP9 only and has no encoders, so it is a template, not a solution. |
| **B. Keep x264; buy x264 (and x265, or drop it) commercial licences** | x264 "is also available under a commercial license" via `x264licensing@videolan.org`; x265 via `license @ x265.com` | Lowest. The current code is unchanged. | Licensor, terms and price unknown **[unverified]**. H.264 patents still apply. |
| **C. OpenH264 Cisco binary, downloaded at install** | BSD-2 source; Cisco covers the royalties for its binary | Medium: install-time download, FFmpeg `libopenh264` | Breaks a fully offline installer. |
| **D. VP9/WebM or AV1 output only** | Royalty-free patent grants (§5.4) | Low for VP9 (Manim supports `.webm`) | Student playback and sharing (WhatsApp, PowerPoint) compatibility **[unverified]**. libvpx encoding speed **[unverified]**. |
| **E. Ship GPL ffmpeg.exe and call it as a separate program** | Aggregate under the GPL FAQ; GPL source offer owed for ffmpeg.exe | Low for the mux and concat steps | Does not fix Manim's in-process x264. Patents unchanged. |

---

## 6. Rendering speed: what is realistic

### 6.1 Measured data

Only two sets exist:

1. **The ChalkDust lab ROG** (`.run/evidence/last-render/REPORT.md`, `logs/summary.tsv`).
   - Setup: Intel Core Ultra 9 275HX, 24 cores. Manim CE 0.21.0, Cairo, default render pool, MiKTeX, Python 3.13.12. Cold cache, one render after another. **[corrected 2026-11-17: REPORT.md does not name the CPU. A read-only `Win32_Processor` query on the lab machine confirms "Intel(R) Core(TM) Ultra 9 275HX", 24 cores and 24 logical processors. The "default render pool" was not 24 workers: each `logs/*.stdout.log` says "rendering 9 beat(s) across 9 processes", or 10 for jee_incline_pulley. That is one worker process per beat, as CLAUDE.md describes the default, not one per core. How many cores those 9 or 10 processes kept busy (encoder threads included) is [unverified]: the logs record the process count only.]**
   - **Three 3-minute specs (9 to 10 beats, 170 to 180 s of video) at final 1080p60 took 123.4 s, 138.5 s\* and 148.7 s.** That is about 72 to 83 encoded frames per second across the machine.
   - At draft 480p15 they took 52.6 s, 37.4 s\* and 71.8 s\*. (\* = slightly inflated by a concurrent analysis process.)
   - **Both runs include the same fixed work.** `pipeline.render` first runs `validate()` (schema, semantic and LaTeX checks, the geometric probe), then TTS for every beat, then the beat render. No 480p15 pass happens inside a final run.
   - Draft has about 20x fewer pixels per second of video than final, yet takes 30 to 50% of final's time. So **the draft figure is mostly fixed cost** (validation with LaTeX compiles, TTS, process start-up), and per spec the extra cost of 1080p60 rasterising and encoding is about 77 to 86 s (final minus draft). **[unverified: the logs have no per-stage timings, so the split between validation, TTS and rendering is inferred, not measured. The arithmetic checks out: 52.6/138.5, 37.4/123.4 and 71.8/148.7 are 30 to 48%, and final minus draft is 76.9 to 86.0 s.]**
   - Best-of-N runs the validator, so the maths change in §4 speeds it up more than any rasteriser change would. That conclusion is an inference from these numbers, not a profile.
2. **The ManimGX vendor benchmark** (§2.3): M4 Pro, 1080p60, short scenes. Not comparable with ChalkDust's beats.

### 6.2 Levers inside Manim CE

- **Render pool.** Beats are independent, and the pool already parallelises them (ARCHITECTURE.md §4). On 24 cores this is the reason a 3-minute 1080p60 video takes about 2 minutes. **[corrected 2026-11-17: the measured runs used 9 or 10 worker processes, one per beat, not 24 (see §6.1). The 2-minute figure is what 9 or 10 parallel beats gave on this CPU. Whether beats per spec rather than cores set the ceiling on the ROG is [unverified]: per-process core use was not measured.]** Cairo is single-threaded per process **[unverified for 0.21; consistent with the pool design]**, so speed scales with core count, and memory caps the number of workers on 8 GB machines **[per-worker RSS unmeasured]**.
- **Cairo vs OpenGL.** No current Cairo-vs-OpenGL measurement for Manim CE was found. ChalkDust's components target Cairo mobjects. 0.21.0's Cairo speedup ("2.2x on animation-heavy scenes") is already in the pin.
- **Parallel encoding** (`max_inflight_encoders` > 1, from 0.21.0): "roughly 8–11%" on larger real renders. Cheap to try.
- **Resolution and frame-rate tiers.** Frames per 3 minutes: 10,800 at 60 fps, 5,400 at 30 fps. Pixels per frame: 2.07 M at 1080p, 0.92 M at 720p, 0.41 M at 480p. Rasterisation and encoding scale roughly with pixels × frames, and per-frame Python work scales with frames only. So **1080p30 is close to half the cost of 1080p60, and 720p30 less again** (not linear; measure). `render/worker.TIERS` already makes the tier part of the render key, and `budget()` works in whole frames at any fps.

### 6.3 Estimates for the target machines

These are back-of-envelope figures, **not measurements**. Method:

- Split each ROG run into two parts:
  - fixed work F, taken as the draft time (37 to 72 s);
  - 1080p60 rasterising and encoding R60, taken as final minus draft (77 to 86 s).
- Scale R60 by the frame count of the target tier. Assume 720p30 costs about 0.5x to 0.6x of 1080p30 rather than the 0.44x pixel ratio, because per-frame Python work does not shrink with resolution.
- F is partly single-threaded (TTS, validation), so scale it by single-core speed. R60 is spread over the pool, so scale it by multi-core throughput.
- The draft row is F plus a small render, so no tier can come out faster than it.
- **[corrected 2026-11-17: the multi-core ratios below compare against all 24 cores, but the ROG runs used 9 or 10 worker processes (§6.1). How many cores those processes kept busy was not measured. If it was well under 24, a 4 to 8 core laptop's ratio would be higher than 0.1 to 0.25, and the B and C render figures would be pessimistic. This is [unverified]; §8.5 measures it.]**

| Machine | Assumed speed vs the ROG (single-core / multi-core) | Draft 480p15 | 3-min 1080p60 | 3-min 1080p30 | 3-min 720p30 |
|---|---|---|---|---|---|
| A. ROG, Core Ultra 9 275HX, 24 cores | 1 / 1 (measured) | **0.6 to 1.2 min (measured)** | **2.0 to 2.5 min (measured)** | about 1.3 to 2 min | about 1 to 1.6 min |
| B. MacBook M1, 16 GB (4P+4E cores) | about 0.6 to 0.8 / 0.25 to 0.4 | about 1 to 2 min | about 4 to 8 min | about 2.5 to 5 min | about 1.5 to 4 min |
| C. Student laptop, 4 to 8 cores, 8 to 16 GB, iGPU | about 0.4 to 0.7 / 0.1 to 0.25 (fewer workers on 8 GB) | about 1 to 3 min | about 6 to 17 min | about 3.5 to 10 min | about 2 to 7 min |

The swap from MiKTeX to Typst (§4) cuts into F, which is the whole of the draft row and a large share of the lower tiers. The token generation for the LLM steps comes on top of all of this (see the October notes). On class C the sensible defaults are 720p30 or 1080p30, a pool capped by free RAM, and the paid cloud tier for 1080p60.

---

## 7. Licence summary for everything that could ship

The four checks are: (a) is commercial use allowed; (b) can a modified copy ship inside a paid closed-source app; (c) for these rendering tools, are the **rendered outputs** encumbered, which decides whether ChalkDust's videos can be sold or used as training data; (d) what obligations come with it.

| Component | Licence | (a) | (b) | (c) | (d) and the operative clause |
|---|---|---|---|---|---|
| Manim CE | MIT | Yes | Yes | No | Keep the notices: "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software." Two notices: 3Blue1Brown LLC (2018) and the Manim Community Developers (2024). |
| ManimGL | MIT | Yes | Yes | No | Same MIT notice. |
| ManimGX source | MIT, "Copyright (c) 2026 Academa, Inc." | Yes | Yes (source) | No | MIT notice. |
| ManimGX native wheels | "GPL-3.0-or-later as a whole" | Yes, but GPL | **Not closed.** In-process linking makes a combined work (GPL FAQ). | No (GPL FAQ, output) | Full corresponding source for the combined work; GPL terms for users. |
| typst-py and the Typst compiler, mitex, whalogen, tylax, MathJax | Apache-2.0 | Yes | Yes | No | §2: "perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of ... and distribute". §4: "(a) You must give any other recipients of the Work or Derivative Works a copy of this License", "(b) ... modified files to carry prominent notices", and "(d)" reproduce any NOTICE file. |
| typsium, KaTeX, ziamath, latex2mathml, Motion Canvas, Revideo | MIT | Yes | Yes | No | MIT notice. |
| FFmpeg, LGPL build | LGPL-2.1-or-later (v3 if `--enable-version3`) | Yes | Yes, if dynamically linked and the §5.2 checklist is followed | No | Source hosting, a notice on download pages, about box, EULA terms, no reverse-engineering ban, no GPL libs "notably libx264". |
| FFmpeg, GPL build / x264 | GPL v2 (or later) | Yes | **No**, unless it is a separate program (aggregate) or under a commercial x264 licence | No | x264 "is also available under a commercial license" (`x264licensing@videolan.org`). |
| PyAV 19.0.1 PyPI wheel | Source BSD-3-Clause; wheel bundles x264/x265 | Yes | **Treat as no** (§5.1) | No | Maintainer: "'LGPLv2.1 or later + Commercial exceptions' OR 'GPLv2 or later'". |
| OpenH264 | BSD-2-Clause, plus Cisco's binary patent coverage | Yes | Yes | No | Royalty coverage only for Cisco's binary, downloaded "at the time the product or project is installed". |
| SVT-AV1 | BSD 3-Clause Clear, plus the AOM Patent License 1.0 | Yes | Yes | No | The AOM patent licence ends on patent aggression. |
| libvpx (VP9) | BSD 3-clause source, plus the Google WebM patent grant | Yes | Yes | No | Keep the copyright notice with binaries; no endorsement by "Google, nor the WebM Project". The patent grant ends on patent litigation. |
| libaom (AV1) | Two-clause BSD-style ("Copyright (c) 2016, Alliance for Open Media"), plus the AOM Patent License 1.0 | Yes | Yes | No | Keep the notice in source and binary documentation. |
| x265 | GPL v2 | Yes | **No**, unless under a commercial licence | No | "This program is also available under a commercial proprietary license ... license @ x265.com." |
| Remotion | Source-available Remotion License | Only free for 3 people or fewer; otherwise a paid Company License | Restricted ("not allowed ... selling ... your own derivate of Remotion") | No stated restriction | $0.01/render, $100/month minimum (Automators); telemetry on client-side rendering "cannot be disabled". |
| H.264 patents (Via LA) | Patent pool | n/a | n/a | n/a | $0 for the first 100,000 units/yr per legal entity, $0.20 after. Whether OS-encoder use needs ChalkDust's own licence is unresolved. Ask counsel. |
| HEVC patents (VCL Advance, ex-Via LA) | Patent pool | n/a | n/a | n/a | $0 for the first 100,000 units, then $0.30 (R1) or $0.20 (R2) per unit, with a $30M cap. Avoid HEVC. |

Font licences for whatever is embedded as glyph paths in the SVGs (Typst's default New Computer Modern, STIX Two in ziamath, the theme fonts) were not checked **[unverified]**.

---

## 8. Measurements to run back at the machine

None of these were run for this note. They are listed in priority order. Each one answers a question above.

1. **Typst vs MiKTeX compile time.**
   - Setup: in a scratch venv (not the repo's), install `manim[typst]==0.21.0`. Collect every maths string in `examples/` and the component `stress()` cases.
   - Measure: time `MathTex` (MiKTeX) against `MathTypst` cold and warm, per expression and for a full 10-beat validator pass. Compare with the about 46 s cold LaTeX figure.
   - Record failures by kind.
2. **LaTeX-to-Typst coverage.** Run the same strings through mitex 0.2.7 and through tylax. Count the ones that fail or render differently: overlay the SVGs and compare bounding boxes with the layout probe. Count the chemistry strings separately.
3. **Small-model maths syntax.** Use the local model candidates from the October notes. On the same prompts, compare the valid-maths rate for LaTeX output against Typst output, with and without grammar-constrained decoding.
4. **Encoder swap.**
   - Encode the three `last-render` finals' frames with `h264_mf` on the ROG (and `h264_videotoolbox` on the M1) at a few bitrates against x264 CRF 23.
   - Compare file size, a VMAF or visual check on text edges, and encode time.
   - Also time `.webm` (libvpx-vp9) partials, which Manim supports without a code change.
5. **Tier timings.** Re-run the three example specs cold at 1080p30 and 720p30 on the ROG, then on an M1 16 GB and on one 8 GB Intel laptop. Record wall time and peak memory per pool worker, so the estimates in §6.3 become measurements.
6. **ManimGX smoke test, optional, in a throwaway venv.**
   - Port two components (EquationDerivation, GraphPlot) to ManimGX on the ROG and on an Iris Xe laptop.
   - Record crashes, render time against Manim CE at 1080p60, and any layout differences.
   - Install the wheel only in that throwaway venv: never ship it, because of the GPL.

---

## 9. Not verified

- Vector-arrow accents in Typst maths: integrals are confirmed by Manim's docstring; arrows are not.
- The exception type typst-py raises on a bad formula.
- Whether `h264_mf` / `h264_videotoolbox` work as Manim 0.22.x `[video_encoder] codec` values, and with which pixel format.
- Typst, MathJax and ziamath time per expression: no source found. See §8.1.
- Whether ManimGX's Direct3D 12 change is on `main` now, and whether it fixes #50 and #78. **[corrected 2026-11-17: the first half is now answered. It is not on `main`: it was reverted on 22 Oct and is waiting as open PR #67 (§2.4). Whether it fixes #50 and #78 remains unverified.]**
- That a pre-filled Typst package cache lets Manim's `Typst` class load `@preview/mitex` offline.
- That mitex cannot handle `\ce`: inferred from "packages not supported" (a paraphrase; see the §4.1 correction), not tested.
- Whether a separate x264 licence is needed if ChalkDust uses only OS encoders. Via LA's terms beyond the fee table were not read.
- x264 and x265 commercial licence price and terms, and who the x264 licensor is.
- HEVC pools other than VCL Advance (ex-Via LA), and how software-only products are counted. The matplotlib licence was not fetched.
- The Apple VideoToolbox overview (the page did not render) and Windows N edition Media Foundation availability.
- Node runtime size for the MathJax route, the MiKTeX installer size and the ziamath package size.
- Remotion 5.0 licence changes (PR #3750, not read).
- tylax using mitex's parser: search snippet only. The DT_NEEDED analysis of PyAV wheels: search snippet only. The local DLL evidence in §5.1 stands on its own.
- Manim CE OpenGL vs Cairo speed for ChalkDust's content.
- All machine B and C timings in §6.3: estimates.

---

## Sources

Fetched 17 Nov 2026.

**ManimGX**
- https://github.com/academa-labs/manimgx
- https://github.com/academa-labs/manimgx/blob/main/LICENSE
- https://github.com/academa-labs/manimgx/tree/main/benchmarks
- https://github.com/academa-labs/manimgx/commits/main
- https://github.com/academa-labs/manimgx/issues
- https://github.com/academa-labs/manimgx/issues?q=is%3Aissue+is%3Aclosed
- https://github.com/academa-labs/manimgx/issues/33
- https://github.com/academa-labs/manimgx/issues/50
- https://github.com/academa-labs/manimgx/issues/60
- https://github.com/academa-labs/manimgx/issues/74
- https://github.com/academa-labs/manimgx/issues/78
- https://github.com/academa-labs/manimgx/issues/97
- https://api.github.com/repos/academa-labs/manimgx (also `/commits` and `/branches`)
- https://pypi.org/project/manimgx/
- https://manimgx.academa.ai
- https://manimgx.academa.ai/user-guide/coming-from-manim-ce/
- https://manimgx.academa.ai/user-guide/rendering/
- https://manimgx.academa.ai/user-guide/text-and-math/
- https://manimgx.academa.ai/changelog/
- https://hn.nuxt.dev/item/49091703 (Show HN thread for Academa's browser Manim)
- Added in verification, 17 Nov 2026:
  - https://raw.githubusercontent.com/academa-labs/manimgx/main/README.md
  - https://api.github.com/repos/academa-labs/manimgx/commits?sha=main&per_page=30
  - https://api.github.com/repos/academa-labs/manimgx/commits/c217a81d0e62b72a2139ea9073a1e4996a087751
  - https://api.github.com/repos/academa-labs/manimgx/commits/e49a64a (the "Take #62 to #66 off main for review" revert)
  - https://api.github.com/repos/academa-labs/manimgx/commits/1eb91c0
  - https://api.github.com/repos/academa-labs/manimgx/commits/59a0719 (head of `fix/windows-direct3d`)
  - https://api.github.com/repos/academa-labs/manimgx/pulls/62
  - https://api.github.com/search/issues?q=repo:academa-labs/manimgx+Direct3D (PR #62 merged and PR #67 open)
  - https://api.github.com/repos/academa-labs/manimgx/pulls?state=all&per_page=30
  - https://api.github.com/repos/academa-labs/manimgx/commits?sha=studio-ready&per_page=1 (also `present-in-window`, `player-failure`, `time/one-evaluator`)

**Manim CE and ManimGL**
- https://github.com/ManimCommunity/manim/releases
- https://github.com/ManimCommunity/manim/blob/main/LICENSE.community
- https://pypi.org/project/manim/
- https://docs.manim.community/en/stable/changelog/0.21.0-changelog.html
- https://docs.manim.community/en/stable/changelog/0.22.0-changelog.html
- https://docs.manim.community/en/stable/reference/manim.mobject.text.typst_mobject.html
- https://docs.manim.community/en/stable/faq/opengl.html
- https://raw.githubusercontent.com/ManimCommunity/manim/v0.21.0/manim/scene/scene_file_writer.py (line 656, `libx264`)
- https://raw.githubusercontent.com/ManimCommunity/manim/v0.21.0/manim/utils/typst_file_writing.py
- https://raw.githubusercontent.com/ManimCommunity/manim/v0.22.1/manim/_config/video_encoder.py
- https://raw.githubusercontent.com/ManimCommunity/manim/v0.22.1/manim/_config/default.cfg (`[video_encoder]`)
- https://raw.githubusercontent.com/ManimCommunity/manim/v0.22.1/manim/_config/utils.py (`video_codec`)
- https://github.com/3b1b/manim
- https://pypi.org/project/manimgl/
- Added in verification, 17 Nov 2026: https://pypi.org/project/manim/#history, https://raw.githubusercontent.com/ManimCommunity/manim/v0.22.1/manim/utils/typst_file_writing.py (still passes only `font_paths`)

**Other engines**
- https://github.com/motion-canvas/motion-canvas
- https://github.com/motion-canvas/motion-canvas/releases
- https://motioncanvas.io/docs/rendering/
- https://github.com/redotvideo/revideo
- https://raw.githubusercontent.com/remotion-dev/remotion/main/LICENSE.md
- https://www.remotion.dev/docs/license/pricing.md
- https://www.remotion.dev/docs/license/faq.md
- npm registry metadata (version, date, licence, unpacked size): https://registry.npmjs.org/remotion, https://registry.npmjs.org/@revideo%2Fcore, https://registry.npmjs.org/@motion-canvas%2Fcore, https://registry.npmjs.org/mathjax, https://registry.npmjs.org/@mathjax%2Fsrc, https://registry.npmjs.org/katex

**Maths renderers**
- https://pypi.org/project/typst/
- https://github.com/messense/typst-py
- https://github.com/typst/typst
- https://github.com/typst/typst/releases
- https://typst.app/docs/reference/math/
- https://github.com/mitex-rs/mitex
- https://typst.app/universe/package/mitex
- https://pypi.org/project/tylax/
- https://typst.app/universe/package/typsium
- https://typst.app/universe/package/whalogen
- https://github.com/mathjax/MathJax-src
- https://github.com/mathjax/MathJax-src/releases
- https://docs.mathjax.org/en/latest/input/tex/extensions/mhchem.html
- https://github.com/KaTeX/KaTeX
- https://katex.org/docs/options
- https://pypi.org/project/ziamath/
- https://ziamath.readthedocs.io/en/latest/
- https://ziamath.readthedocs.io/en/latest/options.html
- https://pypi.org/project/latex2mathml/
- https://matplotlib.org/stable/users/explain/text/mathtext.html
- Added in verification, 17 Nov 2026: https://ziamath.readthedocs.io/en/stable/_sources/options.rst.txt (the latex2mathml and "not a comprehensive implementation" sentences), https://github.com/cdelker/ziamath, https://api.github.com/repos/remotion-dev/remotion/releases?per_page=3

**Encoding, licences and patents**
- https://ffmpeg.org/legal.html
- https://github.com/PyAV-Org/pyav-ffmpeg
- https://raw.githubusercontent.com/PyAV-Org/pyav-ffmpeg/main/scripts/build-ffmpeg.py
- https://github.com/PyAV-Org/PyAV/issues/2270
- https://github.com/JScheumann/pyav-ffmpeg-lgpl
- https://www.videolan.org/developers/x264.html
- https://www.openh264.org/faq.html
- https://www.via-la.com/licensing-2/avc-h%20264/
- https://www.via-la.com/licensing/hevc
- https://bitbucket.org/multicoreware/x265_git/raw/master/COPYING
- https://aomedia.org/license/patent-license/
- https://aomedia.googlesource.com/aom/+/refs/heads/main/LICENSE
- https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/master/LICENSE.md
- https://chromium.googlesource.com/webm/libvpx/+/refs/heads/main/LICENSE
- https://chromium.googlesource.com/webm/libvpx/+/refs/heads/main/PATENTS
- https://learn.microsoft.com/en-us/windows/win32/medfound/h-264-video-encoder
- https://www.gnu.org/licenses/gpl-faq.html (sections MereAggregation, GPLStaticVsDynamic, GPLOutput)
- https://www.apache.org/licenses/LICENSE-2.0.txt
- https://chromium.googlesource.com/webm/libvpx/+/refs/heads/main/LICENSE (fetched in verification)
- https://pypi.org/project/av/ (added in verification: 19.0.2 is the latest)

**Local evidence on the lab machine (paths relative to the repo root)**
- `.venv/Lib/site-packages/manim-0.21.0.dist-info/METADATA` and `licenses/LICENSE`, `licenses/LICENSE.community`
- `.venv/Lib/site-packages/manim/mobject/text/typst_mobject.py`, `.venv/Lib/site-packages/manim/utils/typst_file_writing.py`
- `.venv/Lib/site-packages/manim/scene/scene_file_writer.py` (line 656, `libx264`)
- `.venv/Lib/site-packages/av-19.0.1.dist-info/METADATA`; `.venv/Lib/site-packages/av.libs/` (DLL list, embedded configure line and licence string in `avcodec-63-*.dll` / `avutil-61-*.dll`)
- `ffmpeg -version` / `ffmpeg -encoders` for the gyan.dev 8.1 full build on PATH
- `.run/evidence/last-render/REPORT.md`, `.run/evidence/last-render/logs/summary.tsv`
- `pyproject.toml`, `chalkdust/render/worker.py` (`TIERS`), `chalkdust/pipeline.py` (`render`: validate, then TTS, then beat render), `docs/ARCHITECTURE.md` §4, `CLAUDE.md`
- Added in verification: `.run/evidence/last-render/logs/*.stdout.log` ("rendering 9 beat(s) across 9 processes"), a read-only `Get-CimInstance Win32_Processor` query (CPU model and cores), and a byte search of `av.libs/avcodec-63-*.dll` for the configure and licence strings

---

## Verification (17 Nov 2026)

An adversarial check by a second agent. It fetched every source behind the 15 decision-critical claims and spot-checked about 40 other facts. Nothing was installed, rendered or downloaded. The local checks were read-only file reads, `ffmpeg -version` and one CPU query.

**Outcome.** All 15 decision-critical claims hold in substance. The recommendations in §0 stand. One claim was wrong on a fact that matters: ManimGX's Direct3D 12 fix is **not** on `main`, because it was reverted the day it landed. Several quotes were misattributed or paraphrased as if verbatim. The §6 speed framing assumed 24-way parallelism, but the logs show 9 or 10 worker processes. Per-process core use was not measured.

**Corrected in place** (each marked `[corrected 2026-11-17: ...]`):

| # | Where | What was wrong | Source |
|---|---|---|---|
| 1 | §2.4, §2.7, §9 | The D3D12 fix (c217a81, PR #62) was reverted from `main` by e49a64a on 22 Oct. It is open as PR #67. | GitHub API commits e49a64a, pulls/62, search for "Direct3D" |
| 2 | §2.2 | The 16 Nov `pushed_at` is a dependabot push. The `fix/windows-direct3d` head is from 22 Oct. | GitHub API repo, branches, commit 59a0719 |
| 3 | §1 | The configurable `[video_encoder]` arrived in 0.22.0, not 0.22.1. 0.22.0 also added CLI flags and encoder settings in cache keys. | 0.22.0 changelog, GitHub releases |
| 4 | §1 | "Preserved" should be "Preserve" (PR #4960). | 0.22.0 changelog |
| 5 | §3 | The OpenGL "string comparison bug" quote was not verbatim (PR #4929). | 0.22.0 changelog |
| 6 | §2.1 | The "90.6× faster" and "3.08 seconds" quotes are from the README, not `benchmarks/`. | ManimGX README (raw) |
| 7 | §2.3 | The "encodes static frames once as longer frames" quote was not verbatim. | ManimGX benchmarks page |
| 8 | §4.1 | mitex's "Packages: not supported yet" quote does not exist. Package support is an unchecked "Features to Implement" item. | Typst Universe mitex, GitHub mitex-rs/mitex |
| 9 | §2.2 | Contributors: five author logins in the 24 newest `main` commits, sinaatalay 15 of 24. | GitHub API commits |
| 10 | §4.4 | typsium is 16.4 kB, not about 0.1 MB. | Typst Universe typsium |
| 11 | §5.1 | PyAV 19.0.2 (21 Oct 2026) is newer than the installed 19.0.1. | pypi.org/project/av |
| 12 | §5.4 | The Via LA "FAQ" quote is on the AVC/H.264 licensing page, and it was truncated. | via-la.com AVC page |
| 13 | §0, §6.1, §6.2, §6.3 | The ROG runs used 9 or 10 worker processes, not 24. Per-process core use is unmeasured. The CPU is not named in REPORT.md and was confirmed locally. | local logs, `Win32_Processor` (no URL) |
| 14 | §3 | Remotion's pricing has a Creators tier ($25 per month per seat) that the note omitted. | remotion.dev pricing |

**Confirmed against the cited source** (summary):

- **Manim CE.** MIT licence and condition text. 0.22.1 (27 Oct), 0.22.0 (8 Oct) and 0.21.0 (10 Aug 2026), nothing newer. Requires-Python >=3.11, extras `gui`/`jupyterlab`/`typst`, `typst>=0.14`. 0.21.0's Typst support, 2.2x Cairo speed-up, 56% and "roughly 8–11%" encoding figures, and `pip install "manim[typst]"`. `typst_file_writing.py` passes only `font_paths` in both v0.21.0 and v0.22.1. `libx264` at line 656, locally. `video_encoder.py` `auto` resolves to libx264/yuv420p, and `_validate_profile` works as described. default.cfg has `[video_encoder]`, `[video_encoder.options]`, `[media]` and `format`. The OpenGL FAQ quote.
- **typst-py.** 0.15.2 (5 Nov 2026), Apache-2.0, 29.6 MB win_amd64 and 30.4 MB macOS arm64 abi3 wheels, `package_path`/`package_cache_path`. The Typst compiler is Apache-2.0.
- **mitex and Typst packages.** mitex 0.2.7 (7 Apr 2026), Apache-2.0, 112 kB, about 185 KB WASM, the feature list and the performance figures. typsium 0.3.2 (15 Jun 2026), MIT, minimum Typst 0.13.1. whalogen 0.3.0 (10 Feb 2025), Apache-2.0. tylax 0.3.8 (14 Sep 2026), Apache-2.0, about 1.5 MB.
- **ManimGX.** Code MIT, "Copyright (c) 2026 Academa, Inc.". "Native wheels include x264 and are GPL-3.0-or-later as a whole". The PyPI expression includes `GPL-3.0-or-later WITH Font-exception-2.0`. Requires-Python >=3.13, and only 0.1.0 and 0.1.1, both 7 Oct 2026. Repo created 5 Oct 2026, 2,784 stars, 19 open issues and 1 closed (#16).
- **ManimGX benchmark.** Every number in the §2.3 table, means of three runs for GX and GL against one run for CE, x264 ultrafast CRF 23, M4 Pro, 7 Oct 2026.
- **ManimGX issues and docs.** #50 (Iris Xe, igvk64.dll, 0.1.1, one 👍 from Sina Atalay), #78 (UHD 600, dx12 note, no CPU fallback in the CLI), #97, #74, #60 ("We'll do this! Thanks."), #33, #32 and #37. The docs mention lavapipe only for Linux wheels. The "Coming from Manim CE" list, the text-and-math page, the `inspect` checks and the 34-item "Unreleased" changelog. The Show HN quotes and who said them.
- **Encoding.** The pyav-ffmpeg build flags (x265 always, x264 except arm32, `--enable-version3`, no `--enable-gpl`, mediafoundation on Windows, videotoolbox on Darwin). Locally, `avcodec-63` contains `--enable-libx264`, `--enable-libx265` and "LGPL version 3 or later", and `av.libs` has `libx264-165-*.dll` and `libx265-*.dll`. PyAV #2270: the quotes, WyattBlue (Member), locked 28 Oct 2026. The x264 GPL and commercial licence, `x264licensing@videolan.org`. The local ffmpeg 8.1 gyan.dev build starts `--enable-gpl --enable-version3`. The JScheumann/pyav-ffmpeg-lgpl fork is real and decodes VP8/VP9 only.
- **Licences and patents.** ffmpeg.org/legal.html: the licence paragraph and the checklist in substance. Four strings were confirmed verbatim: "notably libx264", "You might have a problem here", "Remove any prohibition of reverse engineering from your EULA" and the download-page notice. The patent text ("We do not know, we are not lawyers", "MPEG LA is vigilant and diligent"). GPL FAQ: shared address space, pipes and sockets, static or dynamic linking, and output. The "aggregate with non-free programs" sentence was not re-checked. Via LA AVC tiers and cap. Via LA HEVC (Access Advance, 15 Dec 2025, $0.30/$0.20, $30M cap). OpenH264 FAQ. x265 COPYING (GPL v2 plus commercial). AOM Patent License 1.0. SVT-AV1 "BSD 3-Clause Clear". libaom 2-clause BSD. libvpx BSD 3-clause and PATENTS. Apache-2.0 §4(a)(b)(d) verbatim; for §2, only the opening "perpetual, worldwide" was echoed back. Microsoft `Mfh264enc.dll`, its profiles and the certified-hardware sentence.
- **Remotion.** LICENSE.md eligibility, grant, the "derivate" clause and the 5.0 note (PR #3750). Automators pricing, the FAQ's "not open source" and "cannot be disabled". 4.0.544 released 16 Nov 2026 (GitHub releases API).
- **Other versions.** ManimGL 1.7.2 (13 Dec 2024), Python 3.7 to 3.10. Motion Canvas 3.17.2 (14 Dec 2024) and 3.18.0-alpha.0 (16 Feb 2025). @revideo/core 0.11.0 (10 Jul 2026), and the repo now shows midrender/revideo. MathJax 4.1.3 (3 Jul 2026, about 20 MB, Apache-2.0) and its mhchem quote. KaTeX 0.19.1 (23 Oct 2026, about 4 MB, MIT), with output only `html`/`mathml`/`htmlAndMathml`. ziamath 0.13 (6 Feb 2026, Beta, MIT) and its options-page quotes. latex2mathml 3.81.1 (7 Sep 2026). The matplotlib mathtext features.
- **Local evidence (claim 15).** REPORT.md and summary.tsv match every time quoted: 123.4, 138.5\*, 148.7, 52.6, 37.4\* and 71.8\* s. The frame counts give 72 to 83 frames per second. `TIERS` is 854x480@15 and 1920x1080@60. `pipeline.render` runs `validate()`, then `synthesize_beat`, then rendering.
- **Manim's Typst source, locally.** In `typst_mobject.py`: the integral docstring example, "a self-contained Rust binary extension", and `_TYPST_SVG_STROKE_WIDTH_SCALE = 0.5`. In `typst_file_writing.py` lines 18 to 20: the 10 pt compile size and its "post-import scaling" comment.

**Not checked in this pass** (left as the author marked them, or not attempted): MathJax `tex2svg`/`serializeXML` usage, the Motion Canvas rendering docs page, the Revideo "headless browser" behaviour beyond the README sentence, the rest of the `typst_mobject.py` docstrings (`select(label)`, capture groups), the gyan.dev encoder list (`h264_mf`), the GPL FAQ "aggregate" sentence, the Windows N edition and VideoToolbox items, and every item already in §9. The `claude` co-author in §2.2 is [unverified].
