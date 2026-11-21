<!--
PROVENANCE (added 2026-11-17 by the research session):
The original file chalkdust-research-2026-10.md was not found anywhere on this
laptop (full C: search and WSL search, 2026-11-17). Lokavya pasted its contents
into the session instead. The paste is PARTIAL: the text jumps from the
"Running the most capable LLM locally" section (mid-sentence, "no CUDA") straight
into a llama.cpp Vulkan bullet, and again from the "--fit" flag bullet into the
tail of the "licences not checked" list. Everything between those points is
missing. The text below is reproduced verbatim from the paste; the two cut
points are marked [PASTE GAP]. If the full file turns up, replace this one.
-->

# ChalkDust research notes, October 2026

Collected 26 October 2026 in a Claude.ai strategy session. Everything below is a claim from that date: models, tools, prices and licences move fast, so re-verify anything that would change a decision before building on it.

## The direction (decided)

ChalkDust becomes a desktop app students download (Windows and macOS) that generates 3Blue1Brown-style explainer videos fully on their own laptop, to understand topics before exams. Sold as a one-time fee or subscription, with a paid cloud option for slow machines. Shivam has agreed; this replaces the YouTube channel and fleet plan in docs/PRD.md and docs/ROADMAP.md. The founders intend to train their own models.

The core bet: do not compress a huge model, shrink the job. The LLM never writes Manim. It fills a schema-validated spec that selects layout-safe components, and the deterministic validator (schema, geometric layout probe, render check) can select the best of several samples now and serve as a training reward later.

## What is in this file

1. Local models and compression (MoE offloading, quantization, expert pruning, distillation, licences of teacher outputs, Apple and Windows on-device models)
2. Inference speed and test-time compute (speculative decoding, constrained decoding cost, diffusion LMs, small model plus verifier, RL with verifiable rewards, caching, streaming)
3. Pedagogy and content quality (learning science, Grant Sanderson's method, how AI explainer systems are evaluated, grounding and accuracy, personalisation)
4. Narration audio (local TTS options and licences, word timings, maths verbalisation, post-processing)
5. Architecture, packaging and pricing (hybrid split, cloud cost per video, Tauri plus Python sidecar, signing, Indian pricing reality)

## Missing

The rendering and visuals track was never run. Open questions: ManimGX's real compatibility with Manim CE components and independent speed numbers, Typst or other LaTeX-free maths, rendering-free layout verification (the SGA paper, arXiv 2607.18116), small local VLMs as visual critics, and a reusable library of proven visual idioms (for example the 3blue1brown-manim dataset on Hugging Face and 3b1b's own video code, licences to check).

## Findings from the strategy chat not covered in the notes

- Academa (academa.ai, Show HN 31 Aug 2026) is the closest competitor: STEM lectures written as code and compiled to video, a requested lecture in about 25 minutes, Free (1 generation/month) and Plus $20/month (10 generations). Cloud only. Criticised for polished-looking errors.
- ManimGX (academa-labs/manimgx, launched about 10 Oct 2026): Manim CE-compatible Python API on a Rust/wgpu GPU core, claims about 90x faster than Manim CE on an M4 Pro (own benchmark), Typst maths with LaTeX converted via mitex, no LaTeX or ffmpeg install, Python 3.13+, `manimgx inspect` for layout checks. Code MIT but native wheels are GPL-3.0-or-later because of bundled x264. Weeks old and owned by a competitor.
- Research systems (TheoremExplainAgent, Code2Video, PhysicsSolutionAgent) all have frontier models write raw Manim and still report layout problems. TMA.live (Show HN, Jan 2025) was criticised for overlapping text and invented maths. Knowlify (YC S25) started with students and moved to enterprise at about $415/month.
- Earlier estimates, unmeasured: a few minutes per video on the ROG and roughly half an hour on a base M1 16 GB, dominated by token generation, which is why streaming beat by beat matters.

---



<!-- ===== local_models_compression.md ===== -->

# Running the most capable LLM locally on consumer laptops for a narrow task (ChalkDust, October 2026)

Scope reminder: ChalkDust's LLM jobs are (1) lesson outline (pedagogy, the hardest call), (2) narration, (3) filling a validated JSON spec for hand-written Manim components. Target machines: (A) ASUS, RTX 5070 Ti laptop GPU 12 GB VRAM + 32 GB RAM, Windows 11; (B) MacBook Pro M1, 16 GB unified memory; (C) typical student laptops, 8 to 16 GB RAM, integrated graphics, no CUDA

[PASTE GAP]

ulkan (PR #20334), so older llama.cpp builds understate performance — [llama.cpp issue #19480](https://github.com/ggml-org/llama.cpp/issues/19480)
- Radeon 890M iGPU (Vulkan) with 96 GB shared DDR5, all layers on the iGPU: 17.6 tok/s generation, 53.8 tok/s prompt processing. On unified-memory machines `--no-mmap` caused OOM by forcing a second allocation — [llama.cpp issue #19480](https://github.com/ggml-org/llama.cpp/issues/19480)
- RTX 3060 12 GB + DDR4-3200, Q4_K_M: 15–22 tok/s, explicitly an estimate and not a measurement. SpecPicks recommends 64 GB RAM for Q4_K_M and says 32 GB is "tight" even for Q2_K/IQ3_XXS. The article's own RAM formula is internally inconsistent: it implies ~48 GB, yet the table says 64 GB — [SpecPicks](https://specpicks.com/reviews/qwen3-coder-next-80b-a3b-rtx-3060-12gb-offload-quant-table-2026) [estimate]

**Flags and settings**
- `--n-cpu-moe N` keeps the expert tensors of the first N layers in RAM. `-ot ".ffn_.*_exps.=CPU"` pushes all routed experts to CPU. `--fit o

[PASTE GAP]

ikely Apache-2.0, unverified), Gemma 4 terms, Llama community licence (including its India and EU terms). Neither was Manim CE's licence (believed MIT, unverified), nor the licences of ManimGL or any GPU Manim engine.
- The official ffmpeg legal page (ffmpeg.org/legal.html) and H.264 or HEVC patent pool obligations for a commercial desktop encoder were not fetched.
- Obsidian's pricing and model were not fetched as a precedent.
- No source was found on failure modes of "open core + paid app" for very small teams.
