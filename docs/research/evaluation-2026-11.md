# Evaluating ChalkDust videos: teaching quality, accuracy and visuals, November 2026

Collected 17 November 2026. This fills the evaluation gap in `chalkdust-research-2026-10.md`: its section on how explainer systems are evaluated and why model judges are weak was lost in the paste. Every claim about a paper, model, licence, date, number or tool cites a page fetched on 17 Nov 2026 (Sources, part A). Claims seen only in search results, or not checked at all, are marked **[unverified]** and listed in Sources, part B. Nothing was installed, rendered, trained or run for this note.

This is research, not a decision. D-001 to D-012 and the validation ladder in `SCENE_SPEC.md` stand until the owners decide.

---

## 0. Outcome

1. **Hard checks are the measurement; judges are a weak tie-breaker.** In published explainer-video work, model judges agree poorly with people on the things ChalkDust cares most about: layout (Spearman 0.39 to 0.42), accuracy-and-depth and logical flow (0.14 and 0.16), and per-video verdicts. One 2026 system found its VLM scores *negatively* correlated with blind human ratings (r = -0.17). ChalkDust's validator (schema, semantic, geometric probe, render, and sympy where possible) measures layout and maths better than any judge found here. Report it first.
2. **Judges are not useless.** They rank *systems* well when the quality gap is large (VibeVideoBench: system-level Spearman 0.98 with human Elo), and they do well on objective text dimensions with a rubric (EduIllustrate: 0.83 to 0.89). They fail at per-video verdicts, at layout, at separating the top systems, and at certifying that a video is acceptable. Use a judge only for pairwise, both-order, cross-family comparisons on held-out topics, and only on dimensions where it has been checked against Lokavya and Shivam.
3. **Human ratings are noisy too.** Published inter-rater agreement on explainer videos runs from Krippendorff's alpha 0.21 to 0.56, and one study got Fleiss' kappa -0.01 on a binary "usable?" question. So the human protocol must be blind, pairwise where possible, built on binary fatal flags before any 1-to-5 scale, rated by two people independently, with agreement reported.
4. **The SGA paper exists and supports the ChalkDust approach.** arXiv 2607.18116 (Lopez, Hinojosa, Ghanem; v1 20 Jul 2026, v2 10 Sep 2026) checks Manim layout symbolically from bounding boxes, without rendering. Its symbolic checker beat a VLM critic in human frame-level preference (65.0%) and was 6x to 18x faster per iteration. ChalkDust's geometric probe already does a stricter version of this. Borrow two ideas: the occupancy and distribution terms of its MVQS score, and its persistent-versus-transient overlap rule.
5. **Small local VLMs: no evidence that they catch overlap or clipping.** Several 2B to 9B vision models are Apache-2.0 (Qwen3.5-4B, Qwen3-VL-4B/8B, Gemma 4 E4B, MiniCPM-V 4.7 1B, SmolVLM2, InternVL3.5). The three whose files were sized (Qwen3-VL-4B, Qwen3-VL-8B, Gemma 4 E4B: about 3 to 6 GB at 4-bit with the vision projector) fit 12 GB VRAM and a 16 GB Mac by file size; runtime memory is unmeasured. No paper found measures 2B to 8B models on overlap or clipping in rendered frames, and the general evidence says VLMs are poor at exactly this. Recommendation to the owners: treat rung 4 (the VLM check, which `SCENE_SPEC.md` §8 lists as a gate) as advisory until the defect-injection calibration in §4.4 has been run.
6. **Much of good pedagogy can be checked from the JSON spec.** About ten learning-science principles turn into spec-level checks: concrete-before-abstract, no opening definition, one idea per beat, terms introduced before use, signalling, a check-for-understanding beat, a misconception beat, worked-example structure, and length. Mayer's 2024 table gives effect sizes for most of them (§5). These checks are proxies. They give the outline job (job 1) a partial verifier, but they are easy to game, so audit them by hand before using them as a training reward.
7. **Maths: sympy plus an answer key plus units, with "unchecked" reported honestly.** Check derivation steps with sympy, final answers with Math-Verify (Apache-2.0), and units with a units library. A parse failure must count as **unchecked, not pass**. sympy's default ANTLR LaTeX parser can silently drop part of an expression, for example parsing `x -` as `x`. LLM checkers are a poor substitute: GPT-OSS 20B, Qwen3-14B and Phi-4-Reasoning falsely rejected 75.6% to 85.3% of valid, perturbed solution traces.
8. **3b1b material cannot go into a paid product as-is.** `3b1b/videos` is CC BY-NC-SA 4.0: no commercial use, and adaptations must stay under the same licence. The Hugging Face `BibbyResearch/3blue1brown-manim` dataset carries an Apache-2.0 tag, but its card says the components keep their original licences. It aggregates community Manim datasets, not 3b1b's code, and its author says "I would recommend not to use this dataset anymore". Write original components. Do not train on these sources without legal advice.

---

## 1. How published explainer-video systems evaluate

### 1.1 The three systems named in the October notes

| System | What it is | How it is scored | Agreement with people | Failure modes reported |
|---|---|---|---|---|
| **TheoremExplainAgent (TEA)**, arXiv 2502.19400, Feb 2025. MIT code, ACL 2025 main (oral), per its GitHub README | An agent writes raw Manim for theorem videos of 5+ minutes. Benchmark: TheoremExplainBench, 240 theorems (Hugging Face `TIGER-Lab/TheoremExplainBench`, MIT) | Five 0-to-1 dimensions, combined by geometric mean. GPT-4o judges Accuracy & Depth and Logical Flow from the transcript, and Visual Relevance and Element Layout from key frames. Gemini 2.0 Flash judges Visual Consistency from video chunks | **Metric vs human (Spearman):** Accuracy & Depth 0.14, Visual Relevance 0.72, Logical Flow 0.16, Element Layout 0.42, Visual Consistency 0.17. **Human vs human (Krippendorff's alpha, 3 raters):** 0.45, 0.36, 0.56, 0.31, 0.36. The human study used 40 videos rated by 12 STEM students | The paper names Manim code hallucinations as the most common failure. It says most generated videos have minor layout problems: overlapping text, inconsistent sizes and poor object positioning. None of these is quantified |
| **Code2Video**, arXiv 2510.01174, Oct 2025. MIT code. The README says DL4C workshop at NeurIPS 2025 and ICML 2026 | Planner, Coder and Critic agents write Manim. The Critic is a VLM with "visual anchor" grid prompts. Benchmark: MMMC, 117 long 3Blue1Brown videos in 13 subjects, cut into 339 clips | Gemini 2.5 Pro as VLM-as-judge on five aesthetics dimensions; time and tokens; **TeachQuiz**: 10 multiple-choice questions per topic, asked to a VLM after prompt-based "unlearning" (S1) and again after it watches the video (S2), TQ = S2 - S1 | No item-level judge-vs-human correlation is reported. In the human study (5 groups of 8 volunteers, 6 middle-school and 2 undergraduate each, 20 topics), people gave the aesthetics scores and answered the quizzes. The paper says human ratings follow the same overall trends as the VLM scores. People scored Code2Video below 3B1B on aesthetics (81.8 vs 96.5) but about the same on the quiz (80.3 vs 78.8), and gave Code2Video lower layout scores. The r = 0.971 (p = 0.0059) is **aesthetics vs quiz score**, not judge vs human; n is not stated | Paraphrased: humans are very sensitive to brief occlusions, while video LLMs often miss frame-level issues. Overlapping elements, occluded lines and unused regions. The 8x8 anchor grid itself added clutter |
| **PhysicsSolutionAgent (PSA)**, arXiv 2601.13453, 19 Jan 2026 | Solves a physics problem as JSON, plans scenes, writes Manim. Videos up to 6 minutes | 15 automated parameters. The paper never lists all 15; its prompts imply 5 solution (weight 5%), 5 explanation (10%) and 5 visual (60%), plus an error penalty (25%). The visual score averages the *lowest two* of five sub-scores to avoid inflation. A VLM (not named) reads 3 screenshots per scene for feedback | Not reported. Automated: 100% completion, mean 3.8/5 on 32 videos. Human inspection still found "several minor and major issues" | Layout inconsistencies, VLM misreading of the visuals (which undermines the repair loop), unreliable Manim code, redundant narration. Static screenshots miss motion and audio sync |

Two things stand out in these numbers:

- **TEA's automated score barely separates AI videos from human ones** (my reading of Tables 1 and 2, not the authors' claim). Human-made Manim videos scored 0.77 overall and the agents 0.71 to 0.79. Claude 3.5 Sonnet v1 scored 0.79, but it succeeded on only 2.1% of the 240 theorems, so that score rests on about five videos.
- **The judged dimensions that matter most for ChalkDust are weak.** Accuracy & Depth (0.14) and Logical Flow (0.16) barely track people. Element Layout (0.42) is TEA's second-best dimension, but still only moderate. Visual Relevance (0.72) is the only strong one.
- **None of the three systems quantifies how often the maths is wrong.** TEA and PSA report code, LaTeX and layout failures; PSA says its errors were not fundamental physics errors but gives no count. "Invented maths" is therefore unmeasured in this literature.

### 1.2 Newer work (2026)

| Paper | Date | What it adds to the evaluation picture |
|---|---|---|
| **ManimAgent**, arXiv 2606.30296 | 29 Jun 2026 | Self-evolving Manim agent. On 37 paired videos, its VLM keyframe scores vs blind human ratings: **Pearson r = -0.17, Spearman -0.20, Cohen's kappa -0.07, agreement 0.49**. The authors conclude "VLM scores are not reliable headline evaluation evidence" and use them only internally. Human raters also disagreed: Fleiss' kappa for binary usability **-0.01** (153 videos, 3 raters), ICCs -0.09 to 0.33 on the five Likert dimensions. 54 blind raters. Fatal flags used: major factual or formula errors, severe overlap, cropped objects, unreadable text |
| **SGA**, arXiv 2607.18116 | v1 20 Jul 2026, v2 10 Sep 2026 | Rendering-free geometric verification of Manim code (§3). Its symbolic checker beat a VLM critic on layout |
| **VibeVideoBench**, arXiv 2610.35864 | 26 Oct 2026 | 239 code-driven video tasks, 9 agent systems, 2,151 scored rollouts. Separates deterministic L1 checks (1,449 checks: resolution, frame rate, duration, audio) from judged L2 rubric checks and L3 quality. Judge panels vs blind human pairs: **73.3%** agreement on average, about 63% when the score gap is under 0.2 and 86% when it is 0.5 or more. System-level Spearman with human Elo **0.98** (mean panel), but check-level Cohen's kappa only **0.44 to 0.56** and video-level Pearson **0.45 to 0.51**. On pairs humans call "both bad", the panels gave the same scores (3.81) as on other pairs. The authors conclude that judge scores order videos but do not certify them as acceptable. Self-preference: +0.18 and +0.20 for the judge's own model |
| **EduIllustrate**, arXiv 2604.05005 | v1 6 Apr 2026 | K-12 STEM text-plus-diagram explanations built with Manim (static images, not video). 20 expert raters. Judge vs human Spearman: Logical Coherence 0.89, Diagram–Problem Alignment 0.83, Typographic Clarity 0.77, Pedagogical Effectiveness 0.70, Text–Diagram Coordination 0.66, **Visual Consistency 0.47, Element Layout Quality 0.39**. The paper says the judge "is insensitive to low-level visual artifacts" such as overlapping elements and misaligned labels. Self-preference: Gemini rated its own outputs 9.2% higher than GPT-5 did; GPT-5 rated its own 20.6% higher than Gemini did |
| **Teaching Monster Challenge**, arXiv 2608.08852 | 9 Aug 2026 | A competition: 77 teams, 1,612 preliminary videos. An LLM judge screened, then 59 crowd raters did 246 pairwise comparisons, then a 10-member expert panel. Among the ten shortlisted systems, the judge's ranking vs the crowd ranking: **Spearman -0.17**. Their judge Content Accuracy means sat within **4.59 to 4.75**. The judge's top team finished 8th with the crowd; the eventual winner was 4th under the judge. Expert agreement: Kendall's W 0.41. "The LLM-judge separates clearly weak submissions from the rest but ranks the strongest systems poorly" |
| **EduPanel**, arXiv 2607.18529 | v1 20 Jul 2026 | A three-agent LLM judge for teaching videos. MAE 0.85 against human consensus vs 0.87 for the median blind expert (range 0.58 to 1.00), on only 12 videos and 12 experts. Blind expert alpha 0.38. It was more lenient on visual dimensions (+0.67) than on transcript dimensions (+0.24), and it compressed scores (SD 0.77 vs 1.22) |
| **EduVideoBench**, arXiv 2605.26918 | 26 May 2026 | Knowledge-Skills-Attitude benchmark for video *generation* models (pixel video). Expert comments (paraphrased): a single misaligned element, such as pacing, legibility or notation, can invalidate an otherwise correct video. That supports fatal flags in the human rubric |
| **EduVQA**, arXiv 2603.03066 | v3 19 May 2026 | 1,130 text-to-video clips with 310,650 human annotations for concept correctness in early maths. Aimed at pixel video models; not directly usable, but it shows "looks plausible, teaches the wrong thing" is a recognised failure class |

The October notes also said TMA.live (Show HN, Jan 2025) was criticised for overlapping text and invented maths. That was not re-checked here **[unverified]**.

### 1.3 What this means for ChalkDust

- Every system above has a frontier LLM write raw Manim and still reports overlap, clipping and LaTeX failures. ChalkDust removes most of that failure class by construction (components, regions, typed refusals). Its evaluation should show that with numbers: per-sample pass rates by refusal kind (§8).
- None of the systems fetched measured learning in real students against a control. Code2Video's TeachQuiz is a VLM quiz, not a student test. Its 5-group human study is the closest thing, and it is small. Do not cite TeachQuiz-style scores as "learning".
- The deterministic/judged split in VibeVideoBench (L1 checks vs L2/L3) is the right shape for ChalkDust's report.

---

## 2. Why model judges are weak signals (reconstructed)

This section rebuilds the lost October section from primary sources. It covers the counter-evidence too.

### 2.1 General biases of LLM judges

| Bias | Evidence |
|---|---|
| **Position** | Wang et al., arXiv 2305.17926: with ChatGPT as the judge, swapping answer order let Vicuna-13B beat ChatGPT on 66 of 80 queries. Shi et al., arXiv 2406.07791: 15 judges, 22 tasks, about 40 answer models, over 150,000 instances. Position bias is not random chance, varies across judges and tasks, is weakly affected by prompt length, and is strongly affected by the quality gap between the answers |
| **Verbosity / length** | Zheng et al., arXiv 2306.05685 (MT-Bench), name verbosity bias. Dubois et al., arXiv 2404.04475: controlling for length raised AlpacaEval's Spearman with Chatbot Arena from 0.94 to 0.98 |
| **Self-preference** | Panickssery et al., arXiv 2404.13076: GPT-4 and Llama 2 can recognise their own outputs, and self-recognition correlates linearly with self-preference. Measured in this domain: EduIllustrate +9.2% and +20.6%; VibeVideoBench +0.18 and +0.20 (§1.2) |
| **Weak reasoning about errors** | Tyen et al., arXiv 2311.08516 (ACL Findings 2024): LLMs "cannot find reasoning errors" reliably, but can fix them when told where they are. Mazdarani and Toxtli, arXiv 2608.28725: GPT-OSS 20B, Qwen3-14B and Phi-4-Reasoning falsely rejected **75.6% to 85.3%** of valid, perturbed linear-equation traces |

### 2.2 Educational content specifically

- **Tutoring (MRBench, arXiv 2412.09416, NAACL 2025).** Prometheus2 and Llama-3.1-8B as evaluators correlated weakly with human labels on eight pedagogical dimensions, mostly slightly negatively. Most cells sit between about -0.3 and +0.1; the extremes reach -0.72 (Providing guidance) and -0.71 (Tutor tone) for the 60-dialogue novice-tutor set under Prometheus2, and -0.80 under Llama-3.1-8B. The authors: "the annotations from the LLMs are unreliable for the challenging pedagogical dimensions". Humans on the same task agreed well (pairwise Cohen's kappa 0.71).
- **Teaching videos.** Teaching Monster's judge ranked the top ten against the crowd at Spearman -0.17 (§1.2). TEA's transcript-judged accuracy and flow correlate at 0.14 and 0.16.
- **Rubrics help a lot.** MMTutorBench (arXiv 2510.23477): a rubric-based judge reached Spearman 0.652 (Pearson 0.725) with human scores, against 0.563 (0.625) for a rubric-free judge. In its appendix check (100 items), Pearson was 0.89 vs 0.45. So any judge ChalkDust uses should score against explicit, item-level rubric questions, not "rate the pedagogy 1 to 10".

### 2.3 Visual layout specifically

- **VLMs miss simple geometry.** "Vision language models are blind" (arXiv 2407.06581): on seven simple tasks, four strong VLMs averaged 58.07% (best 77.84%), against an expected 100% for humans. The tasks include whether two circles overlap and how many times two lines cross. Accuracy goes up when shapes have more space. Overlap and crowding are exactly ChalkDust's layout defects.
- **Judged layout agrees weakly with people wherever it is measured.** EduIllustrate: 0.39, its lowest dimension. TEA: 0.42, which is moderate there only because TEA's text dimensions were even lower (0.14 to 0.17). EduPanel: most lenient on visual dimensions. ManimAgent: negative overall.
- **SGA's authors:** "perceptual critics remain largely insensitive to geometric inconsistencies", because pixel-space evaluation "does not directly capture geometric violations".
- **Code2Video (paraphrased):** video LLMs often miss frame-level issues, while humans react strongly to brief occlusions.
- **PSA:** VLM misreadings of the frames undermined its own repair loop.
- **Slide work points the same way.** VLM-SlideEval (arXiv 2510.22045) found VLMs weak at pixel-accurate extraction of slide elements. PPTBench (arXiv 2512.02624) reports "systematic layout errors such as misalignment and element overlap" in model outputs. Neither gives overlap-detection accuracy for small models.

### 2.4 Counter-evidence

- MT-Bench: GPT-4 as judge reached "over 80% agreement, the same level of agreement between humans" on open-ended chat (arXiv 2306.05685).
- VibeVideoBench: system-level Spearman 0.98 with human Elo; 86% pair agreement when the score gap is large.
- EduIllustrate: 0.83 to 0.89 on objective text and alignment dimensions.
- EduPanel: MAE comparable to a median expert (on 12 videos), and expert MAE improved from 0.87 to 0.73 when experts saw its feedback.

**Synthesis.** Judges are useful for coarse ranking of systems, for text dimensions with a rubric, and as an assistant to a human reviewer. They are weak per video, weak on layout and maths correctness, weak at separating good from very good, and they do not certify "acceptable". Their errors are biased (order, length, own family, visual leniency), so they also make a gameable reward. This matches the techniques note's rule to keep judges out of the training reward and use them only on held-out topics.

### 2.5 Humans are noisy too

| Study | Human-human agreement |
|---|---|
| TEA | Krippendorff's alpha 0.31 to 0.56 per dimension |
| SGA | alpha 0.407 at frame level, 0.208 at video level |
| EduIllustrate | alpha 0.30 (Visual Consistency) to 0.84 (Logical Coherence) |
| ManimAgent | Fleiss' kappa -0.01 on binary usability; ICC -0.09 to 0.33 |
| EduPanel | blind expert alpha 0.38 |
| Teaching Monster | expert Kendall's W 0.41 |
| MRBench (tutoring, well-defined labels) | Cohen's kappa 0.71 |

The pattern: agreement is high when the question is concrete (MRBench's defined labels, EduIllustrate's logical coherence) and low for holistic visual or "quality" scores. The rubric in §8.5 therefore asks concrete yes/no questions first.

---

## 3. SGA and rendering-free layout verification

### 3.1 The paper (verified)

- **Citation:** "SGA: Plug&Play Geometric Verification for Educational Video Synthesis", Jhon Lopez, Carlos Hinojosa, Bernard Ghanem. arXiv 2607.18116, v1 20 Jul 2026, v2 10 Sep 2026. It exists under the ID the October notes gave.
- **SGA = Symbolic Geometric Agent.** It intercepts LLM-written Manim code before rendering and instruments the AST, wrapping each `construct()` statement in try/except so faulty lines are skipped. It runs the code partially to extract a symbolic scene graph: each mobject's label (Text, Geom, Mark, Deco), centroid, width, height and source line.
- **Conflict detection.** Axis-aligned bounding-box intersection with threshold epsilon = 0. Severity is High when the max overlap ratio exceeds 0.30, Medium above 0.10, and Low otherwise. Five structural patterns (P1 to P5) suppress intended overlaps: decorative enclosures, annotation pointers, explicit centring, containers and maths braces. It samples K = 11 times per animation. Overlaps at rest (t = 0 and t = 1) are "persistent" conflicts; mid-animation overlaps are "transient" and ignored. Out-of-frame is scored as a canvas-boundary term. It has no separate occlusion detector.
- **Repair.** It sends the LLM a minimum separation vector and patches within a +/-5-line window, with AST validation and rollback. One iteration.
- **MVQS** (Manim Visual Quality Score), deterministic and rendering-free:
  `MVQS = 0.5*S_sp + 0.2*S_ly + 0.3*S_ed`
  - S_sp (spatial): overlap penalty 0.50, canvas boundary 0.25, legibility 0.15 (fraction of objects above a minimum scale of 0.15 units), separation 0.10
  - S_ly (layout): density 0.40 (occupancy favoured between 0.10 and 0.35), distribution 0.60 (occupancy of a 4x3 grid)
  - S_ed (educational): labelling completeness 0.40 (geometry linked to MathTex labels by substring), animation quality 0.60 (fraction of animations with no persistent conflict)
- **Benchmark:** MMMC-Code, 117 topics. Backbones: GPT-5.1, GPT-5 mini, Claude 4.6 Sonnet, Gemini 3.0 Flash. Pipelines: TEA and Code2Video.

**MVQS results (Table 1 of the paper)**

| Pipeline + backbone | Raw | VLM critic | SGA |
|---|---|---|---|
| TEA, GPT-5.1 | 58.74 | **61.58** | 59.81 |
| TEA, GPT-5 mini | 59.53 | **63.73** | 63.47 |
| TEA, Claude 4.6 Sonnet | **63.84** | 62.41 | 62.83 |
| TEA, Gemini 3.0 Flash | 63.67 | 62.34 | **67.14** |
| C2V, GPT-5.1 | 62.97 | 64.54 | **73.11** |
| C2V, GPT-5 mini | 63.68 | 63.51 | **66.42** |
| C2V, Claude 4.6 Sonnet | 63.27 | 65.15 | **65.87** |
| C2V, Gemini 3.0 Flash | 67.82 | 69.09 | **71.86** |

Read it carefully:

- "7 of 8" is SGA vs **raw**. Against the VLM critic, SGA lost in two TEA rows.
- The headline 16.1% is one row (C2V + GPT-5.1, 73.11 vs 62.97).
- MVQS is the authors' own metric, so the table partly measures agreement with their own checker.
- **Human study:** 25 recruited, 14 to 16 completed. At **frame** level, SGA was preferred over raw in 84.4% of ratings (93.3% on overlap, 75.6% on canvas) and over the VLM critic in 65.0%. At **video** level the split was SGA 38.5%, VLM critic 26.0%, raw 24.0%, no preference 11.5%. Inter-rater alpha was 0.407 (frame) and 0.208 (video). Clips were not linked to MVQS, so there is no item-level correlation.
- **Cost:** a VLM critic iteration costs 30 to 90 s of rendering; SGA extraction takes under 5 s. Per sample, Code2Video + GPT-5.1 took 6.4 min (raw), 8.8 min (VLM critic) and 6.8 min (SGA).
- **Limits (authors, paraphrased):** 2D only; no check of timing such as a label appearing before its object; English-only benchmark; a visually clean animation can still be factually wrong, and a high MVQS could encourage automation bias.
- **Code and licence:** the paper says the implementation is public via `jhedlp.github.io/projects/sga/`, but that URL returned **404** on 17 Nov 2026. No software licence is stated. Treat the code as unavailable until found.

### 3.2 What ChalkDust should take from it

ChalkDust's rung 3 (`SCENE_SPEC.md` §8) already builds the scene and checks bounding boxes at settle points: safe area, `exclusive()` groups, minimum font size, zero-size and occlusion. That is stricter than SGA in one way, because it refuses rather than filtering with heuristics. ChalkDust also authors components by hand, so intended overlaps are declared rather than guessed. Ideas worth copying:

1. **Occupancy band and grid distribution.** A beat with 3% of STAGE occupied, or with everything crammed in one corner, passes every refusal check today. An S_ly-style score (occupancy within a band, spread over a coarse grid) is a cheap soft metric to report. It should not be a gate until it has been checked against human ratings.
2. **Persistent vs transient overlap.** Sampling within animations and only flagging overlap at rest matches ChalkDust's `settle()` idea. Report "transient overlap time" as a soft metric if it is cheap.
3. **Labelling completeness.** For FreeBodyDiagram, GraphPlot and GeometryConstruct, check from the spec that each force, curve or marked point has a label. This is spec-checkable (§5).

**Other rendering-free checks.** `manimgx inspect` checks for cut-off mobjects, overlapping text, lines or fills over text, and text too small to read, at sampled moments only (see `rendering-2026-11.md`, which verified this). It overlaps with ChalkDust's probe and adds nothing ChalkDust lacks.

---

## 4. Small local VLMs as visual critics

### 4.1 Candidates (2B to 9B class, as listed on Hugging Face on 17 Nov 2026)

| Model | Size | Licence (model card) | Files that would ship (Q4 + projector) | Notes |
|---|---|---|---|---|
| **Qwen3.5-4B** (`Qwen/Qwen3.5-4B`, Feb 2026) | 4B LM (5B with vision, per the Hub) | apache-2.0 | GGUF size not fetched **[unverified]** | Natively multimodal ("early fusion"). Card: OCRBench 85.0, CountBench 96.3, RefCOCO 88.1, ScreenSpot Pro 60.3, HallusionBench 65.0. Qwen3.5-2B, 0.8B and 9B are also tagged image-text-to-text. Newer multimodal Qwen repos were listed but not examined: Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3.8-27B, Qwen3.8-Flash-Next and Qwen-Drive-1.0-4B (Aug 2026) |
| **Qwen3-VL-4B / 8B Instruct** (Oct 2025) | 4B / 8B (9B on the Hub) | apache-2.0 | 4B: Q4_K_M 2.50 GB + mmproj 0.45 GB (Q8) or 0.84 GB (F16). 8B: Q4_K_M 5.03 GB + mmproj 0.75 or 1.16 GB. Official Qwen GGUFs | Card claims GUI-agent use, 2D grounding and occlusion reasoning. The llama.cpp multimodal doc's list does not name Qwen3-VL, although Qwen publishes official GGUF and mmproj files for it **[llama.cpp support not verified from release notes]** |
| **Gemma 4 E4B-it** (Mar 2026; QAT GGUF May 2026) | 4.5B effective, 8B with embeddings | Apache 2.0 | q4_0 5.15 GB + mmproj 0.99 GB (`google/gemma-4-E4B-it-qat-q4_0-gguf`) | Image, audio and text in. Visual token budget 70 to 1,120 per image; the card recommends high budgets for small text. MMMU-Pro 52.6%. Gemma 4 is listed in llama.cpp's multimodal doc |
| **MiniCPM-V 4.7 1B** (Oct 2026) | "1B" (2B on the Hub) | apache-2.0 tag; no registration terms on the card. The LICENSE file was not opened **[unverified]** | GGUF exists (`openbmb/MiniCPM-V-4.7-1B-gguf`), size not fetched | Qwen3.5/3.6 backbone with a SigLIP2 vision encoder. No benchmark numbers in the card text. A 35B-A3B sibling exists |
| **SmolVLM2-2.2B-Instruct** (Feb 2025) | 2.2B | Apache 2.0 | "5.2GB of GPU RAM for video inference" (card) | OCRBench 72.9, ChartQA 68.84. No newer SmolVLM repo appeared in the Hub listing |
| **InternVL3.5-8B** (Aug 2025) | 8.5B | Apache-2.0 (Qwen3 part also Apache-2.0) | Not fetched | No newer InternVL repo appeared in the `InternVL` Hub listing |
| **Moondream 3.1 9B-A2B** (repo Jun 2026) | 9B total, 2B active MoE | Moondream Model License 1.0 | Not fetched | Structured "detect" and "point" outputs, which could in principle return boxes. Licence below |
| LFM3-VL-2B | 2B | LFM Open License v1.0, with a $10M annual-revenue cap per `models-2026-11.md` | — | Not re-checked here. See the models note: the cap travels with derivatives |

**Fit.** Files were sized for three candidates only. Qwen3-VL-4B (about 3.0 to 3.3 GB at Q4_K_M with the projector), Qwen3-VL-8B (about 5.8 to 6.2 GB) and Gemma 4 E4B (about 6.1 GB) fit 12 GB of VRAM (machine A) and 16 GB of unified memory (machine B) by file size. Qwen3-VL-4B is the realistic size for machine C on CPU. The other candidates were not sized **[unverified]**, and runtime memory with KV cache and image tokens was not measured for any of them **[unverified]**. On machine A the critic competes with the LLM and Manim for VRAM, so run it after generation, not alongside.

### 4.2 Licences, plainly

**Apache-2.0** (Qwen3.5, Qwen3-VL, Gemma 4, MiniCPM-V 4.7 per tag, SmolVLM2, InternVL3.5):
- (a) Commercial use: **yes.** Section 2 grants "a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable" copyright licence to reproduce, prepare derivative works, sublicense and distribute.
- (b) Fine-tuned derivative in a paid closed-source app: **yes.** Section 4's conditions are about notices, not source release.
- (c) Outputs used to train a shipped model: **yes.** The licence does not mention outputs, and the cards fetched add no terms.
- (d) Obligations: "You must give any other recipients of the Work or Derivative Works a copy of this License"; "You must cause any modified files to carry prominent notices stating that You changed the files"; keep copyright, patent, trademark and attribution notices; carry the NOTICE file if there is one. Separately, the SmolVLM2 card's *intended-use* section (not the licence) lists evaluating or scoring individuals, for example in employment, education or credit, as a prohibited use. Scoring a generated video is not scoring a person, but do not use it to grade students.

**Moondream Model License 1.0** (moondream.ai/licenses/model/1.0):
- (a) Commercial use: **yes.** Section 2 allows use "in commercial products, applications, and SaaS offerings".
- (b) Derivatives in a paid app: **yes**, with conditions. Section 7 requires you to "retain all copyright, license, attribution, and other notices" and "identify any material modifications". Recipients get the same licence. The catch is Section 3: "You may not provide a General-Purpose Hosted Model Service for the Model Materials or any Derivative Model". Examples include "offering a general-purpose hosted Moondream inference API" (5.1). Using the model as a component inside a product is described as exempt, but ChalkDust's paid cloud tier should get a legal read before it hosts Moondream.
- (c) Outputs: **yes.** "You may use, reproduce, modify, distribute, display, perform, and commercialize Model Outputs". Section 17: "A model trained solely on Model Outputs is not a Derivative Model".
- (d) Obligations: notices and the licence text; no use of the Moondream name as your model's primary name or to imply endorsement (Section 8).

### 4.3 Evidence that small VLMs can see ChalkDust's defects

**None found.** No paper fetched measures a 2B to 8B VLM on overlap, clipping or legibility in rendered frames. The evidence that does exist points against relying on one:

- Frontier VLMs fail simple overlap and intersection questions (BlindTest, §2.3).
- In explainer pipelines, VLM critics underperform symbolic checks (SGA), misread frames (PSA), and correlate negatively with humans (ManimAgent).
- The Qwen3-VL card claims grounding and occlusion reasoning, and Qwen3.5-4B reports strong OCR and counting. Those are general benchmarks, not defect detection in diagrams.

Where a local VLM might still help, because geometry cannot see these:
- **Narration–frame mismatch:** "does this frame show X?", asked with the beat's own key terms.
- **Glyph rendering:** missing-glyph boxes or garbled maths in the actual pixels. ChalkDust's `unrenderable_text` guard covers fonts but not every renderer path.
- **RawScene beats**, which escape the component guarantees. `SCENE_SPEC.md` already scopes rung 4 to sampled beats and RawScene.

### 4.4 Calibration test before any use (proposed, not run)

1. Take 50 known-good rendered beats across components.
2. Inject defects that the geometric probe can label exactly: overlap (shift a mobject onto another), clipping (push past the safe area), tiny text (scale under `MIN_FONT_SIZE`), low contrast, a missing-glyph box and a wrong label, at three severities. Keep untouched copies as negatives.
3. Ask each candidate VLM a fixed yes/no question per defect type, at two image-token budgets, with the frame at 480p and 1080p.
4. Report precision and recall per defect type with intervals, plus seconds per frame on machines A, B and C.
5. **Bar to use it as a warning flag (suggested):** recall at least 0.8 at precision at least 0.8 on that defect type. It should never be a gate on its own. Anything the geometric probe already catches does not need a VLM.

---

## 5. Learning-science rubric as mechanical checks

### 5.1 The evidence base

**Mayer, "The Past, Present, and Future of the Cognitive Theory of Multimedia Learning"**, *Educational Psychology Review* 36:8 (2024), Table 6. These are 15 principles with effect sizes and test counts, drawn from 200+ experiments; boundary conditions are in Mayer (2021).

| # | Principle | Effect size | Tests |
|---|---|---|---|
| 1 | Coherence: exclude extraneous material | 0.86 | 19 |
| 2 | Signaling: add cues that highlight the organisation | 0.69 | 16 |
| 3 | Redundancy: don't add printed text to graphics plus narration (fast-paced lessons) | 0.10 | 12 |
| 4 | Spatial contiguity: words near the pictures they describe | 0.82 | 9 |
| 5 | Temporal contiguity: words and pictures at the same time | 1.31 | 8 |
| 6 | Segmenting: learner-paced segments | 0.67 | 7 |
| 7 | Pretraining: names and characteristics of key concepts first | 0.78 | 10 |
| 8 | Modality: narration rather than on-screen text | 1.00 | 19 |
| 9 | Multimedia: words and pictures, not words alone | 1.35 | 13 |
| 10 | Personalization: conversational, not formal | 1.00 | 15 |
| 11 | Voice: friendly human voice, not a machine voice | 0.74 | 7 |
| 12 | Image: adding the speaker's image doesn't necessarily help | 0.20 | 7 |
| 13 | Embodiment: high-embodiment on-screen instructor | 0.58 | 17 |
| 14 | Immersion: 3D VR not necessarily better than 2D | -0.10 | 9 |
| 15 | Generative activity: guided generative activities | 0.71 | 44 |

**Other sources:**
- **Brame (2016), CBE—Life Sciences Education 15(4) es6**, quoting Guo et al. (2014) on 6.9 million edX sessions: median engagement was close to 100% for videos under 6 minutes, about 50% at 9 to 12 minutes and about 20% at 12 to 40 minutes. "Making videos longer than 6–9 minutes is therefore likely to be wasted effort." Brame also cites interpolated questions (Szpunar et al. 2013) improving later test performance.
- **Dunlosky (American Educator, Fall 2013)**, summarising Dunlosky et al. 2013 (*Psychological Science in the Public Interest*): practice testing and distributed practice are the "most effective". Interleaved practice, elaborative interrogation and self-explanation are "promising". Interleaving "can dramatically improve student achievement, especially in the domain of problem solving."
- **Muller, Bewes, Sharma and Reimann (2008), J. Computer Assisted Learning**, "Saying the wrong thing: improving learning with multimedia by including misconceptions". 364 first-year physics students. "Refutation and Dialogue produced the greatest learning gains, with effect sizes of 0.79 and 0.83" compared with plain exposition. The abstract was read via a blog that reproduces it; the publisher page returned 403.
- **Fyfe, McNeil, Son and Goldstone (2014), Educational Psychology Review**, "Concreteness fading in mathematics and science instruction: a systematic review". Metadata verified; the abstract was not fetched **[findings unverified]**.
- **Worked examples** (Atkinson, Derry, Renkl and Wortham 2000, *Review of Educational Research*): not fetched **[unverified]**.

### 5.2 Grant Sanderson's method, in his own words

From the 3Blue1Brown FAQ, answering "I'd like to start making math videos online, do you have any advice?":
- "Concrete before abstract. Give examples before general frameworks. Open with the key exercise; don't put it at the end."
- "In math especially, topic definitions should not be seen as a starting point, but an ending point."
- "Always ask what picture or visual you could use to elucidate a topic."
- "Every movement on the screen should be deliberate, with an identifiable purpose."
- "Topic choice matters way more than production quality."

From the SoME1 results post, the qualities of the best entries: "clarity, strong motivation, compelling choices of topic matter, and non-obvious insights."

### 5.3 Turning principles into checks on the JSON spec

**Hard** means a deterministic yes/no from the spec. **Mostly mechanical** means a deterministic proxy that needs a list of terms or a cheap classifier. **Judge** and **Human** mean the spec can't show it. Every check below is a proposal, untested.

Field names come from the "key params" table in `SCENE_SPEC.md` §5. For the problem-solving and maths components they were checked against the pydantic `Params` classes in `chalkdust/scenes/components/`: `AnswerBox` (`value`, `units`, `option`), `EquationDerivation` (`steps`, `annotations`), `SolutionStep` (`n`, `claim`, `work`, `work_format`, optional `justification`), `ProblemStatement` (`text`, `given`, `find`), `FreeBodyDiagram` (`body`, `forces`, each force with a required `label`) and `UnitBreakdown` (`quantity`, `decomposition`). The other components' field names were not checked.

| Principle (source) | Check | Type | From spec alone? |
|---|---|---|---|
| Concrete before abstract; no opening definition (Grant; Fyfe [unverified]; PRD §7 "Opens with a concrete hook, not a definition") | The first content beat after TitleCard is a ProblemStatement, an example, GraphPlot, FreeBodyDiagram or DataStructureViz, not a definition-style BulletReveal or EquationDerivation. A general formula's first appearance comes after at least one beat with a concrete instance (numbers, a specific object) | Mostly mechanical | Yes |
| Segmenting (0.67) and PRD "one idea per beat" | Narration 8 to 25 s (already rung 2). One new key term per beat at most, counted against the outline's term list | Hard + mostly mechanical | Yes |
| Pretraining (0.78) | Each technical term is introduced (named and glossed) in or before the first beat that uses it. Needs a term list from the outline | Mostly mechanical | Yes |
| Coherence (0.86); Grant "every movement deliberate" | Every on-screen text item shares a key term with that beat's narration. No decorative components. RawScene rate under 10% (SCENE_SPEC §7) | Mostly mechanical | Yes |
| Signaling (0.69) | Each beat with several items sets an emphasis, verdict, highlight or Callout. Section breaks use TitleCard | Hard | Yes |
| Redundancy (0.10, weak) and modality (1.00) | On-screen body text is short. Flag a beat whose on-screen words largely duplicate the narration, or whose on-screen text exceeds N words while narration runs | Hard | Yes |
| Temporal contiguity (1.31) | Each revealed item's key term is spoken within that item's reveal segment. Needs TTS word timings (ChalkDust has them) | Mostly mechanical | Spec + audio |
| Spatial contiguity (0.82); SGA labelling completeness | Every curve in GraphPlot and point in GeometryConstruct has a label (FreeBodyDiagram's schema already requires a label on each force); Callout and ZoomHighlight point at a real `target_id` | Hard | Yes |
| Multimedia (1.35); Grant "what picture" | Share of beats whose component is visual rather than text-only (TitleCard and BulletReveal count as text). Report it; set a floor per topic type | Hard | Yes |
| Practice testing (Dunlosky "most effective"); generative activity (0.71); interpolated questions (Brame) | At least one "pause and predict" or check-for-understanding beat: a question posed, then answered in a later beat | Mostly mechanical | Yes |
| Misconceptions (Muller: refutation d = 0.79) | At least one beat that states a named common mistake and refutes it. The outline names the misconception explicitly | Mostly mechanical (presence); judge or human (is it a *real* misconception?) | Presence only |
| Worked examples (problem videos) | ProblemStatement, then SolutionStep with `n` running in order, each with a `justification`, then AnswerBox with `units`. Answer checked (§6) | Hard | Yes |
| Interleaving (Dunlosky "promising") | Applies to a series of videos, not one. Out of scope per video | — | No |
| Length (Brame quoting Guo: 6 minutes) | Total narration under about 6 minutes for a single concept; longer topics split into parts | Hard | Yes |
| Personalization (1.00) | Proxy: second-person address and contractions in narration; no long passive sentences | Mostly mechanical | Yes |
| Voice (0.74) | Friendly human-sounding voice. This depends on the TTS, not the spec | Human | No |
| Motivation; non-obvious insight (Grant, SoME1) | Is there a reason to care in the first 30 s? Is there an "aha"? | Judge (rubric item) + human | No |
| Correctness of the explanation | Facts and reasoning right | Hard for maths (§6); human for prose | Partly |

**What the spec cannot tell you:** whether the motivation lands, whether the misconception is the one students actually hold, whether the order is the best one for a JEE or CBSE student, whether the voice is pleasant, and whether the visual metaphor clarifies. Those stay with people (§8.5).

**Gaming risk.** Each check has a cheap way to pass: a token "common mistake" beat, a question nobody needs to think about, a concrete example that is just a number. Before any of these enters a reward or best-of-N selection for the outline, have Lokavya and Shivam audit a sample of passing outlines for hollow passes. Report the audit rate next to the check's pass rate.

---

## 6. Maths accuracy checking

### 6.1 What to check, and with what

| Check | Where in the spec | Tool | Notes |
|---|---|---|---|
| **Final answer vs key** | `AnswerBox.value`, `AnswerBox.option` | **Math-Verify** (Hugging Face, Apache-2.0): extracts the answer, parses LaTeX to SymPy through `latex2sympy2_extended` (ANTLR4), compares by string, then numerically within a tolerance, then by symbolic simplification. It handles sets, intervals, matrices and relations | Its matching is deliberately asymmetric (gold vs prediction). Set the direction on purpose. Needs a key: for exam-style topics, solve once with a computer-algebra script or take the textbook answer |
| **Step equivalence** | `EquationDerivation.steps[]`, `SolutionStep.work` | SymPy `parse_latex`. Then for expressions: `simplify(a - b) == 0`, plus a random-substitution numeric check (fast, and catches simplify's misses). For equations: compare `lhs - rhs` across steps, up to a non-zero factor, or compare solution sets | Not every valid step is an equivalence (squaring both sides, dividing by a variable). Label each step as "equivalent", "implies" or "unchecked" rather than failing it. The idea that step checks must allow one-way steps appeared in search results for arXiv 2608.28725 but was not in the abstract fetched **[unverified]** |
| **Parser honesty** | all LaTeX | SymPy docs: the ANTLR backend "may fail to fully parse an expression, yet not throw a warning", for example `parse_latex(r'x -')` returns `x`. Use `strict=True` (ANTLR) or the Lark backend, which raises on ill-formed input. Both are marked "experimental" | **A parse failure counts as "unchecked", not "pass".** Report coverage (share of steps checked) separately from the pass rate. If maths moves to Typst (`rendering-2026-11.md`), keep a LaTeX or SymPy form in the spec for checking **[design]** |
| **Units / dimensions** | `AnswerBox.units`, `UnitBreakdown`, `ProblemStatement.given[]` | A units library: `sympy.physics.units` (unit conversion and dimensions, SymPy 1.14.0 docs), Pint (BSD) or unyt | Whether each library raises on adding incompatible units was not confirmed in the docs fetched **[unverified]**. Unit checks are necessary but not sufficient: Foster and Tregeagle (arXiv 1807.07643) note code can be "unit-compatible, but still physically meaningless" because many quantities share units. (Torque and energy, both N·m, is my example, not theirs.) |
| **Numbers consistent across beats** | `given[]` values vs their use in later steps | String and number matching | Catches the "polished-looking error" class: a value that changes between the problem and the solution |
| **Physics sanity** | `FreeBodyDiagram.forces[]` vs the stated acceleration | Vector sum with sympy | Proposal only **[design]** |

### 6.2 Evidence that this beats model checking

- BioDimBench (ICML 2026, 500 problems, 3,000 candidate solutions): answer-only verification caught only 60% of invalid solutions (invalid recall 0.600). Numeric-plus-unit and step-aware verification reached 1.000.
- LLM step checking is unreliable: Tyen et al. (models cannot locate errors) and arXiv 2608.28725 (75.6% to 85.3% false rejection of valid traces by GPT-OSS 20B, Qwen3-14B and Phi-4-Reasoning).
- MATH-VF (arXiv 2505.20869) verifies each step with "a Computer Algebra System and an SMT solver" after an LLM formalises it. The abstract does not name SymPy or give numbers. It is the published version of the "LLM formalises, CAS checks" pattern, which ChalkDust does not need: its steps are already structured LaTeX.

**For the PRD bar** ("final answer verified symbolically or against a key"), this means: every problem video gets Math-Verify against a key, every EquationDerivation gets sympy step labels, and the report shows coverage. A video with unchecked steps is "not verified", not "verified".

---

## 7. 3Blue1Brown data and code: what can be used

| Source | What it is | Licence (fetched) | (a) Commercial use | (b) In a paid closed app | (c) As training data for a shipped model | (d) Obligations |
|---|---|---|---|---|---|---|
| **`3b1b/videos`** (GitHub) | Grant's Manim scene code for the channel | **CC BY-NC-SA 4.0** (README badge and `LICENSE.txt`). "NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation." ShareAlike (3(b)): an adapter's licence "must be a Creative Commons license with the same License Elements" | **No** | **No.** Copying or adapting the code into the product is commercial use, and SA would force the same licence | **Not settled by anything fetched.** Whether training on NC code is "sharing Adapted Material" is a legal question. Treat it as no without permission. The FAQ routes other uses to "a licensing inquiry" via the contact form | Attribution, NC, SA |
| **3b1b videos themselves** | The YouTube content | FAQ: clips under 60 s with on-screen attribution and a link; anything else needs a licensing inquiry; no re-uploads | Clips only, with attribution | No | Not covered; ask | Attribution |
| **`BibbyResearch/3blue1brown-manim`** (Hugging Face) | A CSV of prompt-to-Manim-code pairs. The card lists 12 upstream community datasets (bespokelabs/bespoke-manim, generative-manim human feedback, ManimML and others). **None is 3b1b's own code.** `dalle2/3blue1brown-manim` resolves to the same repo | Tag `apache-2.0`, but the card says "Individual components retain their original licenses from their respective sources." Gated (auto-approve). Last modified 31 Oct 2026 | Depends on each upstream source **[unverified]** | Only after auditing all 12 upstream licences | Same | The card's own note: "I would recommend not to use this dataset anymore." Its stats are inconsistent ("2400+" examples, while the per-source counts add to over 60,000) |
| **`pravsels/manim_3b1b_code`** (Hugging Face) | 123 rows with `file_path` and `content` columns (metadata only was fetched) | No licence tag. The name and columns suggest it copies `3b1b/videos`, in which case CC BY-NC-SA 4.0 applies **[unverified: contents not opened]** | No | No | Treat as no | — |
| **MMMC** (`YanzheChen/MMMC`, linked from Code2Video) | 3b1b-derived benchmark topics and clips | No licence tag | Unclear | No | Unclear | Use the **topic list** for internal evaluation at most |
| **TheoremExplainBench** (`TIGER-Lab/TheoremExplainBench`) | 240 theorem prompts (subject, difficulty, description) | MIT | Yes | Yes | Yes | Keep the MIT notice. Good as a held-out **topic list**, though it is university-level, not JEE/CBSE |

**What ChalkDust can do.**
- Watch 3b1b videos, read the FAQ advice, and write original components that teach in the same spirit. Implementing a visual idea from scratch in your own code is a different act from copying code; where the line falls for a specific reimplementation is a legal question this note did not settle.
- Do not paste or adapt `3b1b/videos` code. Do not fine-tune on it or on datasets containing it.
- Manim itself is a separate matter: the 3b1b README says Manim is "open source software and under the MIT license", and `rendering-2026-11.md` verified Manim CE as MIT.

---

## 8. Proposed measurement protocol for ChalkDust experiments

### 8.1 Rules

1. **Held-out topics.** Fix an evaluation set (the techniques note suggests 150 to 300 topics across JEE/CBSE physics and maths, first-year CS and chemistry) before any tuning. Never train or tune prompts on it.
2. **Same topics, same seeds, every arm.** Compare arms on paired items and report paired differences.
3. **Report per-sample rates before selection.** Among accepted beats, the pass rate after best-of-N is 100% by construction and says nothing; the informative numbers are the per-sample rate and the share of beats where all N samples failed and the beat degraded. Report pass@1 per sample, N-to-first-pass, and the cost (samples, tokens, wall clock) per accepted beat.
4. **Counts with intervals.** Report "41/50 (82%, 95% Wilson CI 69 to 90%)", not "82%". With 20 human pairwise votes, a 15-5 split has a 95% interval of about 53% to 89%, so small studies only detect large differences.
5. **Judges stay out of the reward.** This is consistent with `techniques-2026-11.md`. Judges and VLM flags are evaluation-only.
6. **Content floor.** Report component mix, RawScene rate and the content-floor score next to pass rates. A model that passes by emitting minimal specs is failing.

### 8.2 Hard metrics (deterministic)

Use the refusal kinds as the taxonomy, so every failure is attributed to a rung.

| Metric | Computed from | Report |
|---|---|---|
| Schema pass@1 (rung 1) | pydantic, `load_spec` | rate + CI per job and per model |
| Semantic pass@1 by kind (rung 2) | `capacity`, `duration`, `invalid_latex`, `carry_in`, `region_conflict`, `toolchain` | rate per kind; `toolchain` reported separately as a machine problem |
| Geometric pass@1, before and after mechanical repair (rung 3) | `overflow`, `illegible`, `out_of_bounds`, `overlap`, `unrenderable_text` | both rates per kind. PRD target: first-attempt layout pass rate above 90% |
| `build_error` count | any rung | **must be 0**; it is a bug, not a refusal |
| Render success | render stage | rate; failures listed by beat |
| Repair-loop outcomes | repair loop | share fixed by mechanical repair, share fixed by regeneration, share degraded |
| RawScene and degrade rate | spec and logs | PRD target under 10% |
| Samples, tokens and seconds per accepted beat | generation logs | median and 90th percentile, per machine A/B/C |
| Content floor | coverage of outline items, size band, component fit (techniques note) | mean + distribution |
| Maths: answer-key match | Math-Verify | rate over problem videos |
| Maths: step check | sympy | coverage (checked/total) **and** pass among checked |
| Units | units library | rate over answers with units; coverage |
| Pedagogy spec checks (§5.3) | spec | per-check pass rate, plus the hollow-pass audit rate |
| Soft geometry (reported, not gated) | probe output | STAGE occupancy band, grid spread, transient-overlap time |
| A/V timing (PRD §7) | word timings, beat frames | max offset at beat boundaries (target ±150 ms), dead air over 1.5 s |

### 8.3 Soft metrics (judged)

- **LLM judge on the outline and script**: pairwise A vs B, run in **both orders** (count only consistent verdicts), with a model from a **different family** from the generator, against item-level rubric questions such as "Does the first content beat show a concrete example?" or "Is the misconception one a student would really hold?". Report the both-order consistency rate. Use it only on dimensions where it has matched the human raters on a calibration set (§8.6).
- **VLM frame flags**: only after §4.4 calibration, only as warnings, and only for defect types where it met the bar.
- **Do not use**: single-video 1-to-10 quality scores, a judge from the generator's own family, or TeachQuiz-style VLM quizzes reported as "learning".

### 8.4 How to report an experiment

One table per experiment. Columns are arms; rows are the hard metrics above with counts and intervals; then the soft metrics, with the judge's calibration agreement printed next to them; then the human scores (§8.5) with inter-rater agreement. Add a short "failures seen" list with beat IDs. Put the judge row below the hard rows and never fold it into a single overall score.

### 8.5 What Lokavya and Shivam should watch and score

**Set-up.**
- Each rater watches independently, then the two compare notes. Do not discuss before both have scored.
- Hide which arm or model made the video. Randomise the order. A third person, or a script, holds the key.
- When comparing two arms, watch both videos for the same topic back to back, in random order, and answer the pairwise question.
- Watch at 1x, full screen, on the actual target display at least once (one rater on a laptop screen).

**The 5-minute rubric** (per video, filled in while and right after watching; a 3-minute video plus about 2 minutes of scoring):

*A. Fatal flags* (yes/no; note the timestamp of each). Any "yes" means the video fails the PRD bar.

| Flag | Question |
|---|---|
| F1 Wrong | Is any maths, physics, chemistry or CS statement wrong, or is the final answer wrong? |
| F2 Unreadable | Is any text or equation overlapped, cut off, too small to read, or garbled (boxes, broken symbols)? |
| F3 Mismatch | Does the narration describe something that is not on screen, or does the screen show something the narration never explains? |
| F4 Sync | Does audio drift from the animation, cut off mid-word, or leave dead air over about 1.5 s? |
| F5 Lost | At any point, could a student who knows the prerequisites not follow the next step? |

*B. Five 1-to-5 scores*, each with anchors. 1 = would not show a student. 3 = acceptable but unremarkable. 5 = as good as a good teacher's board work.

| Score | Question | Grounding |
|---|---|---|
| S1 Hook | Did it open with a concrete problem or example and a reason to care, not a definition? | Grant FAQ; PRD §7 |
| S2 Build | One idea at a time, each beat building on what is already on screen, ending with a resolution? | Segmenting; PRD §7 |
| S3 Visuals earn their place | Does each visual explain something, rather than decorate? | Multimedia, coherence; Grant "every movement deliberate" |
| S4 Exam use | Would this help a JEE/CBSE student answer an exam question on the topic, including avoiding a common mistake? | Worked examples; misconceptions |
| S5 Pace and voice | Comfortable pace, clear and friendly voice? | Voice principle; Guo/Brame on length |

*C. Overall*
- "Would you send this to a student who has this exam tomorrow?" (yes/no)
- The single biggest problem, in one line.
- For a pairwise comparison: "Which of the two would you rather a student watch?" (A / B / no difference)

**Agreement to report.** Cohen's kappa on each fatal flag and on the send-it question; weighted kappa or Spearman on S1 to S5; the share of identical pairwise picks. Expect moderate agreement at best (§2.5). If kappa on a flag is low, sharpen its wording before the next round rather than averaging the noise.

**How much to watch.** Per comparison, aim for 15 to 20 topics rated by both raters. Use it to detect large differences (§8.1, rule 4). Use the hard metrics for anything finer.

### 8.6 Judge calibration (once, then periodically)

1. Take the first 30 human-rated videos (both raters).
2. Run the candidate judge on the same rubric items, both orders for pairwise items.
3. Report per item: Spearman with the mean human score, pairwise agreement, and both-order consistency. Compare with the human-human agreement on that item.
4. Keep the judge only on items where its agreement with the humans is at least as high as the humans' agreement with each other. Re-check whenever the generator model or prompts change.

---

## 9. Gaps and what was not verified

- No study found measures a 2B to 8B VLM on overlap, clipping or legibility. §4.4 is the test to run.
- GGUF sizes for Qwen3.5-4B, MiniCPM-V 4.7 1B, InternVL3.5-8B and Moondream 3.1; runtime memory with image tokens; speed on machines A, B and C.
- llama.cpp support for Qwen3-VL and MiniCPM-V 4.7: GGUFs exist, but the multimodal doc's list does not name them.
- The MiniCPM-V 4.7 LICENSE file was not opened (the card tag says apache-2.0).
- The SGA code: the project page returned 404, and no licence is stated.
- Fyfe et al. (2014) findings and Atkinson et al. (2000) principles were not fetched. The Muller et al. (2008) abstract came from a blog reproduction.
- Pint's and sympy's behaviour on dimensionally inconsistent addition was not confirmed from docs.
- The "one-directional steps" point attributed to arXiv 2608.28725 came from a search snippet only.
- TMA.live criticism (October notes) was not re-checked.
- Whether training on CC BY-NC-SA code for a commercial model is allowed is a legal question; not answered here.
- Every check in §5.3, the calibration test in §4.4 and the protocol in §8 are designs. None has been run.

---

## Sources

### A. Fetched on 17 Nov 2026

**Explainer-video systems and benchmarks**
- https://arxiv.org/html/2502.19400v2 (TheoremExplainAgent)
- https://github.com/TIGER-AI-Lab/TheoremExplainAgent
- https://huggingface.co/api/datasets/TIGER-Lab/TheoremExplainBench
- https://arxiv.org/abs/2510.01174 and https://arxiv.org/html/2510.01174 (Code2Video)
- https://github.com/showlab/Code2Video
- https://huggingface.co/api/datasets?search=MMMC&limit=20
- https://arxiv.org/abs/2601.13453 and https://arxiv.org/html/2601.13453 (PhysicsSolutionAgent)
- https://arxiv.org/abs/2607.18116 and https://arxiv.org/html/2607.18116 (SGA)
- https://jhedlp.github.io/projects/sga/ (returned 404)
- https://arxiv.org/html/2606.30296v1 (ManimAgent)
- https://arxiv.org/html/2610.35864v1 (VibeVideoBench)
- https://arxiv.org/abs/2604.05005 and https://arxiv.org/html/2604.05005v2 (EduIllustrate)
- https://arxiv.org/abs/2608.08852 and https://arxiv.org/html/2608.08852 (Teaching Monster Challenge)
- https://arxiv.org/abs/2607.18529 and https://arxiv.org/html/2607.18529 (EduPanel)
- https://arxiv.org/abs/2605.26918 (EduVideoBench)
- https://arxiv.org/abs/2603.03066 (EduVQA)

**Judge reliability**
- https://arxiv.org/abs/2306.05685 (MT-Bench, Zheng et al.)
- https://arxiv.org/abs/2305.17926 (Wang et al., position bias)
- https://arxiv.org/abs/2406.07791 (Shi et al., position bias)
- https://arxiv.org/abs/2404.04475 (Length-Controlled AlpacaEval)
- https://arxiv.org/abs/2404.13076 (Panickssery et al., self-preference)
- https://arxiv.org/abs/2412.09416 and https://arxiv.org/html/2412.09416v2 (MRBench)
- https://arxiv.org/abs/2510.23477 and https://arxiv.org/html/2510.23477v2 (MMTutorBench)
- https://arxiv.org/abs/2407.06581 (VLMs are blind)
- https://arxiv.org/abs/2510.22045 (VLM-SlideEval)
- https://arxiv.org/abs/2512.02624 (PPTBench)
- https://arxiv.org/abs/2608.26716 (Beyond Atomic Layouts; consulted, not cited above)
- https://arxiv.org/abs/2312.05291 (GlitchBench; abstract has no numbers; consulted, not cited above)
- https://arxiv.org/abs/2311.08516 (Tyen et al.)
- https://arxiv.org/abs/2608.28725 (Mazdarani and Toxtli)

**Small VLMs and licences**
- https://huggingface.co/api/models?author=Qwen&search=VL&sort=createdAt&direction=-1&limit=40
- https://huggingface.co/api/models?author=Qwen&pipeline_tag=image-text-to-text&sort=createdAt&direction=-1&limit=50
- https://huggingface.co/Qwen/Qwen3.5-4B
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- https://huggingface.co/api/models/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main
- https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct-GGUF/tree/main
- https://huggingface.co/api/models?author=google&search=gemma-4&sort=createdAt&direction=-1&limit=50
- https://huggingface.co/google/gemma-4-E4B-it
- https://huggingface.co/api/models/google/gemma-4-E4B-it-qat-q4_0-gguf/tree/main
- https://huggingface.co/api/models?author=openbmb&search=MiniCPM-V&sort=createdAt&direction=-1&limit=30
- https://huggingface.co/openbmb/MiniCPM-V-4.7-1B
- https://huggingface.co/api/models?author=HuggingFaceTB&search=SmolVLM&sort=createdAt&direction=-1&limit=30
- https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
- https://huggingface.co/api/models?author=OpenGVLab&search=InternVL&sort=createdAt&direction=-1&limit=30
- https://huggingface.co/OpenGVLab/InternVL3_5-8B
- https://huggingface.co/api/models?author=moondream&sort=createdAt&direction=-1&limit=20
- https://huggingface.co/api/models?author=vikhyatk&sort=createdAt&direction=-1&limit=20
- https://huggingface.co/moondream/moondream3.1-9B-A2B
- https://moondream.ai/licenses/model/1.0
- https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- https://www.apache.org/licenses/LICENSE-2.0

**3Blue1Brown data and code**
- https://github.com/3b1b/videos
- https://raw.githubusercontent.com/3b1b/videos/master/LICENSE.txt
- https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en
- https://www.3blue1brown.com/faq/
- https://www.3blue1brown.com/blog/some1-results
- https://huggingface.co/api/datasets?search=3blue1brown&limit=50
- https://huggingface.co/datasets/BibbyResearch/3blue1brown-manim
- https://huggingface.co/api/datasets/BibbyResearch/3blue1brown-manim
- https://huggingface.co/api/datasets/dalle2/3blue1brown-manim
- https://huggingface.co/api/datasets/pravsels/manim_3b1b_code

**Learning science**
- https://link.springer.com/content/pdf/10.1007/s10648-023-09842-1.pdf and https://link.springer.com/article/10.1007/s10648-023-09842-1/tables/6 (Mayer 2024, Table 6)
- https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5132380/fullTextXML (Brame 2016)
- https://api.semanticscholar.org/graph/v1/paper/DOI:10.1187/cbe.16-03-0125 (Brame metadata)
- https://www.aft.org/ae/fall2013/dunlosky (Dunlosky, American Educator 2013)
- https://www.psychologicalscience.org/news/releases/which-study-strategies-make-the-grade.html (APS release on Dunlosky et al. 2013)
- https://api.semanticscholar.org/graph/v1/paper/DOI:10.1111/j.1365-2729.2007.00248.x (Muller et al. metadata)
- https://perticles.blogspot.com/2008/01/muller-sharma-reimann-jcal-2008.html (blog reproducing the Muller et al. abstract; secondary)
- https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/s10648-014-9249-3 (Fyfe et al. metadata only)

**Maths checking**
- https://github.com/huggingface/Math-Verify
- https://docs.sympy.org/latest/modules/parsing.html
- https://docs.sympy.org/latest/modules/physics/units/index.html
- https://github.com/hgrecco/pint
- https://pint.readthedocs.io/en/stable/user/defining-quantities.html (no `DimensionalityError` example found)
- https://arxiv.org/abs/2505.20869 (MATH-VF)
- https://icml.cc/virtual/2026/82501 (BioDimBench)
- https://arxiv.org/abs/1807.07643 (Foster and Tregeagle)

**Local files**
- `docs/research/chalkdust-research-2026-10.md`, `docs/research/techniques-2026-11.md`, `docs/research/rendering-2026-11.md`, `docs/research/models-2026-11.md`, `docs/SCENE_SPEC.md`, `docs/PRD.md`, `CLAUDE.md`

### B. Seen in search results only, not fetched (do not rely on these without checking)

- TeachMaster, arXiv 2601.04204
- LED layout-error benchmark, arXiv 2603.17265
- OCRBench v2, arXiv 2501.00321 (overlapping-text weakness)
- A Qwen3-VL-8B QLoRA "slide examiner" said to abstain on pixel geometry (friendli.ai listing)
- Fyfe et al. 2014 abstract text (ERIC EJ1054508; fetch failed twice)
- Atkinson, Derry, Renkl and Wortham 2000, doi 10.3102/00346543070002181
- Muller et al. 2008 publisher page (403)
- Hard2Verify, arXiv 2510.13744; Putnam-AXIOM, arXiv 2508.08292; SageMath agents, arXiv 2607.06820; TPBench test-time scaling, arXiv 2506.20729
- Chain of Unit-Physics, arXiv 2512.01010; unyt (JOSS 10.21105/joss.00809); CAMEL physics verifier docs
- The "one-directional steps" description of arXiv 2608.28725
