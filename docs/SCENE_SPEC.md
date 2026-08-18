# CHALKDUST — Scene Spec

**The contract between the language model and the renderer.**

This is the load-bearing abstraction of the whole system. Everything upstream (script
generation, LLM prompting) and downstream (rendering, validation, caching) is shaped by
what this spec can express.

---

## 1. The core bet

An LLM cannot see the frame it is producing. It will emit Manim code that compiles
perfectly and looks broken — equations overlapping graphs, labels off-screen, three
objects stacked at ORIGIN. Prompt engineering reduces this rate but never eliminates it,
because the model is reasoning about spatial layout it cannot observe.

**So the LLM does not write Manim.** It fills out a form.

We hand-write a library of scene components. Each one owns its own layout logic, is
tested, and is guaranteed not to break. The LLM chooses which component fits a beat and
supplies its content. Layout correctness becomes a property of the library, verified once,
rather than a property of each generation, verified never.

The trade is expressiveness. We accept it, and buy some back with a controlled escape
hatch (§7).

## 2. Object model

```
Video
 └── Beat[]              one narrative idea; the unit of caching and re-render
      ├── narration      text sent to TTS
      ├── audio          generated wav + measured duration + word timings
      ├── scene          one ComponentSpec
      └── transition     how we get to the next beat
```

A beat is 8–25 seconds of narration. Longer means the visual is static too long; shorter
means the cut rate is exhausting. The script generator enforces this.

**Beats are independently renderable.** This is what makes parallel rendering and
incremental re-render possible, and it constrains component design: a component may not
depend on mobject state left behind by the previous beat. Continuity across beats is
expressed declaratively (§6), not by mutation.

## 3. Spec format

Pydantic models, serialized as JSON. The LLM emits JSON; Pydantic validation is the first
gate. Schema violations are caught before anything renders.

```jsonc
{
  "video_id": "cs-hashmap-collisions",
  "channel": "cs_fundamentals",
  "style": "cs_dark_mono",          // resolves to a theme; not free-form colors
  "beats": [
    {
      "id": "b03",
      "narration": "Two different keys can land in the same bucket. That's a collision.",
      "component": "SplitCompare",
      "params": {
        "left":  { "title": "key: \"cat\"",  "body": "hash → 4" },
        "right": { "title": "key: \"act\"",  "body": "hash → 4" },
        "verdict": "same bucket",
        "emphasis": "verdict"
      },
      "carry_in": ["bucket_array"],   // persists from an earlier beat
      "transition": "cross_fade"
    }
  ]
}
```

Note what the LLM is **not** allowed to specify: coordinates, colors, font sizes, scales,
run times, z-index. Those are the component's business. If the model can't set it, the
model can't break it.

## 4. Layout regions

The Manim frame is fixed (16:9). It is partitioned into named regions with a safe-area
margin. Components declare which regions they occupy; the compiler asserts that no two
simultaneously-active components claim the same region.

```
┌─────────────────────────────────────────┐  ← safe margin
│               TITLE_BAR                 │
├──────────────────┬──────────────────────┤
│                  │                      │
│   STAGE_LEFT     │     STAGE_RIGHT      │
│                  │                      │
│        (STAGE — the union of both)      │
├──────────────────┴──────────────────────┤
│              LOWER_THIRD                │
└─────────────────────────────────────────┘
```

Every component gets a `fit_to_region(mobject, region)` helper that scales down to fit
with padding and raises if the result falls below the minimum legible font size. That
raise is a real failure — it means the content is too dense for the component, and the
right fix is splitting the beat, not shrinking the text.

**Rule: no component ever positions by absolute coordinate.** Everything is relative to a
region or to another mobject.

## 5. Component catalog

Target ~15–20. Start with the five marked ★ — they cover a surprising fraction of real
explainer content.

### Universal
| Component | Purpose | Key params |
|---|---|---|
| ★ `TitleCard` | Open/close, section breaks | `title`, `subtitle`, `kicker` |
| ★ `BulletReveal` | Sequential points | `heading`, `items[]`, `reveal` |
| ★ `SplitCompare` | Two things side by side | `left`, `right`, `verdict` |
| `ZoomHighlight` | Focus on part of existing visual | `target_id`, `callout` |
| `Callout` | Annotate something on screen | `target_id`, `text`, `side` |

### Math / science
| Component | Purpose | Key params |
|---|---|---|
| ★ `EquationDerivation` | Step-by-step algebra with transforms | `steps[]`, `annotations[]` |
| ★ `GraphPlot` | Function plotting, axes, tracing | `functions[]`, `x_range`, `markers[]` |
| `NumberLineWalk` | Discrete stepping / intervals | `range`, `steps[]` |
| `VectorField` | Field visualization | `field_fn`, `sample_density` |
| `GeometryConstruct` | Ruler-and-compass style builds | `shapes[]`, `construction[]` |
| `UnitBreakdown` | Dimensional analysis | `quantity`, `decomposition[]` |

### CS
| Component | Purpose | Key params |
|---|---|---|
| `CodeWalk` | Code with line highlighting | `language`, `source`, `highlights[]` |
| `DataStructureViz` | Array / tree / graph / stack ops | `kind`, `initial`, `operations[]` |
| `StepTrace` | Variable state table over time | `variables[]`, `frames[]` |
| `BoxFlow` | System / pipeline diagram | `nodes[]`, `edges[]`, `animate_flow` |

### Problem-solving (JEE branch)
| Component | Purpose | Key params |
|---|---|---|
| `ProblemStatement` | The question, formatted | `text`, `given[]`, `find` |
| `FreeBodyDiagram` | Physics setup | `body`, `forces[]` |
| `SolutionStep` | Numbered step, work shown | `n`, `claim`, `work`, `justification` |
| `AnswerBox` | Final answer, emphasized | `value`, `units`, `option` |

Every component ships with: a Pydantic param model, a `regions()` declaration, a `build()`
method, a snapshot test, and at least one stress-test fixture with deliberately
oversized content.

## 6. Continuity between beats

Since beats render independently, persistence is declarative. A component may register
named artifacts; a later beat requests them via `carry_in`.

```
b02: DataStructureViz  → registers "bucket_array"
b03: SplitCompare      → carry_in: ["bucket_array"]  (rendered in STAGE, dimmed)
b04: ZoomHighlight     → target_id: "bucket_array"
```

The compiler resolves carry-ins by re-instantiating the artifact deterministically from
its stored construction params — not by serializing mobjects. Same inputs, same output,
which is also what makes the cache hash honest.

Keep the carry-in set small. If a beat needs three carried artifacts, the beat is doing
too much.

## 7. The escape hatch

Some beats genuinely need something the library doesn't have. Those get `RawScene`:

```jsonc
{
  "component": "RawScene",
  "params": {
    "rationale": "Non-standard 3D surface with parametric sweep",
    "code": "class Beat07(Scene): ..."
  }
}
```

Rules:
- Executes in a subprocess with a timeout and constrained imports.
- Subject to the same post-render layout assertions as everything else.
- Failure degrades to `BulletReveal` with the narration as content. **A degraded beat is
  always better than a failed video.**
- **Every use is logged.** The log is the roadmap for the component library. If
  `RawScene` fires three times for similar visuals, that's a missing component, and
  building it is higher priority than whatever else is queued.

Target: under 10% of beats. If it's consistently above that, the library is too small and
we're back to the failure mode we're trying to escape.

## 8. Validation ladder

Cheap checks first. Each rung is a gate; failing rungs feed the repair loop.

**1. Schema (free)** — Pydantic. Unknown component, missing params, wrong types.

**2. Semantic (cheap)** — Narration duration vs. component's animation budget. Carry-in
references a registered artifact. Region conflicts. Text volume vs. component capacity.
LaTeX compiles standalone.

**3. Geometric (post-build, pre-render)** — Build the scene, snapshot mobject bounding
boxes at each settle point:
- every mobject inside safe area
- no intersection between groups tagged mutually exclusive
- effective font size ≥ minimum after all scaling
- no mobject at effectively zero size or fully occluded

**4. Visual (draft render, sampled)** — Render 480p15, sample keyframes, run a VLM check
for "does this look broken." Expensive; use it on sampled beats and on anything that
came through `RawScene`.

**5. Content (parallel)** — Symbolic verification for math. Answer-key cross-check for
problem videos. Blocks publish, not render.

## 9. Repair loop

Bounded at three attempts, escalating:

1. **Mechanical** — compiler auto-fixes: scale to fit, nudge into region, wrap text.
   No LLM involved. Handles most geometric failures.
2. **Regenerate** — send the LLM the beat spec, the failure message, and a rendered
   frame. Ask for a revised spec. Often it picks a different component or splits content.
3. **Degrade** — fall back to the simplest component that carries the narration.

Then always continue. The pipeline never dies on one beat. Failures are logged and
surfaced in the review gate as "beat 7 degraded — check it."

## 10. Versioning

The cache key for a beat includes the component library version. Bumping the library
invalidates every affected beat, which is correct but expensive — so version
deliberately:

- **Patch** — bug fix, no visual change. Cache preserved via explicit allowlist.
- **Minor** — new component or new optional param. Existing beats unaffected.
- **Major** — changed layout behavior of existing components. Full invalidation.

Style themes version independently of the library.

## 11. Design rules for new components

1. It renders correctly, or it raises. It never renders something broken.
2. It never takes a coordinate, color, or font size from the spec.
3. It handles its worst realistic input — test with content 3× the expected volume.
4. Its animation duration is derived from the beat's audio duration, not hardcoded.
5. It's independently renderable; no dependency on prior scene state except via `carry_in`.
6. It has a snapshot test that fails loudly when Manim's behavior shifts underneath it.
