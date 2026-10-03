# CHALKDUST

An educational video generator. Topic in, finished Manim-animated explainer with synced
AI voiceover out. Designed to eventually run unattended across a fleet of YouTube channels.

> **Codename is a placeholder.** Find-and-replace `CHALKDUST` when you pick a real name.

## Document set

| File | What it answers |
|---|---|
| `PRD.md` | What we're building, for whom, what "good" means, what we refuse to build |
| `ARCHITECTURE.md` | How the system is put together and why it's shaped this way |
| `SCENE_SPEC.md` | The contract between the LLM and the renderer — the most important doc here |
| `ROADMAP.md` | Build order, phase exit criteria, what to deliberately not build yet |
| `OPERATIONS.md` | Fleet config, publishing, quota, policy risk, cost model |
| `DECISIONS.md` | Architecture decision log — why we chose X over Y, and when to revisit |

## The one-paragraph version

LLMs write Manim code that compiles but looks broken — text off-screen, objects
overlapping, labels colliding. The model can't see the frame it's producing. So we don't
ask it to write Manim. We give it a **library of hand-written, layout-safe scene
components** and ask it to select and parameterize them via a validated JSON spec. A
deterministic compiler turns that spec into Manim. Audio is generated first and measured,
then animations are stretched to fit. Everything is content-addressed and cached so
editing one beat re-renders one beat.

## Reading order

If you're picking this up cold: `PRD.md` → `SCENE_SPEC.md` → `ARCHITECTURE.md` → `ROADMAP.md`.

`SCENE_SPEC.md` before `ARCHITECTURE.md` is intentional. The spec is the load-bearing
abstraction; the architecture is mostly plumbing around it.
