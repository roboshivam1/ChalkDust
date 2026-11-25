# Animation components: the contract (run 3, November 2026)

The slide components put text up and take it down. An animation component owns
one mathematical picture and changes it while the voice describes the change: a
second point appears, the secant turns into the tangent, the readout settles on
f'(a). This note says what that needed beyond the existing component contract,
what changed and why. The code is `chalkdust/scenes/components/animation.py`,
`chalkdust/scenes/plotting.py`, `chalkdust/core/cues.py` and
`chalkdust/scenes/base.py` (`Moment`, `Timeline`).

## What stays the same

An animation is an ordinary registered `Component`: frozen params with
`extra="forbid"`, no coordinate, colour or size from the spec, `examples()` and
`stress()` walked by `tests/test_layout.py`, render or raise with a typed kind.
The harness reads its schema from the same `Params` model as any other
component. Beats still render independently (D-005), and audio still comes
before animation (D-002).

## What changed, and why

1. **Phases.** One picture usually needs several beats (a beat is 8 to 25 s).
   An animation declares ordered `PHASES`, and each beat plays one. Phase k
   starts exactly where phase k-1 ended, because both are rebuilt from the same
   setup params. Consecutive beats that play consecutive phases therefore join
   with no visible cut, yet no beat reads another beat's mobjects.
   `tests/test_animation.py` checks this per animation on the camera's own
   pixels: the last frame of phase k against the first frame of phase k+1.
   The existing `carry_in` could not do this. It rebuilds a carried artifact
   dimmed, in the free half of STAGE, and refuses a component that claims all
   of STAGE. That is context, not persistence.

2. **Setup and the continuity check.** Every param except `phase` and `cues` is
   the setup. A beat that plays the phase right after the previous beat's phase
   must repeat its setup exactly. Rung 2 refuses it otherwise, as the new kind
   `continuity` (`semantic.check_continuity`). Starting again at the first
   phase, or switching animation, is a deliberate cut and is allowed.

3. **Moments and cues: word-timed motion.** A phase is a list of named moments.
   A moment starts as the voice reaches it:
   - on the first word of its cue phrase, when the spec gives one
     (`"cues": {"slide": "slide the second point"}`);
   - otherwise on sentence k of the narration, when there are at least as many
     sentences as moments, so writing one sentence per moment is enough;
   - otherwise by weight between its anchored neighbours.

   Times come from the voice's real word timings (Kokoro reports them) and are
   estimated from character offsets where there are none (the development
   voices, and every check before speech). A cue phrase that is not in the
   narration, is out of order, or names no moment is refused at rung 2 as the
   new kind `cue`, from the text alone and before any speech is synthesised.

4. **Whole-frame timeline.** `scene.cue_timeline(*moments)` is the word-timed
   sibling of `budget()`. Each moment gets a window in whole frames. Its motion
   runs for its natural length (or fills the window) and the rest is a hold.
   Together the windows are exactly `beat_frames`. Anchors closer than
   `MIN_MOMENT_SECONDS` (0.35 s) are pushed apart, never refused, because the
   textual refusals have already happened. Layout is settled at the end of
   every moment, which is also the start of the next, so both ends of every
   sweep are checked.

5. **Cache key.** Narration stays out of the render key. What the narration now
   changes in the picture is the cue frames, so those are the new key term
   (`beat_render_key(cues=...)`). Beats with no animation keep their keys.

6. **Speech stage.** Kokoro returns per-word timings. A backend that times its
   own words is trimmed by those times (0.05 s before the first word, 0.12 s
   after the last), not by silence detection, so the timings and the audio come
   from one cut. The timings are stored beside the WAV in the TTS cache, and the
   trim rule is part of that backend's TTS key.

7. **Everything drawn is true.** `plotting.Fn` computes values, slopes (a
   five-point central difference) and areas (adaptive quadrature) from the
   setup. A spec cannot type a number onto the screen. Params that would draw a
   false picture are refused as param errors. In SecantToTangent that means a
   point where f is not differentiable (the left and right slopes disagree, as
   for |x| at 0), a secant across a break in f (1/x across 0), and a second
   point too close to see.

8. **Colour roles.** `plotting.ROLES`: function (the curve f), rate (slopes,
   secants, tangents, f'), area (integrals, sums), alert (what breaks), text,
   guide. Each role maps to one palette entry, and a component asks for a role,
   never a colour, so a colour means one thing for a whole video.

9. **Layout.** An animation claims STAGE and LOWER_THIRD: the picture on the
   left 58 percent, and the maths column on the right with one short line at a
   time. Axes fit their window to the function's own values. Tick numbers sit
   outside the plot rectangle, so no curve runs through them.

10. **A 1080p30 tier.** `--quality standard` (1920x1080 at 30 fps) is the
    watchable tier for a student laptop. It sits beside D-006's draft and final
    tiers, which are unchanged. The measured cost of 60 fps against 30 is in the
    run 3 report.

## Adding an animation

Subclass `Animation`. Set `name`, a `Params` subclass of `AnimationParams`
(narrow `phase` to a `Literal` of the phase names and add the setup fields with
validators that refuse false pictures), and `PHASES`. Implement `enter()` (put
the previous phase's end state on screen with `scene.add`, no animation) and
`perform()` (play every moment through the timeline). The cleanest pattern is
the pilot's: one `_state(k)` function builds the end state of phase k from the
setup. `enter` adds `_state(k-1)`, and each phase ends by swapping its animated
pieces for `_state(k)`. `examples()` must start with one example per phase, in
order, with one setup, so the continuity test covers every join.

Before it ships: stress cases that try to make it draw something false, one
reviewer who did not write it (the bar in this document; a failure goes back to
the author once, and a second failure means a different approach or leaving it
out), the snapshot recorded at merge, and a card in `research/animate.py` with
its phases, so the local-model harness can use it (`research/tests/test_animate.py`
fails without one).
