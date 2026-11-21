"""Cues: when, inside a beat, each moment of an animation starts.

An animation component (scenes/components/animation.py) splits its beat into
named *moments*: "the second point appears", "the secant starts to turn". A
moment should start as the voice reaches the words that describe it, not at an
even share of the beat. This module turns words into times. It is pure: no
Manim, no audio, so the semantic rung, the render worker and the tests all get
the same answer from the same inputs.

Where a moment's start comes from, in order of precedence:

  1. A cue phrase from the spec (`cues: {"slide": "move it closer"}`): the
     moment starts as the first word of that phrase is spoken.
  2. Otherwise, sentence order: moment k starts with sentence k of the
     narration, when the narration has at least as many sentences as the
     phase has moments. Writing one sentence per moment is the whole of the
     contract for a writer who gives no cues.
  3. Otherwise, the moment has no anchor and the timeline places it between
     its anchored neighbours by its default weight (scenes/base.Timeline).

A word's time comes from the voice's own word timings when the backend
reports them (Kokoro does; core/models.WordTiming), and is estimated from its
character offset in the narration otherwise (the development voices, and
every check that runs before speech). An estimate never refuses anything: the
only refusals here are about the text (a cue phrase that is not in the
narration, or cues out of order), and those are decidable without audio, so
they are caught at rung 2 before any speech is synthesised (SCENE_SPEC.md §8).
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from chalkdust.core.models import WordTiming

# A token is a run of letters/digits, with inner apostrophes and decimal points
# kept ("can't", "3.5"). Punctuation is never a token: the voice does not say it.
_TOKEN = re.compile(r"[^\W_]+(?:['’.][^\W_]+)*", re.UNICODE)
# A sentence ends at . ! ? (or a run of them) followed by whitespace or the end.
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")

# Cues closer than this many words would give a moment too little time for a
# viewer to register it (scenes/base.Timeline enforces a floor in frames too).
MIN_WORDS_BETWEEN_CUES = 2


def norm(word: str) -> str:
    """The comparable form of a word: lower case, letters and digits only."""
    return "".join(c for c in word.lower() if c.isalnum())


@dataclass(frozen=True)
class Token:
    text: str      # as written
    key: str       # norm(text)
    start: int     # character offset in the narration
    sentence: int  # 0-based sentence index


def tokenize(narration: str) -> list[Token]:
    """The narration's spoken words, with their character offsets and the
    sentence each belongs to."""
    ends = [m.end() for m in _SENTENCE_END.finditer(narration)]
    out = []
    for m in _TOKEN.finditer(narration):
        key = norm(m.group())
        if not key:
            continue
        sentence = sum(1 for e in ends if e <= m.start())
        out.append(Token(m.group(), key, m.start(), sentence))
    return out


def find_phrase(tokens: Sequence[Token], phrase: str, start: int = 0) -> int | None:
    """Index of the first token of `phrase` in `tokens`, at or after `start`,
    matching whole words with case and punctuation ignored. None if absent."""
    want = [norm(w) for w in _TOKEN.findall(phrase)]
    want = [w for w in want if w]
    if not want:
        return None
    keys = [t.key for t in tokens]
    for i in range(start, len(keys) - len(want) + 1):
        if keys[i:i + len(want)] == want:
            return i
    return None


@dataclass(frozen=True)
class CueProblem:
    """A cue the narration cannot honour. Text-only, so decided at rung 2."""

    moment: str
    message: str


def cue_problems(narration: str, moments: Sequence[str],
                 cues: Mapping[str, str]) -> list[CueProblem]:
    """Every reason the given cue phrases cannot be placed in this narration.

    Cues must name moments of the phase, appear in the narration, and come in
    moment order with at least MIN_WORDS_BETWEEN_CUES words between them. A
    phrase is searched for after the previous cue, so a word that occurs twice
    ("the line ... the line") anchors to the occurrence that keeps the order.
    """
    problems = []
    unknown = [m for m in cues if m not in moments]
    for m in unknown:
        problems.append(CueProblem(m, f"cue {m!r} names no moment of this phase; "
                                      f"its moments are {list(moments)}"))
    tokens = tokenize(narration)
    last: int | None = None
    last_name = ""
    for m in moments:
        if m not in cues:
            continue
        phrase = cues[m]
        at = find_phrase(tokens, phrase, 0 if last is None else last + 1)
        if at is None:
            anywhere = find_phrase(tokens, phrase)
            if anywhere is None:
                problems.append(CueProblem(m, f"cue {m!r}: the phrase {phrase!r} is "
                                              "not in the narration; quote words the "
                                              "voice actually says"))
            else:
                problems.append(CueProblem(m, f"cue {m!r}: {phrase!r} is spoken before "
                                              f"the cue {last_name!r}; moments happen in "
                                              f"the order {list(moments)}"))
            continue
        if last is not None and at - last < MIN_WORDS_BETWEEN_CUES:
            problems.append(CueProblem(m, f"cue {m!r}: {phrase!r} starts "
                                          f"{at - last} word(s) after cue {last_name!r}; "
                                          f"leave at least {MIN_WORDS_BETWEEN_CUES} so "
                                          "each moment can be seen"))
        last, last_name = at, m
    return problems


def anchor_tokens(narration: str, moments: Sequence[str],
                  cues: Mapping[str, str]) -> dict[str, int | None]:
    """Moment -> index of the token it starts at, or None for no anchor.

    Explicit cues win. Without any cue, sentence k anchors moment k when there
    are at least as many sentences as moments (rule 2 above). Cues that cannot
    be placed (cue_problems) anchor nothing; rung 2 has already refused them in
    a real run, and a test or a probe still gets a usable timeline.
    """
    tokens = tokenize(narration)
    out: dict[str, int | None] = {m: None for m in moments}
    if cues:
        last = -1
        for m in moments:
            if m in cues:
                at = find_phrase(tokens, cues[m], last + 1)
                if at is not None:
                    out[m] = at
                    last = at
        return out
    if not tokens:
        return out
    n_sentences = tokens[-1].sentence + 1
    if n_sentences >= len(moments) and len(moments) > 1:
        firsts: dict[int, int] = {}
        for i, t in enumerate(tokens):
            firsts.setdefault(t.sentence, i)
        for k, m in enumerate(moments):
            # Moment 0 starts the beat; it is anchored only by an explicit cue.
            out[m] = firsts.get(k) if k > 0 else None
    return out


def word_times(narration: str, words: Sequence[WordTiming],
               duration: float) -> list[float]:
    """Start time, in seconds, of every token of `narration`.

    With the voice's word timings, tokens are aligned to the timed words by
    sequence matching on their normal forms; a token the voice split or merged
    differently ("can't" as "ca" + "n't") is interpolated between its matched
    neighbours. Without timings, each token's start is estimated from its
    character offset, as a share of `duration`.
    """
    tokens = tokenize(narration)
    if not tokens:
        return []
    total_chars = max(1, len(narration))
    estimate = [duration * t.start / total_chars for t in tokens]
    timed = [(norm(w.word), w.start) for w in words if norm(w.word)]
    if not timed:
        return estimate

    matcher = difflib.SequenceMatcher(a=[t.key for t in tokens],
                                      b=[k for k, _ in timed], autojunk=False)
    known: dict[int, float] = {}
    for block in matcher.get_matching_blocks():
        for j in range(block.size):
            known[block.a + j] = timed[block.b + j][1]
    if not known:
        return estimate
    # Interpolate unmatched tokens by character offset between matched ones,
    # with the beat's edges (0 and duration) as the outer anchors.
    anchors = sorted(known.items())
    out = []
    for i, t in enumerate(tokens):
        if i in known:
            out.append(known[i])
            continue
        before = max(((j, s) for j, s in anchors if j < i), default=None)
        after = min(((j, s) for j, s in anchors if j > i), default=None)
        c0, s0 = (tokens[before[0]].start, before[1]) if before else (0, 0.0)
        c1, s1 = (tokens[after[0]].start, after[1]) if after else (total_chars, duration)
        frac = (t.start - c0) / (c1 - c0) if c1 > c0 else 0.0
        out.append(s0 + frac * (s1 - s0))
    return out


def cue_times(narration: str, moments: Sequence[str], cues: Mapping[str, str],
              words: Sequence[WordTiming], duration: float) -> dict[str, float]:
    """Moment -> start time in seconds, for every moment that has an anchor.

    Moments without an anchor are left out; the timeline places them by
    weight. Times are clamped into [0, duration].
    """
    anchors = anchor_tokens(narration, moments, cues)
    if not any(v is not None for v in anchors.values()):
        return {}
    times = word_times(narration, words, duration)
    return {m: min(max(times[i], 0.0), duration)
            for m, i in anchors.items() if i is not None}
