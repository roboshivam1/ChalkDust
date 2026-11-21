"""Extra T2 statistics quoted in FINDINGS item 11 and section 2c (read-only).

    .lab/venv-harness/Scripts/python research/audit/scripts/t2_extra_stats.py

1. Topic-clustered bootstrap: seeds of one topic are not independent, so resample
   the 6 topics (10,000 draws, RNG seed 20261121) over per-topic mean differences,
   with both methods pooled per topic.
2. Calculator use per arm: how often the plan has an `answer_calc` value, how often
   that value is right (research.content._key_close, the checker's own rule), and how
   often it is right while the shown answer is wrong.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import research.analyze as an                      # noqa: E402
from research.content import _key_close            # noqa: E402
from research.report import video_metrics          # noqa: E402
from research.topics import BY_ID                  # noqa: E402

TUNED, BASE = "granite-4.2-3b-sft-v1", "granite-4.2-3b-self"
recs, _ = an.load("T2-heldout")
recs = [r for r in recs if r.get("rescored")]

# 1. topic bootstrap ------------------------------------------------------------
side = {TUNED: {}, BASE: {}}
for r in recs:
    if r["model"] in side:
        side[r["model"]][(r["topic"], r["seed"], r["method"])] = video_metrics(r)
keys = sorted(set(side[TUNED]) & set(side[BASE]))
print("Topic-clustered bootstrap, SFT minus untuned")
for m in ("fs", "fs_all", "allclean", "ans", "spk", "leak"):
    unit = defaultdict(list)
    for k in keys:
        a, b = side[TUNED][k][m], side[BASE][k][m]
        if a is not None and b is not None:
            unit[k[:2]].append(a - b)
    pairs = {u: sum(v) / len(v) for u, v in unit.items()}
    per_topic = defaultdict(list)
    for (t, _s), d in pairs.items():
        per_topic[t].append(d)
    tm = {t: sum(v) / len(v) for t, v in per_topic.items()}
    tops = sorted(tm)
    rng = random.Random(20261121)
    bs = sorted(sum(tm[rng.choice(tops)] for _ in tops) / len(tops) for _ in range(10_000))
    print(f"  {m:8} pairs {len(pairs):2} (+{sum(d > 0 for d in pairs.values())} / -{sum(d < 0 for d in pairs.values())})"
          f"  topics {len(tops)} (+{sum(v > 0 for v in tm.values())} / -{sum(v < 0 for v in tm.values())})"
          f"  mean {sum(tm.values()) / len(tops):+.3f}  topic CI [{bs[250]:+.3f}, {bs[9749]:+.3f}]")

# 2. calculator use ---------------------------------------------------------------
print("\nCalculator use (number-keyed topics)")
c = Counter()
for r in recs:
    key = BY_ID[r["topic"]].key
    if not key or key.kind != "number" or r["model"] not in (TUNED, BASE, "gemma-4-12b"):
        continue
    plan = json.loads((an.REPO / r["trace_dir"] / "trace.json").read_text(encoding="utf-8"))["plan"]
    cv = plan.get("answer_calc_value")
    has = isinstance(cv, (int, float))
    right = has and _key_close(float(cv), float(key.value), key)
    shown_ok = r["answer"]["status"] == "correct"
    arm = r["model"]
    c[arm, "n"] += 1
    c[arm, "has"] += has
    c[arm, "right"] += right
    c[arm, "right_shown_wrong"] += right and not shown_ok
    c[arm, "counterfactual"] += right if has else shown_ok
for arm in (BASE, TUNED, "gemma-4-12b"):
    n, h = c[arm, "n"], c[arm, "has"]
    print(f"  {arm:24} videos {n:3}  calc present {h:3}  calc right {c[arm, 'right']:3}"
          f" ({c[arm, 'right'] / max(h, 1):.2f} of present)  right but shown wrong {c[arm, 'right_shown_wrong']:3}"
          f"  counterfactual 'calculator decides' {c[arm, 'counterfactual']}/{n} = {c[arm, 'counterfactual'] / n:.2f}")
