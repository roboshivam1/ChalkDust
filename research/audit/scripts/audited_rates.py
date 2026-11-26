"""Audited answer rates for the T2/T3 arms, from the blind-grading consensus files.
One rule for every arm (read-only):

    python research/audit/scripts/audited_rates.py

For each arm and field (shown, spoken):
- point estimate = checker passes - audited checker passes the graders reject
  + checker failures the graders accept. Checker passes nobody audited count as right.
- lower bound = the point estimate minus the unaudited checker passes times the
  control false-pass rate (controls are random checker passes).
Spoken uses the spk denominator: videos whose checker spoken status is correct, wrong
or not_found (n/a is outside it, as in report.video_metrics).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import research.analyze as an                      # noqa: E402

FILES = [REPO / "research/audit/t2-heldout/consensus.jsonl", REPO / "research/audit/t3-heldout/consensus.jsonl"]
rows = [json.loads(line) for f in FILES for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
recs, _ = an.load("T2-heldout")
CHECK = {"shown": ("answer", "checker_shown"), "spoken": ("answer_spoken", "checker_spoken")}

for arm in ("granite-4.2-3b-self", "granite-4.2-3b-sft-v1", "granite-4.2-3b-sft-v1-grpo"):
    vids = [r for r in recs if r["model"] == arm]
    audited = [r for r in rows if r["model"] == arm]
    ctrl = [r for r in audited if r["control"]]
    print(arm)
    for field, (rec_key, ck) in CHECK.items():
        denom = [r for r in vids if field == "shown" or r[rec_key]["status"] in ("correct", "wrong", "not_found")]
        passes = sum(r[rec_key]["status"] == "correct" for r in denom)
        aud = [r for r in audited if field == "shown" or r[ck] in ("correct", "wrong", "not_found")]
        aud_pass = [r for r in aud if r[ck] == "correct"]
        rejected = sum(r["graders_" + field] != "correct" for r in aud_pass)
        rescued = sum(r["graders_" + field] == "correct" for r in aud if r[ck] != "correct")
        fp_rate = sum(r["graders_" + field] != "correct" for r in ctrl if r[ck] == "correct") / max(
            1, sum(r[ck] == "correct" for r in ctrl))
        point = passes - rejected + rescued
        lower = point - (passes - len(aud_pass)) * fp_rate
        n = len(denom)
        print(f"  {field:6} n={n:2} checker {passes / n:.2f} | audited passes {len(aud_pass)}/{passes}, rejected {rejected},"
              f" rescued {rescued}, control false-pass {fp_rate:.2f} -> audited {lower / n:.2f}-{point / n:.2f}")
