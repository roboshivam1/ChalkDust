"""The validator as an RL reward, served on localhost.

The trainer runs in .lab/venv-train (torch, trl); the validator needs the
chalkdust package and Manim, which live in .lab/venv-harness. This server
runs in the harness venv and scores beats for the trainer:

    .lab/venv-harness/Scripts/python -m research.reward_server --port 8099

POST /score  {"items": [{"component": "...", "completion": "<model text>"}, ...]}
  -> {"rewards": [float, ...], "details": [{level, clean, flags, words, ...}, ...]}

Reward (the gate x floor in METHODOLOGY.md section 6):
  invalid JSON / wrong shape        0.0
  validator level 0-2 (refused)     0.05 * level        (0, 0.05, 0.10)
  level 3 (valid) with flags        0.5
  level 3 and clean                 1.0
  x content floor: 0.5 if the narration is under 30 words or the params hold
  under 20 characters of text, so the model cannot win by saying little.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from research.llm import parse_json
from research.score import ensure_miktex_first, score_many

_LOCK = threading.Lock()
_N = [0]


def reward_of(sc, beat: dict) -> tuple[float, dict]:
    words = len(str(beat.get("narration", "")).split())
    d = {"level": sc.level, "clean": sc.clean, "flags": list(sc.flags), "words": words,
         "text_chars": sc.text_chars, "findings": [f.get("kind") for f in sc.findings][:4],
         "rung1": sc.rung1_errors[:2], "harness_fault": sc.harness_fault}
    if sc.level < 3:
        r = 0.05 * sc.level
    else:
        r = 1.0 if sc.clean else 0.5
    if words < 30 or sc.text_chars < 20:
        r *= 0.5
        d["floor"] = True
    return r, d


def score_items(items: list[dict], work_root: Path) -> dict:
    beats, bad = [], {}
    for i, it in enumerate(items):
        v, err = parse_json(it.get("completion", ""))
        if not isinstance(v, dict) or not isinstance(v.get("params", {}), dict):
            bad[i] = err or "not an object"
            beats.append(None)
            continue
        beats.append({"component": it["component"], "narration": v.get("narration", ""),
                      "params": v.get("params", {})})
    with _LOCK:
        _N[0] += 1
        batch = _N[0]
    jobs = [(b, "b01", work_root / f"g{batch}" / f"s{i}") for i, b in enumerate(beats) if b is not None]
    scores = iter(score_many(jobs)) if jobs else iter(())
    rewards, details = [], []
    for i, b in enumerate(beats):
        if b is None:
            rewards.append(0.0)
            details.append({"parse_error": bad[i]})
            continue
        r, d = reward_of(next(scores), b)
        rewards.append(r)
        details.append(d)
    return {"rewards": rewards, "details": details}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()
    ensure_miktex_first()
    work_root = Path(tempfile.mkdtemp(prefix="cd-reward-", dir=str(Path(__file__).resolve().parents[1] / ".lab")))

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            out = json.dumps(score_items(body.get("items", []), work_root)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    print(f"reward server on 127.0.0.1:{a.port}, work dir {work_root}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
