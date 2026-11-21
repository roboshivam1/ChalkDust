"""The GPU queue stops at the first failing job and retries it on the next start.

Later jobs depend on earlier ones (teacher -> SFT set -> train -> export -> eval), so a
failure must not let the rest run on missing inputs or be recorded as done."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _job(name: str, code: str) -> dict:
    return {"name": name, "cmd": [sys.executable, "-c", code]}


def _run_queue(qdir: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "QUEUE_DIR": str(qdir),
           "QUEUE_GPU_BUSY_MIB": str(10**9), "QUEUE_MIN_RAM_GIB": "0"}
    return subprocess.run([sys.executable, "-m", "research.gpu_queue"], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)


def test_failure_stops_queue_and_rerun_retries_it(tmp_path):
    marker = tmp_path / "after-ran"
    jobs = [_job("ok", "pass"),
            _job("bad", "import sys; sys.exit(3)"),
            _job("after", f"open({str(marker)!r}, 'w').close()")]
    (tmp_path / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")

    r = _run_queue(tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert (tmp_path / "done.txt").read_text(encoding="utf-8").split() == ["ok"]
    assert not marker.exists()                      # nothing after the failure ran
    assert "FAILED bad: rc=3" in (tmp_path / "queue.log").read_text(encoding="utf-8")

    jobs[1] = _job("bad", "pass")                   # "fix" the job, start the queue again
    (tmp_path / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    r = _run_queue(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "done.txt").read_text(encoding="utf-8").split() == ["ok", "bad", "after"]
    assert marker.exists()
    assert "ALL DONE" in (tmp_path / "queue.log").read_text(encoding="utf-8")
