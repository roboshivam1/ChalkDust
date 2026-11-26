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


def _run_queue(qdir: Path, **extra_env: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "QUEUE_DIR": str(qdir),
           "QUEUE_GPU_BUSY_MIB": str(10**9), "QUEUE_MIN_RAM_GIB": "0",
           "QUEUE_GUARD_MIN_AVAIL_GIB": "0", "QUEUE_GUARD_MIN_COMMIT_GIB": "0",
           "QUEUE_POLL_SECS": "0.2", **extra_env}
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


def test_importing_the_module_starts_no_queue(tmp_path):
    """A review agent once imported research.gpu_queue to read a constant and started
    a second live queue (2026-11-21)."""
    marker = tmp_path / "ran"
    (tmp_path / "jobs.json").write_text(json.dumps([_job("j", f"open({str(marker)!r}, 'w').close()")]),
                                        encoding="utf-8")
    env = {**os.environ, "QUEUE_DIR": str(tmp_path)}
    r = subprocess.run([sys.executable, "-c", "import research.gpu_queue"], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert not marker.exists() and not (tmp_path / "queue.log").exists()


def test_second_queue_on_same_dir_is_refused_other_dir_runs(tmp_path):
    import psutil
    d1, d2 = tmp_path / "q1", tmp_path / "q2"
    d1.mkdir(), d2.mkdir()
    (d1 / "jobs.json").write_text(json.dumps([_job("slow", "import time; time.sleep(60)")]), encoding="utf-8")
    (d2 / "jobs.json").write_text(json.dumps([_job("quick", "pass")]), encoding="utf-8")
    env = {**os.environ, "QUEUE_DIR": str(d1), "QUEUE_GPU_BUSY_MIB": str(10**9), "QUEUE_MIN_RAM_GIB": "0",
           "QUEUE_GUARD_MIN_AVAIL_GIB": "0", "QUEUE_GUARD_MIN_COMMIT_GIB": "0", "QUEUE_POLL_SECS": "0.2"}
    first = subprocess.Popen([sys.executable, "-m", "research.gpu_queue"], cwd=REPO, env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import time
        for _ in range(100):                       # until the first queue has started its job
            if (d1 / "queue.log").exists() and "starting slow" in (d1 / "queue.log").read_text(encoding="utf-8"):
                break
            time.sleep(0.1)
        r = _run_queue(d1)
        assert r.returncode == 2, r.stdout + r.stderr
        assert "REFUSED" in (d1 / "queue.log").read_text(encoding="utf-8")
        assert _run_queue(d2).returncode == 0      # a different queue dir is not blocked
    finally:
        for p in psutil.Process(first.pid).children(recursive=True) + [psutil.Process(first.pid)]:
            try:
                p.kill()
            except psutil.Error:
                pass
        first.wait(timeout=30)


def test_memory_guard_kills_job_tree_and_stops_queue(tmp_path):
    """A job whose machine stays short of memory is killed with its children
    (llama-server, scoring workers) and left out of done.txt."""
    import psutil
    pidfile = tmp_path / "child.pid"
    code = ("import subprocess, sys, time; "
            "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
            f"open({str(pidfile)!r}, 'w').write(str(c.pid)); time.sleep(120)")
    (tmp_path / "jobs.json").write_text(json.dumps([_job("hog", code)]), encoding="utf-8")

    r = _run_queue(tmp_path, QUEUE_GUARD_MIN_AVAIL_GIB="1e9", QUEUE_GUARD_SECS="2")
    assert r.returncode == 1, r.stdout + r.stderr
    log = (tmp_path / "queue.log").read_text(encoding="utf-8")
    assert "FAILED hog" in log and "memory guard" in log
    assert not (tmp_path / "done.txt").exists()
    child = int(pidfile.read_text())
    assert not psutil.pid_exists(child)             # the grandchild died with the job
