"""Sequential GPU queue. One GPU job at a time.

Jobs live in .lab/queue/jobs.json (a list of {"name", "cmd": [...]}) and the
file is re-read before every job, so jobs can be added, removed or reordered
while the queue runs. A finished job's name is appended to done.txt; the
queue exits when every job is done. Console of each job: .lab/queue/<name>.log.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import psutil

REPO = Path(__file__).resolve().parents[1]
Q = REPO / ".lab/queue"
Q.mkdir(exist_ok=True)
JOBS, DONE = Q / "jobs.json", Q / "done.txt"
WAIT_FOR = sys.argv[1] if len(sys.argv) > 1 else ""


def running(tag: str) -> bool:
    # Only python processes: a shell whose command text merely mentions the run
    # (a waiter loop) once matched here and deadlocked the queue.
    for p in psutil.process_iter(["cmdline", "name"]):
        cl = " ".join(p.info["cmdline"] or [])
        if not (p.info["name"] or "").lower().startswith("python"):
            continue
        if "research.run" in cl and tag in cl and "gpu_queue" not in cl and " -c " not in f" {cl} ":
            return True
    return False


while WAIT_FOR and running(WAIT_FOR):
    time.sleep(20)
while True:
    done = set(DONE.read_text(encoding="utf-8").split()) if DONE.exists() else set()
    jobs = json.loads(JOBS.read_text(encoding="utf-8"))
    todo = [j for j in jobs if j["name"] not in done]
    if not todo:
        break
    j = todo[0]
    t = time.time()
    with open(Q / f"{j['name']}.log", "a", encoding="utf-8") as f:
        f.write(f"=== start {time.strftime('%H:%M:%S')} {j['cmd']}\n")
        f.flush()
        cmd = list(j["cmd"])
        if not Path(cmd[0]).is_absolute() and (REPO / cmd[0]).exists():
            cmd[0] = str(REPO / cmd[0])   # Windows CreateProcess ignores cwd= when resolving argv[0]
        rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                                 **j.get("env", {})}).returncode
        f.write(f"=== end rc={rc} {time.time() - t:.0f}s\n")
    with open(DONE, "a", encoding="utf-8") as f:
        f.write(j["name"] + "\n")
    print(f"{time.strftime('%H:%M:%S')} {j['name']}: rc={rc} {time.time() - t:.0f}s", flush=True)
