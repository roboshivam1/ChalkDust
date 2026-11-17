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


GPU_BUSY_MIB = int(os.environ.get("QUEUE_GPU_BUSY_MIB", "4500"))   # desktop alone uses ~2.4 GB
MIN_RAM_GIB = float(os.environ.get("QUEUE_MIN_RAM_GIB", "8"))


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.split()[0])
    except Exception:
        return None


def wait_for_room() -> None:
    """Start a job only when nothing else holds the GPU and there is RAM to spare.
    The GPU is one lane: a WSL voice job once held 5 GB of VRAM and WSL 15 GB of
    RAM while this queue ran, and Claude Code stopped the queue for low memory."""
    last = 0.0
    while True:
        vram, ram = gpu_used_mib(), psutil.virtual_memory().available / 2**30
        busy = []
        if vram is not None and vram > GPU_BUSY_MIB:
            busy.append(f"GPU has {vram} MiB in use by other programs (starts at <= {GPU_BUSY_MIB})")
        if ram < MIN_RAM_GIB:
            busy.append(f"only {ram:.1f} GiB RAM free (needs {MIN_RAM_GIB:g}; try 'wsl --shutdown' or closing apps)")
        if not busy:
            return
        if time.time() - last > 300:
            print(f"{time.strftime('%H:%M:%S')} waiting: " + "; ".join(busy), flush=True)
            last = time.time()
        time.sleep(30)


if os.name == "nt":   # keep the laptop awake while the queue lives (not the display)
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

while WAIT_FOR and running(WAIT_FOR):
    time.sleep(20)
while True:
    done = set(DONE.read_text(encoding="utf-8").split()) if DONE.exists() else set()
    jobs = json.loads(JOBS.read_text(encoding="utf-8"))
    todo = [j for j in jobs if j["name"] not in done]
    if not todo:
        print(f"{time.strftime('%H:%M:%S')} ALL DONE - every queued job has finished.", flush=True)
        break
    wait_for_room()
    j = todo[0]
    print(f"{time.strftime('%H:%M:%S')} starting {j['name']} ({len(todo)} jobs left)", flush=True)
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
