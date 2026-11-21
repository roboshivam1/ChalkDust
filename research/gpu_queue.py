"""Sequential GPU queue. One GPU job at a time.

Jobs live in .lab/queue/jobs.json (a list of {"name", "cmd": [...]}) and the
file is re-read before every job, so jobs can be added, removed or reordered
while the queue runs. A finished job's name is appended to done.txt; the
queue exits when every job is done. Console of each job: .lab/queue/<name>.log;
the queue's own status lines also go to .lab/queue/queue.log.

A job that exits non-zero stops the queue and stays out of done.txt, so the
next start retries it. Later jobs depend on earlier ones (teacher -> SFT set ->
train -> export -> held-out eval), and before this change a single failure
let every later job run on missing inputs and be marked done.

Memory guard: while a job runs, if available RAM or commit headroom stays under
its floor for QUEUE_GUARD_SECS, the job's whole process tree is killed and the
queue stops the same way. On 2026-11-21 E2 took system commit to 51.3 of
53.4 GB within two minutes; unattended, that ends in Windows paging hard or
refusing allocations to the user's own apps. Jobs run with --resume, so a
restart continues from the last saved record.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import psutil

REPO = Path(__file__).resolve().parents[1]
Q = Path(os.environ.get("QUEUE_DIR") or REPO / ".lab/queue")   # override for tests
Q.mkdir(parents=True, exist_ok=True)
JOBS, DONE, LOG = Q / "jobs.json", Q / "done.txt", Q / "queue.log"
WAIT_FOR = sys.argv[1] if len(sys.argv) > 1 else ""


def say(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


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
            say("waiting: " + "; ".join(busy))
            last = time.time()
        time.sleep(30)


GUARD_MIN_AVAIL_GIB = float(os.environ.get("QUEUE_GUARD_MIN_AVAIL_GIB", "1.5"))
GUARD_MIN_COMMIT_GIB = float(os.environ.get("QUEUE_GUARD_MIN_COMMIT_GIB", "2"))
GUARD_SECS = float(os.environ.get("QUEUE_GUARD_SECS", "60"))
POLL_SECS = float(os.environ.get("QUEUE_POLL_SECS", "15"))


def commit_headroom_gib() -> float | None:
    """CommitLimit - CommitTotal (Windows). The limit grows if the page file is
    system-managed, so this is read live rather than once."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class PerfInfo(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t),
                    ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
                    ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
                    ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t),
                    ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
                    ("PageSize", ctypes.c_size_t), ("HandleCount", wintypes.DWORD),
                    ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD)]
    pi = PerfInfo(cb=ctypes.sizeof(PerfInfo))
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        return None
    return (pi.CommitLimit - pi.CommitTotal) * pi.PageSize / 2**30


def memory_short() -> str | None:
    avail = psutil.virtual_memory().available / 2**30
    if avail < GUARD_MIN_AVAIL_GIB:
        return f"{avail:.1f} GiB RAM available (floor {GUARD_MIN_AVAIL_GIB:g})"
    head = commit_headroom_gib()
    if head is not None and head < GUARD_MIN_COMMIT_GIB:
        return f"{head:.1f} GiB commit headroom (floor {GUARD_MIN_COMMIT_GIB:g})"
    return None


def kill_tree(pid: int) -> None:
    try:
        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=30)


def run_guarded(cmd: list[str], log, env: dict) -> tuple[int, str | None]:
    """Run one job; return (rc, reason) where reason is set if the guard killed it."""
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env)
    short_since = None
    while True:
        try:
            return proc.wait(timeout=POLL_SECS), None
        except subprocess.TimeoutExpired:
            pass
        why = memory_short()
        if why is None:
            short_since = None
            continue
        short_since = short_since or time.time()
        if time.time() - short_since >= GUARD_SECS:
            kill_tree(proc.pid)
            proc.wait()
            return proc.returncode, f"memory guard: {why} for {GUARD_SECS:g} s; killed the job's process tree"


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
        say("ALL DONE - every queued job has finished.")
        break
    wait_for_room()
    j = todo[0]
    say(f"starting {j['name']} ({len(todo)} jobs left)")
    t = time.time()
    with open(Q / f"{j['name']}.log", "a", encoding="utf-8") as f:
        f.write(f"=== start {time.strftime('%H:%M:%S')} {j['cmd']}\n")
        f.flush()
        cmd = list(j["cmd"])
        if not Path(cmd[0]).is_absolute() and (REPO / cmd[0]).exists():
            cmd[0] = str(REPO / cmd[0])   # Windows CreateProcess ignores cwd= when resolving argv[0]
        rc, guard = run_guarded(cmd, f, {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                                         **j.get("env", {})})
        f.write(f"=== end rc={rc} {time.time() - t:.0f}s{' ' + guard if guard else ''}\n")
    if rc != 0 or guard:
        say(f"FAILED {j['name']}: rc={rc} after {time.time() - t:.0f}s{' (' + guard + ')' if guard else ''}. "
            f"Queue stopped; see {Q / (j['name'] + '.log')}. Start the queue again to retry this job.")
        sys.exit(1)   # not rc: Windows crash codes (0xC0000005) overflow sys.exit
    with open(DONE, "a", encoding="utf-8") as f:
        f.write(j["name"] + "\n")
    say(f"{j['name']}: rc={rc} {time.time() - t:.0f}s")
