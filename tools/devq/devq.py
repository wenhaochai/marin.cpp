#!/usr/bin/env python3
"""devq: run many short GPU tasks inside ONE interactive allocation (owner rule: debug and tests use one
pli-cp salloc, released as soon as the work ends; never one Slurm job per test).

  devq.py submit --gpus N --name NAME [--cwd DIR] -- CMD...   queue a task; prints its id (CMD is joined
                                                               with spaces: put anything with quotes in a script)
  devq.py wait ID [ID...]                                      block until done; print rc and log path
  devq.py ls                                                   pending / running / done
  devq.py ensure                                               make sure an H100 runner is alive or queued
  devq.py runner --gpus G [--idle-exit 1200] [--deadline S]    run inside the allocation (srun --overlap, or a
                                                               wq job with DEVQ_NO_SCANCEL=1)

Queue directory: $DEVQ_DIR (default $NANOMARIN_ROOT/tmp/devq). The runner gives each task its own GPUs via
CUDA_VISIBLE_DEVICES, a per-task TMPDIR on scratch, `module load proxy/default` and HF offline env.
When nothing is pending or running for --idle-exit seconds, the runner cancels its own allocation, so GPUs
never sit idle long enough for the 30-min idle warning.
Auto runner: submitting an h100/any task (and a runner that exits with runnable tasks left) calls `ensure`: if no
H100 runner has a fresh heartbeat and no wq runner job is queued or running, it queues one with submit_wq.sh
(wq --prio 20). Set DEVQ_AUTO_WQ=0 to turn this off.
"""
import argparse, fcntl, json, os, subprocess, sys, time

ROOT = os.environ.get("NANOMARIN_ROOT")
if not ROOT:
    sys.exit("set NANOMARIN_ROOT")
QDIR = os.environ.get("DEVQ_DIR", os.path.join(ROOT, "tmp", "devq"))
SUB = {k: os.path.join(QDIR, k) for k in ("pending", "running", "done")}
HB = os.path.join(QDIR, "runners")  # one heartbeat file per live runner
HERE = os.path.dirname(os.path.abspath(__file__))


def _mk():
    for d in list(SUB.values()) + [HB]:
        os.makedirs(d, exist_ok=True)


def ensure(a=None):
    """Keep exactly one queued wq runner, sized (1-8 GPUs) to the pending h100/any tasks that do not first-fit into the
    free GPUs of live runners; queued runners of another size are replaced. Runners pack tasks side by side."""
    _mk()
    if os.environ.get("DEVQ_AUTO_WQ", "1") != "1":
        return
    fd = os.open(os.path.join(QDIR, "ensure.lock"), os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        now = time.time()
        tasks = []  # GPU counts of pending tasks a runner could take
        for fn in os.listdir(SUB["pending"]):
            try:
                t = json.load(open(os.path.join(SUB["pending"], fn)))
            except (OSError, ValueError):
                continue
            if t.get("arch", "h100") in ("h100", "any") and int(t.get("gpus", 1)) > 0:
                tasks.append(int(t["gpus"]))
        bins = []  # free GPUs of each live runner
        for fn in os.listdir(HB):
            try:
                hb = json.load(open(os.path.join(HB, fn)))
            except (OSError, ValueError):
                continue
            if hb.get("arch") == "h100" and now - hb.get("t", 0) < 120:
                bins.append(int(hb.get("free", 0)))
        wq = os.path.join(ROOT, "tools", "wq", "wq")
        queued = []  # (wq id, GPUs) of runners queued in wq and not started yet
        for line in subprocess.run([wq, "ls"], capture_output=True, text=True).stdout.splitlines():
            f = line.split()
            if len(f) >= 6 and f[2] == "marincpp-devq" and f[5] == "queued":
                queued.append((f[0], int(f[3])))
        unfit = 0
        for g in sorted(tasks, reverse=True):  # first fit into live runners, largest task first
            for i, room in enumerate(bins):
                if room >= g:
                    bins[i] -= g
                    break
            else:
                unfit += g
        need = 8 if unfit >= 8 else 4 if unfit >= 4 else unfit  # whole 4-GPU blocks; the rest waits for a slot
        if [q[1] for q in queued] == [need] or (need <= 0 and not queued):
            return
        # merge: one runner sized to all waiting work beats several small ones (a full 4-GPU block starts sooner on a
        # pilot host, and the tasks pack side by side), so replace the runners that have not started yet
        for wid, _ in queued:
            subprocess.run([wq, "cancel", wid], capture_output=True, text=True)
        if need <= 0:
            return
        out = subprocess.run(["bash", os.path.join(HERE, "submit_wq.sh"), "1", str(need)], capture_output=True, text=True,
                             env=dict(os.environ, NANOMARIN_ROOT=ROOT, DEVQ_ENSURE="1"))
        wid = (out.stdout.strip().split() or [""])[-1]
        if out.returncode == 0 and wid.startswith("w"):
            print(f"[devq] queued a {need}-GPU wq runner {wid}", file=sys.stderr)
        else:
            print(f"[devq] could not queue a {need}-GPU wq runner: {out.stdout} {out.stderr}", file=sys.stderr)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _next_id():
    path = os.path.join(QDIR, "counter")
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    import fcntl
    fcntl.flock(fd, fcntl.LOCK_EX)
    raw = os.read(fd, 32)
    n = int(raw or b"0") + 1
    os.lseek(fd, 0, 0)
    os.ftruncate(fd, 0)
    os.write(fd, str(n).encode())
    os.close(fd)
    return f"t{n}"


def submit(a):
    _mk()
    tid = _next_id()
    task = {"id": tid, "name": a.name, "gpus": a.gpus, "arch": a.arch, "cwd": os.path.abspath(a.cwd), "cmd": a.cmd,
            "submitted": time.time()}
    tmp = os.path.join(SUB["pending"], f".{tid}.json")
    with open(tmp, "w") as f:
        json.dump(task, f)
    os.rename(tmp, os.path.join(SUB["pending"], f"{tid}.json"))
    print(tid)
    if a.arch in ("h100", "any"):
        ensure()


def _state(tid):
    for k in ("done", "running", "pending"):
        p = os.path.join(SUB[k], f"{tid}.json")
        if os.path.exists(p):
            return k, json.load(open(p))
    return None, None


def wait(a):
    left = set(a.ids)
    while left:
        for tid in sorted(left):
            k, t = _state(tid)
            if k == "done":
                print(f"{tid} {t['name']} rc={t['rc']} log={t['log']} secs={t['end'] - t['start']:.0f}")
                left.discard(tid)
            elif k is None:
                print(f"{tid} unknown")
                left.discard(tid)
        if left:
            time.sleep(10)


def ls(a):
    _mk()
    for k in ("pending", "running", "done"):
        for fn in sorted(os.listdir(SUB[k])):
            if fn.endswith(".json") and not fn.startswith("."):
                t = json.load(open(os.path.join(SUB[k], fn)))
                print(k, t["id"], t["name"], f"gpus={t['gpus']}", t.get("arch", "h100"), f"rc={t.get('rc', '')}", t.get("devices", ""))


IDLE_KILL = int(os.environ.get("DEVQ_IDLE_KILL", "900"))  # a task whose GPU sits at 0% this long is killed


def _gpu_util():
    """{physical GPU index (str): utilization %} from nvidia-smi; {} if it fails."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {l.split(",")[0].strip(): float(l.split(",")[1]) for l in out.splitlines() if "," in l}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def runner(a):
    _mk()
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    vis = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    free = vis[:a.gpus] if vis else [str(i) for i in range(a.gpus)]  # the GPUs this allocation owns
    if len(free) < a.gpus:
        sys.exit(f"runner: asked for {a.gpus} GPUs, CUDA_VISIBLE_DEVICES has {len(free)}")
    procs = {}  # tid -> (Popen, task, devices)
    busy = {}   # GPU -> last time it was seen above 0% (or its task's start)
    last_poll = 0.0
    idle_since = t0 = time.time()
    hbf = os.path.join(HB, f"{job}-{os.getpid()}.json")
    while True:
        json.dump({"arch": a.arch, "gpus": a.gpus, "free": len(free), "job": job, "t": time.time()}, open(hbf + ".tmp", "w"))
        os.replace(hbf + ".tmp", hbf)
        draining = a.deadline and time.time() - t0 > a.deadline
        # reap
        if procs and time.time() - last_poll > 60:  # idle watchdog: never let a hung task idle a GPU for long
            last_poll = time.time()
            util = _gpu_util()
            for tid, (p, t, devs) in list(procs.items()):
                for g in devs:
                    if util.get(g, 100.0) > 0:
                        busy[g] = last_poll
                stale = [g for g in devs if last_poll - busy.get(g, t["start"]) > IDLE_KILL]
                if stale and p.poll() is None:
                    t["error"] = f"killed by the idle watchdog: GPU {','.join(stale)} at 0% for > {IDLE_KILL} s"
                    print(f"[devq] {tid} {t['name']}: {t['error']}", flush=True)
                    try:
                        os.killpg(p.pid, 15)
                    except ProcessLookupError:
                        pass
        for tid, (p, t, devs) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            t.update(rc=rc, end=time.time())
            with open(os.path.join(SUB["done"], f"{tid}.json"), "w") as f:
                json.dump(t, f)
            os.remove(os.path.join(SUB["running"], f"{tid}.json"))
            free.extend(devs)
            del procs[tid]
            print(f"[devq] done {tid} {t['name']} rc={rc}", flush=True)
        # start FIFO tasks that fit (a task that does not fit blocks smaller later ones only if it is first)
        pend = sorted((f for f in os.listdir(SUB["pending"]) if f.endswith(".json") and not f.startswith(".")),
                      key=lambda f: int(f[1:-5]))
        mine = 0  # pending tasks this runner could ever run (others must not keep it alive)
        blocked = False
        for fn in pend:
            try:
                t = json.load(open(os.path.join(SUB["pending"], fn)))
            except FileNotFoundError:
                continue
            if t.get("arch", "h100") not in ("any", a.arch):
                continue  # e.g. an H100-only measurement on an A100 runner
            if t["gpus"] > a.gpus:
                continue  # leave it for a runner with more GPUs (runners pack smaller tasks side by side)
            mine += 1
            if draining or blocked or t["gpus"] > len(free):
                blocked = True  # FIFO: the first task that does not fit waits for GPUs
                continue
            try:
                os.rename(os.path.join(SUB["pending"], fn), os.path.join(SUB["running"], fn))
            except FileNotFoundError:
                continue  # another runner claimed it
            devs, free = free[:t["gpus"]], free[t["gpus"]:]
            tid = t["id"]
            log = os.path.join(SUB["done"], f"{tid}.log")
            tmpdir = os.path.join(ROOT, "tmp", "job", f"{job}-{tid}")
            os.makedirs(tmpdir, exist_ok=True)
            proxy = "" if os.environ.get("DEVQ_NO_PROXY") == "1" else "module load proxy/default >/dev/null 2>&1 || true\n"
            script = (
                proxy +
                f"export TMPDIR={tmpdir} CUDA_VISIBLE_DEVICES={','.join(devs)}\n"
                f"export HF_HOME={ROOT}/cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "
                "TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled\n"
                f"cd {t['cwd']}\n"
                + " ".join(t["cmd"]) + "\n"
                f"rc=$?; rm -rf {tmpdir}; exit $rc\n")
            t.update(start=time.time(), log=log, devices=devs, job=job)
            json.dump(t, open(os.path.join(SUB["running"], fn), "w"))
            p = subprocess.Popen(["bash", "-lc", script], stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                 start_new_session=True)  # own process group, so the watchdog can kill the tree
            for g in devs:
                busy[g] = time.time()
            procs[tid] = (p, t, devs)
            print(f"[devq] start {tid} {t['name']} gpus={devs}", flush=True)
        if draining and not procs:
            print(f"[devq] deadline {a.deadline}s reached, all tasks ended: exiting", flush=True)
            os.remove(hbf)
            if mine and a.arch == "h100":
                ensure()  # runnable tasks remain: queue a successor
            return
        if procs or (mine and not draining):
            idle_since = time.time()
        elif time.time() - idle_since > a.idle_exit:
            print(f"[devq] idle {a.idle_exit}s: releasing allocation {job}", flush=True)
            os.remove(hbf)
            if os.environ.get("DEVQ_NO_SCANCEL") != "1":
                subprocess.run(["scancel", job])
            return
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="c", required=True)
    s = sp.add_parser("submit")
    s.add_argument("--gpus", type=int, required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--cwd", default=os.getcwd())
    s.add_argument("--arch", default="h100", choices=["h100", "a100", "any"],
                   help="where the task may run (default h100: timings and H100-numerics fixtures)")
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    w = sp.add_parser("wait")
    w.add_argument("ids", nargs="+")
    sp.add_parser("ls")
    sp.add_parser("ensure")
    r = sp.add_parser("runner")
    r.add_argument("--gpus", type=int, required=True)
    r.add_argument("--idle-exit", type=int, default=1200)
    r.add_argument("--deadline", type=int, default=0, help="stop claiming tasks after this many seconds")
    r.add_argument("--arch", default="h100", choices=["h100", "a100"])
    a = ap.parse_args()
    if a.c == "submit":
        if a.cmd and a.cmd[0] == "--":
            a.cmd = a.cmd[1:]
        if not a.cmd:
            sys.exit("no command")
    {"submit": submit, "wait": wait, "ls": ls, "runner": runner, "ensure": ensure}[a.c](a)


if __name__ == "__main__":
    main()
