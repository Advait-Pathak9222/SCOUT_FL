"""GPU worker-pool dispatcher: run many units in parallel, one process per unit.

Units are independent, so we pin each worker to a GPU via CUDA_VISIBLE_DEVICES and
pull uids off a shared queue. Per-unit subprocesses give crash isolation (one bad
unit never kills the batch) and free resumability (run_unit skips complete units).

    python -m scout_fl.experiments.dispatch --uids-file uids.txt --num-gpus 4 [--smoke]
    NUM_GPUS=4 python -m scout_fl.experiments.dispatch --uids-file uids.txt

num-gpus 0 -> CPU/MPS: run --workers processes with no GPU pinning. Returns a
non-zero exit code iff any unit failed (so run_all.sh can react).
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _worker(idx, num_gpus, q, results, lock, smoke, extra_env, log_dir):
    env = dict(os.environ)
    if num_gpus > 0:
        env["CUDA_VISIBLE_DEVICES"] = str(idx % num_gpus)
    env.update(extra_env)
    while True:
        try:
            uid = q.get_nowait()
        except queue.Empty:
            return
        cmd = [sys.executable, "-m", "scout_fl.experiments.run_unit", "--uid", uid]
        if smoke:
            cmd.append("--smoke")
        t0 = time.time()
        logf = None
        if log_dir:
            safe = uid.replace(":", "_").replace("/", "_")
            logf = open(Path(log_dir) / f"unit_{safe}.log", "w")
        try:
            r = subprocess.run(cmd, cwd=str(REPO), env=env,
                               stdout=logf or subprocess.DEVNULL,
                               stderr=subprocess.STDOUT if logf else subprocess.DEVNULL)
            ok = (r.returncode == 0)
        except Exception as e:                                # noqa: BLE001
            ok = False
            if logf:
                logf.write(f"\nDISPATCH EXCEPTION: {e}\n")
        finally:
            if logf:
                logf.close()
        with lock:
            results.append((uid, ok, time.time() - t0))
            n = len(results)
        status = "ok " if ok else "FAIL"
        print(f"[dispatch {n}] gpu{idx % max(num_gpus,1)} {status} {uid} ({time.time()-t0:.1f}s)",
              flush=True)
        q.task_done()


def dispatch(uids, num_gpus, workers_per_gpu=1, smoke=False, extra_env=None, log_dir=None):
    q = queue.Queue()
    for u in uids:
        q.put(u)
    n_workers = max(1, num_gpus * workers_per_gpu) if num_gpus > 0 else max(1, workers_per_gpu)
    results, lock, threads = [], threading.Lock(), []
    print(f"[dispatch] {len(uids)} units across {n_workers} worker(s), num_gpus={num_gpus}", flush=True)
    for i in range(n_workers):
        t = threading.Thread(target=_worker,
                             args=(i, num_gpus, q, results, lock, smoke, extra_env or {}, log_dir))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    failed = [u for u, ok, _ in results if not ok]
    print(f"[dispatch] done: {len(results)-len(failed)} ok, {len(failed)} failed", flush=True)
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids-file", required=True)
    ap.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", "0")))
    ap.add_argument("--workers-per-gpu", type=int, default=int(os.environ.get("WORKERS_PER_GPU", "1")))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    uids = [ln.strip() for ln in Path(args.uids_file).read_text().splitlines() if ln.strip()]
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    failed = dispatch(uids, args.num_gpus, args.workers_per_gpu, smoke=args.smoke, log_dir=args.log_dir)
    if failed:
        Path("logs").mkdir(exist_ok=True)
        Path("logs/failed_units.txt").write_text("\n".join(failed) + "\n")
        raise SystemExit(f"{len(failed)} unit(s) failed; see logs/failed_units.txt")


if __name__ == "__main__":
    main()
