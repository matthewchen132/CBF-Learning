"""
Run the Monte Carlo sweep defined by config.SweepConfig and collect results.

    python run_sweep.py                       # full sweep, 80% of CPUs (override with --workers N)
    python run_sweep.py --dry-run             # list runs only
    python run_sweep.py --max-runs 8 --end-time 1.0   # smoke test

Output: <repo>/outputs/sweep_<YYYY-mm-dd_HHMMSS>/{index.csv, <run_name>/{config.json, steps.npz, summary.json}}
Metric definitions: metrics_to_track.md
"""

import argparse
import contextlib
import csv
import io
import multiprocessing as mp
import os
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from config import SweepConfig

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"


def _flatten(d, prefix=""):
    """{'gibo': {'M': 2}} → {'gibo.M': 2}; tuples become strings for CSV."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = str(v) if isinstance(v, (tuple, list)) else v
    return out


def run_one(args):
    """Worker: one run → (run_name, summary). Never raises; failures are recorded in the summary."""
    cfg, run_dir = args
    import torch
    torch.set_num_threads(1)                         # one BLAS thread per process — avoids oversubscription
    import GIBO_Robotic_Manipulator as G             # sets float64 default dtype on import
    from src.core.logger import RunLogger

    logger = RunLogger(cfg, out_dir=run_dir)         # writes config.json immediately
    try:
        with contextlib.redirect_stdout(io.StringIO()):   # silence per-step / solver prints
            summary = G.main(cfg, logger)
    except Exception as e:
        logger.status = f"failed: {type(e).__name__}: {e}"
        summary = logger.save()                      # partial steps.npz + summary.json
    return cfg.run_name(), {**_flatten(cfg.to_dict()), **summary}


def write_index(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))   # union of keys, first-seen order
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) * 4 // 5))   # 80% of cores, rounded down (8 on this Mac): ~3.2 GB peak
    p.add_argument("--max-runs", type=int, default=None, help="only the first N runs (after ordering)")
    p.add_argument("--end-time", type=float, default=None, help="override sim end_time (smoke tests)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    runs = list(SweepConfig().runs())
    if args.end_time is not None:
        runs = [replace(c, sim=replace(c.sim, end_time=args.end_time)) for c in runs]
    runs.sort(key=lambda c: -c.gibo.M)               # longest first → better load balancing
    if args.max_runs:
        # interleave algorithms so a short smoke test covers all of them
        by_algo = {}
        for c in runs:
            by_algo.setdefault(c.algo, []).append(c)
        runs = [c for group in zip(*by_algo.values()) for c in group][:args.max_runs]

    sweep_dir = OUTPUT_ROOT / f"sweep_{datetime.now():%Y-%m-%d_%H%M%S}"
    print(f"{len(runs)} runs → {sweep_dir}  (workers={args.workers})")
    if args.dry_run:
        for c in runs:
            print("  ", c.run_name())
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(c, sweep_dir / c.run_name()) for c in runs]
    rows, t0 = [], time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:   # fresh process per run: no memory build-up
        for i, (name, row) in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows.append(row)
            write_index(sweep_dir / "index.csv", rows)          # crash-safe: rewritten after every run
            elapsed = time.perf_counter() - t0
            eta = timedelta(seconds=int(elapsed / i * (len(jobs) - i)))
            print(f"[{i}/{len(jobs)}] {name:40s} {row['status'][:40]:10s} {row['runtime_total']:6.1f}s  ETA {eta}", flush=True)

    n_fail = sum(not r["status"].startswith("ok") for r in rows)
    print(f"Done in {timedelta(seconds=int(time.perf_counter() - t0))} — {len(rows) - n_fail} ok, {n_fail} failed → {sweep_dir / 'index.csv'}")


if __name__ == "__main__":
    main()
