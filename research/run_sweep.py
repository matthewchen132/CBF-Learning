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
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

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

HEADLINE = ("n_violations", "min_h", "grad_err_mean", "grad_std_auc", "grad_coverage",
            "h_std_mean", "h_coverage", "excess_gap", "intervention_rate", "runtime_total")
ALGO_ORDER = {"oracle": 0, "gibo": 1, "random": 2, "lambda": 3}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None          # "", "None", text → not numeric


def summarize_sweep(sweep_dir):
    """
    Average every numeric summary metric over θ₀ and seeds, grouped by (algo, noise_std, M).
    Adds excess_gap = boundary_gap − boundary_gap of the matching oracle run (same noise, θ₀, seed).
    Writes <sweep_dir>/summary_by_algo.csv (mean and std per metric) and prints a headline table.
    """
    from config import RunConfig
    sweep_dir = Path(sweep_dir)
    rows = list(csv.DictReader(open(sweep_dir / "index.csv")))
    ok   = [r for r in rows if r["status"].startswith("ok")]

    # -- Excess gap vs. the matching oracle run --
    oracle_gap = {(r["noise_std"], r["theta0_id"], r["seed"]): _num(r["boundary_gap"])
                  for r in ok if r["algo"] == "oracle"}
    for r in ok:
        gap, base = _num(r["boundary_gap"]), oracle_gap.get((r["noise_std"], r["theta0_id"], r["seed"]))
        r["excess_gap"] = gap - base if gap is not None and base is not None else ""

    # -- Metric columns = everything that isn't a config field --
    config_cols = set(_flatten(RunConfig().to_dict()))
    metrics     = [k for k in ok[0] if k not in config_cols and k != "status"]

    # -- Group by (algo, noise, M); mean / std over θ₀ × seeds --
    def key(r):
        return (r["algo"], float(r["noise_std"]), int(r["gibo.M"]))
    groups = {}
    for r in rows:
        groups.setdefault(key(r), {"ok": [], "n_failed": 0})
        if r["status"].startswith("ok"):
            groups[key(r)]["ok"].append(r)
        else:
            groups[key(r)]["n_failed"] += 1

    out = []
    for (algo, noise, M) in sorted(groups, key=lambda g: (g[1], g[2], ALGO_ORDER.get(g[0], 9))):
        g   = groups[(algo, noise, M)]
        row = {"algo": algo, "noise_std": noise, "M": M, "n_runs": len(g["ok"]), "n_failed": g["n_failed"]}
        for m in metrics:
            vals = np.array([v for v in (_num(r.get(m)) for r in g["ok"]) if v is not None])
            row[f"{m}_mean"] = float(vals.mean()) if vals.size else ""
            row[f"{m}_std"]  = float(vals.std())  if vals.size else ""
        out.append(row)
    write_index(sweep_dir / "summary_by_algo.csv", out)

    # -- Headline table --
    print(f"\n{'algo':7s} {'noise':>5s} {'M':>2s} {'n':>3s} " + " ".join(f"{m[:14]:>16s}" for m in HEADLINE))
    for row in out:
        cells = [f"{row[m + '_mean']:7.3f}±{row[m + '_std']:<7.3f}" if row.get(m + "_mean") != "" else f"{'—':>16s}"
                 for m in HEADLINE]
        print(f"{row['algo']:7s} {row['noise_std']:5.2f} {row['M']:2d} {row['n_runs']:3d} " + " ".join(cells))
    print(f"\n→ {sweep_dir / 'summary_by_algo.csv'}")
    return out


PLOT_METRICS = (    # (summary column, panel title, y-axis label) — one per metric category
    ("n_violations",  "Safety violations",          "steps with h < 0"),
    ("grad_err_mean", "Gradient error",             "mean ‖μ_∇ − ∇h‖"),
    ("grad_std_auc",  "Gradient uncertainty (AUC)", "∫ √tr Σ_∇ dt"),
    ("h_std_mean",    "h uncertainty",              "mean h_std [rad]"),
    ("excess_gap",    "Excess gap vs. oracle",      "boundary gap − oracle [rad]"),
    ("runtime_total", "Runtime",                    "seconds per run"),
)
PLOT_ALGOS  = ("gibo", "random", "lambda")
ALGO_LABELS = {"gibo": "vanilla GIBO", "random": "random", "lambda": "λ-weighted"}
ALGO_COLORS = {"gibo": "#2a78d6", "random": "#eb6834", "lambda": "#1baf7a"}   # categorical slots 1–3 (CVD-validated)
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"


def plot_metric_comparison(sweep_dir, summary, show=True):
    """2×3 bar grid: one panel per PLOT_METRICS entry; bars = algorithms, groups = (noise, M) cells, error bars = ±1 std."""
    import matplotlib
    import matplotlib.pyplot as plt

    sweep_dir = Path(sweep_dir)
    by_key = {(r["algo"], float(r["noise_std"]), int(r["M"])): r for r in summary}
    cells  = sorted({(n, M) for (a, n, M) in by_key if a in PLOT_ALGOS})
    x, w   = np.arange(len(cells)), 0.26

    def series(algo, col):
        vals = [_num(by_key.get((algo, *c), {}).get(col)) for c in cells]
        return np.array([np.nan if v is None else v for v in vals])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)
    for ax, (col, title, ylabel) in zip(axes.flat, PLOT_METRICS):
        ax.set_facecolor(SURFACE)
        for i, algo in enumerate(PLOT_ALGOS):
            mean, std = series(algo, col + "_mean"), series(algo, col + "_std")
            lower = std if col == "excess_gap" else np.minimum(std, mean)   # non-negative metrics: don't draw error below 0
            ax.bar(x + (i - 1) * w, mean, w, yerr=[lower, std],
                   color=ALGO_COLORS[algo], edgecolor=SURFACE, linewidth=1.5, label=ALGO_LABELS[algo],
                   error_kw=dict(ecolor=INK_2, elinewidth=1, capsize=2), zorder=2)
        ax.axhline(0, color=INK_2, linewidth=0.8, zorder=3)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.set_xticks(x, [f"σ={n:g}\nM={M}" for n, M in cells])
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Acquisition comparison — mean ± 1 std over ~15 runs per bar (5 θ₀ × 3 seeds); lower is better",
                 color=INK, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    path = sweep_dir / "metric_comparison.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"→ {path}")
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
    return path


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

    # -- Average over θ₀ × seeds per (algo, noise, M), then optionally plot --
    summary = summarize_sweep(sweep_dir)
    if not sys.stdin.isatty():                       # background / piped run: no one to ask, save the plot anyway
        plot_metric_comparison(sweep_dir, summary, show=False)
    elif input("Plot metric comparisons across algorithms? [Y/n] ").strip().lower() in ("", "y", "yes"):
        plot_metric_comparison(sweep_dir, summary)


if __name__ == "__main__":
    main()
