"""
RunLogger provides per-run config snapshot, per-iteration log, and summary metrics.
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np

TRUE_GRAD_H = np.array([-1.0, 0.0])     # ∇h for h = θ_max − θ₁


def _git_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty  = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    return {"commit": commit, "dirty": dirty}


def _mean(x):
    """Mean that returns None for empty input (e.g. query stats for the oracle)."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()) if x.size else None


class RunLogger:
    def __init__(self, cfg, out_dir=None):
        self.cfg     = cfg
        self.steps   = []
        self.t_start = time.perf_counter()
        self.status  = "running"
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            snapshot = {**cfg.to_dict(), "git": _git_info()}
            (self.out_dir / "config.json").write_text(json.dumps(snapshot, indent=2))   # written at START

    # -- Per-iteration --
    def log_step(self, **fields):
        self.steps.append({k: np.asarray(v, dtype=float) for k, v in fields.items()})

    def arrays(self) -> dict:
        return {k: np.stack([s[k] for s in self.steps]) for k in self.steps[0]} if self.steps else {}

    # -- Summary metrics --
    def summarize(self) -> dict:
        cfg, a = self.cfg, self.arrays()
        out = {"status": self.status, "n_steps": len(self.steps),
               "runtime_total": time.perf_counter() - self.t_start}
        if not self.steps:
            return out

        dt, n_sig  = cfg.sim.dt, cfg.gp.n_sigma
        theta_max  = np.radians(cfg.cbf.THETA_MAX_deg)
        half_box   = cfg.gibo.box_factor * cfg.gp.l_0

        # 3.1 Safety (h_true = after the step)
        h = a["h_true"]
        out.update(n_violations=int((h < 0).sum()), min_h=float(h.min()),
                   max_violation_depth=float(max(0.0, -h.min())),
                   time_in_violation=float((h < 0).sum() * dt))

        # 3.2 Value learning (at θ_t, where the GP posterior was evaluated)
        h_err = np.abs(a["h_mean"] - a["h_true_t"])
        out.update(h_std_mean=_mean(a["h_std"]), h_std_final=float(a["h_std"][-1]),
                   h_abs_err_mean=_mean(h_err),
                   h_coverage=_mean(h_err <= n_sig * a["h_std"]))

        # 3.3 Gradient learning
        g_err  = np.linalg.norm(a["grad_mean"] - TRUE_GRAD_H, axis=1)
        g_norm = np.linalg.norm(a["grad_mean"], axis=1)
        cosang = np.clip(a["grad_mean"] @ TRUE_GRAD_H / np.maximum(g_norm, 1e-12), -1, 1)
        out.update(grad_err_mean=_mean(g_err), grad_err_final=float(g_err[-1]),
                   grad_angle_err_mean=_mean(np.degrees(np.arccos(cosang))),
                   grad_std_mean=_mean(a["grad_std"]), grad_std_final=float(a["grad_std"][-1]),
                   grad_std_auc=float(a["grad_std"].sum() * dt),
                   grad_coverage=_mean(g_err <= n_sig * a["grad_std"]))

        # 3.4 Conservatism (excess_* vs. oracle is computed in the sweep analysis)
        du         = np.linalg.norm(a["u"] - a["u_nom"], axis=1)
        past_limit = a["ref"][:, 0] > theta_max
        out.update(intervention_rate=_mean(du > 1e-6), intervention_mean=_mean(du),
                   boundary_gap=_mean(h[past_limit]),
                   tracking_rmse=float(np.sqrt(np.mean(np.sum((a["theta"] - a["ref"])**2, axis=1)))))

        # 3.5 Data / queries (empty for the oracle → None)
        q       = a["queries"].reshape(-1, 2) if a["queries"].size else np.empty((0, 2))
        th_rep  = np.repeat(a["theta"], a["queries"].shape[1] if a["queries"].ndim == 3 else 0, axis=0)
        on_edge = np.abs(q - th_rep).max(axis=1) >= half_box * (1 - 1e-9) if q.size else np.empty(0)
        out.update(n_data_final=int(a["n_data"][-1]),
                   query_offset_mean=_mean(a["query_offset"]),
                   query_edge_frac=_mean(on_edge),
                   query_unsafe_frac=_mean(a["query_h_true"] < 0))

        # 3.6 Compute
        out.update(acq_time_mean=_mean(a["acq_time"]), step_time_mean=_mean(a["step_time"]))
        return out

    # -- Output --
    def save(self) -> dict:
        summary = self.summarize()
        if self.out_dir:
            if self.steps:
                np.savez_compressed(self.out_dir / "steps.npz", **self.arrays())
            (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
