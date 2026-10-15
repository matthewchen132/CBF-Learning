# Metrics to Track — Acquisition-Function Comparison

This document defines the metrics and the Monte Carlo setup for comparing the 3 acquisition functions
(vanilla GIBO, random, λ-weighted) on the 2-DOF manipulator with a GP-learned HOCBF.

**Goal:** show whether acquisition 3 (λ-weighted) gives a significant improvement over vanilla GIBO and random.
A `RunLogger` records everything below, and `run_sweep.py` loops over the Monte Carlo grid.

---

## 1. Monte Carlo setup (`research/config.py → SweepConfig`)

| Axis | Values | Notes |
|---|---|---|
| Algorithm | `gibo`, `random`, `lambda` (+ `oracle` baseline) | oracle = true h and ∇h = [−1, 0]; no GP, no queries |
| Sensor noise std σₙ | 0.05, 0.2, 0.5 rad | the GP noise follows automatically: `likelihood.noise = σₙ²` |
| M (queries per step) | 1, 2 | M = 4 ≈ 55 s/run after the fast acquisition, can be re-added |
| θ₀ offset from r(0) | 5 scrambled-Sobol presets in ±0.5 rad | weak axis: PD settles in ~0.4 s (ω_n = 10, ζ = 1) |
| Seed | 0, 1, 2 | **paired**: same seed → same sensor-noise sequence across algorithms |
| Optimizer | SOCP (fixed) | |

Count: 3 noise × 2 M × 5 θ₀ × 3 seeds = 90 runs per algorithm → 270 + 45 oracle (oracle doesn't depend on M) = **315 runs**.

**Pairing:** every comparison between algorithms uses runs with the same (noise, M, θ₀, seed). Sensor noise comes from
`Manipulator.rng` (seed), and random acquisition from a separate `acq_rng` (seed + 1000), so all algorithms see identical noise.

**Fixed (logged in `config.json`, not swept):** dt = 0.04 s, T = 15 s, ℓ = 1.0, σ² = 11.8, jitter = 1e-6, n_sigma = 2,
box = ±2ℓ, grid 8×8, γ = α = 5, class-K linear, KP = 100, KD = 20, θ_max = 150°, reference A = [π, π/2], ω = [1, 4].

---

## 2. Per-iteration log (`steps.npz`, one row per step t)

| Field | Shape | Meaning |
|---|---|---|
| `t` | 1 | time [s] |
| `theta`, `dtheta` | 2, 2 | joint state |
| `ref` | 2 | reference r(t) |
| `u`, `u_nom` | 2, 2 | applied torque / nominal tracking torque |
| `h_true` | 1 | θ_max − θ₁ (after the step) |
| `h_noisy` | 1 | sensor reading at θ_t |
| `h_mean`, `h_std`, `h_safe` | 1 each | GP posterior at θ_t; h_safe = μ − n_sigma·σ |
| `grad_mean` | 2 | μ_∇ at θ_t |
| `grad_var` | 2 | diag Σ_∇ at θ_t |
| `grad_std` | 1 | √tr Σ_∇ (computed every step, also for QP) |
| `queries` | M×2 | query points chosen this step |
| `query_offset` | M | ‖q − θ_t‖ / ℓ |
| `query_h_true` | M | h at each query (< 0 means the query is in the unsafe region) |
| `acq_value` | M | full gradient-info score (gibo), or λ (lambda) |
| `n_data` | 1 | dataset size |
| `cbf_active` | 1 | ‖u − u_nom‖ > 1e-6 |
| `socp_slack` | 1 | SOCP slack (0 for QP) |
| `step_time`, `acq_time` | 1 each | wall time [s] |

---

## 3. Summary metrics (`summary.json`, one per run)

### 3.1 Safety
| Metric | Definition |
|---|---|
| `n_violations` | # steps with h_true < 0 |
| `min_h` | min_t h_true [rad] |
| `max_violation_depth` | max(0, −min_h) |
| `time_in_violation` | n_violations · dt [s] |

### 3.2 Value learning (h)
| Metric | Definition |
|---|---|
| `h_std_mean`, `h_std_final` | mean / last h_std |
| `h_abs_err_mean` | mean \|h_mean − h_true(θ_t)\| |
| `h_coverage` | fraction of steps with \|h_mean − h_true\| ≤ n_sigma·h_std (target ≈ 0.95 for 2σ) |

### 3.3 Gradient learning [!!!] (the quantity GIBO's acquisition optimizes)
True gradient for this task: ∇h = [−1, 0].
| Metric | Definition |
|---|---|
| `grad_err_mean`, `grad_err_final` | ‖μ_∇ − ∇h‖ |
| `grad_angle_err_mean` | angle between μ_∇ and ∇h [deg]; the QP only uses the direction of g |
| `grad_std_mean`, `grad_std_final` | √tr Σ_∇ |
| `grad_std_auc` | Σ_t grad_std · dt (lower = uncertainty reduced faster) |
| `grad_coverage` | fraction of steps with ‖μ_∇ − ∇h‖ ≤ n_sigma·grad_std |

### 3.4 Conservatism
| Metric | Definition |
|---|---|
| `intervention_rate` | fraction of steps with cbf_active |
| `intervention_mean` | mean ‖u − u_nom‖ |
| `boundary_gap` | mean h_true over steps where r₁(t) > θ_max (ideal ≈ 0: ride the limit) |
| `tracking_rmse` | RMSE(θ − r) over the run |
| **`excess_gap`** (headline) | boundary_gap − boundary_gap_oracle (same noise, θ₀, seed) |
| `excess_tracking_rmse` | tracking_rmse − tracking_rmse_oracle |

`excess_gap` measures the conservatism caused by *learning* h, separated from the conservatism built into the CBF
design itself (γ, α, class-K), which the oracle also has. It's computed in the sweep analysis, not per run.

### 3.5 Data / queries
| Metric | Definition |
|---|---|
| `n_data_final` | dataset size; deterministic = (1 + M)·steps, logged for completeness |
| `query_offset_mean` | mean ‖q − θ_t‖/ℓ (the gradient-info optimum is ≈ 1 for a single point) |
| `query_edge_frac` | fraction of queries on the box edge (acquisition is box-limited) |
| `query_unsafe_frac` | fraction of queries with h_true(q) < 0 (matters if queries were physical) |

### 3.6 Compute
| Metric | Definition |
|---|---|
| `runtime_total` | wall time of the run [s] |
| `acq_time_mean` | mean acquisition time per step [s] |

---

## 4. Analysis (after the sweep)
- For each cell (noise, M, θ₀): paired differences per seed, `lambda − gibo` and `lambda − random`; report mean ± std over seeds.
- Headline table: `n_violations`, `grad_err_mean`, `grad_std_auc`, `excess_gap`, `runtime_total` per algorithm × noise × M.
- Diagnostics for λ: log the λ-weighted terms (I_∇ and σ² before normalization) to check which term decides each query.

## 5. Output layout
```
outputs/sweep_<YYYY-mm-dd_HHMMSS>/
  index.csv                         one row per run: config fields + summary metrics
  <algo>_<optim>_M<M>_n<noise>_th<k>_s<seed>/
    config.json                     RunConfig snapshot + git hash/dirty flag (written at start)
    steps.npz                       per-iteration log (section 2)
    summary.json                    summary metrics (section 3), written at the end
```
