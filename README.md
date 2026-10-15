## Gradient-Information from BayesOpt
Gradient Information Paper: https://arxiv.org/pdf/2106.11899

## Context:
Bayesian Optimization is a highly sample-efficient optimization method which focuses on (1) using a surrogate model to fit a desired function, and (2) optimizing samples of each function to balance **exploration** of the search space and **exploitation** of useful data. Each sample is optimized through usage of an **acquisition function**, a user-defined function which provides a ranking of all points in the space. 

> [!NOTE]
Bayesian Optimization has some amazing properties:
- Un-matched sample efficiency
- Globally searching optimization algorithm.
- No reliance on gradients
- Accounts for Noise


## Example: Efficient Global Optimization (EGO):
- Utilizes the Expected Improvement acquisition function to search for points in uncertain areas, and points with high value. By selecting these points optimally with an acquisition function, EGO can effectively find a (pretty close to) optimal outcome for a system with sparse sampling and no reliance on gradients.

<img width="3527" height="1547" alt="image" src="https://github.com/user-attachments/assets/cd0842f7-96c6-4610-979b-6be389a2557f" />

This project is conducted through the _Safe Autonomous Systems Lab_ at the University of California San Diego.


# Monte Carlo Sweep
Compare performance of a new weighted acquisition function vs vanilla GIBO, random querying on a GP-learned HOCBF on a 2-DOF arm (θ₁ ≤ 150°). Full metric definitions: [metrics_to_track.md](metrics_to_track.md).

**Sweep Parameters:** σₙ ∈ {0.05, 0.2, 0.5} × M ∈ {1, 2} × 5 Sobol θ₀ × 3 paired seeds (+ oracle baseline) = 315 runs.

## Metrics (brief — full definitions in [metrics_to_track.md](metrics_to_track.md))
Each run logs ~22 fields per step (`steps.npz`) and ~28 summary metrics (`summary.json`).
The true gradient for this task is known (∇h = [−1, 0]), so the gradient estimate can be graded directly.

| Category | Metrics | Question it answers |
|---|---|---|
| Safety | # violation steps, min h, max violation depth | Did the arm cross θ_max, and by how much? |
| Gradient learning **[main]** | ‖μ_∇ − ∇h‖, angle error, √tr Σ_∇ (mean, AUC), 2σ coverage | How accurate is the slope estimate, how fast does its uncertainty shrink, and is that uncertainty honest? |
| Value learning | h_std, \|μ_h − h\|, 2σ coverage | Is h_safe = μ − 2σ actually below the true h? (Similar across algorithms: h is measured at θ_t every step.) |
| Conservatism | intervention rate, boundary gap, tracking RMSE, **excess gap vs. oracle** | How much tracking performance does safety cost, and how much of that is due to *learning* h? |
| Queries | offset / ℓ, fraction on box edge, fraction in unsafe region | Where does each acquisition put its points? |
| Compute | runtime, acquisition time per step | Is a better acquisition also more expensive? |

The **oracle** runs the same CBF with the true h and ∇h (no GP, no queries). Comparing against it separates the
conservatism caused by learning from the conservatism built into the CBF design (γ, α, class-K).

## Running the Monte Carlo sweep
The sweep axes live in `research/config.py → SweepConfig`; every other constant (dt, ℓ, σ², CBF gains, …) is in the
other dataclasses there, and each run's exact settings are saved to its `config.json`.

| Axis | Default values |
|---|---|
| `algos` | gibo, random, lambda (+ oracle if `include_oracle=True`) |
| `noise_stds` | 0.05, 0.2, 0.5 (sensor noise std; the GP noise follows it) |
| `Ms` | 1, 2 (queries per step; M = 4 ≈ 1 min/run) |
| `theta0_offsets` | 5 scrambled-Sobol offsets from r(0), ±0.5 rad |
| `seeds` | 0, 1, 2 (paired: the same seed gives every algorithm the same sensor noise) |

```bash
cd research
python run_sweep.py --dry-run                        # list the runs that would execute
python run_sweep.py --max-runs 8 --end-time 1.0      # smoke test (~6 s): all algorithms, short sims
MPLBACKEND=Agg python run_sweep.py                   # full sweep (315 runs, ~15–20 min)
python run_sweep.py --workers 4                      # fewer workers while using the machine
```
To change the sweep, edit the `SweepConfig` tuples (e.g. `Ms = (1, 2, 4)`) and rerun.
To run a single configuration with plots: `python GIBO_Robotic_Manipulator.py` (uses `RunConfig()` defaults).

**Output:** `outputs/sweep_<YYYY-mm-dd_HHMMSS>/`
- `index.csv`: one row per run (config fields + summary metrics), rewritten after every run, so it survives interruptions
- `<algo>_<optim>_M<M>_n<noise>_th<k>_s<seed>/`: `config.json` (written at start, with git commit), `steps.npz`, `summary.json`
- A run that crashes (e.g. solver failure) is recorded with `status = "failed: …"` and its partial log; the sweep continues.

Commit before a sweep: `config.json` records the git commit, and a `dirty: true` flag means the code can't be recovered exactly.

## How the parallel sweep works (`research/run_sweep.py`)
- **Processes, not threads.** Python's Global Interpreter Lock (GIL) lets only one thread per interpreter run Python
  code at a time. The simulation loop is mostly Python, so threads wouldn't run in parallel. Each worker is a separate
  Python process with its own interpreter, GIL, and memory (~0.3–0.4 GB each, mostly torch/gpytorch/cvxpy).
- **One main process + a Pool of workers.** The main process builds all run configs (`SweepConfig().runs()`), pairs each with
  its output folder, and hands them to a `multiprocessing.Pool`. It never simulates; it only collects results and writes `index.csv`.
- **Workers default to 80% of the cores**, rounded down (8 on a 10-core Mac). Override with `--workers N`.
- **Data passing.** Each job sends only `(RunConfig, run_dir)` to a worker (pickled). The worker builds the arm, GP, and logger
  itself, writes `steps.npz` to disk, and sends back only a flat summary row.
- **`imap_unordered`** returns results as they finish (live progress + crash-safe index). Runs are sorted longest-first
  (M = 2 → M = 1 → oracle) so no worker is left with a long run at the end.
- **`spawn` start method.** Each worker is a fresh interpreter that re-imports the script, so the
  `if __name__ == "__main__":` guard is required (otherwise every worker would start its own sweep).
- **`maxtasksperchild=1`.** Each worker does one run and exits, so memory is returned to the OS after every run
  (no build-up; ~3 GB flat peak with 8 workers). Cost: ~2–3 s of re-imports per run.
- **`torch.set_num_threads(1)` per worker.** Prevents oversubscription (8 processes × 10 torch threads on 10 cores).

**Fixes this round:** GP hyperparameters tied to the sensor noise and derived from the trajectory (ℓ = 1.0, σ² = 11.8);
σ²/ℓ² prior in Σ_∇; a single noisy Gram matrix for every GP computation; exact Cholesky above 800 points;
exact O(n²)-per-candidate acquisition (~22× faster).

# 4/05: Zero-noise CBF vs noisy CBF vs learned-CBF under varying simulated sensor noise conditions.
<img width="250" height="200" alt="image" src="https://github.com/user-attachments/assets/04d7d6f4-5843-4a32-8320-63e42ae6191a" />

### Summary metrics per run:
 - Safety: # violation steps, min h_true, max violation depth, time in violation.
 - Value learning: mean/final h_std, mean |h_mean − h_true|, 2σ coverage of h.
 - Gradient learning [!!!]: mean/final ‖μ_∇ − ∇h‖, mean angle error (the QP uses only g's direction), mean/final grad_std, Σ_∇ reduction (AUC of grad_std), 2σ coverage.
 - Conservatism (brainstorm → chosen): see the next section.
 - Data/queries: n_data (deterministic: (1+M)·steps, logged for completeness), mean offset/ℓ, fraction at box edge, fraction of queries in the unsafe region (h(q) < 0; matters if queries are ever physical).
 - Compute: total runtime, mean acquisition time per step.

### Conservatism metrics:
 1. Intervention: fraction of steps with u ≠ u_nom; mean ‖u − u_nom‖.
 2. Boundary gap: mean h_true over steps where r₁(t) > θ_max. Ideally ≈ 0: the arm should ride the limit.
 3. Excess tracking error: RMSE(θ − r) − RMSE_oracle.
 4. Learning-induced conservatism (headline): boundary gap − oracle gap (same θ₀, seed and noise). It isolates the cost of not knowing h from the conservatism built into the CBF design (γ, α, class-K).


## Introductory Experiments
### Stage 4 (GIBO)
<img width="485" height="385" alt="image" src="https://github.com/user-attachments/assets/fb720dd6-a6e3-4f89-a8f0-f8ccafc29f86" />

<img width="495" height="391" alt="image" src="https://github.com/user-attachments/assets/e36f2e7b-d980-40d0-9426-0313bc64243e" />

 - GIBO allows us to poll points of high covariance to gain information about our function more effectively.
 - However, polling becomes locally "Trapped" with small step sizes


### 4/05: Zero-noise CBF vs noisy CBF vs learned-CBF under varying simulated sensor noise conditions.
<img width="250" height="200" alt="image" src="https://github.com/user-attachments/assets/04d7d6f4-5843-4a32-8320-63e42ae6191a" />

### 3/24: Conventional CBF performance under noise:
<img width="300" height="251" alt="image" src="https://github.com/user-attachments/assets/173dd130-b97f-491b-bbad-eecf73851c47" />
