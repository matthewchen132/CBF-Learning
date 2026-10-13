## Researching Bayesian Optimization methods to learn Information about the Gradient.
Papers: 
https://arxiv.org/pdf/2106.11899



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
