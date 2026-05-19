import numpy as np
import cvxpy as cp
import torch
import gpytorch as gp


class Noise:
    def __init__(self, rng, noise_std_dev):
        self.rng = rng
        self.noise_std_dev = noise_std_dev


class Manipulator:
    """
    2-DOF planar robotic arm with GIBO + HOCBF safety filter.

    State: (θ₁, θ₂, θ̇₁, θ̇₂)
    GP input domain: joint-angle space (θ₁, θ₂)
    Safety constraint: θ₁ ≤ θ_max  →  h(θ) = θ_max − θ₁  (relative degree 2 → HOCBF)
    GIBO provides ĝ = E[∇_θ J], which feeds the HOCBF-QP in place of the analytic gradient.
    """

    # ── Physical parameters ───────────────────────────────────────────────────
    m1, m2   = 1.0, 1.0
    l1, l2   = 1.0, 1.0
    lc1, lc2 = 0.5, 0.5
    I1 = m1 * l1**2 / 12
    I2 = m2 * l2**2 / 12
    c11 = I1 + m1*lc1**2 + I2 + m2*lc2**2 + m2*l1**2
    c12 = 2 * m2 * l1 * lc2
    c13 = I2 + m2 * lc2**2
    c14 = m2 * l1 * lc2
    c21 = I2 + m2 * lc2**2

    # -- Safety / HOCBF constants --
    THETA_MAX = np.radians(150.0)
    CBF_GAMMA = 5.0   # rate for ψ = ḣ + γh
    CBF_ALPHA = 5.0   # rate for ψ̇ + αψ ≥ 0

    # -- Tracking controller gains --
    KP = 100.0
    KD = 20.0

    def __init__(self, alpha_cbf: float, M_number_of_queries: int, dt: float):
        self.alpha_cbf = alpha_cbf
        self.M_number_of_queries = M_number_of_queries
        self.dt = dt
        self.D = {"thetas": [], "h": []}

        self.rng = np.random.default_rng(seed=2)
        self.Noise = Noise(self.rng, noise_std_dev=0.05)

        # Start near reference at t = 0
        r0, dr0, _ = self.reference(0.0)
        self.theta  = r0  + np.array([0.3, -0.2])
        self.dtheta = dr0 + np.array([0.0,  0.0])

    # -- Arm dynamics --
    def inertia_matrix(self, theta2: float) -> np.ndarray:
        cos2 = np.cos(theta2)
        return np.array([
            [self.c11 + self.c12 * cos2, self.c13 + self.c14 * cos2],
            [self.c13 + self.c14 * cos2, self.c21],
        ])

    def coriolis_vector(self, theta2: float, dtheta: np.ndarray) -> np.ndarray:
        dth1, dth2 = dtheta
        h = self.m2 * self.l1 * self.lc2 * np.sin(theta2)
        return np.array([
            -2.0 * h * dth1 * dth2 - h * dth2**2,
             h * dth1**2,
        ])

    # -- Reference trajectory --
    def reference(self, t: float):
        """r(t) = [π·sin(t), (π/2)·sin(4t)] and its first two derivatives."""
        r   = np.array([np.pi * np.sin(t),    (np.pi / 2) * np.sin(4 * t)])
        dr  = np.array([np.pi * np.cos(t),     2 * np.pi  * np.cos(4 * t)])
        ddr = np.array([-np.pi * np.sin(t),   -8 * np.pi  * np.sin(4 * t)])
        return r, dr, ddr

    # -- Nominal controller --

    def u_nom(self, t: float) -> np.ndarray:
        """
        Torque Control following reference:
         - M(r̈ − Kd·ė − Kp·e) + C·θ̇
         - https://arxiv.org/pdf/2503.10953
         """
        r, dr, ddr = self.reference(t)
        e  = self.theta  - r
        de = self.dtheta - dr
        M  = self.inertia_matrix(self.theta[1])
        C  = self.coriolis_vector(self.theta[1], self.dtheta)
        return M @ (ddr - self.KD * de - self.KP * e) + C

    # ── Safety / objective function --

    def observe_J(self, theta: np.ndarray = None) -> float:
        """Noisy observation of J(θ) = h(θ) + ε = (θ_max − θ₁) + ε."""
        if theta is None:
            theta = self.theta
        h = self.THETA_MAX - theta[0]
        return float(h + self.Noise.rng.normal(0, self.Noise.noise_std_dev))

    # -- Numerical Integration --
    def rk4_step(self, tau: np.ndarray):
        """RK4 integration of  M(θ)θ̈ + C(θ, θ̇)θ̇ = τ."""
        def f(th, dth):
            M = self.inertia_matrix(th[1])
            C = self.coriolis_vector(th[1], dth)
            return dth.copy(), np.linalg.solve(M, tau - C)

        dt = self.dt
        k1_dth, k1_ddth = f(self.theta, self.dtheta)
        k2_dth, k2_ddth = f(self.theta + 0.5*dt*k1_dth, self.dtheta + 0.5*dt*k1_ddth)
        k3_dth, k3_ddth = f(self.theta + 0.5*dt*k2_dth, self.dtheta + 0.5*dt*k2_ddth)
        k4_dth, k4_ddth = f(self.theta + dt*k3_dth,     self.dtheta + dt*k3_ddth)
        self.theta  += (dt / 6) * (k1_dth  + 2*k2_dth  + 2*k3_dth  + k4_dth)
        self.dtheta += (dt / 6) * (k1_ddth + 2*k2_ddth + 2*k3_ddth + k4_ddth)

    # -- GIBO Helper Functions --

    def return_gridspace(self, wander_angle_deg) -> torch.Tensor:
        """8×8 candidate query points centred on current (θ₁, θ₂)."""
        wander = np.deg2rad(wander_angle_deg)  # radians
        t1 = torch.linspace(float(self.theta[0]) - wander, float(self.theta[0]) + wander, 8)
        t2 = torch.linspace(float(self.theta[1]) - wander, float(self.theta[1]) + wander, 8)
        g1, g2 = torch.meshgrid(t1, t2, indexing='ij')
        return torch.stack([g1.flatten(), g2.flatten()], dim=1)  # (64, 2)

    def compute_grad_K(self, train_x: torch.Tensor, GP_model,
                       lengthscale: torch.Tensor) -> torch.Tensor:
        """∂K(θ_curr, θ_i)/∂θ for RBF: −(θ_curr − θ_i)/l² · K(θ_curr, θ_i)"""
        l = lengthscale
        x_curr  = train_x.new_tensor([[self.theta[0], self.theta[1]]])
        diffs   = x_curr - train_x                                         # (N, 2)
        K_x_all = GP_model.covar_module(x_curr, train_x).evaluate().T     # (N, 1)
        grad_K = -(diffs / l**2) * K_x_all
        return grad_K                                   # (N, 2)

    def compute_gradient_update_weights(self, train_x, train_y, GP_model, likelihood):
        """α = (K_XX + σ²I)⁻¹ y — weights for the posterior gradient E[∇J]."""
        K_xx  = GP_model.covar_module(train_x, train_x).evaluate()
        # Clamp noise floor to 1e-2: nearby query points make K_xx nearly rank-1;
        # without this the solve amplifies noise into the gradient estimate.
        jitter     = max(float(likelihood.noise.detach()), 1e-2)
        K_xx_noisy = K_xx + jitter * torch.eye(len(train_x))
        return torch.linalg.solve(K_xx_noisy, train_y)

    def acquisition_function(self, train_x, train_y, GP_model, GP_likelihood,
                             sigma2, l, obs_noise, next_query_point, query_space, m):
        """
        Select the query point that maximises expected decrease in Jacobian variance at θ_t.
        Criterion: Tr( ∇K(θ_t, X̂)ᵀ · (K(X̂,X̂) + σ²I)⁻¹ · ∇K(θ_t, X̂) )
        where X̂ = [train_x, qp] — the extended dataset including the candidate.
        """
        x_curr  = train_x.new_tensor([[self.theta[0], self.theta[1]]])  # (1, 2) — fixed

        best_val = -float("inf")
        next_qp  = query_space[0]
        for qp in query_space:
            # Extended dataset X̂ = [train_x, qp]
            X_hat   = torch.cat([train_x, qp.unsqueeze(0)], dim=0)          # (n+1, 2)
            n_hat   = X_hat.shape[0]

            # K(X̂, X̂) + σ²I
            K_hat       = GP_model.covar_module(X_hat, X_hat).evaluate().detach()
            K_hat_noisy = K_hat + GP_likelihood.noise.detach() * torch.eye(n_hat)

            # ∇_{θ_t} K(θ_t, X̂)  — shape (n+1, 2)
            diffs      = x_curr - X_hat                                      # (n+1, 2)
            K_curr_hat = GP_model.covar_module(x_curr, X_hat).evaluate().detach().T  # (n+1, 1)
            grad_K_hat = -(1 / l**2) * diffs * K_curr_hat                   # (n+1, 2)

            # Tr( ∇K^T · K̂⁻¹ · ∇K ) via solve (more stable than explicit inverse)
            K_hat_inv_grad = torch.linalg.solve(K_hat_noisy, grad_K_hat)    # (n+1, 2)
            tr_val = torch.trace(grad_K_hat.T @ K_hat_inv_grad)             # scalar

            if tr_val > best_val:
                best_val = tr_val
                next_qp  = qp
        return next_qp, float(best_val)
    
    def modified_acq_function(self, train_x, train_y, GP_model, GP_likelihood,
                             sigma2, l, obs_noise, next_query_point, query_space, m):
        """
        Select a query point in M that maximises information gain about ∇J(θ_t).
        Criterion: tr( ∇_θ K(θ_t, X)ᵀ (K_XX + σ²I)⁻¹ ∇_θ K(θ_t, X) )
        θ_t is fixed for the entire inner loop — gradient is always w.r.t. current arm state.
        """
        x_curr     = train_x.new_tensor([[self.theta[0], self.theta[1]]])  # θ_t — fixed
        K_xx       = GP_model.covar_module(train_x, train_x).evaluate().detach()
        K_xx_noisy = K_xx + GP_likelihood.noise.detach() * torch.eye(len(train_x))
        K_inv      = torch.inverse(K_xx_noisy)

        best_val = -float("inf")
        next_qp  = query_space[0]
        for qp in query_space:
            dist   = x_curr - qp.unsqueeze(0)                                                 # (1, 2) — θ_t − qp
            K_curr_qp = GP_model.covar_module(x_curr, qp.unsqueeze(0)).evaluate().detach()    # (1, 1)
            grad_k    = -(1 / l**2) * dist * K_curr_qp                                        # (1, 2) — ∇_θ k(θ_t, qp)
            tr_val    = (grad_k @ grad_k.T).squeeze()                                         # ||∇_θ k(θ_t, qp)||²
            if tr_val > best_val:
                best_val = tr_val
                next_qp  = qp
        return next_qp, float(best_val)

    def random_acq_function(self, train_x, train_y, GP_model, GP_likelihood,
                             sigma2, l, obs_noise, next_query_point, query_space, m):
        """
        Used to benchmark performance:
         - randomly selects query points in the gridspace vs acquisition function
        """

        next_qp  = query_space[np.random.randint(0,query_space.shape[0])]
        return next_qp,  0.0

    # -- HOCBF-QP --
    def solve_hocbf_qp(self, tau_nom: np.ndarray, h: float,
                   grad_h: torch.Tensor) -> np.ndarray:

        M_mat = self.inertia_matrix(self.theta[1])
        C_vec = self.coriolis_vector(self.theta[1], self.dtheta)  # (2,) vector C(θ,θ̇)·θ̇
        Minv  = np.linalg.inv(M_mat)
        g     = grad_h.numpy()

        g_norm = np.linalg.norm(g)
        if g_norm < 1e-6:
            g = np.array([-1.0, 0.0])
        else:
            g = g / g_norm          # normalize — only direction enters the constraint

        # ψ = g·θ̇ + γ·h
        psi = float(g @ self.dtheta) + self.CBF_GAMMA * h

        # A·τ ≥ b  from  ψ̇ + α·ψ ≥ 0
        A_row = g @ Minv                                    # (2,)
        b_rhs = (float(g @ (Minv @ C_vec))                 # g·M⁻¹·C(θ,θ̇)θ̇
                - (self.CBF_GAMMA + self.CBF_ALPHA) * float(g @ self.dtheta)
                - self.CBF_ALPHA * self.CBF_GAMMA * h)

        # Sanity checks
        assert np.isfinite(b_rhs), f"b_rhs blew up: {b_rhs:.3e} | psi={psi:.3e} | dtheta={self.dtheta}"
        assert np.isfinite(A_row).all(), f"A_row invalid: {A_row}"

        # Closed-form projection — no OSQP needed for single constraint
        if A_row @ tau_nom >= b_rhs:
            return tau_nom                                  # already safe

        # Project tau_nom onto constraint boundary A·τ = b
        tau_safe = tau_nom + ((b_rhs - A_row @ tau_nom) / (A_row @ A_row)) * A_row
        return tau_safe
