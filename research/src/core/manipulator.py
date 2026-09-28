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
    CBF_GAMMA = 5.0   # rate for ψ = ḣ + γh ; FIX AT 5 for consistent results
    CBF_ALPHA = 5.0   # rate for ψ̇ + αψ ≥ 0 ; FIX AT 5 for consistent results

    # -- Tracking controller gains --
    KP = 100.0
    KD = 20.0

    def __init__(self, alpha_cbf: float, M_number_of_queries: int, dt: float,
                 class_k: str = "linear"):
        self.alpha_cbf = alpha_cbf
        self.M_number_of_queries = M_number_of_queries
        self.dt = dt
        self.D = {"thetas": [], "h": []}
        self.rng = np.random.default_rng(seed=2)
        self.Noise = Noise(self.rng, noise_std_dev=0.05)
        self.class_K = class_k

        # Class K function α₁(h) and its derivative α₁'(h) for the HOCBF
        g = self.CBF_GAMMA
        if class_k == "linear":
            self.alpha1      = lambda h: g * h
            self.alpha1_grad = lambda h: g
        elif class_k == "sqrt":
            self.alpha1      = lambda h: g * np.sqrt(max(h, 1e-6))
            self.alpha1_grad = lambda h: g / (2 * np.sqrt(max(h, 1e-6)))
        elif class_k == "cubic":
            self.alpha1      = lambda h: g * h**3
            self.alpha1_grad = lambda h: 3 * g * h**2
        elif class_k == "quadratic":
            self.alpha1      = lambda h: g * h **2
            self.alpha1_grad = lambda h: 2 * g * h
        elif class_k == "tanh":
            self.alpha1      = lambda h: np.tanh(h)
            self.alpha1_grad = lambda h: 1 - np.tanh(h) ** 2
        else:
            raise ValueError(f"Unknown class_k '{class_k}'. Choose: linear, sqrt, cubic, quadratic, tanh")

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

    def compute_grad_h_uncertainty(self, train_x: torch.Tensor, GP_model,
                                   likelihood, lengthscale: torch.Tensor) -> float:
        """Posterior std of ∇h at current θ: sqrt(tr(Σ_∇)) where
           Σ_∇ = (1/l²)·I  −  grad_K.T · K_XX_noisy⁻¹ · grad_K  (2×2)."""
        l = lengthscale
        grad_K  = self.compute_grad_K(train_x, GP_model, l)          # (N, 2)
        K_xx    = GP_model.covar_module(train_x, train_x).evaluate()
        noise  = max(float(likelihood.noise.detach()), 1e-2)
        K_xx_noisy = K_xx + noise * torch.eye(len(train_x))
        K_inv_gradK = torch.linalg.solve(K_xx_noisy, grad_K)         # (N, 2)
        posterior_cov_grad  = (1.0 / l**2) * torch.eye(2, dtype=train_x.dtype) \
                      - grad_K.T @ K_inv_gradK                        # (2, 2)
        posterior_cov_grad  = posterior_cov_grad.clamp(min=0.0)
        return float(posterior_cov_grad.trace().sqrt())

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

            K_hat_inv_grad = torch.linalg.solve(K_hat_noisy, grad_K_hat)    # (n+1, 2)
            tr_val = torch.trace(grad_K_hat.T @ K_hat_inv_grad)             # scalar

            if tr_val > best_val:
                best_val = tr_val
                next_qp  = qp
        return next_qp, float(best_val)
    
    def compute_lambda(self, full_norm=True) -> float:
        """
        full_norm: True for consideration of theta1, theta2 / False for just theta1
        lambda(x) = V_{∇h} / (V_{∇h} + V_h) 
        V_{∇h} = ||dθdt|| — Our system's "speed" is the rate of change of theta
        V_h    = alpha_cbf — sensitivity to CBF value error is constant (eq. 17: V_h = -α).
        """
        V_grad_h = np.linalg.norm(self.dtheta)
        V_h      = self.alpha_cbf
        return float(V_grad_h / (V_grad_h + V_h))

    def composite_acq_function(self, train_x, GP_model, GP_likelihood, l, query_space):
        """
        Selects a query point via eq. (14):
          x* = argmax (1-λ)·σ²_h(x)/σ²_max  +  λ·I_∇(x)/I_max
        λ(x) adapts based on joint velocity: fast motion → prioritise gradient learning.
        """
        x_curr  = train_x.new_tensor([[self.theta[0], self.theta[1]]])  # (1, 2) — fixed
        n_train = train_x.shape[0]
        lam     = self.compute_lambda()

        # Precompute K(train_x, train_x) + σ²I — shared across all candidates
        K_XX_noisy = (GP_model.covar_module(train_x, train_x).evaluate().detach()
                      + GP_likelihood.noise.detach() * torch.eye(n_train))

        I_vals      = torch.zeros(query_space.shape[0])
        sigma2_vals = torch.zeros(query_space.shape[0])

        for idx, qp in enumerate(query_space):
            qp_2d = qp.unsqueeze(0)  # (1, 2) — GPyTorch requires 2D inputs

            # -- (1) Gradient variance: Tr( ∇K(θ_t, X̂)ᵀ · K̂⁻¹ · ∇K(θ_t, X̂) ) --
            X_hat       = torch.cat([train_x, qp_2d], dim=0)                # (n+1, 2)
            n_hat       = X_hat.shape[0]
            K_hat       = GP_model.covar_module(X_hat, X_hat).evaluate().detach()
            K_hat_noisy = K_hat + GP_likelihood.noise.detach() * torch.eye(n_hat)
            diffs       = x_curr - X_hat                                     # (n+1, 2)
            K_curr_hat  = GP_model.covar_module(x_curr, X_hat).evaluate().detach().T  # (n+1, 1)
            grad_K_hat  = -(1 / l**2) * diffs * K_curr_hat                  # (n+1, 2)
            K_hat_inv_grad = torch.linalg.solve(K_hat_noisy, grad_K_hat)    # (n+1, 2)
            I_vals[idx] = torch.trace(grad_K_hat.T @ K_hat_inv_grad)

            # -- (2) Posterior variance at qp: σ²(qp|train_x) --
            k_qq   = GP_model.covar_module(qp_2d, qp_2d).evaluate().detach().squeeze()
            k_qX   = GP_model.covar_module(qp_2d, train_x).evaluate().detach()          # (1, n)
            alpha  = torch.linalg.solve(K_XX_noisy, k_qX.T)                             # (n, 1)
            sigma2_vals[idx] = (k_qq - (k_qX @ alpha).squeeze()).clamp(min=0.0)

        # -- Normalise then combine with adaptive λ (eq. 14) --
        max_I      = I_vals.max().clamp(min=1e-12)
        max_sigma2 = sigma2_vals.max().clamp(min=1e-12)
        acq_vals   = (1 - lam) * (sigma2_vals / max_sigma2) + lam * (I_vals / max_I)
        best_idx   = acq_vals.argmax()
        next_qp    = query_space[best_idx]
        return next_qp, lam

    def random_acq_function(self, query_space):
        """
        Used to benchmark performance:
         - randomly selects query points in the gridspace vs acquisition function
        """

        next_qp  = query_space[np.random.randint(0,query_space.shape[0])]
        return next_qp,  0.0

    # -- HOCBF-QP --
    def solve_hocbf_qp(self, torque_nom: np.ndarray, h: float,
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

        # ψ = g·θ̇ + α·h
        psi = (g @ self.dtheta) + self.alpha1(h)

        # g·Minv·τ ≥ b  from  ψ̇ + α·ψ ≥ 0
        A_row = g @ Minv                                    # (2,)
        b_rhs = (float(g @ (Minv @ C_vec))                 # g·M⁻¹·C(θ,θ̇)θ̇
                - (self.alpha1_grad(h) + self.CBF_ALPHA) * float(g @ self.dtheta)
                - self.CBF_ALPHA * self.alpha1(h))

        # Sanity checks
        assert np.isfinite(b_rhs), f"b_rhs blew up: {b_rhs:.3e} | psi={psi:.3e} | dtheta={self.dtheta}"
        assert np.isfinite(A_row).all(), f"A_row invalid: {A_row}"

        # Closed-form projection — no OSQP needed for single constraint
        if A_row @ torque_nom >= b_rhs:
            return torque_nom                                  # already safe

        # Project torque_nom onto constraint boundary A·τ = b
        tau_safe = torque_nom + ((b_rhs - A_row @ torque_nom) / (A_row @ A_row)) * A_row
        return tau_safe
    
    # -- HOCBF-SOCP --
    def solve_hocbf_SOCP(self, torque_nom: np.ndarray, 
                        h: float,
                        grad_h: torch.Tensor,
                        e_h: torch.Tensor,
                        e_grad_h: torch.Tensor) -> np.ndarray:

        # -- SOCP Higher Order CBF --
        '''
        Our state h(x) is not able to map to our control input (Non-holonomic). 
        Therefore, we need to form a HOCBF with:
        ψ = ḣ , ψ̇ = dḣ/dt

        -- Conceptually: -- 
            (1) ψ = grad_h_unit · θ̇ + alpha_K·h
            (2) ψ̇ ≥ -α·ψ
        '''
        M_mat  = self.inertia_matrix(self.theta[1])
        C_vec  = self.coriolis_vector(self.theta[1], self.dtheta)  # (2,) vector C(θ,θ̇)·θ̇
        Minv   = np.linalg.inv(M_mat)
        grad_h = grad_h.numpy()

        psi = (grad_h @ self.dtheta) + self.alpha1(e_h) # h_dot = psi
        tau = cp.Variable(2)

        f_x = - Minv @ C_vec # <-- Dynamics (correlated w/ Drift)
        g_x = Minv           # <-- Dynamics (correlated w/ Control)

        # -- || A·τ + b || <= c·τ + d --
        # e_∇h · ||Minv·τ − Minv·C||  ≤  (μ_g @ Minv)·τ  −  b_rhs
        A_cone = e_grad_h * g_x      # (2,2)
        b_cone = e_grad_h * (f_x + self.alpha1_grad(h) * self.dtheta)        # (2,1)
        c_linear = g_x @ grad_h   # (2,1)

        # -- d = μ_∇h^T · f(x)  +  α·[μ_h(x) -  e_h(x)] --
        d_linear = float(grad_h @ f_x) + self.alpha1_grad(h) * float(grad_h @ self.dtheta) + self.CBF_ALPHA * (psi)

        # Sanity checks
        assert np.isfinite(f_x).all(), f"f_x blew up: {f_x} | psi={psi:.3e} | dtheta={self.dtheta}"
        assert np.isfinite(g_x).all(), f"g_x invalid: {g_x}"

        # -- Optimize || A·τ + b || <= c·τ+d --
        slack = cp.Variable(nonneg=True)
        constraint = [cp.norm(A_cone @ tau + b_cone) <= c_linear @ tau + d_linear + slack ] 
        
        objective = cp.Minimize(cp.sum_squares(tau - torque_nom) + 1e6 * cp.square(slack)) # min || u - u* ||^2
        prob = cp.Problem(objective=objective, constraints=constraint)
        prob.solve()
        print(f"{tau.value} \n")

        if prob.status not in ["optimal", "optimal_inaccurate"] or tau.value is None:
            raise Exception(f"problem is not optimally solved, {tau.value}")
        print(f"Slack: {slack.value}, psi: {psi}")
        return tau.value