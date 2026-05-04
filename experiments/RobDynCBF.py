import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ── Physical parameters ──────────────────────────────────────────────────────
m1, m2   = 1.0, 1.0          # link masses  [kg]
l1, l2   = 1.0, 1.0          # link lengths [m]
lc1, lc2 = 0.5, 0.5          # CoM distances from joint [m]
I1 = m1 * l1**2 / 12         # rod moment of inertia
I2 = m2 * l2**2 / 12

# Paper's constant coefficients cij (from Spong, Hutchinson & Vidyasagar [31])
c11 = I1 + m1*lc1**2 + I2 + m2*(l2**2 + lc2**2)  # wait — should be l1 not l2 in m2 term
# Recompute properly:
c11 = I1 + m1*lc1**2 + I2 + m2*lc2**2 + m2*l1**2
c12 = 2 * m2 * l1 * lc2          # coefficient of cos(θ₂) in M[0,0]
c13 = I2 + m2 * lc2**2            # constant part of M[0,1] = M[1,0]
c14 = m2 * l1 * lc2               # coefficient of cos(θ₂) in M[0,1]
c21 = I2 + m2 * lc2**2            # M[1,1]   (= c13 for symmetric arm)
# c25 = 0 (horizontal links, no gravity)


def inertia_matrix(theta2: float) -> np.ndarray:
    """Mass/inertia matrix  M(θ₂)  — symmetric, positive definite."""
    cos2 = np.cos(theta2)
    M = np.array([
        [c11 + c12 * cos2,  c13 + c14 * cos2],
        [c13 + c14 * cos2,  c21              ]
    ])
    return M


def coriolis_vector(theta2: float, dtheta: np.ndarray) -> np.ndarray:
    """
    Centripetal / Coriolis term  C(θ,θ̇)·θ̇
    in the standard EL form:  M·θ̈ + C·θ̇ = τ

    Returns the 2-vector  C(θ,θ̇)·[θ̇₁, θ̇₂]ᵀ.
    """
    dth1, dth2 = dtheta
    h = m2 * l1 * lc2 * np.sin(theta2)
    c_vec = np.array([
        -2.0 * h * dth1 * dth2 - h * dth2**2,
         h * dth1**2
    ])
    return c_vec


# ── Reference trajectory ─────────────────────────────────────────────────────

def reference(t: float):
    """
    r(t), ṙ(t), r̈(t)  for the paper's tracking target:
        r₁ = π·sin(t),      r₂ = (π/2)·sin(4t)
    """
    r   = np.array([ np.pi * np.sin(t),     (np.pi/2) * np.sin(4*t)  ])
    dr  = np.array([ np.pi * np.cos(t),      2*np.pi  * np.cos(4*t)  ])
    ddr = np.array([-np.pi * np.sin(t),     -8*np.pi  * np.sin(4*t)  ])
    return r, dr, ddr


# ── Nominal controller ────────────────────────────────────────────────────────

KP = 100.0   # proportional gain
KD = 20.0    # derivative gain

def u_nom(t: float, theta: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
    """
    Computed-torque nominal controller with tunable PD gains:
        u_nom = M(θ)(r̈ + Kd·ė + Kp·e) + C(θ,θ̇)·θ̇  ... wait, signs
        ddtheta_des = r̈ - Kd·ė - Kp·e
    Closed-loop error: ë + Kd·ė + Kp·e = 0.
    """
    r, dr, ddr = reference(t)
    e   = theta  - r
    de  = dtheta - dr
    theta2 = theta[1]
    M     = inertia_matrix(theta2)
    c_vec = coriolis_vector(theta2, dtheta)
    ddtheta_des = ddr - KD * de - KP * e
    return M @ ddtheta_des + c_vec


# ── HOCBF safety filter ───────────────────────────────────────────────────────
# Constraint: θ₁ ≤ θ_max  →  h(x) = θ_max − θ₁ ≥ 0
#
# h has relative degree 2 (u appears only in θ̈₁), so we use a HOCBF:
#   ψ(x) = ḣ + γ·h = −θ̇₁ + γ(θ_max − θ₁)          (must stay ≥ 0)
#   ψ̇ + α·ψ ≥ 0   →   −θ̈₁ − γθ̇₁ + α·ψ ≥ 0
#
# Substituting θ̈ = M⁻¹(u − C·θ̇):
#   −[M⁻¹u]₁ ≥ [M⁻¹C·θ̇]₁ + γθ̇₁ − α·ψ
#   A·u ≤ b     (single linear constraint, affine in u)
#
# QP:  min ‖u − u_nom‖²  s.t.  A·u ≤ b
# With a scalar constraint this has a closed-form projection solution.

THETA_MAX = np.radians(150.0)   # safety bound [rad]
CBF_GAMMA = 5.0                 # class-K rate for ψ = ḣ + γh
CBF_ALPHA = 5.0                 # class-K rate for ψ̇ + α·ψ ≥ 0


def cbf_qp(t: float, theta: np.ndarray, dtheta: np.ndarray) -> np.ndarray:

    u0 = u_nom(t, theta, dtheta)

    theta2 = theta[1]
    M     = inertia_matrix(theta2)
    c_vec = coriolis_vector(theta2, dtheta)
    Minv  = np.linalg.inv(M)

    h   =  THETA_MAX - theta[0]          # ≥ 0 when safe
    dh  = -dtheta[0]                     # ḣ = −θ̇₁
    psi =  dh + CBF_GAMMA * h            # intermediate barrier

    A = Minv[0, :]
    b = (Minv @ c_vec)[0] - CBF_GAMMA * dtheta[0] + CBF_ALPHA * psi  # sign fixed

    if A @ u0 <= b:
        return u0

    # Closed-form projection onto the constraint hyperplane
    u_safe = u0 - ((A @ u0 - b) / (A @ A)) * A
    return u_safe


# ── ODE helpers ───────────────────────────────────────────────────────────────

def make_ode(controller):
    """Returns an ODE function using the given controller."""
    def ode(t, state):
        theta  = state[:2]
        dtheta = state[2:]
        tau    = controller(t, theta, dtheta)
        M      = inertia_matrix(theta[1])
        c_vec  = coriolis_vector(theta[1], dtheta)
        ddtheta = np.linalg.solve(M, tau - c_vec)
        return np.concatenate([dtheta, ddtheta])
    return ode


# ── Simulation ────────────────────────────────────────────────────────────────

T_END  = 30.0
t_eval = np.linspace(0, T_END, 2000)

r0, dr0, _ = reference(0.0)
theta0  = r0  + np.array([0.3, -0.2])
dtheta0 = dr0 + np.array([0.0,  0.0])
state0  = np.concatenate([theta0, dtheta0])

# Nominal (no safety filter)
sol_nom = solve_ivp(make_ode(u_nom), [0, T_END], state0,
                    method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-10)

# Safety-filtered
sol_cbf = solve_ivp(make_ode(cbf_qp), [0, T_END], state0,
                    method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-10)

t       = sol_nom.t
th_nom  = sol_nom.y[:2]
th_cbf  = sol_cbf.y[:2]
dth_cbf = sol_cbf.y[2:]

r_hist  = np.array([reference(ti)[0] for ti in t]).T
n_cbf   = th_cbf.shape[1]
t_cbf   = sol_cbf.t
u_cbf   = np.array([cbf_qp(t_cbf[i], th_cbf[:, i], dth_cbf[:, i])
                    for i in range(n_cbf)]).T

# ── Plots ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
(ax1, ax2), (ax3, ax4) = axes
fig.suptitle("2-DOF Planar Arm — CBF-QP Safety Filter ($\\theta_1 \\leq 150°$)",
             fontsize=12, fontweight="bold")

COLORS = ["#2563EB", "#DC2626"]

# ── (a) θ₁ tracking ──────────────────────────────────────────────────────────
ax1.plot(t,     np.degrees(r_hist[0]),  "--", color="gray",      lw=1.5, label="reference $r_1$")
ax1.plot(t,     np.degrees(th_nom[0]),  ":",  color="gray",      lw=1.2, label="nominal (unsafe)")
ax1.plot(t_cbf, np.degrees(th_cbf[0]),        color=COLORS[0],  lw=1.5, label="CBF-filtered")
ax1.axhline(150, color=COLORS[1], lw=1.2, ls="--", label="constraint $150°$")
ax1.set_ylabel("angle (deg)")
ax1.set_title("$\\theta_1$ tracking")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# ── (b) θ₂ tracking ──────────────────────────────────────────────────────────
ax2.plot(t,     np.degrees(r_hist[1]),  "--", color="gray",      lw=1.5, label="reference $r_2$")
ax2.plot(t_cbf, np.degrees(th_cbf[1]),        color=COLORS[1],  lw=1.5, label="CBF-filtered")
ax2.set_ylabel("angle (deg)")
ax2.set_title("$\\theta_2$ tracking")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ── (c) u₁ ───────────────────────────────────────────────────────────────────
ax3.plot(t_cbf, u_cbf[0], color=COLORS[0], lw=1.5)
ax3.set_xlabel("time (s)")
ax3.set_ylabel("torque (N·m)")
ax3.set_title("Control input $u_1$ (CBF-filtered)")
ax3.grid(alpha=0.3)

# ── (d) u₂ ───────────────────────────────────────────────────────────────────
ax4.plot(t_cbf, u_cbf[1], color=COLORS[1], lw=1.5)
ax4.set_xlabel("time (s)")
ax4.set_ylabel("torque (N·m)")
ax4.set_title("Control input $u_2$ (CBF-filtered)")
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.show()      

