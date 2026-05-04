import matplotlib.pyplot as plt
import numpy as np

class Simulation():
    '''
    Simulation Class
     - times
     - obstacles
     - logging
    
    TODO: add config.py as an input to initialize
    '''
    def __init__(self, dt, end_time):
        self.curr_time = 0.0
        self.dt = dt
        self.end_time = end_time

        self.logs = {
            "state": [],
            "u": [],
            "dist": [],
            "times": [],
            "query_points": []
        }
    def is_running(self):
        return self.curr_time < self.end_time

    def arrived_at_goal(self, goal_radius, GIBO):
        if GIBO.dist_to_goal <= goal_radius:
            GIBO.goal_reached = True

    def step(self, print_t):
        self.curr_time += self.dt
        if print_t:
            print(f"Current Time: {self.curr_time}")

    def log_data(self, X, u, dist, time, query_points=None):
        self.logs["state"].append(X)
        self.logs["u"].append(u)
        self.logs["dist"].append(dist)
        self.logs["times"].append(time)
        if query_points is not None:
                self.logs["query_points"].append(query_points.flatten().tolist())

    def plot_trajectory(self, GIBO):
        X = []
        X_p = []
        Y = []
        Y_p = []
        for State in self.logs["state"]:
            X.append(State.x)
            X_p.append(State.xp)
            Y.append(State.y)
            Y_p.append(State.yp)
        fig, ax = plt.subplots(figsize=(8, 8))

        # Robot center and lookahead point trajectories
        ax.plot(X, Y, color='blue', linewidth=1.5, label='Robot center')
        ax.plot(X_p, Y_p, color='green', label='Lookahead point')

        # Start marker
        ax.scatter(X[0], Y[0], color='blue', marker='o', s=80, zorder=5)

        # Obstacle (close the loop by appending first point)
        obs = GIBO.Obstacle.circle_points
        ax.plot(np.append(obs[:, 0], obs[0, 0]),
                np.append(obs[:, 1], obs[0, 1]),
                color='red', linewidth=2, label='Obstacle')
        ax.fill(obs[:, 0], obs[:, 1], color='red', alpha=0.15)

        # Waypoint
        ax.scatter(GIBO.waypoint[0], GIBO.waypoint[1],
                   color='green', marker='*', s=250, zorder=5, label='Waypoint')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f"GIBO Trajectory | M samples: {GIBO.M_number_of_queries}")
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()



    def log_data_manipulator(self, theta, tau, h, t, query_points, grad):
        self.logs["state"].append(theta.tolist())
        self.logs["u"].append(tau.tolist())
        self.logs["dist"].append(float(h))
        self.logs["times"].append(t)
        self.logs["query_points"].append(query_points.flatten().tolist())
        self.logs.setdefault("grad", []).append(grad)

    def plot_manipulator(self, arm):
        t_arr    = np.array(self.logs["times"])
        th_arr   = np.array(self.logs["state"])         # (N, 2) — [θ₁, θ₂]
        tau_arr  = np.array(self.logs["u"])             # (N, 2) — [τ₁, τ₂]
        h_arr    = np.array(self.logs["dist"])          # (N,)   — true h
        qp_arr   = np.array(self.logs["query_points"])  # (N, 2) — query points in joint space
        grad_arr = np.array(self.logs.get("grad", []))  # (N, 2) — GP gradient estimate

        ref = np.array([arm.reference(t)[0] for t in t_arr])  # (N, 2)

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(
            f"GIBO + HOCBF — 2-DOF Planar Arm   ($\\theta_1 \\leq 150°$,  M={arm.M_number_of_queries})",
            fontweight="bold",
        )
        BLUE, RED = "#2563EB", "#DC2626"

        # ── (a) θ₁ tracking ──────────────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(t_arr, np.degrees(ref[:, 0]),  "--", color="gray", lw=1.2, label="reference")
        ax.plot(t_arr, np.degrees(th_arr[:, 0]), color=BLUE, lw=1.5, label="GIBO+HOCBF")
        ax.axhline(150, color=RED, ls="--", lw=1.2, label="theta_max")
        ax.set_ylabel("θ₁ (deg)")
        ax.set_title("Joint 1 angle")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ── (b) θ₂ tracking ──────────────────────────────────────────────────
        ax = axes[0, 1]
        ax.plot(t_arr, np.degrees(ref[:, 1]),  "--", color="gray", lw=1.2, label="reference")
        ax.plot(t_arr, np.degrees(th_arr[:, 1]), color=RED, lw=1.5, label="GIBO+HOCBF")
        ax.set_ylabel("θ₂ (deg)")
        ax.set_title("Joint 2 angle")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ── (c) Safety function h ─────────────────────────────────────────────
        ax = axes[1, 0]
        ax.plot(t_arr, np.degrees(h_arr), color=BLUE, lw=1.5)
        ax.axhline(0, color=RED, ls="--", lw=1.2, label="h = 0 (boundary)")
        ax.set_ylabel("h = θ_max − θ₁ (deg)")
        ax.set_title("Safety function h(t)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ── (d) Control torques ───────────────────────────────────────────────
        ax = axes[1, 1]
        ax.plot(t_arr, tau_arr[:, 0], color=BLUE, lw=1.5, label="τ₁")
        ax.plot(t_arr, tau_arr[:, 1], color=RED,  lw=1.5, label="τ₂")
        ax.set_ylabel("Torque (N·m)")
        ax.set_title("Control inputs τ")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ── (e) GIBO query points in joint space ──────────────────────────────
        ax = axes[2, 0]
        sc = ax.scatter(
            np.degrees(qp_arr[:, 0]), np.degrees(qp_arr[:, 1]),
            c=t_arr, cmap="viridis", s=8, alpha=0.7,
        )
        ax.axvline(150, color=RED, ls="--", lw=1.2, label="$\\theta_{max}$")
        ax.set_xlabel("θ₁ (deg)")
        ax.set_ylabel("θ₂ (deg)")
        ax.set_title("GIBO query points (joint space)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label="time (s)")

        # ── (f) GP gradient estimate vs analytic ──────────────────────────────
        ax = axes[2, 1]
        if len(grad_arr):
            ax.plot(t_arr, grad_arr[:, 0], color=BLUE, lw=1.5, label="∂J/∂θ₁  (→ −1)")
            ax.plot(t_arr, grad_arr[:, 1], color=RED,  lw=1.5, label="∂J/∂θ₂  (→  0)")
            ax.axhline(-1.0, color=BLUE, ls=":", lw=1.0)
            ax.axhline( 0.0, color=RED,  ls=":", lw=1.0)
        ax.set_ylabel("gradient estimate")
        ax.set_title("GP gradient $\\mathbb{E}[\\nabla J]$ vs analytic $[-1,\\ 0]$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def list_to_np(self):
        processed_logs = {}
        for key, value in self.logs.items():
            if key == "state":
                # Extract x and y attributes from each State object
                # This turns a list of Objects into a [N, 2] float matrix
                processed_logs["state"] = np.array([[s.x, s.y] for s in value])
            elif key == "query_points":
                # Ensure query points are also a 2D array [N, 2]
                processed_logs["query_points"] = np.array(value)
            else:
                processed_logs[key] = np.array(value)
        return processed_logs