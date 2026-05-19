import numpy as np
import cvxpy as cp
import gpytorch as gp
import torch

import src.helpers.helper_functions as hf
import src.helpers.gp_helpers as gp_hf
from src.dynamics.augmented_state import LookaheadState
from src.dynamics.state import State


class Noise():
    def __init__(self, rng, noise_std_dev, num_std_deviations):
        self.rng = rng
        self.noise_std_dev = noise_std_dev
        self.num_std_deviations = num_std_deviations
    
class Obstacle():
    def __init__(self, Noise):
        self.Noise = Noise
        self.circle_points = hf.spawn_circle(3.5, 3.5, r=1.6)

    
class GIBO_control():
    '''
    Initials:
     - X = 0,0,0

    Class managing:
     - Obstacle object
     - State {x, y theta}
     - control constants
     - optimization functions
     - noise
    '''
    def __init__(self, Kp, V, alpha_cbf, M_number_of_queries, dt):
        self.D = {"X": [], "Y": []}
        self.dist_to_goal = np.inf
        self.goal_reached = False
        self.rng = np.random.default_rng(seed=2)
        self.actuator_limit = hf.actuator_limit(4.0) # 4 rad/s actuator limiting.
        self.M_number_of_queries = M_number_of_queries

        # -- Control --
        self.V = V
        self.alpha_cbf = alpha_cbf
        self.u = 0.0
        self.Kp = Kp

        # -- Larger Objects --
        self.Noise = Noise(self.rng, noise_std_dev=0.05, num_std_deviations=1.0)
        self.Obstacle = Obstacle(self.Noise)
        self.waypoint = np.array([5.0, 5.0])
        self.X = LookaheadState(0.0, 0.0, 0.0, l=V*dt)

    def sdf(self, X, Noise):
        assert isinstance(X, LookaheadState), "Invalid type for X -- must be of type 'Lookahead State'. "

        '''
        -- Line 4: Sample Noisy Objective Function (Signed Distance Function) --
        . Injects Noise into obstacle to return a noisy signed distance
        . returns distance from the closest point defining the obstacle, along with the index to access the closest point
        . robot_x, robot_y: current x,y position
        . circle_pts: Nx2 numpy array
        '''
        injected_noise = Noise.rng.normal(loc=0, scale=Noise.noise_std_dev, size=self.Obstacle.circle_points.shape[0] )
        dx = X.xp - self.Obstacle.circle_points[:, 0]
        dy = X.yp - self.Obstacle.circle_points[:, 1]
        distances = np.array([dx,dy]).T
        norms = np.linalg.norm(distances, axis=1)
        min_dist_idx = np.argmin(norms)
        min_dist = np.min(norms)
        return min_dist + injected_noise[min_dist_idx], min_dist_idx

    def optimal_control(self, goal_xy):
        assert isinstance(self.X, LookaheadState), "Invalid type for X -- must be of type 'Lookahead State'. "

        '''
        Solves for the optimal control input u
        by finding the instantaneous angular error and doing Kp * angular_error
        . returns in rads2deg
        '''
        dx = goal_xy[0] - self.X.xp
        dy = goal_xy[1] - self.X.yp
        theta_target = np.atan2(dy, dx)
        theta_error = theta_target - self.X.theta
        theta_error = (theta_error + np.pi) % (2 * np.pi) - np.pi
        return self.Kp * theta_error # 

    def return_gridspace(self):

        '''
        Line 7 of GIBO (loop over m = 1,2,...M)
        - Returns gridspace of points "M" to query information gain over.
        '''
        wander_range = .5
        grid_x, grid_y = hf.make_gridspace(self.X, wander_range, num_steps=8)
        grid_x = grid_x.flatten(0) # (8,8) -> (64, 1)
        grid_y = grid_y.flatten(0) # (8,8) -> (64, 1)
        gridspace = torch.stack([grid_x, grid_y], dim = 1) # (64, 2)
        return gridspace
    def return_dist_vec(self):
        '''

        X: object of interest, type LookaheadState
        waypoint: 2d list of desired XY waypoint (2,1)

        returns r_norm
        - r_norm: norm of r
        '''
        r = [self.X.x - self.waypoint[0], self.X.y - self.waypoint[1]]
        self.dist_to_goal = np.linalg.norm(r)


    def CBF_SDF(self, posterior):
        '''
        Returns a signed distance based on h_safe - variance

        params:
        posterior = GP_model(self.D["X"], self.D["Y"])
        '''
        h_mean = posterior.mean             # SDF estimate
        h_variance  = posterior.variance         # SDF variance
        h_safe = h_mean - h_variance
        return h_safe
    
    def compute_grad_K(self, train_x, GP_model, lengthscale):
        l = lengthscale
        x_curr = train_x.new_tensor([[self.X.xp, self.X.yp]])
        diffs = x_curr - train_x
        K_x_all = GP_model.covar_module(x_curr, train_x).evaluate().T # (N+M, 1)
        grad_K =  -(diffs/l**2) * K_x_all # (N+M, 2)
        return grad_K
    
    def compute_gradient_update_weights(self, train_x, train_y, GP_model, likelihood):
        '''
        Computes weights after extending training set with GIBO Steps 7-10
        '''

        K_xx = GP_model.covar_module(train_x, train_x).evaluate()
        obs_noise = likelihood.noise
        K_xx_noisy = K_xx + obs_noise * torch.eye(len(train_x))
        weights = torch.linalg.solve(K_xx_noisy, train_y) # (N+M, 1)
        return weights

    def acquisition_function(self, train_x, train_y, 
                            GP_model, GP_likelihood, waypoint, 
                            sigma2, l, obs_noise, next_query_point, query_space, m):
        """
        Evaluates gridspace (M = 1,2,...M) to select a query point based on moving to the point of maximal gradient covariance.
        Also returns
        """

        # -- Noisy K --
        K_xx = GP_model.covar_module(train_x, train_x).evaluate().detach()
        obs_noise = GP_likelihood.noise.detach()
        K_xx_noisy = K_xx + obs_noise*torch.eye(len(train_x)) # K + variance*I
        K_inv = torch.inverse(K_xx_noisy)

        # Prior Gradient Covariance (Constant for RBF) <- Constant, can leave out of trace maximization loop.
        grad2_K_prior = (sigma2 / l**2) * torch.eye(2)

        best_acq_value = -float("inf")
        next_qp = query_space[0]  # safe default
        for qp in query_space:
            dist = train_x - qp.unsqueeze(0)
            K_qp_x = GP_model.covar_module(qp.unsqueeze(0), train_x).evaluate().detach()
            grad_K_x_qp = -(1/l**2) * (K_qp_x.T * dist)

            tr_query_point = torch.trace(grad_K_x_qp.T @ K_inv @ grad_K_x_qp)

            if tr_query_point > best_acq_value:
                best_acq_value = tr_query_point
                next_qp = qp
        print(f"Next Query Point: {next_qp}, m: {m}, Posterior Covariance of QP: {best_acq_value}")
        return next_qp, best_acq_value
    
    def solve_cbf_qp(self, u_GIBO, h_safe, grad_h, alpha_cbf, X):
        '''
        Optimizes a QP to find a control input "u_safe"

        params:
        - u_GIBO: angular velocity commanded by GIBO to minimize posterior covariance
        - h_safe: Predicted signed distance with Safety Margin
        - alpha_cbf: selected constant for CBF condition
        '''
        l = X.l
        theta = X.theta
        u_safe = cp.Variable(1) # variable to optimize
        QP_objective = cp.Minimize(0.5 * cp.sum_squares(u_safe - u_GIBO))

        dhdx = grad_h[0]
        dhdy = grad_h[1]
        # h_dot = (dhdx*np.cos(theta) + dhdy*np.sin(theta))*self.V + (dhdx*(-l*np.sin(theta)) +dhdy*l*np.cos(theta)) * u_safe

        V_term = (dhdx*np.cos(theta) + dhdy*np.sin(theta))
        angular_vel_term = (dhdx*(-l*np.sin(theta)) +dhdy*l*np.cos(theta))
        h_dot = V_term * self.V + (dhdx*(-l*np.sin(theta)) +dhdy*l*np.cos(theta)) * u_safe
        constraints = [h_dot >= -alpha_cbf * h_safe, u_safe <= self.actuator_limit, u_safe >= -self.actuator_limit]
        optimization_problem = cp.Problem(QP_objective, constraints)
        try:
            optimization_problem.solve(solver=cp.OSQP, verbose=False)
            if u_safe.value[0] is not None:
                return u_safe.value[0]
            else:
                # print(f"Infeasible QP, using nominal control (GP): {u.value[0]}")
                return(float(u_GIBO))
        except Exception:
            pass

        # QP infeasible — most common cause: Lg_h ≈ 0 (heading directly at obstacle).
        # Emergency: turn in the direction that grows Lg_h fastest.
        # d(Lg_h)/dθ = −Lf_h/V, so the best turn direction is sign(−Lf_h).
        print(f"[CBF] infeasible  h={V_term:.3f}  Lf_h={V_term:.3f}  Lg_h={angular_vel_term:.4f}, Generating an emergency turn")
        return self.actuator_limit * np.sign(-V_term) if abs(V_term) > 1e-6 else float(u_GIBO)

