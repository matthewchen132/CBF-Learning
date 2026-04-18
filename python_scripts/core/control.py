import numpy as np
import cvxpy as cp
import torch

import helpers.helper_functions as hf
import helpers.gp_helpers as gp_hf
from dynamics.state import State

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
    def __init__(self, Kp, V, alpha1):
        self.D = {"X": [], "sdf" : []} # x, y, theta
        self.goal_reached = False
        self.rng = np.random.default_rng(seed=2)
        self.max_ang_vel = hf.actuator_limit(4.0) # 4 rad/s actuator limiting.

        self.V = V
        self.alpha1 = alpha1
        self.Kp = Kp

        # -- Larger Objects --
        self.Noise = Noise(self.rng, noise_std_dev=0.05, num_std_deviations=1.0)
        self.Obstacle = Obstacle(self.Noise)
        self.waypoint = np.array([5.0, 5.0])
        self.X = State(0.0, 0.0, 0.0)

    def sdf(self, X, Noise):
        '''
        -- Line 4: Sample Noisy Objective Function (Signed Distance Function) --
        . Injects Noise into obstacle to return a noisy signed distance
        . returns distance from the closest point defining the obstacle, along with the index to access the closest point
        . robot_x, robot_y: current x,y position
        . circle_pts: Nx2 numpy array
        '''
        injected_noise = Noise.rng.normal(loc=0, scale=Noise.noise_std_dev, size=self.Obstacle.circle_points.shape[0] )
        dx = X.x - self.Obstacle.circle_points[:, 0]
        dy = X.y - self.Obstacle.circle_points[:, 1]
        distances = np.array([dx,dy]).T
        norms = np.linalg.norm(distances, axis=1)
        min_dist_idx = np.argmin(norms)
        min_dist = np.min(norms)
        return min_dist + injected_noise, min_dist_idx

    def weighted_control(weight):
        '''
        Weighted CBF control + GIBO active safety control
        '''

    def return_gridspace(X):
        '''
        Line 7 of GIBO (loop over m = 1,2,...M)
        - Returns gridspace of points "M" to query information gain over.
        '''
        wander_range = .5
        grid_x, grid_y = hf.make_gridspace(X, wander_range, num_steps=8)
        grid_x = grid_x.flatten(0) # (8,8) -> (64, 1)
        grid_y = grid_y.flatten(0) # (8,8) -> (64, 1)
        gridspace = torch.stack([grid_x, grid_y], dim = 1) # (64, 2)
        return gridspace
    
    def SE_acq_func_grad_covariance(self, X, V, train_x, train_y, 
                                    GP_model, GP_likelihood, waypoint, 
                                    sigma2, l, obs_noise, next_query_point, M):
        """
        Evaluates gridspace (M = 1,2,...M) to select a query point based on moving to the point of maximal gradient covariance.
        """

        # -- Noisy K --
        K_xx = GP_model.covar_module(train_x, train_x).evaluate().detach()
        obs_noise = GP_likelihood.noise.detach()
        K_xx_noisy = K_xx + obs_noise*torch.eye(len(train_x)) # K + variance*I
        K_inv = torch.inverse(K_xx_noisy)

        # 2. Prior Gradient Covariance (Constant for RBF) <- Constant, can compute out of loop.
        grad2_K_prior = (sigma2 / l**2) * torch.eye(2)

        for qp in M:
            dist = qp - train_x
            K_qp_x = GP_model.covar_module((qp.unsqueeze(0), train_x)).evaluate().detach()
            grad_K_x_qp = -(1/l**2) * (K_qp_x.T * dist)

            tr_query_point = torch.trace(grad_K_x_qp.T @ K_inv & grad_K_x_qp)
            
            best_acq_value = -float("inf")
            if tr_query_point > best_acq_value:
                best_acq_value = tr_query_point
                next_qp = qp
        return next_qp