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

    def sdf(self, Noise):
        '''
        -- Step 4: Sample Noisy Objective Function (Signed Distance Function) --
        . Injects Noise into obstacle to return a noisy signed distance
        . returns distance from the closest point defining the obstacle, along with the index to access the closest point
        . robot_x, robot_y: current x,y position
        . circle_pts: Nx2 numpy array
        '''
        Noise.rng.normal(loc=0, scale=Noise.noise_std_dev, size=self.Obstacle.circle_points.shape[0] )
        dx = self.X.x - self.Obstacle.circle_points[:, 0]
        dy = self.X.y - self.Obstacle.circle_points[:, 1]
        distances = np.array([dx,dy]).T
        norms = np.linalg.norm(distances, axis=1)
        min_dist_idx = np.argmin(norms)
        min_dist = np.min(norms)
        return min_dist, min_dist_idx

    def weighted_control(weight):
        '''
        Weighted CBF control + GIBO active safety control
        '''

    def return_gridspace(X):
        '''
        Step 7 of GIBO (loop over m = 1,2,...M)
        - Returns gridspace of points "M" to query information gain over.
        '''
        wander_range = .5
        grid_x, grid_y = hf.make_gridspace(X, wander_range, num_steps=8)
        grid_x = grid_x.flatten(0) # (8,8) -> (64, 1)
        grid_y = grid_y.flatten(0) # (8,8) -> (64, 1)
        gridspace = torch.stack([grid_x, grid_y], dim = 1) # (64, 2)
        return gridspace
    