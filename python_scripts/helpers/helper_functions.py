import torch
import gpytorch as gp
import numpy as np
import matplotlib.pyplot as plt
import math

import cvxpy as cp

def in_hemisphere(X, V, grid_point):
    '''
    Filters 2d points based on whether the point is in the direction of the velocity vector..
        - returns True if is in direction of vector.
        - returns False if not in direction of vector.

    params:
    r: vector between grid_point and position
    V: Velocity
    X: State of robot
        - x,y, theta
        - returns True or False.
    '''
    X_vec = [V*np.cos(X.theta), V*np.sin(X.theta)]
    r = [grid_point[0] -X.x , grid_point[1] - X.y]
    if np.dot(X_vec, r) <= 0:
        return False
    return True
    
def spawn_circle(x, y, r):
    '''
    Spawns a 2d circle based on x-y position, radius, and 
    360 pts used to define the circle
    '''
    theta = 0
    pts = []
    while(theta <= 360):
        circle_x = r * np.cos(np.deg2rad(theta)) + x
        circle_y = r * np.sin(np.deg2rad(theta)) + y
        pts.append([circle_x, circle_y])
        theta += 1
    
    pts = np.array(pts)
    return pts

def make_gridspace(X, wander_range, num_steps):
    '''
    Takes in a State X and wander_range to generate around (+/- wander_range)
    '''
    x_fwd = X.x
    y_fwd = X.y
    x_range = torch.linspace(start=x_fwd-wander_range, end=x_fwd+wander_range, steps=num_steps)
    y_range = torch.linspace(start=y_fwd-wander_range, end=y_fwd+wander_range, steps=num_steps)
    grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij') # 8x8 meshgrid
    return grid_x, grid_y

def gen_waypoint(gridsize):
    '''
    Generates a random 2d waypoint in gridsize
    '''
    waypoint_x = np.random.uniform(0.5*gridsize, -0.5*gridsize)
    waypoint_y = np.random.uniform(0.5*gridsize, -0.5*gridsize)
    waypoint = np.array([waypoint_x, waypoint_y])
    return waypoint

def actuator_limit(ang_vel_rads):
    '''TODO, user-defined actuator limit for angular velocity'''
    return np.abs(ang_vel_rads)

def return_visible_pts(circle_obstacle, closest_idx):
    N = len(circle_obstacle)
    circle_pts_copy = circle_obstacle.copy()
    # -- Create pairs 180 deg from each other. --
    # . we can see the closest 160 deg based on the closest visual index.
    indices = np.arange(closest_idx-80, closest_idx+80) % N; # assuming we miss the edges of the hemisphere (retains 160 deg)
    return circle_pts_copy[indices]

def return_dist_vec(X, waypoint):
    '''

    X: object of interest, type State
    waypoint: 2d list of desired XY waypoint (2,1)

    returns r, r_norm
     - r: 2d list of distance between X and waypoint (2,1)
     - r_norm: norm of r
    '''
    r = [X.x - waypoint[0], X.y - waypoint[1]]
    return r, np.linalg.norm(r)

def sdf(robot_x, robot_y, circle_pts, Noisy, noise):
    noise_copy = noise
    circle_pts_copy = circle_pts.copy()
    '''
    -- Signed Distance Function --
    . returns distance from the closest point defining the obstacle, along with the index to access the closest point
    . robot_x, robot_y: current x,y position
    . circle_pts: Nx2 numpy array
    '''
    rng = np.random.default_rng(0)
    if Noisy:
        noise_copy = noise[0:circle_pts.shape[0]]
        circle_pts_copy[:,0] += noise_copy
        circle_pts_copy[:,1] += noise_copy

    dx = robot_x - circle_pts_copy[:, 0]
    dy = robot_y - circle_pts_copy[:, 1]
    distances = np.array([dx,dy]).T
    norms = np.linalg.norm(distances, axis=1)
    min_dist_idx = np.argmin(norms)
    return [norms[min_dist_idx], min_dist_idx]

def filter_relevant_data(X, train_x, train_y, max_pts):
    """
    Keep the points closest to the robot to prevent 
    old GIBO queries from biasing the current control.
    """
    if train_x.shape[0] <= max_pts:
        return train_x, train_y
        
    # Calculate distances
    robot_pos = torch.tensor([X.x, X.y])
    dists = torch.norm(train_x - robot_pos, dim=1)
    
    # Get indices of the 'max_pts' closest points
    _, closest_indices = torch.topk(dists, k=max_pts, largest=False)
    
    return train_x[closest_indices], train_y[closest_indices]

def optimal_control(X, Kp, goal_xy):
    '''
    Solves for the optimal control input u
    by finding the instantaneous angular error and doing Kp * angular_error
    . returns in rads2deg
    '''
    dx = goal_xy[0] - X.x
    dy = goal_xy[1] - X.y
    theta_target = np.atan2(dy, dx)
    theta_error = theta_target - X.theta
    theta_error = (theta_error + np.pi) % (2 * np.pi) - np.pi
    return Kp * theta_error # 


def conventional_cbf(alpha1, alpha2, X, closest_idx, h, circle_obstacle, Kp, V, waypoint):
    '''
    -- Control Barrier Function for Dubin's Car
    - Returns the control "u" at that point
    '''
    u_data = []
# -- dhdx >= -alpha * h(x) --
    alpha1 = 1.0
    grad_x = (X.x - circle_obstacle[closest_idx, 0]) / h
    grad_y = (X.y - circle_obstacle[closest_idx, 1]) / h
    # -- dhdt --
    dhdt = grad_x*V*np.cos(X.theta) + grad_y*V*np.sin(X.theta)
    # -- Let "b(x) = h_dot(x) + alpha1 * h(x)" --
    b = dhdt + alpha1 * h
    # -- Forcing b_dot(x) >= - alpha2 * b(x) brings "u" term back --
    coeff_u = V * (-grad_x * np.sin(X.theta) + grad_y * np.cos(X.theta))
    # -- The Remainder of b_dot (The 'drift' term LgLf_h) --
    dbdt = alpha1 * dhdt 
    # -- The CBF Condition: b_dot >= -alpha2 * b --
    # . (coeff_u * u) + dbdt(x) >= -alpha2 * b
    u_min_allowed = (-alpha2 * b - dbdt) / (coeff_u)
    u_nom = optimal_control(X, Kp, waypoint)
    # -- Minimally Invasive Filter -- 
    # . (coeff_u * u) + dbdt(x) >= -alpha2 * b
    # . u >= (-alpha2 * b(x) - dbdt(x) ) / coeff_u
    if coeff_u > 0:
        # Base case with positive signed coeff_u
        u_final = max(u_nom, (-alpha2 * b - dbdt) / (coeff_u + 1e-3))
    elif coeff_u < 0:
        # u must be LESS than some value
        u_final = min(u_nom, (-alpha2 * b - dbdt) / (coeff_u + 1e-3))
    else:
        u_final = u_nom
    # -- find u_nom -- 
    return np.clip(u_final, -4.0, 4.0)

def solve_cbf_qp(u_nom, Lf_h, Lg_h, h_safe, alpha1, u_max):
    '''
    Uses cvxpy to optimize a control input "u"
    based on the Lie derivatives learned by GP

    params:
    - Lf_h, Lf_g: Lie derivatives w.r.t (f,g)  s.t  f_dot(x) = f(x) + g(x) * u)
    - h_safe: h - N * std_dev  |  gp_predicted signed distance with some margin
    - alpha1: selected constant for CBF condition
    - u_max: actuator_limit defined
    '''
    u = cp.Variable(1)
    QP_objective = cp.Minimize(0.5 * cp.sum_squares(u - u_nom))

    # Lf * h(x,y) + Lg_h * u + alpha1(h(x,y))  >= 0 
    constraints = [Lf_h + Lg_h*u+ alpha1*h_safe >= 0, u <= u_max, u >= -u_max]
    optimization_problem = cp.Problem(QP_objective, constraints)
    try:
        optimization_problem.solve(solver=cp.OSQP, verbose=False)

        if u.value[0] is not None:
            # print(f"using CBF control (GP): {u.value[0]}")
            return u.value[0]
        else:
            # print(f"Infeasible QP, using nominal control (GP): {u.value[0]}")
            return(float(u_nom))
    except:
        # print("using nominal control (GP)")
        return float(u_nom)

def check_if_reached_waypoint(X, r_norm, waypoint, sim_time, reached):
    '''
    returns [sim_time, True] if waypoint is reached.
    '''

    if r_norm < 0.1 or reached:
        # -- Stay at Waypoint once reached --
        X.x = waypoint[0]
        X.y = waypoint[1]
        return [sim_time, True]

    return [sim_time, False]