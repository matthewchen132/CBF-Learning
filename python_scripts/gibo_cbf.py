import torch
import gpytorch as gp
import numpy as np
import matplotlib.pyplot as plt
import math
import cvxpy as cp


# from GIBO
from pkg.gp_simple import m52_example
from pkg.gp_helpers import gp_helpers
'''
 - General Workflow of this Script --
1) Train the GP on the 160 LiDAR points (0 distance) + 1 Robot point (current distance).
2) Predict at the Robot's point to get h_mean and h_std.
3) Calculate the Gradient.
4) Plug h_safe and grad_h into the CBF.


03/24 Deliverable
Safe navigation of robot with obstacles, (get familiar with signed distance functions)
 - implement learning cbf
 - compare performance with Azra's weighted acq funciton, gibo
 - Maybe we can design an acquisition function that minimizes the posterior variance rather than the gradient and see how it performs.

What am I trying to do?
 - See if learning a NOISY control bar using a GP and GIBO provides 
    1) comparable Results than conventional methods with PERFECT data 
    2) Better Results than conventional methods with NOISY, perfect data.
 
IDEAS:
    Adaptive step size to not get caught in local minima. With very small cov change of each step, Take a leap 
    

 1) spawn robot <- Done
 2) spawn obstacle <- Done
 3) make signed distance function <- Done
 4) traditional navigation to waypoint <- Done

 5) GP Learning based navigation to waypoint <- IP
'''

class State:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def __add__(self, other):
        return State(
            self.x + other.x,
            self.y + other.y,
            self.theta + other.theta
        )

    def __mul__(self, scalar):
        return State(
            self.x * scalar,
            self.y * scalar,
            self.theta * scalar
        )
def in_hemisphere(X, V, grid_point):
    '''
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
    '''
    Returns a filtered gridspace
    '''


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



def actuator_limit(ang_vel_rads):
    '''TODO, user-defined actuator limit for angular velocity'''
    return np.abs(ang_vel_rads)

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
def dXdt(t, X, V, u):
    '''
    Modeling as a Dubin's Car
    
    -- inputs --
    . velocity (Pre-defined Constat)
    . u: angular velocity (control input)

    -- State Equations -- (X is state, x is position variable)
    . X_dot = AX + Bu 
    . x_dot = V * cos(theta)
    . y_dot = V * sin(theta)
    . theta_dot = u
    '''

    x_dot = V * np.cos(X.theta)
    y_dot = V * np.sin(X.theta)
    theta_dot = u
    return State(x_dot, y_dot, theta_dot)

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

def return_visible_pts(circle_obstacle, closest_idx):
    N = len(circle_obstacle)
    circle_pts_copy = circle_obstacle.copy()
    # -- Create pairs 180 deg from each other. --
    # . we can see the closest 160 deg based on the closest visual index.
    indices = np.arange(closest_idx-80, closest_idx+80) % N; # assuming we miss the edges of the hemisphere (retains 160 deg)
    return circle_pts_copy[indices]

def rk4_step(X, V, t, dt, u):
    '''
    Runge Kutta 4 Numerical Integration
    
    '''
    k1 = dXdt(t, X, V, u)
    k2 = dXdt(t + 0.5*dt, X + k1*0.5*dt, V, u)
    k3 = dXdt(t + 0.5*dt, X + k2*0.5*dt, V, u)
    k4 = dXdt(t + dt, X + k3*dt, V, u)
    X = X + (k1 + k2*2 + k3*2 + k4) * (dt/6.0)
    # print(f"X : {round(X.x,2)}, Y : {round(X.y,2)} theta : {round(X.theta * 180/math.pi, 2)}")
    return X;
def gp_sdf():
    print("")
def gen_waypoint(gridsize):
    '''
    Generates a random waypoint in gridsize
    '''
    waypoint_x = np.random.uniform(0.5*gridsize, -0.5*gridsize)
    waypoint_y = np.random.uniform(0.5*gridsize, -0.5*gridsize)
    waypoint = np.array([waypoint_x, waypoint_y])
    return waypoint
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
    # -- The Remainder of b_dot (The 'drift' term LfLf_h) --
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

def GIBO(gridspace, X, V, train_x, train_y, GP_model, waypoint, 
                        l, sigma2, obs_noise, cov_old):
    """
    Evaluates gridspace to find the point that maximizes the Trace reduction 
    of the GP Covariance (Information Gain).
    """
    best_acq_value = -float('inf')
    next_qp = None
    tr_cov_old = torch.trace(cov_old)
    rt5 = np.sqrt(5)

    # Convert waypoint to tensor for distance calc if needed
    waypoint_t = torch.tensor(waypoint, dtype=torch.float32)

    for query_point in gridspace:
        # 1. Hemisphere Filter (Keep it relative to robot velocity)
        if not in_hemisphere(X, V, grid_point=query_point):
            continue  
        
        # 2. Kernel Math for the Augmented Set (Current + Candidate)
        train_x_new = torch.cat((train_x, query_point.unsqueeze(0)), dim=0)
        
        # -- grad2 K(query_point, query_point) --
        grad2_K_tt = -(5 * sigma2 / (3 * l**2)) * torch.eye(2)

        # -- grad1 K (query_point, train_x_new) --
        diff = query_point.unsqueeze(0) - train_x_new # (N+1, 2)
        r = torch.norm(diff, dim=1, keepdim=True) # (N+1, 1)

        # Matern 5/2 Gradient logic
        K_xx_new = GP_model.covar_module(train_x_new, train_x_new).evaluate().detach()
        K_xx_noisy_new = K_xx_new + obs_noise * torch.eye(len(train_x_new))
        
        # Gradient of Kernel: dK/dx
        grad_K_tx_new = -sigma2 * (5.0 * diff / (3 * l**2)) * (1 + rt5 * r / l) * torch.exp(-rt5 * r / l)
        K_inv_Ktx = torch.linalg.solve(K_xx_noisy_new, grad_K_tx_new) 
        
        # -- Covariance --
        cov_new = grad2_K_tt - grad_K_tx_new.T @ K_inv_Ktx 
        tr_cov_new = torch.trace(cov_new)

        acq_function = tr_cov_old - tr_cov_new
        if acq_function > best_acq_value: 
            best_acq_value = acq_function
            next_qp = query_point.clone().reshape(1, 2)
    return next_qp, best_acq_value


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

def return_dist_vec(X, waypoint):
    r = [X.x - waypoint[0], X.y - waypoint[1]]
    return r, np.linalg.norm(r)

def generate_gp_training_data(gp_obstacle_data, X, visible_pts):
    gp_obstacle_data = torch.tensor(visible_pts) # (N, 2)
    pos_x = torch.tensor([X.x, X.y]).reshape(shape=[1,2]) # (N,2)
    train_x = torch.cat([gp_obstacle_data, pos_x], dim=0).squeeze(-1) # (N+1, 2)

    surface_y = torch.zeros(visible_pts.shape[0], 1) # (N,1) <-- surface of SDF is marked as zeros
    dist_to_obj = torch.min(torch.norm(gp_obstacle_data - pos_x, dim=1)).reshape(shape=[1,1]) # (1,1)
    train_y = torch.cat([surface_y, dist_to_obj], dim=0).squeeze(-1) # (N+1, 1)
    return train_x, train_y

def main():
    # -- Simulation Setup --
    t = 0.0
    dt = 0.1
    Kp = 4.0
    end_time = 20.0

    # -- Create a Waypoint --
    waypoint = np.array([5.0, 5.0])

    # -- Spawn Robot --
    [x_i, y_i, theta_i] = [0.0, 0.0, 0.0]
    X = State(x_i, y_i, theta_i)
    X_noisy = State(x_i, y_i, theta_i)
    X_gp_noisy = State(x_i, y_i, theta_i)
    V = 1.0

    # -- Spawn Obstacle --
    circle_obstacle = spawn_circle(3.5, 3.5, r=1.6)
    noisy_circle_obstacle = spawn_circle(3.5, 3.5, r=1.6)
    noisy_gp_circle_obstacle = spawn_circle(3.5, 3.5, r=1.6)

    # -- Position (Plotting Only) --
    robot_pos = []
    noisy_robot_pos = []
    gp_robot_pos = []
    goal1_reached = False
    goal2_reached = False
    gp_reached = False

    # -- Control --
    u_data = []
    u_noisy_data = []
    u_gp_data = []
    query_points = []
    dist_data = []
    dist_noisy_data = []
    dist_gp_data = []

    alpha1 = 1.0
    alpha2 = 1.5

    times = []
    rng = np.random.default_rng(seed=1)
    noise_std_dev = 0.05
    max_ang_vel = actuator_limit(4.0)

    # -- GIBO --
    step_size = 0.1
    # next_qp: point of interest for polling
    next_qp = torch.tensor([[0.2, 0.2]], dtype=float) #  <- set initially to a step_size towards the way_point

    # -- Simulation Loop --
    while(t < end_time):
        robot_pos.append([X.x, X.y])
        noisy_robot_pos.append([X_noisy.x, X_noisy.y])
        gp_robot_pos.append([X_gp_noisy.x, X_gp_noisy.y])

        noise = rng.normal(0, noise_std_dev, size=circle_obstacle.size)
        times.append(t)
        # -- Conventional CBF --
        # . h(x) = sqrt ( (x-xc)^2 + (y-yc)^2 ) - r_c
        h, closest_idx = sdf(X.x, X.y, circle_obstacle, Noisy=False, noise=noise)
        h_noisy, closest_noisy_idx = sdf(X_noisy.x, X_noisy.y, noisy_circle_obstacle, Noisy=True, noise=noise) 
        # -- Construct Training Data for GIBO / GP -- 
        closest_gp_idx = sdf(X_gp_noisy.x, X_gp_noisy.y, noisy_gp_circle_obstacle, Noisy=True, noise=noise)[1]
        visible_pts = return_visible_pts(noisy_gp_circle_obstacle, closest_gp_idx)
        noise_vector = rng.normal(0, noise_std_dev, size=visible_pts.shape)
        noisy_visible_pts = visible_pts + noise_vector
        gp_obstacle_data = torch.tensor(visible_pts) # (N, 2)

        # -- GP Training Data --
        train_x, train_y = generate_gp_training_data(gp_obstacle_data, X, visible_pts)

        # -- Set up GP Model -- 
        m52_gaussian_likelihood = gp.likelihoods.GaussianLikelihood()
        GP_model = m52_example(train_x=torch.tensor(train_x), train_y=torch.tensor(train_y), likelihood=m52_gaussian_likelihood)
        m52_mll = gp.mlls.ExactMarginalLogLikelihood(m52_gaussian_likelihood, GP_model) # finds probability of the function found by GP by comparing to sampled data.
        m52_optimizer = torch.optim.Adam(
                list(GP_model.parameters()) + list(m52_gaussian_likelihood.parameters()),
                lr=0.05
            )
        # - Cut down training samples on later samples -
        if t == 0.0:
            n_samples = 200
        else:
            n_samples = 50

        # -- Train GP -- 
        for i in range(n_samples):
            # -- Provides the posterior GP -- 
            m52_optimizer.zero_grad() # needed to feed the m52_model only the CURRENTLY accumulated gradient
            m52_output = GP_model(torch.tensor(train_x))
            m52_loss = -m52_mll(m52_output, torch.tensor(train_y)).sum()
            m52_loss.backward()
            m52_optimizer.step()

        GP_model.eval()
        m52_gaussian_likelihood.eval()
        with torch.enable_grad(), gp.settings.fast_pred_var(): 
            # (1) -- Create a grid around current position for the robot to select within using GIBO --
            '''
            Explanation:
             - First, GPs provide an estimate of the function mean, and the covariance associated with it.
             - High Covariance = High uncertainty, Polling information at that point will give us large gains.  
            '''
            wander_range = .5
            grid_x, grid_y = make_gridspace(X_gp_noisy, wander_range, num_steps=8)
            test_x = torch.tensor([X_gp_noisy.x, X_gp_noisy.y]).reshape(1, 2) # (1, 3)
            # (2) --  Find Gradient of the Signed Distance Function --
            '''
             - To create a good control barrier function, we must not only estimate the surface of the obstacle, 
               but also the rate of change with which we are approaching dangerous regions.
             - By finding the gradient, we can enforce a CBF which is safe, yet also allows us to explore the region in a smart manner.
            '''
            sigma2 = GP_model.covar_module.outputscale.detach()
            l = GP_model.covar_module.base_kernel.lengthscale.detach()
            rt5 = math.sqrt(5)
            obs_noise = m52_gaussian_likelihood.noise.detach()

            # -- Noisy K --
            K_xx = GP_model.covar_module(train_x, train_x).evaluate().detach()
            obs_noise = m52_gaussian_likelihood.noise.detach()
            K_xx_noisy = K_xx + obs_noise*torch.eye(len(train_x)) # K + variance*I

        # -- GIBO Acquisition function (Find next x,y based on most uncertain point)--  
        diff_r_old = next_qp.unsqueeze(1) - train_x.unsqueeze(0)
        r_old = torch.norm(diff_r_old, dim=-1, keepdim=True)
        grad2_K_tt = -(5*sigma2 /(3*l**2)) # next query_p - next query_p = 0 -> simplified gradient
        grad2_K_tt = torch.eye(2) * grad2_K_tt # <-- convert to I, 2x2 to fit num cols of data
        grad_K_tx_old = -sigma2 * (5.0 * diff_r_old/(3*l**2)) * (1 + rt5*r_old/l) * torch.exp(-rt5*r_old/l) # (1, 161, 2)
        grad_K_tx_old = grad_K_tx_old.squeeze(0) # (161,2)
        K_xx_inv_K_tx = torch.linalg.solve(K_xx_noisy, grad_K_tx_old) # (2, 2)
        cov_old = grad2_K_tt - grad_K_tx_old.T @ K_xx_inv_K_tx


        # -- Format Gridspace --
        grid_x = grid_x.flatten(0) # (8,8) -> (64, 1)
        grid_y = grid_y.flatten(0) # (8,8) -> (64, 1)
        gridspace = torch.stack([grid_x, grid_y], dim = 1) # (64, 2)

        # -- Find Query Point with GIBO --
        next_qp, best_acq_value = GIBO(gridspace=gridspace, X= X_gp_noisy, V=V, train_x=train_x, train_y=train_y, 
                                           GP_model=GP_model, waypoint=waypoint, l=l, sigma2=sigma2, obs_noise=obs_noise, cov_old=cov_old)

        u_search_point = optimal_control(X_gp_noisy, Kp, next_qp.flatten().tolist())
        # print(f"Next Query point: {torch.round(next_qp[0,0], decimals=2)} , {torch.round(next_qp[0,1], decimals=2)}")
        query_points.append(next_qp[0].tolist())
        # print(f"u to reach query point {round(u_search_point,2)}")

        # -- Extend Data Set to include next query point --
        train_x, train_y = filter_relevant_data(X_gp_noisy, train_x, train_y, max_pts=160)

        # -- Updata next_qp (next point to query) at n+1 guess --
        K_xx = GP_model.covar_module(train_x, train_x).evaluate().detach()
        K_xx_noisy = K_xx + obs_noise * torch.eye(len(train_x))
        diff_r_theta = next_qp.unsqueeze(1) - train_x.unsqueeze(0)
        r_theta = torch.norm(diff_r_theta, dim=-1, keepdim=True)
        grad_K_theta_x = -sigma2 * (5.0 * diff_r_theta/(3*l**2)) * (1 + rt5*r_theta/l) * torch.exp(-rt5*r_theta/l)
        grad_K_theta_x = grad_K_theta_x.squeeze(0) # (162, 2)
        
        # -- (10) Update the posterior probability distribution of ∇θJ. -- 
        alpha = torch.linalg.solve(K_xx_noisy, train_y)
        grad_mean_next_qp = grad_K_theta_x.T @ alpha
        
        # -- move sample forward -- 
        update = (step_size * grad_mean_next_qp).reshape(1, 2)
        next_qp = (next_qp.reshape(1, 2) + update).detach()        
        # -- Find u_nom with GIBO-CBF -- (GP) 
        posterior = GP_model(torch.tensor([X_gp_noisy.x, X_gp_noisy.y], dtype=torch.float32).reshape(1,2))     
        h_mean = posterior.mean             # SDF estimate
        h_var  = posterior.variance         # SDF variance
        h_std  = torch.sqrt(h_var)
        grad_h, grad_h_covariance = gp_helpers.sdf_gp_gradient(GP_model, m52_gaussian_likelihood, test_x.float(), train_x.float(), train_y.float())
        grad_h = torch.tensor(grad_h, dtype=torch.float32)

        # -- PROBABILISTIC SAFE SDF --
        num_gp_std_devs = 1.
        h_safe = h_mean - num_gp_std_devs * h_std # 2 std deviations -> 95% confidence
        # -- CBF Condition --
            # . h_dot(x,y) >= - alpha1 * h
            # . grad_h(x,y) * [dxdt , dydt].T >= - alpha1 * h
        cos_t = torch.cos(torch.tensor(X_gp_noisy.theta))
        sin_t = torch.sin(torch.tensor(X_gp_noisy.theta))
        vel_xy = torch.tensor([V*cos_t, V*sin_t], dtype=torch.float32)
        Lf_h = grad_h @ vel_xy
        Lg_h = V * (-grad_h[0] * sin_t + grad_h[1] * cos_t)

        # -- GP-GIBO-CBF -- 
        u_gp_nom = optimal_control(X_gp_noisy, Kp=Kp, goal_xy=waypoint)
        r, r_norm = return_dist_vec(X_gp_noisy, waypoint)
        k_dist = 1. # Tuning parameter
        weight = 1.0 - np.exp(-r_norm * k_dist) 
        u_blended_nom = (1-weight) * u_gp_nom + weight * u_search_point
        u_gp_cbf = solve_cbf_qp(u_blended_nom, Lf_h.item(), Lg_h.item(), h_safe.item(), alpha1, max_ang_vel)

        # -- Take the maximum of GP-defined control and the optimal control input "u_gp_optimal" -- 
        # -- Find u_nom with CBF -- (Conventional)
        u_final = conventional_cbf(alpha1, alpha2, X, closest_idx, h, circle_obstacle, Kp, V, waypoint)
        u_noisy_final = conventional_cbf(alpha1, alpha2, X_noisy, closest_noisy_idx, h_noisy, circle_obstacle, Kp, V, waypoint)
        # - Append "u" data -
        u_data.append(u_final)
        u_noisy_data.append(u_noisy_final)
        u_gp_data.append(u_gp_cbf)
        dist_data.append(np.linalg.norm([X.x - waypoint[0], X.y - waypoint[1]]))
        dist_noisy_data.append(np.linalg.norm([X_noisy.x - waypoint[0], X_noisy.y - waypoint[1]]))
        dist_gp_data.append(np.linalg.norm([X_gp_noisy.x - waypoint[0], X_gp_noisy.y - waypoint[1]]))

        # -- Update Position w/ RK4 --
        X = rk4_step(X, V, t, dt, u_final)
        X_noisy = rk4_step(X_noisy, V, t, dt, u_noisy_final)
        X_gp_noisy = rk4_step(X_gp_noisy, V, t , dt, u_gp_cbf)
        print(f"u (GIBO / GP): {u_gp_cbf} | Weight: {round(weight,2)}")

        # -- Check if Waypoint Reached -- (Function this after gibo) TODO
        r, r_norm = return_dist_vec(X=X, waypoint=waypoint)
        r_noisy, r_noisy_norm  = return_dist_vec(X=X, waypoint=waypoint)
        r_gp, r_gp_norm = return_dist_vec(X=X, waypoint=waypoint)

        if r_norm < 0.05 or goal1_reached:
            # -- Stay at Waypoint once reached --
            X.x = waypoint[0]
            X.y = waypoint[1]
            goal1_reached = True
        else:
            zero_noise_cbf_time = t
        if r_noisy_norm < 0.1 or goal2_reached:
            # -- Stay at Waypoint once reached --
            X_noisy.x = waypoint[0]
            X_noisy.y = waypoint[1]
            goal2_reached = True
        else:
            noisy_cbf_time = t
        if r_gp_norm < 0.1 or gp_reached:
            # -- Stay at Waypoint once reached --
            X_gp_noisy.x = waypoint[0]
            X_gp_noisy.y = waypoint[1]
            gp_reached = True
        else:
            gp_cbf_time = t
        if goal1_reached and goal2_reached and gp_reached:
            break
        # -- Increase Sim. Time --
        t += dt
    
    print(f"Zero-noise finish time: {zero_noise_cbf_time}, Noisy CBF finish time: {noisy_cbf_time}, GP-CBF finish time: {gp_cbf_time}")

    robot_pos = np.array(robot_pos)
    noisy_robot_pos = np.array(noisy_robot_pos)
    gp_robot_pos = np.array(gp_robot_pos)
    query_points = np.array(query_points)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12)) 
    # -- Trajectory
    axes[0].plot(circle_obstacle[:,0], circle_obstacle[:,1], 'k', label="Obstacle")
    axes[0].plot(waypoint[0], waypoint[1], 'g*', label="waypoint")
    axes[0].plot(robot_pos[:,0], robot_pos[:,1], 'b--', label="Ideal (Zero Noise)")
    axes[0].plot(noisy_robot_pos[:,0], noisy_robot_pos[:,1], 'r-', label="Ideal (Zero Noise)")
    axes[0].plot(gp_robot_pos[:,0], gp_robot_pos[:,1], 'g-', linewidth=2, label=f"GP-GIBO-CBF with {num_gp_std_devs} Deviations")

    axes[0].scatter(query_points[:,0], query_points[:,1], 
                    c=np.linspace(0, 1, len(query_points)), 
                    cmap='Blues', s=10, alpha=0.3, label="GIBO Pts (Darker blue = Latest)")
    axes[0].set_title("Trajectories (Ideal CBF vs. Noisy CBF vs. GP-GIBO-CBF)")
    axes[0].axis("equal")
    axes[0].legend()
    axes[0].grid(True)

    # -- Distance Error Norm vs Time -- 
    # Ensure all arrays are the same length as t_arr to avoid Shape Mismatch
    t_arr = np.array(times)
    n = len(t_arr)
    d_ideal = np.array(dist_data[:n])
    d_noisy = np.array(dist_noisy_data[:n])
    d_gp    = np.array(dist_gp_data[:n])

    axes[1].plot(t_arr, d_ideal, 'b-', label="Ideal")
    axes[1].plot(t_arr, d_noisy, 'g-', label="Noisy")
    axes[1].plot(t_arr, d_gp, 'y-', label="GP-GIBO")
    axes[1].set_title("Error (Dist. to Waypoint)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Distance (m)")
    axes[1].grid(True)
    axes[1].legend()

    # -- Control Effort (u) --
    # Clipping u_data to match time length
    u_ideal_plot = np.array(u_data[:n])
    u_noisy_plot = np.array(u_noisy_data[:n])
    u_gp_plot    = np.array(u_gp_data[:n])

    axes[2].plot(t_arr, u_ideal_plot, 'b', label="Nominal u")
    axes[2].plot(t_arr, u_noisy_plot, 'g', label="Noisy u")
    axes[2].plot(t_arr, u_gp_plot, 'y', label="GP-CBF u")
    axes[2].set_title("Control Input (u) Comparison")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Angular Velocity (rad/s)")
    axes[2].legend()

    plt.tight_layout() # Prevents label overlap
    plt.show()
    print("End Function")

if __name__ == "__main__":
    main()