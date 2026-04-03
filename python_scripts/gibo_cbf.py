import torch
import gpytorch as gp
import numpy as np
import matplotlib.pyplot as plt
import pybullet_data
import math

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

def actuator_limit():
    '''TODO, user-defined actuator limit for angular velocity'''
    print("todo")

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

def make_gridspace(X, wander_range):
    '''
    Takes in a State X and wander_range to generate around (+/- wander_range)
    '''
    x_fwd = X.x
    y_fwd = X.y
    wander_range = 0.4
    x_range = torch.linspace(start=x_fwd-wander_range, end=x_fwd+wander_range, steps=8)
    y_range = torch.linspace(start=y_fwd-wander_range, end=y_fwd+wander_range, steps=8)
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
    print(f"X : {round(X.x,2)}, Y : {round(X.y,2)} theta : {round(X.theta * 180/np.pi, 2)}")
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
        u_final = max(u_nom, (-alpha2 * b - dbdt) / (coeff_u + 1e-6))
    elif coeff_u < 0:
        # u must be LESS than some value
        u_final = min(u_nom, (-alpha2 * b - dbdt) / (coeff_u + 1e-6))
    else:
        u_final = u_nom
    # -- find u_nom -- 
    return u_final
def main():
    # -- Simulation Setup --
    t = 0.0
    dt = 0.1
    u = 0.0
    Kp = 4.0
    end_time = 20.0

    # -- Create a Waypoint --
    waypoint = np.array([5.0, 5.0])

    # -- Spawn Robot --
    x_i = 0.0
    y_i = 0.0
    z_i = 0.0
    X = State(x_i, y_i, z_i)
    X_noisy = State(x_i, y_i, z_i)
    X_gp_noisy = State(x_i, y_i, z_i)
    V = 1.0

    # -- Spawn Obstacle --
    circle_obstacle = spawn_circle(3.5, 3.5, r=1.0)
    noisy_circle_obstacle = spawn_circle(3.5, 3.5, r=1.0)
    noisy_gp_circle_obstacle = spawn_circle(3.5, 3.5, r=1.0)

    # -- Position (Plotting Only) --
    robot_pos = []
    noisy_robot_pos = []
    gp_robot_pos = []

    goal1_reached = False
    goal2_reached = False
    # -- Control --
    u_data = []
    noisy_u_data = []
    times = []
    rng = np.random.default_rng(seed=1)
    noise_std_dev = 0.01

    # -- GIBO --
    step_size = 0.1
    # - theta_xy: point of interest for polling
    # . In GIBO, theta_xy = point of high covariance
    theta_xy = torch.tensor([[0.2, 0.2]], dtype=float) #  <- set initially to a step_size towards the way_point

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
        h_noisy, closest_noisy_idx = sdf(X.x, X.y, noisy_circle_obstacle, Noisy=True, noise=noise) 
        # -- Construct Training Data for GIBO / GP -- 
        closest_gp_idx = sdf(X_gp_noisy.x, X_gp_noisy.y, noisy_gp_circle_obstacle, Noisy=True, noise=noise)[1]
        # -- Logic to Detect Visible Points (Mock Lidar) --
        visible_pts = return_visible_pts(noisy_gp_circle_obstacle, closest_gp_idx)

        # -- GP Training Data --
        gp_obstacle_data = torch.tensor(visible_pts) # (N, 2)
        pos_x = torch.tensor([X_gp_noisy.x, X_gp_noisy.y]).reshape(shape=[1,2]) # (N,2)
        train_x = torch.cat([gp_obstacle_data, pos_x], dim=0).squeeze(-1) # (N+1, 2)

        surface_y = torch.zeros(visible_pts.shape[0], 1) # (N,1) <-- surface of SDF is marked as zeros
        dist_to_obj = torch.min(torch.norm(gp_obstacle_data - pos_x, dim=1)).reshape(shape=[1,1]) # (1,1)
        train_y = torch.cat([surface_y, dist_to_obj], dim=0).squeeze(-1) # (N+1, 1)

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
            n_samples = 10

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
            wander_range = 0.4
            grid_x, grid_y = make_gridspace(X_gp_noisy, wander_range)
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
            print(f"K_xx_noisy {K_xx_noisy.size()}\n")

            posterior = GP_model(torch.tensor([X_gp_noisy.x, X_gp_noisy.y], dtype=torch.float32).reshape(1,2))     
            h_mean = posterior.mean             # SDF estimate
            h_var  = posterior.variance         # SDF variance
            h_std  = torch.sqrt(h_var)

            grad_h, grad_h_covariance = gp_helpers.sdf_gp_gradient(GP_model, m52_gaussian_likelihood, test_x.float(), train_x.float(), train_y.float())
            grad_h = torch.tensor(grad_h, dtype=torch.float32)

            # -- PROBABILISTIC SAFE SDF --
            h_safe = h_mean - 2.0 * h_std # 2 std deviations -> 95% confidence
            # -- CBF Condition --
                # . h_dot(x,y) >= - alpha1 * h
                # . grad_h(x,y) * [dxdt , dydt].T >= - alpha1 * h
                # . Lf * h(x,y) + Lg * h(x,y) + alpha1 * h(x,y)  >= 0 
            vel_xy = torch.tensor([V*torch.cos(torch.tensor(X_gp_noisy.theta)), V*torch.sin(torch.tensor(X_gp_noisy.theta))], dtype=float)
            vel_xy = torch.tensor(vel_xy, dtype=torch.float32)
            Lf_h = grad_h @ vel_xy
            Lg_h = 0.0
            u_gp_optimal = optimal_control(X_gp_noisy, Kp=Kp, goal_xy=noisy_gp_circle_obstacle)
            
        print(f"Computed gradient mean {1} and gradient Covariance {1}")
        # -- GIBO Acquisition function (Find next x,y based on most uncertain point)--
        diff_r_old = theta_xy.unsqueeze(1) - train_x.unsqueeze(0)
        r_old = torch.norm(diff_r_old, dim=-1, keepdim=True)
        grad2_K_tt = -(5*sigma2 /(3*l**2)) # theta_xy - theta_xy = 0 -> simplified gradient
        grad2_K_tt = torch.eye(2) * grad2_K_tt # <-- convert to I, 2x2 to fit num cols of data
        grad_K_tx_old = -sigma2 * (5.0 * diff_r_old/(3*l**2)) * (1 + rt5*r_old/l) * torch.exp(-rt5*r_old/l) # (1, 161, 2)
        grad_K_tx_old = grad_K_tx_old.squeeze(0) # (161,2)
        K_xx_inv_K_tx = torch.linalg.solve(K_xx_noisy, grad_K_tx_old) # (2, 2)
        cov_old = grad2_K_tt - grad_K_tx_old.T @ K_xx_inv_K_tx
        best_acq_value = -torch.inf
        next_qp = None
        # -- (7) query over the new grid space -- TODO: maybe sort these poll points to only be in the direction of the obstacle to save iteration loops

        grid_x = grid_x.flatten(0) # (8,8) -> (64, 1)
        grid_y = grid_y.flatten(0) # (8,8) -> (64, 1)
        gridspace = torch.stack([grid_x, grid_y], dim = 1) # (64, 2)
        for i, query_point in enumerate(gridspace): # using the test data as the points to sample.   
            train_x_new = torch.cat((train_x, query_point.unsqueeze(0)), dim=0) # append the query point to the end
            K_xx_new = GP_model.covar_module(train_x_new, train_x_new).evaluate().detach()

            # -- grad2 K(theta_xy, theta_xy)
            grad2_K_tt = -(5*sigma2 /(3*l**2)) # theta_xy - theta_xy = 0 -> simplified gradient
            grad2_K_tt = grad2_K_tt * torch.eye(2)
            # -- grad1 K (theta_xy, X) --
            diff_r_new = theta_xy.unsqueeze(1) - train_x_new.unsqueeze(0) # ??
            r_new = torch.abs(diff_r_new)

            K_xx_noisy_new = K_xx_new + obs_noise*torch.eye(len(train_x_new))
            grad_K_tx_new  = -sigma2 * (5.0 * diff_r_new/(3*l**2)) * (1 + rt5*r_new/l) * torch.exp(-rt5*r_new/l) # (1, 162, 2)
            grad_K_tx_new = grad_K_tx_new.squeeze(0) # (162,2)

            K_inv_Ktx = torch.linalg.solve(K_xx_noisy_new,grad_K_tx_new) # (1, 162, 2)
            K_inv_Ktx = K_inv_Ktx.squeeze(0) # (162, 2)

            # -- New Covariance -- 
            cov_new = grad2_K_tt - grad_K_tx_new.T @ K_inv_Ktx # (2,2)
            tr_cov_new = torch.trace(cov_new)
            tr_cov_old = torch.trace(cov_old)

            acq_function = tr_cov_old - tr_cov_new
            # -- (8) get query point "theta_xy" --
            if acq_function > best_acq_value: 
                # Extract the max acquisition function value + store the kernel/ grad_kernel values
                best_acq_value = acq_function
                next_qp = query_point
        # -- (9) Sample Noisy Objective Function -- 
        sdf_qp = sdf(robot_x=next_qp[0].item(), robot_y=next_qp[1].item(), circle_pts=noisy_gp_circle_obstacle, Noisy=True, noise=noise)[0]
        h_gp = torch.tensor([sdf_qp])
        # -- (10) Extend Data Set to include next query point --
        train_x = torch.cat((train_x, next_qp.unsqueeze(0)), dim=0) # (163, 2)
        train_y = torch.cat((train_y, h_gp), dim=0)

        # -- Updata theta_xy (next point to query) at n+1 guess --
        K_xx = GP_model.covar_module(train_x, train_x).evaluate().detach()
        K_xx_noisy = K_xx + obs_noise * torch.eye(len(train_x))
        diff_r_theta = theta_xy.unsqueeze(1) - train_x.unsqueeze(0)
        r_theta = torch.norm(diff_r_theta, dim=-1, keepdim=True)
        grad_K_theta_x = -sigma2 * (5.0 * diff_r_theta/(3*l**2)) * (1 + rt5*r_theta/l) * torch.exp(-rt5*r_theta/l)
        grad_K_theta_x = grad_K_theta_x.squeeze(0) # (162, 2)
        # -- (10) Update the posterior probability distribution of ∇θJ. -- 
        alpha = torch.linalg.solve(K_xx_noisy, train_y)
        grad_mu_theta_xy = grad_K_theta_x.T @ alpha
        breakpoint()

        theta_xy = torch.clamp((theta_xy + step_size * grad_mu_theta_xy.squeeze()), 
                              min=start_x, 
                              max=end_x).detach()
        # -- move sample forward -- 
        print(f"Point: {samples_fwd} theta_(t + 1): {theta_xy}")      
        samples_fwd = samples_fwd + 1
        
        # -- Find u_nom with CBF -- 
        alpha1 = 1.0
        alpha2 = 1.5
        u_final = conventional_cbf(alpha1, alpha2, X, closest_idx, h, circle_obstacle, Kp, V, waypoint)
        noisy_u_final = conventional_cbf(alpha1, alpha2, X_noisy, closest_noisy_idx, h_noisy, circle_obstacle, Kp, V, waypoint)
        u_data.append(u_final)
        noisy_u_data.append(noisy_u_final)

        # -- Update Position w/ RK4 --
        X = rk4_step(X, V, t, dt, u_final)
        X_noisy = rk4_step(X_noisy, V, t, dt, noisy_u_final)
        print(f"Current u (no noise): {u_final}")
        print(f"Current u (noisy): {noisy_u_final}")

        # -- Check if Waypoint Reached -- (Function this after gibo) TODO
        dx = abs(waypoint[0] - X.x)
        dy = abs(waypoint[1] - X.y)
        dx_noisy = abs(waypoint[0] - X_noisy.x)
        dy_noisy = abs(waypoint[1] - X_noisy.y)
        dx_gp = abs(waypoint[0] - X_gp_noisy.x)
        dy_gp = abs(waypoint[1] - X_gp_noisy.y)
        if math.sqrt(dx**2 + dy**2) < 0.05 or goal1_reached:
            # -- Stay at Waypoint once reached --
            X.x = waypoint[0]
            X.y = waypoint[1]
            goal1_reached = True
        if math.sqrt(dx_noisy**2 + dy_noisy**2) < 0.1 or goal2_reached:
            # -- Stay at Waypoint once reached --
            X_noisy.x = waypoint[0]
            X_noisy.y = waypoint[1]
            goal2_reached = True
        if math.sqrt(dx_gp**2 + dy_gp**2) < 0.1 or gp_reached:
            # -- Stay at Waypoint once reached --
            X_gp_noisy.x = waypoint[0]
            X_gp_noisy.y = waypoint[1]
            gp_reached = True
        if goal1_reached and goal2_reached and gp_reached:
            break
        # -- Increase Sim. Time --
        t += dt
    robot_pos = np.array(robot_pos)
    noisy_robot_pos = np.array(noisy_robot_pos)

    # -- Plot Config
    plt.plot(circle_obstacle[:,0], circle_obstacle[:,1], label="Obstacle", color='k', linewidth=1)
    plt.plot(noisy_circle_obstacle[:,0], noisy_circle_obstacle[:,1], label="Noisy Obstacle", color='r', linewidth=1)
    plt.plot(waypoint[0], waypoint[1], 'g*', markersize=10, label="Waypoint")
    plt.plot(robot_pos[:,0], robot_pos[:,1], 'b-', markersize=2, label="Zero Noise Position")
    plt.plot(noisy_robot_pos[:,0], noisy_robot_pos[:,1], 'g-', markersize=2, label="Noisy Position")
    plt.xlabel("X")
    plt.xlim(-2,6)
    plt.ylabel("Y")
    plt.ylim(-2,6)
    plt.title(f"Ideal CBF vs CBF with Noise Std. Dev. of {noise_std_dev}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()
    print("End Function")

if __name__ == "__main__":
    main()