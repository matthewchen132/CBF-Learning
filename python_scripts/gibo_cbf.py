import torch
import gpytorch as gp
import numpy as np
import matplotlib.pyplot as plt
import pybullet_data
import math

# from GIBO
from gp_simple import RBF_example
'''
03/24 Deliverable
Safe navigation of robot with obstacles, (get familiar with signed distance functions)
 - implement learning cbf
 - compare performance with Azra's weighted acq funciton, gibo

What am I trying to do?
 - See if learning a NOISY control bar using a GP and GIBO provides a betteer

 1) spawn robot <- Done
 2) spawn obstacle <- Done
 3) make signed distance function <- Done
 4) traditional navigation to waypoint
 5) GP Learning based navigation to waypoint
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
def cbf(alpha1, alpha2, X, closest_idx, h, circle_obstacle, Kp, V, waypoint):
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
    Kp = 4
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

    # -- Position --
    robot_pos = []
    noisy_robot_pos = []
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

    # -- Simulation Loop --
    while(t < end_time):
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

        # -- Creating Training Data --
        gp_obstacle_data = torch.tensor(visible_pts) # (N, 2)
        pos_x = torch.tensor([X_gp_noisy.x, X_gp_noisy.y]).reshape(shape=[1,2]) # (N,2)
        train_x = torch.cat([gp_obstacle_data, pos_x], dim=0) # (N+1, 2)
        surface_y = torch.zeros(visible_pts.shape[0], 1) # (N,1) <-- surface of SDF is marked as zeros
        dist_to_obj = torch.min(torch.norm(gp_obstacle_data - pos_x, dim=1)).reshape(shape=[1,1]) # (1,1)
        train_y = torch.cat([surface_y, dist_to_obj], dim=0) # (N+1, 1)

        rbf_gaussian_likelihood = gp.likelihoods.GaussianLikelihood()
        GP_model = RBF_example(train_x=torch.tensor(train_x), train_y=torch.tensor(train_y), likelihood=rbf_gaussian_likelihood)
        rbf_mll = gp.mlls.ExactMarginalLogLikelihood(rbf_gaussian_likelihood, GP_model) # finds probability of the function found by GP by comparing to sampled data.
        rbf_optimizer = torch.optim.Adam(
                list(GP_model.parameters()) + list(rbf_gaussian_likelihood.parameters()),
                lr=0.05
            )
        
        if t == 0.0:
            n_samples = 200
        else:
            n_samples = 10
        
        for i in range(n_samples):
            # -- Provides the posterior GP
            rbf_optimizer.zero_grad() # needed to feed the rbf_model only the CURRENTLY accumulated gradient
            rbf_output = GP_model(torch.tensor(train_x))
            rbf_loss = -rbf_mll(rbf_output, torch.tensor(train_y)).sum()
            rbf_loss.backward()
            rbf_optimizer.step()

        GP_model.eval()
        rbf_gaussian_likelihood.eval()
        with torch.enable_grad(), gp.settings.fast_pred_var(): 
            # Adjust test_x throughout the range and create a prediction Y_pred_rbf
            x_fwd = X_gp_noisy.x
            y_fwd = X_gp_noisy.y
            wander_range = 0.4
            x_range = torch.linspace(start=x_fwd-wander_range, end=x_fwd+wander_range, steps=8)
            y_range = torch.linspace(start=y_fwd-wander_range, end=y_fwd+wander_range, steps=8)

            test_x = torch.tensor([x_range, y_range]) # (M, 2)
            breakpoint()

            # --  Matern 5/2 Gradient --
            K_xstar_x = GP_model.covar_module(test_x,training_x).evaluate().detach()
            l = GP_model.covar_module.base_kernel.lengthscale.detach()
            diff_r  = (test_x.unsqueeze(1) - training_x.unsqueeze(0))
            print(f"diff_r {diff_r.size()}")
            r = torch.abs(diff_r) 
            sigma2 = GP_model.covar_module.outputscale.detach()
            rt5 = math.sqrt(5)
            # -- Grad K --
            grad_K_xstar_x = -sigma2 * (5.0 * diff_r/(3*l**2)) * (1 + rt5*r/l) * torch.exp(-rt5*r/l)

            # -- Noisy K --
            K_xx = GP_model.covar_module(training_x,training_x).evaluate().detach()
            obs_noise = rbf_gaussian_likelihood.noise.detach()
            K_xx_noisy = K_xx + obs_noise*torch.eye(len(training_x)) # K + variance*I
            print(f"K_xx_noisy {K_xx_noisy.size()}\n")

            # alpha = K^-1 * y
            alpha = torch.linalg.solve(K_xx_noisy, training_y) 
            grad_mean = grad_K_xstar_x @ alpha # [grad_K][alpha]
            # Finding d2K(x*, x*)
            r_star = torch.abs(test_x.unsqueeze(1) - test_x.unsqueeze(0))
            d2K_dxstar_xstar = -(5/(3*l**2))*sigma2 * (1 + rt5*r_star/l - 5*r_star**2/l**2) * torch.exp(-rt5*r_star/l)
            grad2_K_xstar_xstar = - d2K_dxstar_xstar 
            middle_term =torch.linalg.solve(K_xx_noisy, grad_K_xstar_x.T) # K^-1 ^ grad_K(x*,x)
            cov_grad_K = grad2_K_xstar_xstar -grad_K_xstar_x @ middle_term
            
        # -- GIBO Acquisition function --
        diff_r_old = theta_t.unsqueeze(1) - training_x.unsqueeze(0)
        r_old = torch.abs(diff_r_old)
        grad2_K_tt = -(5*sigma2 /(3*l**2)) # theta_t - theta_t = 0 -> simplified gradient
        grad_K_tx_old = -sigma2 * (5.0 * diff_r_old/(3*l**2)) * (1 + rt5*r_old/l) * torch.exp(-rt5*r_old/l)
        K_xx_inv_K_tx = torch.linalg.solve(K_xx_noisy, grad_K_tx_old.T)
        cov_old = grad2_K_tt - grad_K_tx_old @ K_xx_inv_K_tx
        best_acq_value = -torch.inf
        next_qp = None
        # -- (7) For m = 1, ... M -- 
        for i, query_point in enumerate(test_x): # using the test data as the points to sample.   
            training_x_new = torch.cat((training_x, query_point.unsqueeze(0)), dim=0) # append the query point to the end
            K_xx_new = GP_model.covar_module(training_x_new, training_x_new).evaluate().detach()

            # -- grad2 K(theta_t, theta_t)
            grad2_K_tt = -(5*sigma2 /(3*l**2)) # theta_t - theta_t = 0 -> simplified gradient

            # -- grad1 K (theta_t, X) --
            diff_r_new = theta_t.unsqueeze(1) - training_x_new.unsqueeze(0) # ??
            r_new = torch.abs(diff_r_new)

            K_xx_noisy_new = K_xx_new + obs_noise*torch.eye(len(training_x_new))
            grad_K_tx_new  = -sigma2 * (5.0 * diff_r_new/(3*l**2)) * (1 + rt5*r_new/l) * torch.exp(-rt5*r_new/l)

            K_inv_Ktx = torch.linalg.solve(K_xx_noisy_new,grad_K_tx_new.T)
            cov_new = grad2_K_tt - grad_K_tx_new @ K_inv_Ktx
            tr_cov_new = torch.trace(cov_new)
            tr_cov_old = torch.trace(cov_old)
            acq_function = tr_cov_old - tr_cov_new
            # -- (8) get query point "theta_t" --
            if acq_function > best_acq_value: 
                # Extract the max acquisition function value + store the kernel/ grad_kernel values
                best_acq_value = acq_function
                next_qp = query_point
        # -- (9) Sample Noisy Objective Function -- 
        # -- (10) Extend Data Set --
        # -- (12) End for loop
        training_x = torch.cat((training_x, next_qp.unsqueeze(0)), dim=0)
        new_noise = torch.tensor(rng.normal(0.0, 0.5), dtype=training_x.dtype)
        y_gp = torch.sin(math.pi*next_qp) + next_qp + new_noise
        training_y = torch.cat((training_y, y_gp.unsqueeze(0)), dim=0)
        # -- Updata theta_t at n+1 guess --
        K_xx = GP_model.covar_module(training_x, training_x).evaluate().detach()
        K_xx_noisy = K_xx + obs_noise * torch.eye(len(training_x))
        diff_r_theta = theta_t.unsqueeze(1) - training_x.unsqueeze(0)
        r_theta = torch.abs(diff_r_theta)
        grad_K_theta_x = -sigma2 * (5.0 * diff_r_theta/(3*l**2)) * (1 + rt5*r_theta/l) * torch.exp(-rt5*r_theta/l)
        # -- (11) Update the posterior probability distribution of ∇θJ.  (Done implicitly??)-- 
        alpha = torch.linalg.solve(K_xx_noisy, training_y)
        grad_mu_theta_t = grad_K_theta_x @ alpha
        theta_t = torch.clamp((theta_t + step_size * grad_mu_theta_t.squeeze()), 
                              min=start_x, 
                              max=end_x).detach()
        # -- move sample forward -- 
        print(f"Point: {samples_fwd} theta_(t + 1): {theta_t}")      
        samples_fwd = samples_fwd + 1
        
        # print(f"h:{h} , h_noisy: {h_noisy}")

        # -- Find u_nom with CBF -- 
        alpha1 = 1.0
        alpha2 = 1.5
        u_final = cbf(alpha1, alpha2, X, closest_idx, h, circle_obstacle, Kp, V, waypoint)
        noisy_u_final = cbf(alpha1, alpha2, X_noisy, closest_noisy_idx, h_noisy, circle_obstacle, Kp, V, waypoint)
        u_data.append(u_final)
        noisy_u_data.append(noisy_u_final)

        # -- Update Position w/ RK4 --
        X = rk4_step(X, V, t, dt, u_final)
        X_noisy = rk4_step(X_noisy, V, t, dt, noisy_u_final)
        print(f"Current u (no noise): {u_final}")
        print(f"Current u (noisy): {noisy_u_final}")
        robot_pos.append([X.x, X.y])
        noisy_robot_pos.append([X_noisy.x, X_noisy.y])
        # -- Check if Waypoint Reached --
        dx = abs(waypoint[0] - X.x)
        dy = abs(waypoint[1] - X.y)
        dx_noisy = abs(waypoint[0] - X_noisy.x)
        dy_noisy = abs(waypoint[1] - X_noisy.y)
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
        if goal1_reached and goal2_reached:
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