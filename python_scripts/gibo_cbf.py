import torch
import gpytorch as gp
import numpy as np
import matplotlib.pyplot as plt
import math
import cvxpy as cp


from helpers.gp_simple import m52_example
from helpers.gp_helpers import sdf_gp_gradient, GIBO, generate_gp_training_data, train_GP, probabilistic_sdf
import helpers.helper_functions as hf
import plotting.plots as plots

'''
Learning based waypoint navigation with Gaussian Processes 
and intelligent point selection via acquisition function.

 -- General Workflow of this Script --
1) Train the GP on the 160 LiDAR points (0 distance) + 1 Robot point (current distance).
2) Predict at the Robot's point to get h_mean and h_std.
3) Calculate the Gradient.
4) Plug h_safe and grad_h into the CBF.

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

def rk4_step(X, V, t, dt, u):
    '''
    Runge Kutta 4 Numerical Integration
    
    '''
    k1 = dXdt(t, X, V, u)
    k2 = dXdt(t + 0.5*dt, X + k1*0.5*dt, V, u)
    k3 = dXdt(t + 0.5*dt, X + k2*0.5*dt, V, u)
    k4 = dXdt(t + dt, X + k3*dt, V, u)
    X = X + (k1 + k2*2 + k3*2 + k4) * (dt/6.0)
    return X;

def main():
    # - Setup Times -
    t = 0.0
    times = []
    dt = 0.1
    end_time = 20.0

    # - Tracking Arrival at Goal
    ideal_reached = False
    noisy_reached = False
    gp_reached = False

    # -- Create goal --
    waypoint = np.array([5.0, 5.0])

    # -- Spawn Robots --
    [x_i, y_i, theta_i] = [0.0, 0.0, 0.0] # State [x, y, theta]
    V = 1.0 # Initial velocity
    max_ang_vel = hf.actuator_limit(4.0) # 4 rad/s actuator limiting.
    Kp = 4.0

    # -- Constants for CBF inequality and definitiion --
    alpha1 = 1.0
    alpha2 = 1.5

    X = State(x_i, y_i, theta_i) # Noiseless conventional CBF navigation
    X_noisy = State(x_i, y_i, theta_i) # Noisy conventional CBF navigation
    X_gp_noisy = State(x_i, y_i, theta_i) # GP-GIBO-based CBF and navigation

    # -- Spawn Obstacles (Circular) --
    circle_obstacle = hf.spawn_circle(3.5, 3.5, r=1.6)
    noisy_circle_obstacle = hf.spawn_circle(3.5, 3.5, r=1.6)
    noisy_gp_circle_obstacle = hf.spawn_circle(3.5, 3.5, r=1.6)

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

    rng = np.random.default_rng(seed=2)
    noise_std_dev = 0.05

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
        h, closest_idx = hf.sdf(X.x, X.y, circle_obstacle, Noisy=False, noise=noise)
        h_noisy, closest_noisy_idx = hf.sdf(X_noisy.x, X_noisy.y, noisy_circle_obstacle, Noisy=True, noise=noise) 
        
        # -- Construct Training Data for GIBO / GP -- 
        closest_gp_idx = hf.sdf(X_gp_noisy.x, X_gp_noisy.y, noisy_gp_circle_obstacle, Noisy=True, noise=noise)[1]
        visible_pts = hf.return_visible_pts(noisy_gp_circle_obstacle, closest_gp_idx)
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
        if t == 0.0:
            train_GP(n=200, GP_optimizer=m52_optimizer, GP_model=GP_model, GP_mll=m52_mll, train_x=train_x, train_y=train_y)
        else:
            train_GP(n=50, GP_optimizer=m52_optimizer, GP_model=GP_model, GP_mll=m52_mll, train_x=train_x, train_y=train_y)


        # if t == 0.0:
        #     n_samples = 200
        # else:
        #     n_samples = 50

        # # -- Train GP -- 
        # for i in range(n_samples):
        #     # -- Provides the posterior GP -- 
        #     m52_optimizer.zero_grad() # needed to feed the m52_model only the CURRENTLY accumulated gradient
        #     m52_output = GP_model(torch.tensor(train_x))
        #     m52_loss = -m52_mll(m52_output, torch.tensor(train_y)).sum()
        #     m52_loss.backward()
        #     m52_optimizer.step()

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
            grid_x, grid_y = hf.make_gridspace(X_gp_noisy, wander_range, num_steps=8)
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

        u_query_point = hf.optimal_control(X_gp_noisy, Kp, next_qp.flatten().tolist())
        query_points.append(next_qp[0].tolist())

        # -- Extend Data Set to include next query point --
        train_x, train_y = hf.filter_relevant_data(X_gp_noisy, train_x, train_y, max_pts=160)

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

        # -- Extract learned SDF  as h_safe) -- 
        posterior = GP_model(torch.tensor([X_gp_noisy.x, X_gp_noisy.y], dtype=torch.float32).reshape(1,2))     
        h_mean = posterior.mean             # SDF estimate
        h_variance  = posterior.variance         # SDF variance
        n_std_deviations = 1.0              # h_mean - n_std_deviations * h_std
        h_safe = probabilistic_sdf(h_mean, h_variance, n_std_deviations)

        grad_h, grad_h_covariance = sdf_gp_gradient(GP_model, m52_gaussian_likelihood, test_x.float(), train_x.float(), train_y.float())
        grad_h = torch.tensor(grad_h, dtype=torch.float32)


        # -- CBF Condition --
        cos_t = torch.cos(torch.tensor(X_gp_noisy.theta))
        sin_t = torch.sin(torch.tensor(X_gp_noisy.theta))
        vel_xy = torch.tensor([V*cos_t, V*sin_t], dtype=torch.float32)
        Lf_h = grad_h @ vel_xy
        Lg_h = V * (-grad_h[0] * sin_t + grad_h[1] * cos_t) # grad_h * dVdtheta

        # -- GP-GIBO-CBF -- 
        u_gp_nom = hf.optimal_control(X_gp_noisy, Kp=Kp, goal_xy=waypoint)
        r, r_norm = hf.return_dist_vec(X_gp_noisy, waypoint)
        k_dist = 1. # Tuning parameter
        weight = 1.0 - np.exp(-r_norm * k_dist) 
        u_blended_nom = (1-weight) * u_gp_nom + weight * u_query_point
        u_gp_cbf = hf.solve_cbf_qp(u_blended_nom, Lf_h.item(), Lg_h.item(), h_safe.item(), alpha1, max_ang_vel)

        # -- Take the maximum of GP-defined control and the optimal control input "u_gp_optimal" -- 
        # -- Find u_nom with CBF -- (Conventional)
        u_final = hf.conventional_cbf(alpha1, alpha2, X, closest_idx, h, circle_obstacle, Kp, V, waypoint)
        u_noisy_final = hf.conventional_cbf(alpha1, alpha2, X_noisy, closest_noisy_idx, h_noisy, circle_obstacle, Kp, V, waypoint)
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

        # -- Check if Waypoint Reached -- (
        r, r_norm = hf.return_dist_vec(X=X, waypoint=waypoint)
        r_noisy, r_noisy_norm  = hf.return_dist_vec(X=X_noisy, waypoint=waypoint)
        r_gp, r_gp_norm = hf.return_dist_vec(X=X_gp_noisy, waypoint=waypoint)

        [ideal_time, ideal_reached] = hf.check_if_reached_waypoint(X, r_norm=r_norm, waypoint=waypoint, sim_time=t, reached=ideal_reached)
        [noisy_time, noisy_reached] = hf.check_if_reached_waypoint(X_noisy, r_norm=r_noisy_norm, waypoint=waypoint, sim_time=t, reached=noisy_reached)
        [gp_time, gp_reached] = hf.check_if_reached_waypoint(X_gp_noisy, r_norm=r_gp_norm, waypoint=waypoint, sim_time=t, reached=gp_reached)
     
        # -- End Simulation if ALL Goals Reached --
        if ideal_reached and noisy_reached and gp_reached: 
            break

        # -- Increase Sim. Time Every Loop --
        t += dt
    
    print(f"Zero-noise finish time: {ideal_time}, Noisy CBF finish time: {noisy_time}, GP-CBF finish time: {gp_time}")

    robot_pos = np.array(robot_pos)
    noisy_robot_pos = np.array(noisy_robot_pos)
    gp_robot_pos = np.array(gp_robot_pos)
    query_points = np.array(query_points)
    times = np.array(times)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12)) 

    # -- Plot #1: Trajectories-- 
    plots.plot_trajectories(axes, circle_obstacle, waypoint, robot_pos, noisy_robot_pos, gp_robot_pos, n_std_deviations, query_points)

    # -- Plot #2: Distance Error Norm vs Time -- 
    plots.plot_error_wrt_time(axes, times, dist_data, dist_noisy_data, dist_gp_data)

    # -- Control Effort (u) --
    plots.plot_control_wrt_time(axes, u_data, u_noisy_data, u_gp_data, times)

    plt.tight_layout() # Prevents label overlap
    plt.show()
    print("End Function")

if __name__ == "__main__":
    main()