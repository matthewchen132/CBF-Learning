import torch
import gpytorch as gp
import numpy as np
import matplotlib.pyplot as plt
import math
import cvxpy as cp

from dynamics.state import State
from dynamics.rk4_integrator import rk4_step
from helpers.gp_simple import m52_example
from helpers.gp_helpers import sdf_gp_gradient, acq_func_grad_covariance, azra_acq_function, acq_func_posterior_covariance, generate_gp_training_data, train_GP, probabilistic_sdf
import helpers.helper_functions as hf
import plotting.plots as plots

'''
Learning based waypoint navigation with Gaussian Processes 
and intelligent point selection via selected acquisition function.

'''

def main():
    # - Setup Times -
    t = 0.0
    times = []
    dt = 0.1
    end_time = 20.0

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
    X_gp_noisy = State(x_i, y_i, theta_i) # GP-GIBO-based CBF and navigation | Trace Covariance Selection
    X_gp_post_variance = State(x_i, y_i, theta_i) # GP-GIBO-based CBF and navigation | Posterior Covariance Selection

    # -- Spawn Obstacles (Circular) --
    circle_obstacle = hf.spawn_circle(3.5, 3.5, r=1.6)
    noisy_circle_obstacle = hf.spawn_circle(3.5, 3.5, r=1.6)
    noisy_gp_circle_obstacle = hf.spawn_circle(3.5, 3.5, r=1.6)

    # -- Position Logging --
    robot_pos = []
    noisy_robot_pos = []
    gp_robot_pos = []
    gp_post_variance_pos = []

    # -- Booleans to Check if Goal is Reached --
    ideal_reached = False
    noisy_reached = False
    gp_reached = False

    # -- More Logging --
    u_data = []
    u_noisy_data = []
    u_gp_data = []
    query_points = []
    dist_data = []
    dist_noisy_data = []
    dist_gp_data = []


    # -- Noise injected to measurement reading of Obstacle--
    rng = np.random.default_rng(seed=2)
    noise_std_dev = 0.05

    # -- Define first Query Point Before Running Acquisition Function --
    step_size = 0.1
    next_query_point = torch.tensor([[0.2, 0.2]], dtype=float) #  <- set initially to a step_size towards the way_point

    # -- Simulation Loop --
    while(t < end_time):

        # -- Logging --
        robot_pos.append([X.x, X.y])
        noisy_robot_pos.append([X_noisy.x, X_noisy.y])
        gp_robot_pos.append([X_gp_noisy.x, X_gp_noisy.y])
        noise = rng.normal(0, noise_std_dev, size=circle_obstacle.size)
        times.append(t)

        # -- Conventional CBF / Signed Distance (h, h_noisy) --
        h, closest_idx = hf.sdf(X.x, X.y, circle_obstacle, Noisy=False, noise=noise)
        h_noisy, closest_noisy_idx = hf.sdf(X_noisy.x, X_noisy.y, noisy_circle_obstacle, Noisy=True, noise=noise) 
        
        # -- Construct Training Data for GIBO / GP -- 
        closest_gp_idx = hf.sdf(X_gp_noisy.x, X_gp_noisy.y, noisy_gp_circle_obstacle, Noisy=True, noise=noise)[1]
        visible_pts = hf.return_visible_pts(noisy_gp_circle_obstacle, closest_gp_idx)
        noise_vector = rng.normal(0, noise_std_dev, size=visible_pts.shape)
        noisy_visible_pts = visible_pts + noise_vector
        gp_obstacle_data = torch.tensor(noisy_visible_pts) # (N, 2)

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


        # -- Set our GP and likelihood into evaluation mode (training done) --
        GP_model.eval()
        m52_gaussian_likelihood.eval()


        test_x = torch.tensor([X.x, X.y]).reshape(1, 2) # (1, 3)
        sigma2 = GP_model.covar_module.outputscale.detach()
        l = GP_model.covar_module.base_kernel.lengthscale.detach()
        rt5 = math.sqrt(5)
        obs_noise = m52_gaussian_likelihood.noise.detach() 

        # -- Acquisition Function using Gradient Variances --
        # next_query_point, best_acq_value = acq_func_grad_covariance(X=X_gp_noisy, V=V, train_x=train_x, train_y=train_y, 
        #                                    GP_model=GP_model, GP_likelihood=m52_gaussian_likelihood, 
        #                                    waypoint=waypoint, sigma2=sigma2, l=l, obs_noise=obs_noise, next_query_point=next_query_point)

        # -- Acquisition Function using Posterior Variances --
        next_query_point, best_acq_value = acq_func_posterior_covariance(X=X_gp_noisy, GP_model=GP_model, 
                                    GP_likelihood=m52_gaussian_likelihood, prior_covariance=sigma2)
        next_query_point, best_acq_value = azra_acq_function(X, V, train_x, GP_model, m52_gaussian_likelihood, sigma2, 
                                                             sigma2, l, obs_noise, next_query_point, weight=0.5)

        u_query_point = hf.optimal_control(X_gp_noisy, Kp, next_query_point.flatten().tolist())
        query_points.append(next_query_point[0].tolist())

        # -- Extend Data Set to include next query point --
        train_x, train_y = hf.filter_relevant_data(X_gp_noisy, train_x, train_y, max_pts=160)

        # -- Updata next_query_point --
        K_xx = GP_model.covar_module(train_x, train_x).evaluate().detach()
        K_xx_noisy = K_xx + obs_noise * torch.eye(len(train_x))
        diff_r_theta = next_query_point.unsqueeze(1) - train_x.unsqueeze(0)
        r_theta = torch.norm(diff_r_theta, dim=-1, keepdim=True)
        grad_K_theta_x = -sigma2 * (5.0 * diff_r_theta/(3*l**2)) * (1 + rt5*r_theta/l) * torch.exp(-rt5*r_theta/l)
        grad_K_theta_x = grad_K_theta_x.squeeze(0) # (162, 2)
        
        # -- Update the posterior probability distribution of ∇θJ. -- 
        alpha = torch.linalg.solve(K_xx_noisy, train_y)
        grad_mean_next_query_point = grad_K_theta_x.T @ alpha        
        # -- move sample forward -- 
        update = (step_size * grad_mean_next_query_point).reshape(1, 2)
        next_query_point = (next_query_point.reshape(1, 2) + update).detach()   

        # -- Extract learned SDF  as h_safe) -- 
        posterior = GP_model(torch.tensor([X_gp_noisy.x, X_gp_noisy.y], dtype=torch.float32).reshape(1,2))     
        h_mean = posterior.mean             # SDF estimate
        h_variance  = posterior.variance         # SDF variance
        n_std_deviations = 1.0              # h_mean - n_std_deviations * h_std
        h_safe = probabilistic_sdf(h_mean, h_variance, n_std_deviations)

        # -- Gradients from GP --
        grad_h, grad_h_covariance = sdf_gp_gradient(GP_model, m52_gaussian_likelihood, test_x.float(), train_x.float(), train_y.float())
        grad_h = torch.tensor(grad_h, dtype=torch.float32)

        # -- CBF Condition --
        cos_t = torch.cos(torch.tensor(X_gp_noisy.theta))
        sin_t = torch.sin(torch.tensor(X_gp_noisy.theta))
        vel_xy = torch.tensor([V*cos_t, V*sin_t], dtype=torch.float32)
        f_x = vel_xy
        Lf_h = grad_h @ f_x
        # -- Recover the effect of Angular Velocity (u) -- 
        LgLf_h = V * (-grad_h[0] * sin_t + grad_h[1] * cos_t) # d(Lf_h)d_theta -> Chain rule gets us theta_dot = angular velocity

        # -- GP-GIBO-CBF -- (Adaptive Weight)
        u_gp_nom = hf.optimal_control(X_gp_noisy, Kp=Kp, goal_xy=waypoint)
        r, r_norm = hf.return_dist_vec(X_gp_noisy, waypoint)
        k_dist = 1. # Tuning parameter for weight
        weight = 1.0 - np.exp(-r_norm * k_dist) 
        u_blended_nom = (1-weight) * u_gp_nom + weight * u_query_point
        # u_gp_cbf = hf.solve_cbf_qp(u_blended_nom, Lf_h.item(), LgLf_h.item(), h_safe.item(), alpha1, max_ang_vel) # u_safe
        # TODO: solve_SOCP is WIP
        u_gp_cbf = hf.solve_SOCP(u_blended_nom, Lf_h.item(), LgLf_h.item(), h_safe.item(), h_variance.item(), f_x, alpha1, max_ang_vel) # u_safe

        # -- Take the maximum of GP-defined control and the optimal control input "u_gp_optimal" -- 
        # -- Find u_nom with CBF -- (Conventional)
        u_final = hf.conventional_cbf(alpha1, alpha2, X, closest_idx, h, circle_obstacle, Kp, V, waypoint)
        u_noisy_final = hf.conventional_cbf(alpha1, alpha2, X_noisy, closest_noisy_idx, h_noisy, circle_obstacle, Kp, V, waypoint)

        # -- Append Logging Data --
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
     
        # -- End Sim if Goals Reached --
        if ideal_reached and noisy_reached and gp_reached: 
            break

        # -- Increase Sim. Time by "dt" --
        t += dt

    print(f"Zero-noise finish time: {round(ideal_time, 2)}, Noisy CBF finish time: {round(noisy_time)}, GP-CBF finish time: {round(gp_time,2)}")

    # - Sort Arrays for Logging - 
    robot_pos = np.array(robot_pos)
    noisy_robot_pos = np.array(noisy_robot_pos)
    gp_robot_pos = np.array(gp_robot_pos)
    query_points = np.array(query_points)
    times = np.array(times)

    # -- PLOTTING -- 
    fig, axes = plt.subplots(3, 1, figsize=(10, 12)) 
    plots.plot_trajectories(axes, circle_obstacle, waypoint, robot_pos, 
                            noisy_robot_pos, gp_robot_pos, n_std_deviations, query_points) # Plots Trajectory and obstacle
    plots.plot_error_wrt_time(axes, times, dist_data, dist_noisy_data, dist_gp_data) # Plots distance from goal w.r.t time
    plots.plot_control_wrt_time(axes, u_data, u_noisy_data, u_gp_data, times) # Plots control effort w.r.t time
    plt.tight_layout() # Prevents label overlap
    plt.show()

if __name__ == "__main__":
    main()
    print("End Function")
