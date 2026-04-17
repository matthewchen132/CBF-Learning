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

from core.simulation import Simulation
'''

-- New Gibo --
1) State to X Y theta
2) Verify Lie derivative math
3) SE Kernel

'''


def main():
    # i. Initialize Sim
    sim = Simulation(dt=0.1, end_time=20.0)

    # ii. Initialize Objects
    waypoint = np.array([5.0, 5.0])
    X_gp_noisy = State(0.0, 0.0, 0.0) # GP-GIBO-based CBF and navigation | Trace Covariance Selection
    V = 1.0 # Initial velocity
    max_ang_vel = hf.actuator_limit(4.0) # 4 rad/s actuator limiting.
    Kp = 4.0
    alpha1 = 1.0
    noisy_gp_circle_obstacle = hf.spawn_circle(3.5, 3.5, r=1.6)
    gp_reached = False

    '''
    1) Setup GIBO Hyperparameters:
        a. stepsize η, 
        b. hyperpriors for GP hyperparameters, 
        c. number of iterations N 
        d. number of samples for a gradient estimate M
    '''
    # a:
    step_size = 0.1
    # b:
    
    # c: N = 201, 0 to 20 seconds with 0.1 dt.
    rng = np.random.default_rng(seed=2)
    noise_std_dev = 0.05
    next_query_point = torch.tensor([[0.2, 0.2]], dtype=float) #  <- set initially to a step_size towards the way_point
    n_std_deviations = 1.0              # h_mean - n_std_deviations * h_std

    while sim.is_running():
        noise = rng.normal(0, noise_std_dev, size=noisy_gp_circle_obstacle.size)
        
        # -- Construct Training Data for GIBO / GP -- 
        closest_gp_idx = hf.sdf(X_gp_noisy.x, X_gp_noisy.y, noisy_gp_circle_obstacle, Noisy=True, noise=noise)[1]
        visible_pts = hf.return_visible_pts(noisy_gp_circle_obstacle, closest_gp_idx)
        injected_noise = rng.normal(0, noise_std_dev, size=visible_pts.shape)
        noisy_visible_pts = visible_pts + injected_noise
        gp_obstacle_data = torch.tensor(noisy_visible_pts) # (N, 2)

        # -- GP Training Data --
        train_x, train_y = generate_gp_training_data(gp_obstacle_data, X_gp_noisy, visible_pts)

        # -- Set up GP Model -- 
        m52_gaussian_likelihood = gp.likelihoods.GaussianLikelihood()
        GP_model = m52_example(train_x=torch.tensor(train_x), train_y=torch.tensor(train_y), likelihood=m52_gaussian_likelihood)
        m52_mll = gp.mlls.ExactMarginalLogLikelihood(m52_gaussian_likelihood, GP_model) # finds probability of the function found by GP by comparing to sampled data.
        m52_optimizer = torch.optim.Adam(
                list(GP_model.parameters()) + list(m52_gaussian_likelihood.parameters()),
                lr=0.05
            )
        if sim.curr_time == 0.0:
            train_GP(n=200, GP_optimizer=m52_optimizer, GP_model=GP_model, GP_mll=m52_mll, train_x=train_x, train_y=train_y)
        else:
            train_GP(n=50, GP_optimizer=m52_optimizer, GP_model=GP_model, GP_mll=m52_mll, train_x=train_x, train_y=train_y)

        # Set our GP and likelihood into evaluation mode (training done) 
        GP_model.eval()
        m52_gaussian_likelihood.eval()

        test_x = torch.tensor([X_gp_noisy.x, X_gp_noisy.y]).reshape(1, 2) # (1, 3)
        sigma2 = GP_model.covar_module.outputscale.detach()
        l = GP_model.covar_module.base_kernel.lengthscale.detach()
        rt5 = math.sqrt(5)
        obs_noise = m52_gaussian_likelihood.noise.detach() 

        # -- Acquisition Function using Posterior Variances --
        next_query_point, best_acq_value = acq_func_posterior_covariance(X=X_gp_noisy, GP_model=GP_model, 
                                    GP_likelihood=m52_gaussian_likelihood, prior_covariance=sigma2)

        # -- Azra's Acquisition Function -- 
        # next_query_point, best_acq_value = azra_acq_function(X_gp_noisy, V, train_x, GP_model, m52_gaussian_likelihood, sigma2, 
        #                                                      sigma2, l, obs_noise, next_query_point, weight=1.0)

        u_query_point = hf.optimal_control(X_gp_noisy, Kp, next_query_point.flatten().tolist())
        next_query_point = (next_query_point[0].tolist())

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
        h_safe = probabilistic_sdf(h_mean, h_variance, n_std_deviations)

        # -- Gradients from GP --
        grad_h, grad_h_covariance = sdf_gp_gradient(GP_model, m52_gaussian_likelihood, test_x.float(), train_x.float(), train_y.float())
        grad_h = torch.tensor(grad_h, dtype=torch.float32)

        # -- CBF Condition --
        cos_t = torch.cos(torch.tensor(X_gp_noisy.theta))
        sin_t = torch.sin(torch.tensor(X_gp_noisy.theta))
        vel_xy = torch.tensor([V*cos_t, V*sin_t], dtype=torch.float32)
        f_x = vel_xy
        # NOTE:
        Lf_h = grad_h @ f_x
        # -- Recover the effect of Angular Velocity (u) -- 
        LgLf_h = V * (-grad_h[0] * sin_t + grad_h[1] * cos_t) # d(Lf_h)d_theta -> Chain rule gets us theta_dot = angular velocity

        # -- GP-GIBO-CBF -- (Adaptive Weight)
        u_gp_nom = hf.optimal_control(X_gp_noisy, Kp=Kp, goal_xy=waypoint)
        r, r_norm = hf.return_dist_vec(X_gp_noisy, waypoint)
        k_dist = 1. # Tuning parameter for weight
        weight = 1.0 - np.exp(-r_norm * k_dist) 
        u_blended_nom = (1-weight) * u_gp_nom + weight * u_query_point
        u_gp_cbf = hf.solve_cbf_qp(u_blended_nom, Lf_h.item(), LgLf_h.item(), h_safe.item(), alpha1, max_ang_vel) # u_safe
        # TODO: solve_SOCP is WIP
        # u_gp_cbf = hf.solve_SOCP(u_blended_nom, Lf_h.item(), LgLf_h.item(), h_safe.item(), h_variance.item(), f_x, alpha1, max_ang_vel) # u_safe


        # -- Append Logging Data --
        dist = np.linalg.norm([X_gp_noisy.x - waypoint[0], X_gp_noisy.y - waypoint[1]])
        sim.log_data(X=X_gp_noisy, u=u_gp_cbf, dist=dist, time=sim.curr_time, query_point=next_query_point)

        # -- Update Position w/ RK4 --
        X_gp_noisy = rk4_step(X_gp_noisy, V, sim.curr_time , sim.dt, u_gp_cbf)
        sim.step()
        print(f"u (GIBO / GP): {u_gp_cbf} | Weight: {round(weight,2)}")

        # -- Check if Waypoint Reached -- (
        r_gp, r_gp_norm = hf.return_dist_vec(X=X_gp_noisy, waypoint=waypoint)
        [gp_time, gp_reached] = hf.check_if_reached_waypoint(X_gp_noisy, r_norm=r_gp_norm, waypoint=waypoint, sim_time=sim.curr_time, reached=gp_reached)
     
        # -- End Sim if Goals Reached --
        if gp_reached: 
            break

    print(f"GP-CBF finish time: {round(sim.curr_time,2)}")

    # -- PLOTTING -- 
    logs = sim.list_to_np()
    fig, axes = plt.subplots(3, 1, figsize=(10, 12)) 
    breakpoint()
    plots.plot_trajectory(axes, noisy_gp_circle_obstacle, waypoint, logs["state"], logs["query_points"], 
                              fig_num=0, label=f"GP-GIBO-CBF with {n_std_deviations} Deviations") # Plots Trajectory and obstacle
    plots.plot_distance_error(axes, logs["times"], logs["dist"], 
                              fig_num=1, label="GP distance Error")
    plots.plot_control(axes, logs["u"], logs["times"], 
                              fig_num=2, label="GIBO-control") 
    plt.tight_layout() # Prevents label overlap
    plt.show()

if __name__ == "__main__":
    main()
    print("End Function")
