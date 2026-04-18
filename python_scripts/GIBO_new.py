import torch
import gpytorch as gp
from gpytorch.priors import GammaPrior
import numpy as np
import matplotlib.pyplot as plt
import math
import cvxpy as cp

from dynamics.state import State
from dynamics.rk4_integrator import rk4_step
from helpers.gp_simple import RBF_example
from helpers.gp_helpers import SE_acq_func_grad_covariance, azra_acq_function, acq_func_posterior_covariance
import helpers.helper_functions as hf
import plotting.plots as plots

from core.simulation import Simulation
from core.control import GIBO_control, Noise
'''

-- New Gibo --
1) State to X Y theta
2) Verify Lie derivative math
3) SE Kernel
'''


'''
Questions for Azra:
 - Should I train off of X,Y or X,Y, theta?
   -> As we talked about, with a signed distance, there is not a clear correlation between theta and distance.
   -> for now, I am using X,Y because it simplifies the logic, improves performance, and eliminates any unclear coupling between theta and signed distance.
 - GIBO Line 6: 
'''

def main():
    # i. Initialize Sim and Objects
    sim = Simulation(dt=0.1, end_time=20.0)
    GIBO_Controller = GIBO_control(V=1.0, Kp=4.0, alpha1 = 1.0)
    noise = Noise(GIBO_Controller.rng, noise_std_dev=0.05, num_std_deviations=1.0)

    # -- GIBO Line 1: Setup GIBO Hyperparameters -- 
    # a. stepsize η
    step_size = 0.1
    # b. hyperpriors for GP hyperparameters 
    RBF_gaussian_likelihood = gp.likelihoods.GaussianLikelihood()
    lengthscale_prior = GammaPrior(3.0, 1) # (a,b) : lengthscale = a/b
    outputscale_prior = GammaPrior(4.0, .17) # (a,b) : outputscale = a/b
    GP_model = RBF_example(train_x=[], train_y=[], likelihood=RBF_gaussian_likelihood, l_prior=lengthscale_prior, output_prior=outputscale_prior)
    # c. number of iterations N = 201 (20 seconds, 0.1s timestep)
    # d. number of samples for a gradient estimate M = 

    GP_model.eval()
    RBF_gaussian_likelihood.eval()

    # -- GIBO Line 2: Set a theta_initial (next_query_point) and emtpy dataset D = {} -- 
    GIBO_Controller.D = {"X": [], "Y" : []}
    next_query_point = torch.tensor([[0.2, 0.2]], dtype=float) #

    # -- GIBO Line 3: for t = 0, ..., N do --
    while sim.is_running():
        # -- GIBO Line 4: Sample Noisy Objective Function | sdf = J(X_t) + ϵt --
        signed_distance, _ = GIBO_Controller.sdf(X=GIBO_Controller.X, Noise=GIBO_Controller.Noise)

        # -- GIBO Line 5: Extend Dataset D (NOTE: Using X,Y over X,Y, theta.  See comment above.)
        GIBO_Controller.D["X"].append([GIBO_Controller.X.x, GIBO_Controller.X.y])
        GIBO_Controller.D["Y"].append([signed_distance])
        train_x = torch.tensor(GIBO_Controller.D["X"])
        train_y = torch.tensor(GIBO_Controller.D["Y"])

        # -- GIBO Line 6: Construct Training Data for GIBO / GP (Leave Out?) --
        GP_model.set_train_data(inputs=train_x, 
                                targets=train_y,
                                strict=False)
        sigma2 = GP_model.covar_module.outputscale.detach()
        l = GP_model.covar_module.base_kernel.lengthscale.detach()
        obs_noise = RBF_gaussian_likelihood.noise.detach() 

        # TODO: Maybe readjust lengthscale and outputscale on the first few points??
        # RBF_mll = gp.mlls.ExactMarginalLogLikelihood(RBF_gaussian_likelihood, GP_model) # finds probability of the function found by GP by comparing to sampled data.
        # RBF_optimizer = torch.optim.Adam(
        #         list(GP_model.parameters()) + list(RBF_gaussian_likelihood.parameters()),
        #         lr=0.05
        #     )
        # train_GP(n=200, GP_optimizer=RBF_optimizer, GP_model=GP_model, GP_mll=RBF_mll, train_x=GIBO_Controller.D["X"], train_y=GIBO_Controller.D["X"])

        # -- Step 7: For m = 1,2, ... M, -> 2nd Loop is baked into "SE_acq_func_grad_covariance"
        # -> SLACK me if this doesn't make sense.
        M = GIBO_Controller.return_gridspace(GIBO_Controller.X)

        # -- Step 8: Get query point = argmax(acq_function)  
        next_query_point, _ = GIBO_Controller.SE_acq_func_grad_covariance(X=GIBO_Controller.X, V=GIBO_Controller.V, 
                                                                          train_x=train_x, train_y=train_y, 
                                                                        GP_model=GP_model, GP_likelihood=RBF_gaussian_likelihood, 
                                                                        waypoint=GIBO_Controller.waypoint, sigma2=sigma2, l=l, 
                                                                        obs_noise=obs_noise, next_query_point=next_query_point)

        # -- Step 9: Sample Noisy Objective Funmction sdf = J(query_point) + noise
        sdf_GIBO_point = GIBO_Controller.sdf(X=State(next_query_point[0], next_query_point[1], 0.0), Noise=GIBO_Controller.Noise)
        # - 9i: Corresponding control input to get to query point. -
        u_query_point = hf.optimal_control(GIBO_Controller.X, GIBO_Controller.Kp, next_query_point.flatten().tolist())

        # -- Step 10: Extend Data Set to include next query point --
        GIBO_Controller.D["X"].append(next_query_point)
        GIBO_Controller.D["Y"].append(sdf_GIBO_point)

        # -- Step 11: Update the posterior probability distribution of ∇θJ.

        # Step 12: End Inner For Loop
    # Step 13: Gradient ascent, or any other gradient based optimizer. X_t+1 = X_t + η·E ∇_X J X=Xt
    next_query_point = next_query_point + step_size * grad_h

    #     # -- Updata next_query_point --
    #     K_xx = GP_model.covar_module(train_x, train_x).evaluate().detach()
    #     K_xx_noisy = K_xx + obs_noise * torch.eye(len(train_x))
        
    #     diff_r_theta = next_query_point.unsqueeze(1) - train_x.unsqueeze(0)
    #     r_theta = torch.norm(diff_r_theta, dim=-1, keepdim=True)
    #     # grad_K_theta_x = -sigma2 * (5.0 * diff_r_theta/(3*l**2)) * (1 + rt5*r_theta/l) * torch.exp(-rt5*r_theta/l) <- Matern5/2
    #     K_x_qp = GP_model.covar_module(train_x, next_query_point).evaluate().detach()
    #     grad_K_theta_x = -(1/l**2) * K_x_qp * diff_r_theta
    #     grad_K_theta_x = grad_K_theta_x.squeeze(0) # (162, 2)

    #     # -- Update the posterior probability distribution of ∇θJ. -- 
    #     alpha = torch.linalg.solve(K_xx_noisy, train_y)
    #     grad_mean_next_query_point = grad_K_theta_x.T @ alpha        
        
    #     # -- move sample forward -- 
    #     update = (step_size * grad_mean_next_query_point).reshape(1, 2)
    #     next_query_point = (next_query_point.reshape(1, 2) + update).detach()   
    #     posterior = GP_model(torch.tensor([GIBO_Controller.X.x, GIBO_Controller.X.y], dtype=torch.float32).reshape(1,2))     
    #     h_mean = posterior.mean             # SDF estimate
    #     h_variance  = posterior.variance         # SDF variance
    #     h_safe = probabilistic_sdf(h_mean, h_variance, Noise.num_std_deviations)

    #     # -- Gradients from GP --
    #     grad_h, grad_h_covariance = sdf_gp_gradient(GP_model, RBF_gaussian_likelihood, test_x.float(), train_x.float(), train_y.float())
    #     grad_h = torch.tensor(grad_h, dtype=torch.float32)

    #     # -- CBF Condition --
    #     cos_t = torch.cos(torch.tensor(GIBO_Controller.X.theta))
    #     sin_t = torch.sin(torch.tensor(GIBO_Controller.X.theta))
    #     vel_xy = torch.tensor([GIBO_Controller.V*cos_t, GIBO_Controller.V*sin_t], dtype=torch.float32)
    #     f_x = vel_xy
    #     Lf_h = grad_h @ f_x
    #     # -- Recover the effect of Angular Velocity (u) -- 
    #     LgLf_h = GIBO_Controller.V * (-grad_h[0] * sin_t + grad_h[1] * cos_t) # d(Lf_h)d_theta -> Chain rule gets us theta_dot = angular velocity

    #     # -- GP-GIBO-CBF -- (Adaptive Weight)
    #     u_gp_nom = hf.optimal_control(GIBO_Controller.X, Kp=GIBO_Controller.Kp, goal_xy=GIBO_Controller.waypoint)
    #     r, r_norm = hf.return_dist_vec(GIBO_Controller.X, GIBO_Controller.waypoint)
    #     k_dist = 1. # Tuning parameter for weight
    #     weight = 1.0 - np.exp(-r_norm * k_dist) 
    #     u_blended_nom = (1-weight) * u_gp_nom + weight * u_query_point
    #     u_gp_cbf = hf.solve_cbf_qp(u_blended_nom, Lf_h.item(), LgLf_h.item(), h_safe.item(), GIBO_Controller.alpha1, GIBO_Controller.max_ang_vel) # u_safe

    #     # -- Append Logging Data --
    #     dist = np.linalg.norm([GIBO_Controller.X.x - GIBO_Controller.waypoint[0], GIBO_Controller.X.y - GIBO_Controller.waypoint[1]])
    #     sim.log_data(X=GIBO_Controller.X, u=u_gp_cbf, dist=dist, time=sim.curr_time, query_point=next_query_point)

    #     # -- Update Position w/ RK4 --
    #     GIBO_Controller.X = rk4_step(GIBO_Controller.X, GIBO_Controller.V, sim.curr_time , sim.dt, u_gp_cbf)
    #     sim.step()
    #     print(f"u (GIBO / GP): {u_gp_cbf} | Weight: {round(weight,2)}")

    #     # -- Check if Waypoint Reached -- (
    #     r_gp, r_gp_norm = hf.return_dist_vec(X=GIBO_Controller.X, waypoint=GIBO_Controller.waypoint)
    #     [gp_time, GIBO_Controller.goal_reached] = hf.check_if_reached_waypoint(GIBO_Controller.X, r_norm=r_gp_norm, waypoint=GIBO_Controller.waypoint, sim_time=sim.curr_time, reached=GIBO_Controller.goal_reached)
     
    #     # -- End Sim if Goals Reached --
    #     if GIBO_Controller.goal_reached: 
    #         break

    # # -- PLOTTING -- 
    # logs = sim.list_to_np()
    # fig, axes = plt.subplots(3, 1, figsize=(10, 12)) 
    # plots.plot_trajectory(axes, noisy_gp_circle_obstacle, GIBO_Controller.waypoint, logs["state"], logs["query_points"], 
    #                           fig_num=0, label=f"GP-GIBO-CBF with {Noise.num_std_deviations} Deviations") # Plots Trajectory and obstacle
    # plots.plot_distance_error(axes, logs["times"], logs["dist"], 
    #                           fig_num=1, label="GP distance Error")
    # plots.plot_control(axes, logs["u"], logs["times"], 
    #                           fig_num=2, label="GIBO-control") 
    # plt.tight_layout() # Prevents label overlap
    # plt.show()

if __name__ == "__main__":
    main()
    print("End Function")
