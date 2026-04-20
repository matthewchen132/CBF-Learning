import torch
torch.set_default_dtype(torch.float64)

import gpytorch as gp
from gpytorch.priors import GammaPrior
import numpy as np
import matplotlib.pyplot as plt
import math
import cvxpy as cp

from dynamics.state import State
from dynamics.augmented_state import LookaheadState
from dynamics.rk4_integrator import rk4_step, GIBO_rk4_step
from helpers.gp_simple import RBF_example
from helpers.gp_helpers import azra_acq_function, acq_func_posterior_covariance
import helpers.helper_functions as hf
import plotting.plots as plots

from core.simulation import Simulation
from core.control import GIBO_control, Noise

'''
GIBO Algorithm Lines 1-10 are roughly implemented, with a rough outline for lines 11-13. 

Questions for Azra:
 - Should I train off of X,Y or X,Y, theta?
   -> As we talked about, with a signed distance, there is not a clear correlation between theta and distance.
   -> for now, I am using X,Y because it simplifies the logic, improves performance, and eliminates any unclear coupling between theta and signed distance.

 - Should we do some form of pre-training or calibration before running GIBO on an empty dataset?
'''

def main():
    # i. Initialize Sim and Objects
    sim = Simulation(dt=0.1, end_time=20.0)
    GIBO = GIBO_control(V=1.0, Kp=4.0, alpha_cbf=1.0, M_number_of_queries=5, dt=sim.dt)
    noise = Noise(GIBO.rng, noise_std_dev=0.05, num_std_deviations=1.0)

    # -- GIBO Line 1: Setup GIBO Hyperparameters -- 
    # a. stepsize η
    step_size = 0.1
    # b. hyperpriors for GP hyperparameters 
    RBF_gaussian_likelihood = gp.likelihoods.GaussianLikelihood()
    lengthscale_prior = GammaPrior(3.0, 1) # (a,b) : lengthscale = a/b
    outputscale_prior = GammaPrior(4.0, .17) # (a,b) : outputscale = a/b
    GP_model = RBF_example(train_x=[], train_y=[], likelihood=RBF_gaussian_likelihood, l_prior=lengthscale_prior, output_prior=outputscale_prior)
    # c. number of iterations -> N = 201 (20 seconds, 0.1s timestep)
    # d. number of samples for gradient estimate -> M = 8x8 (Gridspace size)

    GP_model.eval()
    RBF_gaussian_likelihood.eval()

    # -- GIBO Line 2: Set a theta_initial (next_query_point) and emtpy dataset D = {} -- 
    GIBO.D = {"X": [], "Y" : []}
    next_query_point = torch.tensor([[0.2, 0.2]], dtype=float)

    # -- GIBO Line 3: for t = 0, ..., N do --
    while sim.is_running():
        # -- GIBO Line 4: Sample Noisy Objective Function | sdf = J(X_t) + ϵt --
        signed_distance, _ = GIBO.sdf(X=GIBO.X, Noise=GIBO.Noise)

        # -- GIBO Line 5: Extend Dataset D (NOTE: Using X,Y over X,Y, theta.  See comment above.)
        GIBO.D["X"].append([GIBO.X.xp, GIBO.X.yp])
        GIBO.D["Y"].append(signed_distance)
        train_x = torch.tensor(GIBO.D["X"]) # <- Slow, optimize
        train_y = torch.tensor(GIBO.D["Y"]) # <- Slow, optimize

        # -- GIBO Line 6: Construct Training Data for GIBO / GP (Leave Out?) --
        GP_model.set_train_data(inputs=train_x, 
                                targets=train_y,
                                strict=False)
        sigma2 = GP_model.covar_module.outputscale.detach()
        l = GP_model.covar_module.base_kernel.lengthscale.detach().squeeze(0)
        obs_noise = RBF_gaussian_likelihood.noise.detach() 

        # -- Step 7: For m = 1,2, ... M, -> 2nd Loop is baked into "SE_acq_func_grad_covariance"
        M = GIBO.return_gridspace()

        # -- Step 8: Get query point = argmax(acq_function)  
        for m in range(GIBO.M_number_of_queries):
            next_query_point = GIBO.acquisition_function(train_x=train_x, train_y=train_y, 
                                                    GP_model=GP_model, GP_likelihood=RBF_gaussian_likelihood, 
                                                    waypoint=GIBO.waypoint, sigma2=sigma2, l=l, 
                                                    obs_noise=obs_noise, next_query_point=next_query_point, M=M)

            # -- Step 9: Sample Noisy Objective Function sdf = J(query_point) + noise
            sdf_query_point, _ = GIBO.sdf(X=LookaheadState(next_query_point[0], next_query_point[1], 0.0, l=GIBO.V), Noise=GIBO.Noise)

            # - 9i: Generate the corresponding control input to get to query point. -
            u_GIBO = GIBO.optimal_control(next_query_point.flatten().tolist())

            # -- Step 10: Extend Data Set to include next query point --
            GIBO.D["X"].append(next_query_point.squeeze().tolist())
            GIBO.D["Y"].append(sdf_query_point)
            train_x = torch.tensor(GIBO.D["X"], dtype=torch.float64)
            train_y = torch.tensor(GIBO.D["Y"], dtype=torch.float64)

            # -- Step 11: Update the posterior of ∇θJ <-> (∇ Signed Distance)
            # E[∇J]= ∑​ w_i * ∇K(x,x) <- Expected value of gradient is the sum of weighted gradient_K_xx\
            
            weights = GIBO.compute_gradient_update_weights(train_x, train_y, GP_model, GP_model.likelihood).unsqueeze(-1) # (3,1)
            grad_K = GIBO.compute_grad_K(train_x, GP_model, lengthscale=l)
            expected_grad_SDF = torch.sum(weights * grad_K, dim=0).detach()

            GIBO.u_final = GIBO.solve_cbf_qp(u_GIBO=u_GIBO,h_safe=sdf_query_point, grad_h=expected_grad_SDF,
                                            alpha_cbf=GIBO.alpha_cbf, X=GIBO.X)
        
        breakpoint()
        GIBO.X = GIBO_rk4_step(GIBO.X, expected_grad_SDF, step_size, sim.dt)
        sim.step() # Steps time to t+1

    # Step 12: End Inner For Loop

    # Step 13: Gradient ascent, or any other gradient based optimizer. X_t+1 = X_t + η·E ∇_X J X=Xt









    # TODO: Will be converted or deleted next session -- 4/18

    # # -- PLOTTING -- 
    # logs = sim.list_to_np()
    # fig, axes = plt.subplots(3, 1, figsize=(10, 12)) 
    # plots.plot_trajectory(axes, noisy_gp_circle_obstacle, GIBO.waypoint, logs["state"], logs["query_points"], 
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
