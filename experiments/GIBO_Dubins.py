import torch
torch.set_default_dtype(torch.float64)

import gpytorch as gp
from gpytorch.priors import GammaPrior
import numpy as np
import matplotlib.pyplot as plt
import math
import cvxpy as cp

from src.dynamics.state import State
from src.dynamics.augmented_state import LookaheadState
from src.dynamics.rk4_integrator import rk4_step
from src.helpers.gp_simple import RBF_example
from src.helpers.gp_helpers import azra_acq_function, acq_func_posterior_covariance
import src.helpers.helper_functions as hf
import src.plotting.plots as plots

from src.core.simulation import Simulation
from src.core.control import GIBO_control, Noise

'''
GIBO Algorithm Implementation

Notes:
 - Increasing number of inner loop searches "m" drastically DECREASES speed but also improves performance
    - Sometimes performance improves, other times it does not. m=5 mproved smoothness, while m=7 didnt.
 - Tweaking lengthscale_prior and output_scale prior doesnt affect the results at all.
 - Decreasing Velocity sometimes has us crash into the CBF, sometimes it results in very strong performance.
 - On the inner loop (m=1,2...M) we essentially do a ghost sample of the point without controlling to the point. This may be difficult to implement IRL

 
A Note on Step 6 and Step 13 of GIBO Algorithm
 - [OLD] Step 13: θt+1 = θt + η·E ∇θJ θ=θt ◃Gradient ascent, or any other gradient based optimizer.
 - [NEW] Step 13: Solve the QP for the control input necessary to step forward the state (θt+1)
 - [?] GIBO Line 6: No GP hyperparameter optimization (Frozen in step 1)
'''

'''
mtg:
 - Query Points on edge of boundary (turning points are most uncertain)
 
Performance criteria
 1) posterior covariance reduction along with query points w.r.t time.
 - plot of time s

 
Mathematically guarantess of safety
Walkthrough of manipulator
    - SOCP explanation
'''
def main():
    # i. Initialize Sim and Objects
    sim = Simulation(dt=0.1, end_time=20.0)
    GIBO = GIBO_control(V=0.6, Kp=4.0, alpha_cbf=.5, M_number_of_queries=6, dt=sim.dt)

    # -- GIBO Line 1: Setup GIBO Hyperparameters -- 
    # a. stepsize η ()
    step_size = 0.1
    # b. hyperpriors for GP hyperparameters 
    RBF_gaussian_likelihood = gp.likelihoods.GaussianLikelihood()
    lengthscale_prior = GammaPrior(1.0, 1) # (a,b) : lengthscale = a/b
    outputscale_prior = GammaPrior(2.0, .17) # (a,b) : outputscale = a/b
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

        # -- GIBO Line 6 (SKIPPED for Safety) : Construct Training Data for GIBO / GP --
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
            next_query_point, query_covariance_gain = GIBO.acquisition_function(train_x=train_x, train_y=train_y, 
                                                    GP_model=GP_model, GP_likelihood=RBF_gaussian_likelihood, 
                                                    waypoint=GIBO.waypoint, sigma2=sigma2, l=l, 
                                                    obs_noise=obs_noise, next_query_point=next_query_point, M=M, m=m)

            # -- Step 9: Sample Noisy Objective Function sdf = J(query_point) + noise
            sdf_query_point, _ = GIBO.sdf(X=LookaheadState(next_query_point[0], next_query_point[1], theta=GIBO.X.theta, l=0.0), Noise=GIBO.Noise)

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

        # -- Step 12: End Inner For Loop --
        # -- Step 13 (ADAPTED -> Solves CBF-QP): X_{t+1} = X_t + η·E[∇J | X=X_t] --
        u_nominal = GIBO.optimal_control(GIBO.waypoint.tolist())
        GIBO.u_final = GIBO.solve_cbf_qp(u_GIBO=u_nominal, h_safe=signed_distance,    # SDF at the robot's current lookahead position
            grad_h=expected_grad_SDF,  # GP gradient estimate refined over M queries
            alpha_cbf=GIBO.alpha_cbf, X=GIBO.X,
        )
        X_next = rk4_step(GIBO.X, GIBO.V, sim.curr_time, sim.dt, GIBO.u_final)
        GIBO.X = LookaheadState(X_next.x, X_next.y, X_next.theta, l=GIBO.X.l)

        # ----------------------------------------------------------
        # -- End of Algorithm. Repeat Loops and plot. -- 
        # ----------------------------------------------------------
        sim.log_data(X=GIBO.X, u=GIBO.u_final, dist=GIBO.return_dist_vec(),
                     time=sim.curr_time, query_points=next_query_point)
        sim.arrived_at_goal(0.2, GIBO)
        sim.step(print_t=True)
        if GIBO.goal_reached:
            break        

    sim.plot_trajectory(GIBO)

if __name__ == "__main__":
    main()
    print("End Function")
