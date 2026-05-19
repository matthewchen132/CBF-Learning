import torch
import time
torch.set_default_dtype(torch.float64)

import gpytorch as gp
from gpytorch.priors import GammaPrior
import numpy as np

from src.helpers.gp_simple import RBF_example
from src.core.simulation import Simulation
from src.core.manipulator import Manipulator

'''
GIBO Algorithm Implementation
[ ] Robotic manipulator, implement dynamics and GIBO
[ ] CBF based on angle limit:  r = pisin(t), pi/2 sin(4t)

Notes on Paper
 - 2nd order System Dynamics
 - Construct a control-invariant subset of the full state space which contains
    all positions allowed by the original positional constraints.
3) derive a condition which, if satisfied, proves that our constructed set is 
safe for general second-order systems and provide an associated 

QP safeguarding controller.
'''

def main():
    start_time = time.perf_counter()
    print("1 for GIBO | 2 for random sample | 3 for adjusted")
    algo = input()
    # i. Initialize Sim
    sim = Simulation(dt=0.04, end_time=11)
    # NOTE: dt must be sufficiently small to not explode

    # -- GIBO Line 1: Setup GIBO Hyperparameters -- 
    # a. stepsize η (Not applicable, step forward using numerical integration)

    # b. hyperpriors for GP hyperparameters 
    RBF_gaussian_likelihood = gp.likelihoods.GaussianLikelihood()
    a = 0.2
    b = 1.0
    lengthscale_prior = GammaPrior(a, b) # (a,b) : lengthscale = a/b
    outputscale_prior = GammaPrior(2.0, .17) # (a,b) : outputscale = a/b
    GP_model = RBF_example(train_x=[], train_y=[], likelihood=RBF_gaussian_likelihood, l_prior=lengthscale_prior, output_prior=outputscale_prior)
    GP_model.eval()
    RBF_gaussian_likelihood.eval()
    
    # c. number of iterations -> N = 201 (20 seconds, 0.1s timestep)
    N = sim.dt
    
    # d. number of samples for gradient estimate -> M =
    M_samples = 1
    Arm = Manipulator(alpha_cbf=.5, M_number_of_queries=M_samples, dt=N)
    n_sigma = 2.0

    # -- GIBO Line 2: Set theta1 and theta2 (next_query_point) and empty dataset D = {} --
    Arm.D = {"thetas": [], "h": []}
    # expected_grad_ang_dist = torch.tensor([-1.0, 0.0])  # analytic default before GP warms up
    next_query_point = torch.tensor([Arm.theta.tolist()])  # initial query at arm's joint state

    # -- Safety metrics --
    cbf_violations = 0
    min_h = float("inf")

    # -- GIBO Line 3: for t = 0, ..., N do --
    while sim.is_running():

        # -- GIBO Line 4: Sample Noisy Objective Function | J(θ_t) + ε_t --
        h_noisy = Arm.observe_J(theta=Arm.theta)  # noisy J — added to GP dataset

        # -- GIBO Line 5: Extend Dataset D --
        Arm.D["thetas"].append(Arm.theta.tolist())  # .tolist() avoids reference sharing with rk4_step
        Arm.D["h"].append(h_noisy)
        train_x = torch.tensor(Arm.D["thetas"])
        train_y = torch.tensor(Arm.D["h"])

        # -- GIBO Line 6 (Retraining Skipped for Safety) : Construct Training Data for GIBO / GP --
        GP_model.set_train_data(inputs=train_x, 
                                targets=train_y,
                                strict=False)
        sigma2 = GP_model.covar_module.outputscale.detach()
        l = GP_model.covar_module.base_kernel.lengthscale.detach().squeeze(0)
        obs_noise = RBF_gaussian_likelihood.noise.detach() 

        # -- GIBO Step 7: For m = 1,2, ... M:
        query_space = Arm.return_gridspace(wander_angle_deg=np.rad2deg(.55*a/b))
        # -- Step 8: Get query point = argmax(acq_function)  
        for m in range(Arm.M_number_of_queries):
            # -- GIBO (Original) --

            if algo == "1":
                next_query_point, query_covariance_gain = Arm.acquisition_function(train_x=train_x, train_y=train_y, 
                                                        GP_model=GP_model, GP_likelihood=RBF_gaussian_likelihood, 
                                                        sigma2=sigma2, l=l, obs_noise=obs_noise, 
                                                        next_query_point=next_query_point, query_space=query_space, m=m)
            elif algo == "2":
                next_query_point, query_covariance_gain = Arm.random_acq_function(train_x=train_x, train_y=train_y, 
                                                        GP_model=GP_model, GP_likelihood=RBF_gaussian_likelihood, 
                                                        sigma2=sigma2, l=l, obs_noise=obs_noise, 
                                                        next_query_point=next_query_point, query_space=query_space, m=m)
            elif algo == "3":
                next_query_point, query_covariance_gain = Arm.composite_acq_function(train_x, GP_model, RBF_gaussian_likelihood,
                                                        sigma2, l, obs_noise, query_space)  
            else:
                raise Exception("Invalid Index for acquisition function.")
            
            print(f"QP Covariance Gain: {query_covariance_gain}")
            
            # -- Step 9: Sample Noisy Objective Function J(query_point) + noise
            angular_dist_qp = Arm.observe_J(theta=next_query_point.squeeze().numpy())

            # -- Step 10: Extend Data Set to include next query point --
            Arm.D["thetas"].append(next_query_point.squeeze().tolist())
            Arm.D["h"].append(angular_dist_qp)
            train_x = torch.tensor(Arm.D["thetas"], dtype=torch.float64)
            train_y = torch.tensor(Arm.D["h"], dtype=torch.float64)

            # -- Step 11: Update the posterior of ∇θJ <-> (∇ Signed Distance)
            # E[∇J]= ∑​ w_i * ∇K(x,x) <- Expected value of gradient is the sum of weighted gradient_K_xx\
            
            weights = Arm.compute_gradient_update_weights(train_x, train_y, GP_model, GP_model.likelihood).unsqueeze(-1) # (3,1)
            grad_K = Arm.compute_grad_K(train_x, GP_model, lengthscale=l)
            expected_grad_ang_dist = torch.sum(weights * grad_K, dim=0).detach()

        # -- Step 12: End Inner For Loop  (Also add safety margin to reading) --
        GP_model.set_train_data(inputs=train_x, targets=train_y, strict=False)
        x_curr = torch.tensor([[Arm.theta[0], Arm.theta[1]]]) # Evaluate
        with torch.no_grad():
            pred   = GP_model(x_curr)
            h_mean = pred.mean.item()
            h_std  = pred.variance.sqrt().item()
        h_safe = h_mean - n_sigma * h_std

        # -- Step 13:  Solves CBF-QP X_{t+1} = X_t + η·E[∇J | X=X_t] --
        torque_nom = Arm.u_nom(sim.curr_time)
        Arm.u_final = Arm.solve_hocbf_qp(tau_nom=torque_nom, h=h_safe, grad_h=expected_grad_ang_dist)
        Arm.rk4_step(Arm.u_final)  # updates self.theta, self.dtheta
        h_true = Arm.THETA_MAX - Arm.theta[0]
        if h_true < 0:
            cbf_violations += 1
        if h_true < min_h:
            min_h = h_true

        # ----------------------------------------------------------
        # -- End of Algorithm. Repeat and plot. --
        # ----------------------------------------------------------
        
        sim.log_data_manipulator(
            theta=Arm.theta, tau=Arm.u_final, h=h_noisy,
            t=sim.curr_time, query_points=next_query_point,
            grad=expected_grad_ang_dist.tolist(),
            h_mean=h_mean, h_std=h_std,
        )
        sim.step(print_t=True)
    end_time = time.perf_counter()
    print(f"Total Runtime (s): {(end_time - start_time):.4f}")
    print(f"CBF Violations:    {cbf_violations}")
    print(f"Min h value (deg): {np.degrees(min_h):.3f}")
    if algo == "1":
        sim.plot_acq_function(Arm, n_sigma=n_sigma)
    elif algo == "2":
        sim.plot_random(Arm, n_sigma=n_sigma)
    elif algo == "3":
        sim.plot_composite(Arm, n_sigma=n_sigma)
    else:
        raise Exception("Invalid Index for acquisition function.")

if __name__ == "__main__":
    main()
    print("End Function")
