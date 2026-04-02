import torch
import gpytorch as gp
import numpy as np
import matplotlib.pyplot as plt
import pybullet_data
import math

import pkg.gp_simple as gp_simple

class gp_helpers():
    def __init__():
        print("Hi")
    def sdf_gp_gradient(GP_model, likelihood, test_x, training_x, training_y):
        '''
        params:
         - GP_model: gpytorch model
         - test_x: data point to recalculate GP gradient over. (Current x,y position)
         - train_x: lidar data (N,2) + current x,y (1,2) <- [N+1,2]
         - train_y: surface of obstacle (M,1) + min_signed_distance (1,1) [M+1, 1]
        '''
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
        obs_noise = likelihood.noise.detach()
        K_xx_noisy = K_xx + obs_noise*torch.eye(len(training_x)) # K + variance*I
        print(f"K_xx_noisy {K_xx_noisy.size()}\n")

        # -- alpha = K^-1 * y --
        alpha = torch.linalg.solve(K_xx_noisy, training_y) 
        # NOTE: Gradient Mean
        grad_mean = grad_K_xstar_x @ alpha # [grad_K][alpha]
        print(f" SDF Gradient Mean: {grad_mean}") 

        # -- Finding d2K(x*, x*) -- 
        r_star = torch.abs(test_x.unsqueeze(1) - test_x.unsqueeze(0))
        d2K_dxstar_xstar = -(5/(3*l**2))*sigma2 * (1 + rt5*r_star/l - 5*r_star**2/l**2) * torch.exp(-rt5*r_star/l)
        grad2_K_xstar_xstar = - d2K_dxstar_xstar 
        middle_term =torch.linalg.solve(K_xx_noisy, grad_K_xstar_x.T) # K^-1 ^ grad_K(x*,x)
        
        # NOTE: Gradient Covariance
        cov_grad_K = grad2_K_xstar_xstar -grad_K_xstar_x @ middle_term
        print(f" SDF Gradient Covariance: {cov_grad_K}")
