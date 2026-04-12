import torch
import gpytorch as gp
import numpy as np
import matplotlib.pyplot as plt
import math

import helpers.gp_simple as gp_simple
from helpers.helper_functions import in_hemisphere

def train_GP(n, GP_optimizer, GP_model, GP_mll, train_x, train_y ):
    '''
    n_i: Number of training samples
    '''

        # -- Train GP -- 
    for i in range(n):
        # -- Provides the posterior GP -- 
        GP_optimizer.zero_grad() # needed to feed the m52_model only the CURRENTLY accumulated gradient
        GP_output = GP_model(torch.tensor(train_x))
        GP_loss = -GP_mll(GP_output, torch.tensor(train_y)).sum()
        GP_loss.backward()
        GP_optimizer.step()
        


def sdf_gp_gradient(GP_model, likelihood, test_x, train_x, train_y):
    '''
    params:
        - GP_model: gpytorch model
        - test_x: data point to recalculate GP gradient over. (Current x,y position)
        - train_x: lidar data (N,2) + current x,y (1,2) <- [N+1,2]
        - train_y: surface of obstacle (M,1) + min_signed_distance (1,1) [M+1, 1]
    '''
    K_xstar_x = GP_model.covar_module(test_x,train_x).evaluate().detach()
    l = GP_model.covar_module.base_kernel.lengthscale.detach()
    diff_r  = (test_x.unsqueeze(1) - train_x.unsqueeze(0))
    # print(f"diff_r {diff_r.size()}")
    r = torch.norm(diff_r, dim=-1, keepdim=True)

    sigma2 = GP_model.covar_module.outputscale.detach()
    rt5 = math.sqrt(5)
    # -- Grad K --
    grad_K_xstar_x = -sigma2 * (5.0 * diff_r/(3*l**2)) * (1 + rt5*r/l) * torch.exp(-rt5*r/l)
    grad_K_xstar_x = grad_K_xstar_x.squeeze(0).transpose(0, 1) # Resize for matrix ops


    # -- Noisy K --
    K_xx = GP_model.covar_module(train_x,train_x).evaluate().detach()
    obs_noise = likelihood.noise.detach()
    K_xx_noisy = K_xx + obs_noise*torch.eye(len(train_x)) # K + variance*I
    # print(f"K_xx_noisy {K_xx_noisy.size()}\n")

    # -- alpha = K^-1 * y --
    alpha = torch.linalg.solve(K_xx_noisy, train_y) 
    # NOTE: Gradient Mean
    grad_mean = grad_K_xstar_x @ alpha # [grad_K][alpha]

    # -- Finding d2K(x*, x*) -- 
    r_star = torch.abs(test_x.unsqueeze(1) - test_x.unsqueeze(0))
    d2K_dxstar_xstar = -(5/(3*l**2))*sigma2 * (1 + rt5*r_star/l - 5*r_star**2/l**2) * torch.exp(-rt5*r_star/l)
    grad2_K_xstar_xstar = - d2K_dxstar_xstar 
    middle_term =torch.linalg.solve(K_xx_noisy, grad_K_xstar_x.T) # K^-1 ^ grad_K(x*,x)
    
    # NOTE: Gradient Covariance
    cov_grad_K = grad2_K_xstar_xstar -grad_K_xstar_x @ middle_term
    return grad_mean, cov_grad_K

def GIBO(gridspace, X, V, train_x, train_y, GP_model, waypoint, 
                        l, sigma2, obs_noise, cov_old):
    """
    Evaluates gridspace to find the point that maximizes the Trace reduction 
    of the GP Covariance (Information Gain).
    """
    best_acq_value = -float('inf')
    next_qp = None
    tr_cov_old = torch.trace(cov_old)
    rt5 = np.sqrt(5)

    # Convert waypoint to tensor for distance calc if needed
    waypoint_t = torch.tensor(waypoint, dtype=torch.float32)

    for query_point in gridspace:
        # 1. Hemisphere Filter (Keep it relative to robot velocity)
        if not in_hemisphere(X, V, grid_point=query_point):
            continue  
        
        # 2. Kernel Math for the Augmented Set (Current + Candidate)
        train_x_new = torch.cat((train_x, query_point.unsqueeze(0)), dim=0)
        
        # -- grad2 K(query_point, query_point) --
        grad2_K_tt = -(5 * sigma2 / (3 * l**2)) * torch.eye(2)

        # -- grad1 K (query_point, train_x_new) --
        diff = query_point.unsqueeze(0) - train_x_new # (N+1, 2)
        r = torch.norm(diff, dim=1, keepdim=True) # (N+1, 1)

        # Matern 5/2 Gradient logic
        K_xx_new = GP_model.covar_module(train_x_new, train_x_new).evaluate().detach()
        K_xx_noisy_new = K_xx_new + obs_noise * torch.eye(len(train_x_new))
        
        # Gradient of Kernel: dK/dx
        grad_K_tx_new = -sigma2 * (5.0 * diff / (3 * l**2)) * (1 + rt5 * r / l) * torch.exp(-rt5 * r / l)
        K_inv_Ktx = torch.linalg.solve(K_xx_noisy_new, grad_K_tx_new) 
        
        # -- Covariance --
        cov_new = grad2_K_tt - grad_K_tx_new.T @ K_inv_Ktx 
        tr_cov_new = torch.trace(cov_new)

        acq_function = tr_cov_old - tr_cov_new # NOTE: try with sigma_^2 instead of gradient covariance
        if acq_function > best_acq_value: 
            best_acq_value = acq_function
            next_qp = query_point.clone().reshape(1, 2)
    return next_qp, best_acq_value

def generate_gp_training_data(gp_obstacle_data, X, visible_pts):
    gp_obstacle_data = torch.tensor(visible_pts) # (N, 2)
    pos_x = torch.tensor([X.x, X.y]).reshape(shape=[1,2]) # (N,2)
    train_x = torch.cat([gp_obstacle_data, pos_x], dim=0).squeeze(-1) # (N+1, 2)

    surface_y = torch.zeros(visible_pts.shape[0], 1) # (N,1) <-- surface of SDF is marked as zeros
    dist_to_obj = torch.min(torch.norm(gp_obstacle_data - pos_x, dim=1)).reshape(shape=[1,1]) # (1,1)
    train_y = torch.cat([surface_y, dist_to_obj], dim=0).squeeze(-1) # (N+1, 1)
    return train_x, train_y

def probabilistic_sdf(h_mean, h_variance, n_std_deviations):
    '''
    Returns a Signed Distance = h_mean - (Std. Deviation) x (user-defined #)
    '''

    h_std  = torch.sqrt(h_variance)
    h_safe = h_mean - n_std_deviations * h_std
    return h_safe

