import math
import numpy as np
import torch
import gpytorch as gp
import matplotlib.pyplot as plt
from gp_simple import m52_example # my GP

'''
Stage 2 Takeaways:
 - In the posterior mean function, only the kernel is dependent on the input variables, and to get gradient we just differentiate kernel.
 - Accomplish gradient calc by running torch.backwards() on our model at different test_x.
 - X = linalg.solve(A,y) -> X = A^-1 y

'''


def det(tensor):
    return tensor.detach().numpy()

def main():
    # underlying function + noise
    start_x = 0
    end_x = 6
    n_training_points = 50
    training_x = torch.linspace(start_x,end_x,n_training_points) # requires_grad to track gradient
    rng = np.random.default_rng(0)  # random reproducible noise
    noise = rng.normal(loc=0.0, scale=0.5, size=training_x.shape)
    noise = torch.tensor(noise, dtype=training_x.dtype)
    training_y = torch.sin(math.pi*training_x) + training_x + noise

    # define our GPs and gaussian likelihood
    m52_gauss_likelihood = gp.likelihoods.GaussianLikelihood()
    m52_model = m52_example(training_x, training_y, m52_gauss_likelihood)


    # find the optimal hyperparameters (mean, cov, etc.)
    m52_model.train()
    m52_gauss_likelihood.train()
    m52_optimizer = torch.optim.Adam(
        list(m52_model.parameters()) + list(m52_gauss_likelihood.parameters()),
        lr=0.05
    )
    # Marginal Log Likelihood:
    # - finds probability of the function found by GP by comparing to sampled data.
    m52_mll = gp.mlls.ExactMarginalLogLikelihood(m52_gauss_likelihood,m52_model)
    n_samples = 200
    # optimize
    for i in range(n_samples):
        m52_optimizer.zero_grad() # needed to feed the m52_model only the CURRENTLY accumulated gradient
        m52_output = m52_model(training_x)
        m52_loss = -m52_mll(m52_output, training_y)
        m52_loss.backward()

        print(f"Iteration: {i}, m52 Loss: {round(m52_loss.item(),4)}")
        m52_optimizer.step()

    # 1) Evaluate the Gradient now that we have our trained GPR
    m52_model.eval()
    m52_gauss_likelihood.eval()
    with torch.enable_grad(), gp.settings.fast_pred_var(): 
        # 2) Adjust test_x throughout the range and create a prediction Y_pred_m52
        test_x = torch.linspace(start_x, end_x + 2,n_samples, dtype=training_x.dtype, requires_grad=True)        
        # 3) get predicted output Y  and f(x) trained on test_x data
        y_pred_m52 = m52_gauss_likelihood(m52_model(test_x))
        y_lower_m52, y_upper_m52 = y_pred_m52.confidence_region()
        f_pred_m52 = m52_model(test_x)
        f_lower_m52, f_upper_m52 = f_pred_m52.confidence_region()
        # ----  Matern 5/2 Gradient ----
        # 4) find "grad_K", gradient of k(x,x_test) w.r.t "r"
        K_xstar_x = m52_model.covar_module(test_x,training_x).evaluate().detach()
        l = m52_model.covar_module.base_kernel.lengthscale.detach()
        diff_r  = (test_x.unsqueeze(1) - training_x.unsqueeze(0))
        print(f"diff_r {diff_r.size()}")
        r = torch.abs(diff_r) 
        sigma2 = m52_model.covar_module.outputscale.detach()
        # kernel derivation
        rt5 = math.sqrt(5)
        grad_K_xstar_x = -sigma2 * (5.0 * diff_r/(3*l**2)) * (1 + rt5*r/l) * torch.exp(-rt5*r/l)

        # 5) alpha = [K_noisy(x_train, x_train) + sigma^2 * I]^-1 @ y 
        K_xx = m52_model.covar_module(training_x,training_x).evaluate().detach()
        noise = m52_gauss_likelihood.noise.detach()
        K_xx_noisy = K_xx + noise*torch.eye(len(training_x)) # K + variance*I
        print(f"K_xx_noisy {K_xx_noisy.size()}\n")

        # K alpha = y
        # alpha = K^-1 * y
        alpha = torch.linalg.solve(K_xx_noisy, training_y) 
        print(f"alpha {alpha.size()}\n")
        # Grad mean
        grad_mean = grad_K_xstar_x @ alpha # [grad_K][alpha]
        K_xstar_xstar = m52_model.covar_module(test_x,test_x).evaluate().detach()
        print(f"K_xstar_xstar {K_xstar_xstar.size()}\n")

        # Finding d2K(x*, x*)
        r_star = torch.abs(test_x.unsqueeze(1) - test_x.unsqueeze(0))
        d2K_dxstar_xstar = -(5/(3*l**2)) * sigma2 * (1 + rt5*r_star/l - 5*r_star**2/l**2) * torch.exp(-rt5*r_star/l)
        grad2_K_xstar_xstar = - d2K_dxstar_xstar 

        middle_term =torch.linalg.solve(K_xx_noisy, grad_K_xstar_x.T) # K^-1 ^ grad_K(x*,x)

        cov_grad_K = grad2_K_xstar_xstar -grad_K_xstar_x @ middle_term
        gradient_conf_bounds = 2 * torch.sqrt(cov_grad_K.diag())
        conf_bound_low = -gradient_conf_bounds
        conf_bound_high = gradient_conf_bounds
        print(gradient_conf_bounds)
        
        
    # Plotting
    # m52 / SE
    f, ax = plt.subplots(2, 1, figsize = (10,8))
    ax[0].plot(training_x, training_y, 'b.', label="Observations")
    ax[0].plot(det(test_x), det(y_pred_m52.mean), 'r', label="Posterior Mean")
    ax[0].plot(det(test_x), np.sin(math.pi*det(test_x)) + np.array(det(test_x)), 'k--', label="True f(x)")
    ax[0].fill_between(
        det(test_x), det(f_lower_m52), det(f_upper_m52),
        alpha=0.7,label="Function Confidence Interval"
    )
    ax[0].fill_between(
        det(test_x), det(y_lower_m52), det(y_upper_m52),
        alpha=0.15,label="Y Confidence Interval"
    )
    ax[0].axvline(x = end_x, color = "g", linestyle = "--")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_title("GPR with m52 Kernel (y = sin(x) + x)")
    ax[0].legend()
    # Derivative Plot
    ax[1].plot(det(test_x), det(grad_mean), 'r-', label="Gradient")
    ax[1].plot(det(test_x), math.pi*np.cos(math.pi*det(test_x)) + 1, 'k--', label="True Gradient")
    ax[1].fill_between(
        det(test_x),det(conf_bound_low+grad_mean), det(conf_bound_high +grad_mean), 
        alpha = 0.15, label="Gradient Confidence Interval (2sig)"
    )
    ax[1].axvline(x = end_x, color = "g", linestyle = "--")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    ax[1].set_title("Gradent determined from m52 Kernel ")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()