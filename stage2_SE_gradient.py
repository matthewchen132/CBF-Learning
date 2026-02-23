import math
import numpy as np
import torch
import gpytorch as gp
import matplotlib.pyplot as plt
from gp_simple import RBF_example, m52_example, hybrid_example # my GP

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
    rbf_gauss_likelihood = gp.likelihoods.GaussianLikelihood()
    rbf_model = RBF_example(training_x,training_y, likelihood=rbf_gauss_likelihood)


    # find the optimal hyperparameters (mean, cov, etc.)
    rbf_model.train()
    rbf_gauss_likelihood.train()
    rbf_optimizer = torch.optim.Adam(
        list(rbf_model.parameters()) + list(rbf_gauss_likelihood.parameters()),
        lr=0.05
    )
    # Marginal Log Likelihood:
    # - finds probability of the function found by GP by comparing to sampled data.
    rbf_mll = gp.mlls.ExactMarginalLogLikelihood(rbf_gauss_likelihood,rbf_model)
    n_samples = 200
    # optimize
    for i in range(n_samples):
        rbf_optimizer.zero_grad() # needed to feed the rbf_model only the CURRENTLY accumulated gradient
        rbf_output = rbf_model(training_x)
        rbf_loss = -rbf_mll(rbf_output, training_y)
        rbf_loss.backward()

        print(f"Iteration: {i}, RBF Loss: {round(rbf_loss.item(),4)}")
        rbf_optimizer.step()

    # 1) Evaluate the Gradient now that we have our trained GPR
    rbf_model.eval()
    rbf_gauss_likelihood.eval()
    with torch.enable_grad(), gp.settings.fast_pred_var(): 
        # torch.enable_grad() : Explicitly calculate gradients
        # 2) Adjust test_x throughout the range and create a prediction Y_pred_rbf
        test_x = torch.linspace(start_x, end_x + 2,n_samples, dtype=training_x.dtype, requires_grad=True)
        # 3) get predicted output Y  and f(x) trained on test_x data
        y_pred_rbf = rbf_gauss_likelihood(rbf_model(test_x))
        y_lower_rbf, y_upper_rbf = y_pred_rbf.confidence_region()
        f_pred_rbf = rbf_model(test_x)
        f_lower_rbf, f_upper_rbf = f_pred_rbf.confidence_region()
        # ----  RBF KERNEL ----
        # 4) find "grad_K", gradient of k(x,x_test) 
        K_xstar_x = rbf_model.covar_module(test_x,training_x).evaluate().detach()
        l = rbf_model.covar_module.base_kernel.lengthscale.detach()
        grad_K_xstar_x= -K_xstar_x*(test_x.unsqueeze(1) - training_x.unsqueeze(0))/l**2
        # 5) alpha = [K_noisy(x_train, x_train) + sigma^2 * I]^-1 @ y 
        K_xx = rbf_model.covar_module(training_x,training_x).evaluate_kernel()
        noise = rbf_gauss_likelihood.noise.detach()
        K_xx_noisy = K_xx + noise*torch.eye(len(training_x)) # K + variance*I
        # Rather than inverting, we do a numerical solve:
        # K alpha = y
        alpha = torch.linalg.solve(K_xx_noisy, training_y)
        # 6) with a derived gradient of the kernel, we can use the same alpha from K+train to get the mean function "grad_mean"
        grad_mean = grad_K_xstar_x @ alpha # [grad_K][alpha]
        K_xstar_xstar = rbf_model.covar_module(test_x,test_x).evaluate().detach()

        grad2_K_xstar_xstar = (1.0/l**2 - (test_x.unsqueeze(1)-test_x.unsqueeze(0))**2/l**4) * K_xstar_xstar
        middle_term =torch.linalg.solve(K_xx_noisy, grad_K_xstar_x.T) # K^-1 ^ grad_K(x*,x)
        cov_grad_K = grad2_K_xstar_xstar -grad_K_xstar_x @ middle_term
        gradient_conf_bound_vals = 2 * torch.sqrt(cov_grad_K.diag())
        conf_bound_low = -gradient_conf_bound_vals
        conf_bound_high = gradient_conf_bound_vals
        
    # Plotting
    # RBF / SE
    f, ax = plt.subplots(2, 1, figsize = (10,8))
    ax[0].plot(training_x, training_y, 'b.', label="Observations")
    ax[0].plot(det(test_x), det(y_pred_rbf.mean), 'r', label="Posterior Mean")
    ax[0].plot(det(test_x), np.sin(math.pi*det(test_x)) + np.array(det(test_x)), 'k--', label="True f(x)")
    ax[0].fill_between(
        det(test_x), det(f_lower_rbf), det(f_upper_rbf),
        alpha=0.7,label="Function Confidence Interval"
    )
    ax[0].fill_between(
        det(test_x), det(y_lower_rbf), det(y_upper_rbf),
        alpha=0.15,label="Y Confidence Interval"
    )
    ax[0].axvline(x = end_x, color = "g", linestyle = "--")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_title("GPR with RBF Kernel (y = sin(x) + x)")
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
    ax[1].set_title("Gradent determined from RBF Kernel ")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()