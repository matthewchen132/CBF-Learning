import math
import numpy as np
import torch
import gpytorch as gp
import matplotlib.pyplot as plt
from gp_simple import hybrid_example # my GP


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
    hybrid_gauss_likelihood = gp.likelihoods.GaussianLikelihood()
    hybrid_model = hybrid_example(training_x, training_y, hybrid_gauss_likelihood)

    # find the optimal hyperparameters (mean, cov, etc.)
    hybrid_model.train()
    hybrid_gauss_likelihood.train()
    hybrid_optimizer = torch.optim.Adam(
        list(hybrid_model.parameters()) + list(hybrid_gauss_likelihood.parameters()),
        lr=0.05
    )
    # Marginal Log Likelihood:
    # - finds probability of the function found by GP by comparing to sampled data.
    hybrid_mll = gp.mlls.ExactMarginalLogLikelihood(hybrid_gauss_likelihood,hybrid_model)
    n_samples = 200
    for i in range(n_samples):
        hybrid_optimizer.zero_grad() # needed to feed the hybrid_model only the CURRENTLY accumulated gradient
        hybrid_output = hybrid_model(training_x)
        hybrid_loss = -hybrid_mll(hybrid_output, training_y)
        hybrid_loss.backward()
        print(f"Iteration: {i}, hybrid Loss: {round(hybrid_loss.item(),4)}")
        hybrid_optimizer.step()

    # 1) Evaluate the Gradient now that we have our trained GPR
    hybrid_model.eval()
    hybrid_gauss_likelihood.eval()

    with torch.enable_grad(), gp.settings.fast_pred_var(): 
        # 2) Adjust test_x throughout the range and create a prediction Y_pred_hybrid
        test_x = torch.linspace(start_x, end_x + 2,n_samples, dtype=training_x.dtype, requires_grad=True)        
        # 3) get predicted output Y  and f(x) trained on test_x data
        y_pred_hybrid = hybrid_gauss_likelihood(hybrid_model(test_x))
        y_lower_hybrid, y_upper_hybrid = y_pred_hybrid.confidence_region()
        f_pred_hybrid = hybrid_model(test_x)
        f_lower_hybrid, f_upper_hybrid = f_pred_hybrid.confidence_region()
        # ----  Hybrid (Matern 5/2 + Linear) Gradient ----
        additive = hybrid_model.covar_module.base_kernel
        linear_kernel = additive.kernels[0]
        matern_kernel = additive.kernels[1]
        linear_var = linear_kernel.variance.detach()
        c = linear_kernel.offset.detach() if hasattr(linear_kernel, 'offset') else torch.tensor(0.0)
        l = matern_kernel.lengthscale.detach().squeeze()
        outputscale = hybrid_model.covar_module.outputscale.detach()

        diff_r = test_x.unsqueeze(1) - training_x.unsqueeze(0) 
        r = torch.abs(diff_r)
        rt5 = math.sqrt(5)
        matern_grad = -(5.0/(3*l**2))*diff_r*(1+rt5*r/l)*torch.exp(-rt5*r/l)

        # find gradients individually
        linear_grad = linear_var * (training_x.unsqueeze(0) - c)  # [1, n_train]
        grad_K_xstar_x = outputscale*matern_grad + linear_grad

        # grad_mean
        obs_noise = hybrid_gauss_likelihood.noise.detach()
        K_xx = hybrid_model.covar_module(training_x, training_x).evaluate().detach()
        K_xx_noisy = K_xx + obs_noise * torch.eye(len(training_x))
        alpha = torch.linalg.solve(K_xx_noisy, training_y)
        grad_mean = grad_K_xstar_x @ alpha

        # grad covariance (for bounds)
        r_star = torch.abs(test_x.unsqueeze(1) - test_x.unsqueeze(0))
        d2K_dxstar_xstar = outputscale*(5.0/(3*l**2))*(1-rt5*r_star/l)*torch.exp(-rt5*r_star/l)
        # linear kernel has 0 to second derivative

        middle_term = torch.linalg.solve(K_xx_noisy, grad_K_xstar_x.T)
        cov_grad = d2K_dxstar_xstar - grad_K_xstar_x @ middle_term
        gradient_conf_bounds = 2 * torch.sqrt(cov_grad.diag().clamp(min=0))  # clamp for numerical safety
        conf_bound_low = -gradient_conf_bounds
        conf_bound_high = gradient_conf_bounds
        
        
    # Plotting
    # hybrid / SE
    f, ax = plt.subplots(2, 1, figsize = (10,8))
    ax[0].plot(training_x, training_y, 'b.', label="Observations")
    ax[0].plot(det(test_x), det(y_pred_hybrid.mean), 'r', label="Posterior Mean")
    ax[0].plot(det(test_x), np.sin(math.pi*det(test_x)) + np.array(det(test_x)), 'k--', label="True f(x)")
    ax[0].fill_between(
        det(test_x), det(f_lower_hybrid), det(f_upper_hybrid),
        alpha=0.7,label="Function Confidence Interval"
    )
    ax[0].fill_between(
        det(test_x), det(y_lower_hybrid), det(y_upper_hybrid),
        alpha=0.15,label="Y Confidence Interval"
    )
    ax[0].axvline(x = end_x, color = "g", linestyle = "--")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_title("GPR with hybrid Kernel (y = sin(x) + x)")
    ax[0].legend()
    # Derivative Plot
    ax[1].plot(det(test_x), det(grad_mean), 'r-', label="Gradient")
    ax[1].plot(det(test_x), math.pi*np.cos(math.pi*det(test_x)) + 1, 'k--', label="True Gradient")
    ax[1].fill_between(
        det(test_x),det(conf_bound_low+grad_mean), det(conf_bound_high + grad_mean), 
        alpha = 0.15, label="Gradient Confidence Interval (2sig)"
    )
    ax[1].axvline(x = end_x, color = "g", linestyle = "--")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    ax[1].set_title("Gradent determined from Hybrid Kernel ")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()