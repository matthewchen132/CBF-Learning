import math
import numpy as np
import torch
import gpytorch as gp
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
import matplotlib.pyplot as plt
from gp_simple import RBF_example, m52_example# my GP

def main():
    # underlying function + noise
    start_x = 0
    end_x = 4
    n_training_points = 50
    training_x = torch.linspace(start_x,end_x,n_training_points)
    rng = np.random.default_rng(0)  # random reproducible noise
    noise = rng.normal(loc=0.0, scale= 0.5, size=training_x.shape)
    noise = torch.tensor(noise, dtype=training_x.dtype)
    training_y = torch.sin(math.pi*training_x) +training_x+ noise

    # define our GPs and gaussian likelihood
    gauss_likelihood = gp.likelihoods.GaussianLikelihood()
    rbf_model = RBF_example(training_x,training_y, likelihood=gauss_likelihood)
    m52_model = m52_example(training_x,training_y, likelihood=gauss_likelihood)
    # find the optimal hyperparameters (mean, cov, etc.)
    rbf_model.train()
    m52_model.train()
    gauss_likelihood.train()

    # learn the GP using Adam's optimizer (gradient descent)
    rbf_optimizer = torch.optim.Adam(
        list(rbf_model.parameters()) + list(gauss_likelihood.parameters()),
        lr=0.05
    )
    m52_optimizer = torch.optim.Adam(
        list(m52_model.parameters()) + list(gauss_likelihood.parameters()),
        lr=0.05
    )
    # Marginal Log Likelihood:
    # - finds probability of the function found by GP by comparing to sampled data.
    rbf_mll = gp.mlls.ExactMarginalLogLikelihood(gauss_likelihood,rbf_model)
    m52_mll = gp.mlls.ExactMarginalLogLikelihood(gauss_likelihood,m52_model)

    n_samples = 200
    for i in range(n_samples):
        rbf_optimizer.zero_grad()
        rbf_output = rbf_model(training_x)
        rbf_loss = -rbf_mll(rbf_output, training_y)
        rbf_loss.backward()

        m52_optimizer.zero_grad()
        m52_output = m52_model(training_x)
        # m52_loss = -m52_mll(m52_output, training_y)



        print(f"Iteration: {i}, rbf_loss: {round(rbf_loss.item(),4)}")
        rbf_optimizer.step()
    # evaluate results
    rbf_model.eval()
    gauss_likelihood.eval()
    with torch.no_grad(), gp.settings.fast_pred_var():
        test_x = torch.linspace(start_x, end_x + 1,n_samples, dtype=training_x.dtype)
        y_pred = gauss_likelihood(rbf_model(test_x))
        f_pred = rbf_model(test_x)
        y_lower, y_upper = y_pred.confidence_region()
        f_lower, f_upper = f_pred.confidence_region()

    # plot
    f, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(training_x, training_y, 'b.', label="Observations")
    ax[0].plot(test_x, y_pred.mean.numpy(), 'r', label="Posterior Mean")
    ax[0].fill_between(
        test_x.numpy(), f_lower.numpy(), f_upper.numpy(),
        alpha=0.7,label="Function Confidence Interval"
    )
    ax[0].fill_between(
        test_x.numpy(), y_lower.numpy(),y_upper.numpy(),
        alpha=0.15,label="Y Confidence Interval"
    )
    ax[0].axvline(x = end_x, color = "g", linestyle = "--")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_title("GP Regression (y = sin(x) + x)")
    ax[0].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

