Tasks
Stage 0: Setup

• Create a repo.
• Pick one GP library -> GPyTorch


Stage 1: Basic GP regression implementation

• GP regression on 1D function.
• Implement experiments with at least 3 kernels (SE, Matern kernels).
• For each kernel, fit hyperparameters, then compare the posterior mean and predictive variance.

Deliverables: Plot the mean function together with its uncertainty. Discuss the effect of kernel smoothness.



# Questions
 - is there a general good amount of training data to use? i notice with less data we get tight bounds, more data we get wider ones.
 - what is this? lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0) 
        outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15) 