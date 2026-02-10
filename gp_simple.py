import gpytorch

class RBF_example(gpytorch.models.ExactGP): # AKA squared exponential
    def __init__(self, train_x, train_y, likelihood):
        super(RBF_example, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # initial ccnditions
        lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0) 
        outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15) 
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel
                                                         (lengthscale_prior=lengthscale_prior,
                                                         ),outputscale_prior=outputscale_prior
                                                         )
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    

class m52_example(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(m52_example, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0) # set by tutorial
        outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15) # set by tutorial
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel
                                                         (lengthscale_prior=lengthscale_prior,
                                                         ),outputscale_prior=outputscale_prior
                                                         )
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)