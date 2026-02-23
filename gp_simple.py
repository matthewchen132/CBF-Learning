import gpytorch as gp

class RBF_example(gp.models.ExactGP): # AKA squared exponential
    '''
    Squared Exponential Kernel
    '''
    def __init__(self, train_x, train_y, likelihood):
        super(RBF_example, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        
        # initial ccnditions
        # large l -> strong effects from all data
        # small l -> strong effects from close points
        lengthscale_prior = gp.priors.GammaPrior(3.0, 1) # (a,b) : lengthscale = a/b

        # Vertical scaling
        outputscale_prior = gp.priors.GammaPrior(4.0, .17) # (a,b) : outputscale = a/b
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel
                                                         (lengthscale_prior=lengthscale_prior,
                                                         ),outputscale_prior=outputscale_prior
                                                         )
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean, covar)    


class m52_example(gp.models.ExactGP):
    '''
    Matern5/2 Kernel
    '''
    def __init__(self, train_x, train_y, likelihood):
        super(m52_example, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        lengthscale_prior = gp.priors.GammaPrior(concentration=3.0, rate=1.0) # (a,b) : a/b = mean, a/b^2 = covariance
        outputscale_prior = gp.priors.GammaPrior(4.0, 0.17) # (a,b) : a/b = covariance
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.MaternKernel
                                                        (lengthscale_prior=lengthscale_prior,
                                                        ),outputscale_prior=outputscale_prior
                                                        )
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean, covar)
    
class hybrid_example(gp.models.ExactGP):
    '''Hybrid Kernel with SE and Linear components'''
    def __init__(self, train_x, train_y, likelihood):
        super(hybrid_example, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        lengthscale_prior = gp.priors.GammaPrior(concentration=3.0, rate=6.0) # set by tutorial
        outputscale_prior = gp.priors.GammaPrior(2.0, 0.15) # set by tutorial
        linear_kern = gp.kernels.LinearKernel()
        matern_kern = gp.kernels.MaternKernel(lengthscale_prior = lengthscale_prior, outputscale_prior = outputscale_prior)
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.AdditiveKernel(linear_kern + matern_kern))
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean, covar)