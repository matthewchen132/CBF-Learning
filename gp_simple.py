import gpytorch
class gp_simple(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
        # train_x = ...; train_y = ...
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = MyGP(train_x, train_y, likelihood)
                # test_x = ...;
        model(test_x)  # Returns the GP latent function at test_x
        likelihood(model(test_x))