import numpy as np
import torch
import gpytorch
# Importing our custom module(s)
import means
import priors

class GPPowerLawMatern(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood, epsilon_min=0.0, with_priors=True, nu=2.5):
        """
        Gaussian Process with Power Law mean function and Matern kernel
        
        Args:
            X (torch.Tensor): Training data inputs
            y (torch.Tensor): Training data outputs
            likelihood (gpytorch.likelihoods): GP likelihood
            epsilon_min (float): Minimum value for epsilon parameter
            with_priors (bool): Whether to use priors
            nu (float): Smoothness parameter for Matern kernel (0.5, 1.5, or 2.5)
        """
        super(GPPowerLawMatern, self).__init__(X, y, likelihood)
        # Mean module
        self.mean_module = means.PowerLawPriorMean(torch.max(y).item(), epsilon_min)
        
        # Covariance module - Matern kernel
        assert nu in [0.5, 1.5, 2.5], "nu must be one of 0.5, 1.5, or 2.5"
        self.nu = nu
        if nu == 0.5:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        elif nu == 1.5:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        elif nu == 2.5:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
            
        if with_priors:
            # Register priors similar to the RBF kernel case
            self.register_prior(
                'noise_std_prior', 
                priors.TruncatedNormalPrior(0, np.inf, 0.01, 0.01), 
                lambda module: module.likelihood.noise.sqrt()
            )
            # Calculate optimal outputscale prior
            tau_prior = priors.my_truncnorm(0, np.inf, 0.01, 0.01)
            desired_low = (1/2)*((1-epsilon_min)-torch.max(y).item())
            desired_high = (3/4)*((1-epsilon_min)-torch.max(y).item())
            m, s = priors.calc_outputscale_prior(tau_prior, desired_low, desired_high)
            self.register_prior(
                'outputscale_std_prior', 
                priors.TruncatedNormalPrior(0, np.inf, m, s), 
                lambda module: module.covar_module.outputscale.sqrt()
            )
            self.register_prior(
                'lengthscale_prior', 
                priors.TruncatedNormalPrior(0, np.inf, -1.23, 2.12), 
                lambda module: module.covar_module.base_kernel.lengthscale
            )
            self.register_prior(
                'epsilon_prior',
                priors.UniformPrior(self.mean_module.epsilon_min, (1.0-self.mean_module.y_max)), 
                lambda module: module.mean_module.epsilon_min + (1.0-module.mean_module.y_max-module.mean_module.epsilon_min)*torch.sigmoid(module.mean_module.epsilon)
            )
            
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(torch.log10(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPArctanMatern(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood, epsilon_min=0.0, with_priors=True, nu=2.5):
        """
        Gaussian Process with Arctan mean function and Matern kernel
        
        Args:
            X (torch.Tensor): Training data inputs
            y (torch.Tensor): Training data outputs
            likelihood (gpytorch.likelihoods): GP likelihood
            epsilon_min (float): Minimum value for epsilon parameter
            with_priors (bool): Whether to use priors
            nu (float): Smoothness parameter for Matern kernel (0.5, 1.5, or 2.5)
        """
        super(GPArctanMatern, self).__init__(X, y, likelihood)
        # Mean module
        self.mean_module = means.ArctanPriorMean(torch.max(y).item(), epsilon_min)
        
        # Covariance module - Matern kernel
        assert nu in [0.5, 1.5, 2.5], "nu must be one of 0.5, 1.5, or 2.5"
        self.nu = nu
        if nu == 0.5:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        elif nu == 1.5:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        elif nu == 2.5:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
            
        if with_priors:
            # Same priors as in the original implementation
            self.register_prior(
                'noise_std_prior', 
                priors.TruncatedNormalPrior(0, np.inf, 0.01, 0.01), 
                lambda module: module.likelihood.noise.sqrt()
            )
            tau_prior = priors.my_truncnorm(0, np.inf, 0.01, 0.01)
            desired_low = (1/2)*((1-epsilon_min)-torch.max(y).item())
            desired_high = (3/4)*((1-epsilon_min)-torch.max(y).item())
            m, s = priors.calc_outputscale_prior(tau_prior, desired_low, desired_high)
            self.register_prior(
                'outputscale_std_prior', 
                priors.TruncatedNormalPrior(0, np.inf, m, s), 
                lambda module: module.covar_module.outputscale.sqrt()
            )
            self.register_prior(
                'lengthscale_prior', 
                priors.TruncatedNormalPrior(0, np.inf, -1.23, 2.12), 
                lambda module: module.covar_module.base_kernel.lengthscale
            )
            self.register_prior(
                'epsilon_prior',
                priors.UniformPrior(self.mean_module.epsilon_min, (1.0-self.mean_module.y_max)), 
                lambda module: module.mean_module.epsilon_min + (1.0-module.mean_module.y_max-module.mean_module.epsilon_min)*torch.sigmoid(module.mean_module.epsilon)
            )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(torch.log10(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp_matern(likelihood, model, X_train, y_train, max_iters=1000, lr=0.01):
    """Training function for Matern kernel GPs"""
    likelihood.train()
    model.train()
    losses = np.zeros(max_iters)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Move model and likelihood to the appropriate device
    model = model.to(device)
    likelihood = likelihood.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(max_iters):
        if device.type == 'cuda': X_train, y_train = X_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step() 
        if device.type == 'cuda': loss = loss.cpu()
        losses[i] = loss
    if device.type == 'cuda': 
        model = model.to('cpu')
        likelihood = likelihood.to('cpu')
    model.eval()
    likelihood.eval()
    return likelihood, model, losses


import torch
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel

class GPMatern12(gpytorch.models.ExactGP):
    """Gaussian Process with Matern 1/2 kernel (less smooth)"""
    def __init__(self, train_x, train_y, likelihood, epsilon_min=0.05, with_priors=True):
        super(GPMatern12, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=0.5, ard_num_dims=1)
        )
        
        # Set priors if requested
        if with_priors:
            self.mean_module.constant.constraint = gpytorch.constraints.Interval(0.5, 1.0)
            self.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
            self.covar_module.outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
            
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPMatern32(gpytorch.models.ExactGP):
    """Gaussian Process with Matern 3/2 kernel (medium smoothness)"""
    def __init__(self, train_x, train_y, likelihood, epsilon_min=0.05, with_priors=True):
        super(GPMatern32, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=1.5, ard_num_dims=1)
        )
        
        # Set priors if requested
        if with_priors:
            self.mean_module.constant.constraint = gpytorch.constraints.Interval(0.5, 1.0)
            self.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
            self.covar_module.outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
            
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPMatern52(gpytorch.models.ExactGP):
    """Gaussian Process with Matern 5/2 kernel (more smooth)"""
    def __init__(self, train_x, train_y, likelihood, epsilon_min=0.05, with_priors=True):
        super(GPMatern52, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=1)
        )
        
        # Set priors if requested
        if with_priors:
            self.mean_module.constant.constraint = gpytorch.constraints.Interval(0.5, 1.0)
            self.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
            self.covar_module.outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
            
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)