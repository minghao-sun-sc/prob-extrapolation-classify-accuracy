import numpy as np
import torch
import gpytorch
import means
import priors

class GPCompositeKernel(gpytorch.models.ExactGP):
    """
    Gaussian Process with a composite kernel (RBF + Periodic + Matern)
    for potentially capturing both short and long-range dependencies
    """
    def __init__(self, X, y, likelihood, epsilon_min=0.01, with_priors=True,
                 periodic_period=1.0, matern_nu=2.5):
        super(GPCompositeKernel, self).__init__(X, y, likelihood)
        
        # Mean module
        self.mean_module = means.PowerLawPriorMean(torch.max(y).item(), epsilon_min)
        
        # Create composite kernel - combine RBF, Periodic, and Matern
        self.rbf_kernel = gpytorch.kernels.RBFKernel()
        self.periodic_kernel = gpytorch.kernels.PeriodicKernel(period_length_constraint=gpytorch.constraints.Positive())
        
        if matern_nu == 0.5:
            self.matern_kernel = gpytorch.kernels.MaternKernel(nu=0.5)
        elif matern_nu == 1.5:
            self.matern_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        else:
            self.matern_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
            
        # Set initial period length
        self.periodic_kernel.period_length = torch.tensor(periodic_period)
        
        # Combine kernels (additive composition)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.rbf_kernel + self.periodic_kernel + self.matern_kernel
        )
        
        # Register priors if requested
        if with_priors:
            # Noise prior
            self.register_prior(
                'noise_prior', 
                priors.TruncatedNormalPrior(0, np.inf, 0.01, 0.01), 
                lambda module: module.likelihood.noise.sqrt()
            )
            
            # Kernel priors
            self.register_prior(
                'rbf_lengthscale_prior',
                gpytorch.priors.LogNormalPrior(0.0, 1.0),
                lambda module: module.rbf_kernel.lengthscale
            )
            
            self.register_prior(
                'periodic_lengthscale_prior',
                gpytorch.priors.LogNormalPrior(0.0, 1.0),
                lambda module: module.periodic_kernel.lengthscale
            )
            
            self.register_prior(
                'matern_lengthscale_prior',
                gpytorch.priors.LogNormalPrior(0.0, 1.0),
                lambda module: module.matern_kernel.lengthscale
            )
            
            # Set prior on period length to reflect data pattern
            self.register_prior(
                'period_length_prior',
                gpytorch.priors.LogNormalPrior(np.log(periodic_period), 0.5),
                lambda module: module.periodic_kernel.period_length
            )
            
            # Outputscale prior
            self.register_prior(
                'outputscale_prior',
                gpytorch.priors.GammaPrior(2.0, 0.15),
                lambda module: module.covar_module.outputscale
            )
            
            # Epsilon prior
            self.register_prior(
                'epsilon_prior',
                priors.UniformPrior(self.mean_module.epsilon_min, 1.0-self.mean_module.y_max),
                lambda module: module.mean_module.epsilon_min + 
                              (1.0-module.mean_module.y_max-module.mean_module.epsilon_min)*
                              torch.sigmoid(module.mean_module.epsilon)
            )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        # Using log transformation which is common for this data pattern
        x_transformed = torch.log10(x)
        covar_x = self.covar_module(x_transformed)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPSpectralMixture(gpytorch.models.ExactGP):
    """
    Gaussian Process with Spectral Mixture kernel for flexible pattern modeling
    """
    def __init__(self, X, y, likelihood, epsilon_min=0.01, with_priors=True, num_mixtures=4):
        super(GPSpectralMixture, self).__init__(X, y, likelihood)
        
        # Mean module
        self.mean_module = means.PowerLawPriorMean(torch.max(y).item(), epsilon_min)
        
        # Spectral Mixture kernel
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=num_mixtures
        )
        
        # Initialize mixture parameters sensibly
        # This is important for SM kernels to converge well
        self.covar_module.initialize_from_data(X, y)
        
        # Register priors if requested
        if with_priors:
            # Noise prior
            self.register_prior(
                'noise_prior', 
                priors.TruncatedNormalPrior(0, np.inf, 0.01, 0.01), 
                lambda module: module.likelihood.noise.sqrt()
            )
            
            # Epsilon prior
            self.register_prior(
                'epsilon_prior',
                priors.UniformPrior(self.mean_module.epsilon_min, 1.0-self.mean_module.y_max),
                lambda module: module.mean_module.epsilon_min + 
                              (1.0-module.mean_module.y_max-module.mean_module.epsilon_min)*
                              torch.sigmoid(module.mean_module.epsilon)
            )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        # Using log transformation
        x_transformed = torch.log10(x)
        covar_x = self.covar_module(x_transformed)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPDeepKernel(gpytorch.models.ExactGP):
    """
    Gaussian Process with a Deep Kernel - combines neural networks with GPs
    """
    def __init__(self, X, y, likelihood, epsilon_min=0.01, with_priors=True, hidden_dims=[32, 16]):
        super(GPDeepKernel, self).__init__(X, y, likelihood)
        
        # Mean module
        self.mean_module = means.PowerLawPriorMean(torch.max(y).item(), epsilon_min)
        
        # Feature extractor - a simple MLP
        modules = []
        input_dim = 1  # For our single feature
        
        for output_dim in hidden_dims:
            modules.append(torch.nn.Linear(input_dim, output_dim))
            modules.append(torch.nn.ReLU())
            input_dim = output_dim
            
        self.feature_extractor = torch.nn.Sequential(*modules)
        
        # Base kernel operates on extracted features
        self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=hidden_dims[-1])
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        
        # Register priors if requested
        if with_priors:
            # Noise prior
            self.register_prior(
                'noise_prior', 
                priors.TruncatedNormalPrior(0, np.inf, 0.01, 0.01), 
                lambda module: module.likelihood.noise.sqrt()
            )
            
            # Outputscale prior
            self.register_prior(
                'outputscale_prior',
                gpytorch.priors.GammaPrior(2.0, 0.15),
                lambda module: module.covar_module.outputscale
            )
            
            # Epsilon prior
            self.register_prior(
                'epsilon_prior',
                priors.UniformPrior(self.mean_module.epsilon_min, 1.0-self.mean_module.y_max),
                lambda module: module.mean_module.epsilon_min + 
                              (1.0-module.mean_module.y_max-module.mean_module.epsilon_min)*
                              torch.sigmoid(module.mean_module.epsilon)
            )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        
        # Log transform the input
        x_log = torch.log10(x).view(-1, 1)
        
        # Extract features using the neural network
        projected_x = self.feature_extractor(x_log)
        
        # Apply GP kernel to the projected features
        covar_x = self.covar_module(projected_x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_composite_kernel(X, y, epsilon_min=0.01, lr=0.01, training_iter=50000, 
                          periodic_period=1.0, matern_nu=2.5):
    """Train a GP with composite kernel"""
    losses = np.zeros(training_iter)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.train()
    
    model = GPCompositeKernel(X, y, likelihood, epsilon_min=epsilon_min, 
                              with_priors=True, periodic_period=periodic_period,
                              matern_nu=matern_nu)
    
    model = model.to(device)
    likelihood = likelihood.to(device)
    model.train()
    
    # Use AdamW for better regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=1000,
                                                          min_lr=1e-5)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        if device.type == "cuda": X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        if device.type == "cuda": loss = loss.cpu()
        losses[i] = loss
    
    if device.type == "cuda":
        model = model.to('cpu')
        likelihood = likelihood.to('cpu')
    
    model.eval()
    likelihood.eval()
    return likelihood, model, losses


def train_spectral_mixture(X, y, epsilon_min=0.01, lr=0.01, training_iter=50000, num_mixtures=4):
    """Train a GP with spectral mixture kernel"""
    losses = np.zeros(training_iter)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.train()
    
    model = GPSpectralMixture(X, y, likelihood, epsilon_min=epsilon_min, 
                             with_priors=True, num_mixtures=num_mixtures)
    
    model = model.to(device)
    likelihood = likelihood.to(device)
    model.train()
    
    # Use AdamW with warm-up and restart for spectral mixture kernels
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # LR warm-up is helpful for spectral mixture kernels
    def lr_lambda(epoch):
        if epoch < 1000:  # Warm-up phase
            return epoch / 1000
        else:
            # Cosine annealing
            return 0.5 * (1 + np.cos(np.pi * (epoch - 1000) / (training_iter - 1000)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        if device.type == "cuda": X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if device.type == "cuda": loss = loss.cpu()
        losses[i] = loss
    
    if device.type == "cuda":
        model = model.to('cpu')
        likelihood = likelihood.to('cpu')
    
    model.eval()
    likelihood.eval()
    return likelihood, model, losses


def train_deep_kernel(X, y, epsilon_min=0.01, lr=0.01, training_iter=50000, hidden_dims=[32, 16]):
    """Train a GP with deep kernel"""
    losses = np.zeros(training_iter)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.train()
    
    model = GPDeepKernel(X, y, likelihood, epsilon_min=epsilon_min, 
                        with_priors=True, hidden_dims=hidden_dims)
    
    model = model.to(device)
    likelihood = likelihood.to(device)
    model.train()
    
    # Different optimizer setup for neural networks
    # Separate parameter groups for neural network and GP
    feature_params = [p for n, p in model.named_parameters() if 'feature_extractor' in n]
    gp_params = [p for n, p in model.named_parameters() if 'feature_extractor' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': feature_params, 'lr': lr, 'weight_decay': 1e-4},
        {'params': gp_params, 'lr': lr/10}  # Lower LR for GP parameters
    ])
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        if device.type == "cuda": X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        
        if device.type == "cuda": loss = loss.cpu()
        losses[i] = loss
    
    if device.type == "cuda":
        model = model.to('cpu')
        likelihood = likelihood.to('cpu')
    
    model.eval()
    likelihood.eval()
    return likelihood, model, losses 