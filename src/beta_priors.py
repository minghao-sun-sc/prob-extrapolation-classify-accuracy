import numpy as np
import torch
import gpytorch
import scipy.stats
import matplotlib.pyplot as plt
from scipy import stats
import torch
import gpytorch
from torch.distributions import Beta
import sys
sys.path.append('../src/')
from initial_models import GPPowerLaw
from matern_kernels import GPPowerLawMatern as GPMatern

class BetaPrior(gpytorch.priors.Prior):
    """
    Beta distribution prior for saturation parameters.
    
    Particularly useful for parameters constrained between 0 and 1,
    with more informed shape based on domain knowledge.
    """
    arg_constraints = {}  # Disable validation for simplicity
    
    def __init__(self, alpha, beta):
        """
        Initialize Beta prior.
        
        Args:
            alpha (float): First shape parameter of the Beta distribution
            beta (float): Second shape parameter of the Beta distribution
        """
        super(BetaPrior, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.distribution = scipy.stats.beta(alpha, beta)
        
    def forward(self, x):
        """
        Calculate the probability density at x.
        
        Args:
            x (torch.Tensor): Input value
        
        Returns:
            torch.Tensor: Probability density at x
        """
        if not (0 <= x <= 1):
            return torch.tensor(0.0)
        return torch.tensor(self.distribution.pdf(x.item()))
        
    def log_prob(self, x):
        """
        Calculate the log probability density at x.
        
        Args:
            x (torch.Tensor): Input value
        
        Returns:
            torch.Tensor: Log probability density at x
        """
        if not (0 <= x <= 1):
            return torch.tensor(-float('inf'))
        return torch.log(self.forward(x))

def plot_beta_prior(alpha, beta, title=None):
    """
    Plot a Beta distribution with given alpha and beta parameters.
    
    Args:
        alpha (float): The alpha parameter of the Beta distribution.
        beta (float): The beta parameter of the Beta distribution.
        title (str, optional): The title of the plot.
    """
    x = np.linspace(0, 1, 1000)
    y = stats.beta.pdf(x, alpha, beta)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'r-', lw=2, alpha=0.8)
    plt.fill_between(x, 0, y, alpha=0.2, color='r')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    if title is None:
        title = f'Beta({alpha}, {beta}) Distribution'
    plt.title(title, fontsize=14)
    
    plt.tight_layout()
    plt.show()

def truncated_beta_uncertainty(loc, scale, alpha, beta, lower_percentile=0.025, upper_percentile=0.975):
    """
    Calculate the uncertainty bounds using the Beta distribution.
    
    Args:
        loc (np.ndarray): The mean of the distribution.
        scale (np.ndarray): The standard deviation of the distribution.
        alpha (float): The alpha parameter of the Beta distribution.
        beta (float): The beta parameter of the Beta distribution.
        lower_percentile (float, optional): The lower percentile for the uncertainty bound.
        upper_percentile (float, optional): The upper percentile for the uncertainty bound.
    
    Returns:
        tuple: The lower and upper bounds of the uncertainty.
    """
    # Calculate the parameters of the Beta distribution
    # We use a method of moments approximation to convert (mean, std) to (alpha, beta)
    mean = loc
    var = scale**2
    
    # Prevent zero or negative variance which would cause division by zero
    var = np.maximum(var, 1e-10)
    
    # Convert from the Gaussian parameters to Beta parameters
    # Using the fact that for Beta distribution:
    # mean = alpha / (alpha + beta)
    # var = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))
    
    # Original alpha and beta are prior parameters
    # We want to combine with our posterior (loc, scale)
    # Simple approach: use mean and variance to calculate implied alpha' and beta'
    
    # From the mean and variance equations:
    # alpha' = mean * (mean * (1 - mean) / var - 1)
    # beta' = (1 - mean) * (mean * (1 - mean) / var - 1)
    
    # Ensure mean is within [0, 1] for Beta distribution
    mean = np.clip(mean, 0.01, 0.99)
    
    # Calculate effective alpha and beta
    a_factor = mean * (1 - mean) / var - 1
    alpha_effective = mean * a_factor
    beta_effective = (1 - mean) * a_factor
    
    # Add the prior alpha and beta (optional, simple addition approach)
    alpha_posterior = alpha_effective + alpha - 1  # -1 to avoid double counting
    beta_posterior = beta_effective + beta - 1  # -1 to avoid double counting
    
    # Ensure alpha and beta are positive
    alpha_posterior = np.maximum(alpha_posterior, 0.01)
    beta_posterior = np.maximum(beta_posterior, 0.01)
    
    # Calculate the bounds using the Beta distribution
    lower_bound = np.array([stats.beta.ppf(lower_percentile, a, b) 
                          for a, b in zip(alpha_posterior, beta_posterior)])
    upper_bound = np.array([stats.beta.ppf(upper_percentile, a, b) 
                          for a, b in zip(alpha_posterior, beta_posterior)])
    
    return lower_bound, upper_bound

class GPPowerLawBetaPrior(gpytorch.models.ExactGP):
    """
    Gaussian Process Power Law model with Beta prior for saturation parameter.
    
    Extends the standard GP model with a power law kernel and a Beta prior
    for the saturation parameter.
    """
    def __init__(self, x, y, likelihood, epsilon_min=0.01, with_priors=True, alpha=2, beta=2):
        """
        Initialize the GP model with a Beta prior.
        
        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The output tensor.
            likelihood (gpytorch.likelihoods.Likelihood): The likelihood function.
            epsilon_min (float, optional): The minimum value for epsilon.
            with_priors (bool, optional): Whether to use priors.
            alpha (float): Alpha parameter for the Beta prior.
            beta (float): Beta parameter for the Beta prior.
        """
        super(GPPowerLawBetaPrior, self).__init__(x, y, likelihood)
        
        # Store Beta prior parameters
        self.alpha = alpha
        self.beta = beta
        
        # The saturation parameter represents the maximum achievable performance
        self.epsilon_min = epsilon_min
        self.register_parameter(
            name="saturation_raw", 
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        
        # The decay parameter controls the learning rate
        self.register_parameter(
            name="decay", 
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        
        # Set up the kernel for the GP
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        # Register priors if requested
        if with_priors:
            # Use Beta prior for saturation parameter
            # We use the Beta transformation to ensure the saturation is between 0 and 1
            self.register_constraint("saturation_raw", gpytorch.constraints.Interval(-3, 3))
            self.register_prior(
                "saturation_prior",
                gpytorch.priors.NormalPrior(0, 1),
                lambda module: module.saturation_raw
            )
            
            # Decay should be positive
            self.register_constraint("decay", gpytorch.constraints.Positive())
            self.register_prior(
                "decay_prior",
                gpytorch.priors.GammaPrior(2, 0.15),
                lambda module: module.decay
            )
            
            # Kernel lengthscale
            self.register_prior(
                "lengthscale_prior",
                gpytorch.priors.GammaPrior(2.0, 0.3),
                lambda module: module.covar_module.base_kernel.lengthscale
            )
            
            # Kernel outputscale
            self.register_prior(
                "outputscale_prior",
                gpytorch.priors.GammaPrior(2.0, 0.3),
                lambda module: module.covar_module.outputscale
            )
        
    def forward(self, x):
        """
        Forward pass of the GP model.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            gpytorch.distributions.MultivariateNormal: The output distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    @property
    def saturation(self):
        """
        Get the saturation parameter, transformed using the Beta CDF.
        
        Returns:
            torch.Tensor: The saturation parameter.
        """
        # Transform the raw parameter through a sigmoid scaled to [0, 1-epsilon_min]
        # and then add epsilon_min to ensure it's in [epsilon_min, 1]
        # Using Beta CDF transformation approximated with sigmoid
        
        # Convert raw parameter to a value between 0 and 1
        sigmoid_value = torch.sigmoid(self.saturation_raw)
        
        # Apply Beta CDF transformation (approximated with scaled sigmoid)
        # This transforms according to the Beta prior's shape
        
        # For simplicity, we're using a scaled sigmoid here, but a true Beta CDF
        # transformation would be more accurate for matching the Beta prior
        scaled_value = self.epsilon_min + (1 - self.epsilon_min) * sigmoid_value
        
        return scaled_value 

class GPPowerLawBetaPriorWithLog(gpytorch.models.ExactGP):
    """
    Gaussian Process Power Law model with Beta prior for saturation parameter.
    Uses log-transformed inputs for potentially better extrapolation.
    """
    def __init__(self, x, y, likelihood, epsilon_min=0.01, with_priors=True, alpha=2, beta=2, 
                 log_transform=True, output_scale_prior_mean=1.0, output_scale_prior_variance=0.5):
        """
        Initialize the GP model with a Beta prior and log transform option.
        
        Args:
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The output tensor.
            likelihood (gpytorch.likelihoods.Likelihood): The likelihood function.
            epsilon_min (float, optional): The minimum value for epsilon.
            with_priors (bool, optional): Whether to use priors.
            alpha (float): Alpha parameter for the Beta prior.
            beta (float): Beta parameter for the Beta prior.
            log_transform (bool): Whether to use log transform for inputs.
            output_scale_prior_mean (float): Mean for output scale prior
            output_scale_prior_variance (float): Variance for output scale prior
        """
        super(GPPowerLawBetaPriorWithLog, self).__init__(x, y, likelihood)
        
        # Store Beta prior parameters
        self.alpha = alpha
        self.beta = beta
        self.log_transform = log_transform
        
        # The saturation parameter represents the maximum achievable performance
        self.epsilon_min = epsilon_min
        self.register_parameter(
            name="saturation_raw", 
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        
        # The decay parameter controls the learning rate
        self.register_parameter(
            name="decay", 
            parameter=torch.nn.Parameter(torch.zeros(1))
        )
        
        # Set up the kernel for the GP
        self.mean_module = gpytorch.means.ConstantMean()
        # Spectral Mixture Kernel might capture more complex patterns
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        # Register priors if requested
        if with_priors:
            # Use Beta prior for saturation parameter
            # We use the Beta transformation to ensure the saturation is between 0 and 1
            self.register_constraint("saturation_raw", gpytorch.constraints.Interval(-3, 3))
            self.register_prior(
                "saturation_prior",
                BetaPrior(alpha, beta),  # Use our custom Beta prior
                lambda module: module.saturation  # Apply to the saturation property
            )
            
            # Decay should be positive
            self.register_constraint("decay", gpytorch.constraints.Positive())
            self.register_prior(
                "decay_prior",
                gpytorch.priors.GammaPrior(3, 0.1),  # Different shape/rate
                lambda module: module.decay
            )
            
            # Kernel lengthscale with more informative prior
            self.register_prior(
                "lengthscale_prior",
                gpytorch.priors.LogNormalPrior(0.0, 1.0),  # Log-normal prior
                lambda module: module.covar_module.base_kernel.lengthscale
            )
            
            # Kernel outputscale with more informative prior
            self.register_prior(
                "outputscale_prior",
                gpytorch.priors.GammaPrior(output_scale_prior_mean**2 / output_scale_prior_variance,
                                          output_scale_prior_mean / output_scale_prior_variance),
                lambda module: module.covar_module.outputscale
            )
        
    def forward(self, x):
        """
        Forward pass of the GP model.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            gpytorch.distributions.MultivariateNormal: The output distribution.
        """
        mean_x = self.mean_module(x)
        
        # Apply log transform if specified
        if self.log_transform:
            x_transformed = torch.log10(x)
        else:
            x_transformed = x
            
        covar_x = self.covar_module(x_transformed)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    @property
    def saturation(self):
        """
        Get the saturation parameter, transformed using the Beta CDF.
        
        Returns:
            torch.Tensor: The saturation parameter.
        """
        # Transform the raw parameter to match Beta distribution
        sigmoid_value = torch.sigmoid(self.saturation_raw)
        
        # Create a more informed transformation based on Beta prior
        # This is a simplified approximation of the Beta CDF
        scaled_value = self.epsilon_min + (1 - self.epsilon_min) * sigmoid_value
        
        return scaled_value
        
def train_gp_with_beta_prior(X, y, alpha=2, beta=2, epsilon_min=0.01, lr=0.01, 
                             training_iter=50000, log_transform=True, 
                             output_scale_prior_mean=1.0, output_scale_prior_variance=0.5):
    """
    Train a GP model with Beta prior and various hyperparameter configurations
    
    Args:
        X (torch.Tensor): Input data
        y (torch.Tensor): Target values
        alpha (float): Alpha parameter of Beta prior
        beta (float): Beta parameter of Beta prior
        epsilon_min (float): Minimum value for epsilon
        lr (float): Learning rate
        training_iter (int): Number of training iterations
        log_transform (bool): Whether to use log transform
        output_scale_prior_mean (float): Mean for output scale prior
        output_scale_prior_variance (float): Variance for output scale prior
        
    Returns:
        tuple: Likelihood, model, and losses
    """
    losses = np.zeros(training_iter)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.train()
    
    # Initialize with custom hyperparameters
    model = GPPowerLawBetaPriorWithLog(X, y, likelihood, epsilon_min=epsilon_min, 
                                    with_priors=True, alpha=alpha, beta=beta,
                                    log_transform=log_transform,
                                    output_scale_prior_mean=output_scale_prior_mean,
                                    output_scale_prior_variance=output_scale_prior_variance)
    
    # Move to device
    model = model.to(device)
    likelihood = likelihood.to(device)
    model.train()
    
    # Use different optimizer or learning rate schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_iter, eta_min=1e-5)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        if device.type == "cuda": X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        if device.type == "cuda": loss = loss.cpu()
        losses[i] = loss
        
    if device.type == "cuda":
        model = model.to('cpu')
        likelihood = likelihood.to('cpu')
        
    model.eval()
    likelihood.eval()
    return likelihood, model, losses


class BetaPrior(gpytorch.priors.Prior):
    """Beta prior distribution for parameters that are bounded between 0 and 1"""
    def __init__(self, alpha, beta, validate_args=None):
        self.alpha = alpha
        self.beta = beta
        self._beta_dist = Beta(alpha, beta, validate_args=validate_args)
        super().__init__(validate_args=validate_args)
    
    def log_prob(self, x):
        return self._beta_dist.log_prob(x)
    
    def sample(self, sample_shape=torch.Size()):
        return self._beta_dist.sample(sample_shape)
    
    @property
    def _extended_shape(self):
        return torch.Size([])

class GPPowerLawWithBetaPrior(gpytorch.models.ExactGP):
    """Gaussian Process with Power Law mean function and Beta prior for saturation parameter"""
    def __init__(self, train_x, train_y, likelihood, epsilon_min=0.05, with_priors=True):
        super(GPPowerLawWithBetaPrior, self).__init__(train_x, train_y, likelihood)
        
        # Power law mean function parameters
        self.register_parameter(
            name="epsilon", 
            parameter=torch.nn.Parameter(torch.tensor(0.1))
        )
        self.register_parameter(
            name="alpha", 
            parameter=torch.nn.Parameter(torch.tensor(0.5))
        )
        self.register_parameter(
            name="saturation", 
            parameter=torch.nn.Parameter(torch.tensor(0.9))
        )
        
        # Covariance module
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=1)
        )
        
        # Constraints
        self.register_constraint("epsilon", gpytorch.constraints.Interval(epsilon_min, 1.0))
        self.register_constraint("alpha", gpytorch.constraints.Positive())
        self.register_constraint("saturation", gpytorch.constraints.Interval(0.5, 1.0))
        
        # Set priors if requested
        if with_priors:
            # Use Beta prior for saturation parameter (favoring higher values)
            self.saturation_prior = BetaPrior(8.0, 2.0)
            self.register_prior(
                "saturation_prior",
                self.saturation_prior,
                lambda m: m.saturation,
                lambda m, v: m._set_saturation(v)
            )
            
            # Other priors
            self.epsilon_prior = gpytorch.priors.GammaPrior(1.0, 10.0)
            self.register_prior(
                "epsilon_prior",
                self.epsilon_prior,
                lambda m: m.epsilon,
                lambda m, v: m._set_epsilon(v)
            )
            
            self.alpha_prior = gpytorch.priors.GammaPrior(1.0, 2.0)
            self.register_prior(
                "alpha_prior",
                self.alpha_prior,
                lambda m: m.alpha,
                lambda m, v: m._set_alpha(v)
            )
            
            # Kernel priors
            self.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
            self.covar_module.outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
    
    def _set_epsilon(self, value):
        self.initialize(epsilon=value)
    
    def _set_alpha(self, value):
        self.initialize(alpha=value)
    
    def _set_saturation(self, value):
        self.initialize(saturation=value)
    
    def forward(self, x):
        # Power law mean function: saturation - epsilon * (x ** -alpha)
        mean_x = self.saturation - self.epsilon * (x ** -self.alpha)
        
        # Covariance
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPArctanWithBetaPrior(gpytorch.models.ExactGP):
    """Gaussian Process with Arctan mean function and Beta prior for saturation parameter"""
    def __init__(self, train_x, train_y, likelihood, epsilon_min=0.05, with_priors=True):
        super(GPArctanWithBetaPrior, self).__init__(train_x, train_y, likelihood)
        
        # Arctan mean function parameters
        self.register_parameter(
            name="epsilon", 
            parameter=torch.nn.Parameter(torch.tensor(0.1))
        )
        self.register_parameter(
            name="alpha", 
            parameter=torch.nn.Parameter(torch.tensor(0.5))
        )
        self.register_parameter(
            name="saturation", 
            parameter=torch.nn.Parameter(torch.tensor(0.9))
        )
        
        # Covariance module
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=1)
        )
        
        # Constraints
        self.register_constraint("epsilon", gpytorch.constraints.Interval(epsilon_min, 1.0))
        self.register_constraint("alpha", gpytorch.constraints.Positive())
        self.register_constraint("saturation", gpytorch.constraints.Interval(0.5, 1.0))
        
        # Set priors if requested
        if with_priors:
            # Use Beta prior for saturation parameter (favoring higher values)
            self.saturation_prior = BetaPrior(8.0, 2.0)
            self.register_prior(
                "saturation_prior",
                self.saturation_prior,
                lambda m: m.saturation,
                lambda m, v: m._set_saturation(v)
            )
            
            # Other priors
            self.epsilon_prior = gpytorch.priors.GammaPrior(1.0, 10.0)
            self.register_prior(
                "epsilon_prior",
                self.epsilon_prior,
                lambda m: m.epsilon,
                lambda m, v: m._set_epsilon(v)
            )
            
            self.alpha_prior = gpytorch.priors.GammaPrior(1.0, 2.0)
            self.register_prior(
                "alpha_prior",
                self.alpha_prior,
                lambda m: m.alpha,
                lambda m, v: m._set_alpha(v)
            )
            
            # Kernel priors
            self.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
            self.covar_module.outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
    
    def _set_epsilon(self, value):
        self.initialize(epsilon=value)
    
    def _set_alpha(self, value):
        self.initialize(alpha=value)
    
    def _set_saturation(self, value):
        self.initialize(saturation=value)
    
    def forward(self, x):
        # Arctan mean function: saturation - epsilon * (2/pi) * arctan(alpha/x)
        mean_x = self.saturation - self.epsilon * (2/torch.pi) * torch.atan(self.alpha/x)
        
        # Covariance
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Your data points
X = torch.tensor([60, 94, 147, 230, 360])
y = torch.tensor([0.6067, 0.6240, 0.6429, 0.6574, 0.6664])

# Initialize the model with power law and standard priors
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPPowerLaw(X, y, likelihood, epsilon_min=0.05, with_priors=True)

# Train the model
likelihood, model, losses = models.train_gp(likelihood, model, X, y, max_iters=50000, lr=0.01)

# Make predictions
with torch.no_grad():
    test_x = torch.linspace(50, 30000, 29950).long()
    predictions = likelihood(model(test_x))
    mean = predictions.mean.numpy()
    std = predictions.stddev.numpy()
    
    # Use beta priors for better bounded predictions (between 0 and 1)
    lower, upper = beta_priors_uncertainty(mean, std, lower_percentile=0.025, upper_percentile=0.975)

# Standard model
model_standard = GPPowerLaw(X, y, likelihood, epsilon_min=0.05, with_priors=True)
likelihood_standard, model_standard, _ = models.train_gp(likelihood, model_standard, X, y)

# Matérn kernel model
model_matern = GPMatern(X, y, likelihood, epsilon_min=0.05, with_priors=True)
likelihood_matern, model_matern, _ = models.train_gp(likelihood, model_matern, X, y)

# Compare predictions
with torch.no_grad():
    test_x = torch.linspace(50, 30000, 29950).long()
    
    # Standard model predictions
    predictions_standard = likelihood_standard(model_standard(test_x))
    mean_standard = predictions_standard.mean.numpy()
    std_standard = predictions_standard.stddev.numpy()
    lower_standard, upper_standard = priors.truncated_normal_uncertainty(0.0, 1.0, mean_standard, std_standard, 0.025, 0.975)
    
    # Matérn model predictions
    predictions_matern = likelihood_matern(model_matern(test_x))
    mean_matern = predictions_matern.mean.numpy()
    std_matern = predictions_matern.stddev.numpy()
    lower_matern, upper_matern = priors.truncated_normal_uncertainty(0.0, 1.0, mean_matern, std_matern, 0.025, 0.975)