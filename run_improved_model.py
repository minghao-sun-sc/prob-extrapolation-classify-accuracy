#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pandas as pd
from scipy import stats

# Import the custom modules
import sys
sys.path.append('src')
# Import modules from src directory
from src import models
from src import means 
from src import priors
from src import matern_kernels

# Set up the directory paths
DATA_DIR = "prob-extrapolation-classify-accuracy/dataset"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define a custom class for Beta priors implementation to avoid circular imports
class GPPowerLawBeta(gpytorch.models.ExactGP):
    """
    Gaussian Process with Power Law mean function and Beta priors for bounded outputs.
    
    This implementation constrains the output to be between 0 and 1, which is appropriate
    for accuracy metrics like AUROC.
    """
    def __init__(self, X, y, likelihood, epsilon_min=0.05, with_priors=True):
        super(GPPowerLawBeta, self).__init__(X, y, likelihood)
        # Mean module
        self.mean_module = means.PowerLawPriorMean(torch.max(y).item(), epsilon_min)
        # Covariance module
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # For Beta distribution output
        self.register_parameter("a_param", torch.nn.Parameter(torch.ones(1) * 10.0))
        self.register_parameter("b_param", torch.nn.Parameter(torch.ones(1) * 10.0))
        
        if with_priors:
            # Register priors for kernel parameters
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
        # Return MultivariteNormal distribution
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def log_marginal_likelihood(self, y):
        """
        Compute log marginal likelihood with Beta distribution
        """
        # Get the standard Gaussian GP distribution
        gp_dist = self(self.train_inputs[0])
        mean = gp_dist.mean
        var = gp_dist.variance
        
        # Transform to Beta parameters
        # Ensure mean is within (0,1)
        mean = torch.clamp(mean, 0.01, 0.99)
        # Ensure variance is positive but small
        var = torch.clamp(var, 1e-6, 0.1)
        
        # Convert to Beta parameters via method of moments
        a = mean * (mean * (1 - mean) / var - 1)
        b = (1 - mean) * (mean * (1 - mean) / var - 1)
        
        # Ensure a, b are positive
        a = torch.abs(a) + 0.1
        b = torch.abs(b) + 0.1
        
        # Store for later use in prediction
        self.a = a
        self.b = b
        
        # Compute log probability under Beta distribution
        from torch.distributions import Beta
        beta_dist = Beta(a, b)
        log_prob = beta_dist.log_prob(y)
        
        return log_prob.sum()

def load_data(dataset_name):
    """Load training data from a specified dataset"""
    # Read performance data at different training set sizes
    try:
        data_path = os.path.join(DATA_DIR, f"{dataset_name}_performance.csv")
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            X = torch.tensor(data['train_size'].values, dtype=torch.float32)
            y = torch.tensor(data['auroc'].values, dtype=torch.float32)
            return X, y
        else:
            print(f"Dataset file {data_path} not found.")
            # List available datasets
            available_files = [f for f in os.listdir(DATA_DIR) if f.endswith('_performance.csv')]
            print(f"Available datasets: {available_files}")
            return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def train_and_evaluate_models(X, y, dataset_name):
    """Train and evaluate both standard and improved models"""
    
    # Make sure we have some data to work with
    if X is None or y is None:
        print("No data available to train models.")
        return
    
    # Scale the inputs for better numerical stability
    X_scaled = X / 1000.0
    
    # Split data into training and testing (use first 70% for training)
    train_size = int(0.7 * len(X))
    X_train, y_train = X_scaled[:train_size], y[:train_size]
    X_test, y_test = X_scaled[train_size:], y[train_size:]
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Define the prediction range for extrapolation
    max_train_size = X.max().item()
    X_predict = torch.linspace(X.min().item(), max_train_size * 10, 100) / 1000.0
    
    # -----------------------------
    # 1. Train standard GP with RBF kernel and power law mean
    # -----------------------------
    likelihood_rbf = gpytorch.likelihoods.GaussianLikelihood()
    model_rbf = models.GPPowerLaw(X_train, y_train, likelihood_rbf, epsilon_min=0.05, with_priors=True)
    
    # Train the model
    print("Training standard model with RBF kernel...")
    model_rbf.train()
    likelihood_rbf.train()
    
    # Use Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model_rbf.parameters()},
    ], lr=0.1)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_rbf, model_rbf)
    
    training_iterations = 100
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model_rbf(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        if i % 20 == 0:
            print(f'Iter {i+1}/{training_iterations} - Loss: {loss.item():.3f}')
        optimizer.step()
    
    # -----------------------------
    # 2. Train improved GP with Matérn kernel and power law mean
    # -----------------------------
    likelihood_matern = gpytorch.likelihoods.GaussianLikelihood()
    
    # We use the Matérn 2.5 kernel which provides a good balance
    model_matern = matern_kernels.GPPowerLawMatern(X_train, y_train, likelihood_matern, epsilon_min=0.05, with_priors=True, nu=2.5)
    
    # Train the model
    print("Training improved model with Matérn 2.5 kernel...")
    model_matern.train()
    likelihood_matern.train()
    
    # Use Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model_matern.parameters()},
    ], lr=0.1)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_matern, model_matern)
    
    training_iterations = 100
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model_matern(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        if i % 20 == 0:
            print(f'Iter {i+1}/{training_iterations} - Loss: {loss.item():.3f}')
        optimizer.step()
    
    # -----------------------------
    # 3. Train improved GP with Matérn kernel and beta priors
    # -----------------------------
    likelihood_beta = gpytorch.likelihoods.GaussianLikelihood()
    
    # We use the Beta prior instead of normal prior for bounded output
    model_beta = GPPowerLawBeta(X_train, y_train, likelihood_beta, epsilon_min=0.05)
    
    # Train the model
    print("Training improved model with Beta priors...")
    model_beta.train()
    likelihood_beta.train()
    
    # Use Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model_beta.parameters()},
    ], lr=0.1)
    
    # Custom loss for Beta prior model
    def beta_loss(model, likelihood, x, y):
        output = model(x)
        return -model.log_marginal_likelihood(y)
    
    training_iterations = 100
    for i in range(training_iterations):
        optimizer.zero_grad()
        loss = beta_loss(model_beta, likelihood_beta, X_train, y_train)
        loss.backward()
        if i % 20 == 0:
            print(f'Iter {i+1}/{training_iterations} - Loss: {loss.item():.3f}')
        optimizer.step()
    
    # --------------------------
    # Evaluate the models
    # --------------------------
    model_rbf.eval()
    likelihood_rbf.eval()
    model_matern.eval()
    likelihood_matern.eval()
    model_beta.eval()
    likelihood_beta.eval()
    
    # Generate predictions for standard RBF model
    with torch.no_grad():
        predictions_rbf = likelihood_rbf(model_rbf(X_predict))
    mean_rbf = predictions_rbf.mean.numpy()
    std_rbf = predictions_rbf.stddev.numpy()
    lower_rbf, upper_rbf = priors.truncated_normal_uncertainty(0.0, 1.0, mean_rbf, std_rbf, lower_percentile=0.025, upper_percentile=0.975)
    
    # Generate predictions for Matérn kernel model
    with torch.no_grad():
        predictions_matern = likelihood_matern(model_matern(X_predict))
    mean_matern = predictions_matern.mean.numpy()
    std_matern = predictions_matern.stddev.numpy()
    lower_matern, upper_matern = priors.truncated_normal_uncertainty(0.0, 1.0, mean_matern, std_matern, lower_percentile=0.025, upper_percentile=0.975)
    
    # Generate predictions for Beta prior model
    with torch.no_grad():
        predictions_beta = model_beta(X_predict)
    mean_beta = predictions_beta.mean.numpy()
    
    # For the Beta model, we need to transform predictions
    gp_dist = model_beta(X_predict)
    mean = gp_dist.mean
    var = torch.clamp(gp_dist.variance, 1e-6, 0.1)
    
    # Convert to Beta parameters via method of moments
    mean = torch.clamp(mean, 0.01, 0.99)
    a = mean * (mean * (1 - mean) / var - 1)
    b = (1 - mean) * (mean * (1 - mean) / var - 1)
    
    # Ensure a, b are positive
    a = torch.abs(a) + 0.1
    b = torch.abs(b) + 0.1
    
    # Calculate confidence bounds using Beta distribution
    lower_beta = np.array([stats.beta.ppf(0.025, aa, bb) for aa, bb in zip(a.detach().numpy(), b.detach().numpy())])
    upper_beta = np.array([stats.beta.ppf(0.975, aa, bb) for aa, bb in zip(a.detach().numpy(), b.detach().numpy())])
    
    # Evaluate on test data
    with torch.no_grad():
        test_pred_rbf = likelihood_rbf(model_rbf(X_test)).mean.numpy()
        test_pred_matern = likelihood_matern(model_matern(X_test)).mean.numpy()
        test_pred_beta = model_beta(X_test).mean.numpy()
    
    # Calculate MSE for each model
    mse_rbf = np.mean((test_pred_rbf - y_test.numpy())**2)
    mse_matern = np.mean((test_pred_matern - y_test.numpy())**2)
    mse_beta = np.mean((test_pred_beta - y_test.numpy())**2)
    
    print(f"\nTest set mean squared error:")
    print(f"Standard RBF model: {mse_rbf:.6f}")
    print(f"Improved Matérn model: {mse_matern:.6f}")
    print(f"Improved Beta model: {mse_beta:.6f}")
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Plot training and test data
    plt.scatter(X_train.numpy() * 1000, y_train.numpy(), c='black', label='Training data', zorder=5)
    plt.scatter(X_test.numpy() * 1000, y_test.numpy(), c='red', marker='x', label='Test data', zorder=5)
    
    # Plot predictions with uncertainty
    X_plot = X_predict.numpy() * 1000
    
    # Standard model
    plt.plot(X_plot, mean_rbf, c='blue', label='Standard model (RBF kernel)')
    plt.fill_between(X_plot, lower_rbf, upper_rbf, alpha=0.1, color='blue')
    
    # Matérn model
    plt.plot(X_plot, mean_matern, c='green', label='Improved model (Matérn kernel)')
    plt.fill_between(X_plot, lower_matern, upper_matern, alpha=0.1, color='green')
    
    # Beta model
    plt.plot(X_plot, mean_beta, c='purple', label='Improved model (Beta priors)')
    plt.fill_between(X_plot, lower_beta, upper_beta, alpha=0.1, color='purple')
    
    plt.ylim([0.5, 1.0])
    plt.xlim([X_plot.min(), X_plot.max()])
    plt.xscale('log')
    plt.xlabel('Training set size', fontsize=14)
    plt.ylabel('Performance (AUROC)', fontsize=14)
    plt.title(f'Learning Curve Extrapolation for {dataset_name}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, f'{dataset_name}_extrapolation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the predictions to CSV
    predictions_df = pd.DataFrame({
        'train_size': X_plot,
        'standard_mean': mean_rbf,
        'standard_lower': lower_rbf,
        'standard_upper': upper_rbf,
        'matern_mean': mean_matern,
        'matern_lower': lower_matern,
        'matern_upper': upper_matern,
        'beta_mean': mean_beta,
        'beta_lower': lower_beta,
        'beta_upper': upper_beta
    })
    predictions_df.to_csv(os.path.join(OUTPUT_DIR, f'{dataset_name}_predictions.csv'), index=False)
    
    print(f"Results saved to {OUTPUT_DIR}/{dataset_name}_extrapolation.png and {OUTPUT_DIR}/{dataset_name}_predictions.csv")
    
    # Return results for potential further analysis
    return {
        'model_rbf': model_rbf,
        'model_matern': model_matern,
        'model_beta': model_beta,
        'mse_rbf': mse_rbf,
        'mse_matern': mse_matern,
        'mse_beta': mse_beta
    }

def main():
    # Check what datasets are available
    available_files = [f[:-16] for f in os.listdir(DATA_DIR) if f.endswith('_performance.csv')]
    print(f"Available datasets: {available_files}")
    
    if not available_files:
        print("No datasets found in the data directory. Please make sure the data is correctly placed.")
        return
    
    # Process each available dataset
    for dataset_name in available_files:
        print(f"\nProcessing dataset: {dataset_name}")
        X, y = load_data(dataset_name)
        if X is not None and y is not None:
            results = train_and_evaluate_models(X, y, dataset_name)
            print(f"Completed analysis for {dataset_name}\n")
        else:
            print(f"Skipping dataset {dataset_name} due to loading issues\n")

if __name__ == "__main__":
    main() 