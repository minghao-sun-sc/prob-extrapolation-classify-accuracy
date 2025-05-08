#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import sys
sys.path.append('src')

# Import custom modules
from . import initial_models
from src import means
from src import priors
from src import matern_kernels

# Directory setup
OUTPUT_DIR = "output/architecture_generalization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def power_law_function(x, a, b, c):
    """Power law function: a - b/x^c"""
    return a - b * np.power(x, -c)

def fit_power_law(x, y):
    """Fit a power law to the data"""
    # Initial guess for parameters (a, b, c)
    p0 = [0.9, 0.3, 0.5]
    
    # Fit curve
    popt, _ = curve_fit(power_law_function, x, y, p0=p0, bounds=([0.5, 0, 0], [1.0, 1.0, 1.0]))
    
    return popt

def simulate_architecture_data():
    """
    Simulate learning curves for different architectures.
    
    This simulates as if we had trained different model architectures
    on increasing dataset sizes and measured their performance.
    """
    # Define architectures with their power law parameters
    architecture_params = {
        'CNN': {'a': 0.92, 'b': 0.3, 'c': 0.5},
        'ViT': {'a': 0.95, 'b': 0.25, 'c': 0.55},
        'ResNet': {'a': 0.93, 'b': 0.28, 'c': 0.52},
        'MLP': {'a': 0.87, 'b': 0.33, 'c': 0.45}
    }
    
    # Create dataset sizes
    dataset_sizes = np.logspace(2, 5, 20)  # 100 to 100,000
    
    # Generate data for each architecture
    architecture_data = {}
    
    for arch_name, params in architecture_params.items():
        # Calculate AUROC values using power law
        auroc = params['a'] - params['b'] * np.power(dataset_sizes, -params['c'])
        
        # Add some random noise to make it realistic
        np.random.seed(42 + hash(arch_name) % 1000)
        noisy_auroc = auroc + np.random.normal(0, 0.01, len(dataset_sizes))
        
        # Store data
        architecture_data[arch_name] = {
            'train_size': dataset_sizes,
            'auroc': noisy_auroc,
            'true_params': params
        }
    
    return architecture_data

def test_apex_gp_on_architectures():
    """
    Test whether APEx-GP can accurately model and predict performance
    for different neural network architectures.
    """
    # Simulate data for different architectures
    architecture_data = simulate_architecture_data()
    
    # Define different pilot sizes to test
    pilot_sizes = [100, 500, 1000, 5000]
    
    # Test APEx-GP on each architecture
    results = {}
    
    for arch_name, data in architecture_data.items():
        arch_results = []
        
        # For each pilot size
        for pilot_size in pilot_sizes:
            # Get pilot data (simulate small dataset)
            pilot_indices = data['train_size'] <= pilot_size
            X_train = torch.tensor(data['train_size'][pilot_indices], dtype=torch.float32)
            y_train = torch.tensor(data['auroc'][pilot_indices], dtype=torch.float32)
            
            # Create testing data (all sizes, for evaluating predictions)
            X_test = torch.tensor(data['train_size'], dtype=torch.float32)
            y_test = torch.tensor(data['auroc'], dtype=torch.float32)
            
            # Scale the inputs for better numerical stability
            X_train_scaled = X_train / 1000.0
            X_test_scaled = X_test / 1000.0
            
            # Initialize APEx-GP models
            
            # 1. Standard RBF kernel
            likelihood_rbf = gpytorch.likelihoods.GaussianLikelihood()
            model_rbf = initial_models.GPPowerLaw(X_train_scaled, y_train, likelihood_rbf, epsilon_min=0.05, with_priors=True)
            
            # 2. Matérn 2.5 kernel
            likelihood_matern = gpytorch.likelihoods.GaussianLikelihood()
            model_matern = matern_kernels.GPPowerLawMatern(X_train_scaled, y_train, likelihood_matern, epsilon_min=0.05, with_priors=True, nu=2.5)
            
            # Train the models
            print(f"Training models for {arch_name} with pilot size {pilot_size}...")
            
            # Train RBF model
            model_rbf.train()
            likelihood_rbf.train()
            optimizer = torch.optim.Adam(model_rbf.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_rbf, model_rbf)
            
            for i in range(100):
                optimizer.zero_grad()
                output = model_rbf(X_train_scaled)
                loss = -mll(output, y_train)
                loss.backward()
                optimizer.step()
            
            # Train Matérn model
            model_matern.train()
            likelihood_matern.train()
            optimizer = torch.optim.Adam(model_matern.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_matern, model_matern)
            
            for i in range(100):
                optimizer.zero_grad()
                output = model_matern(X_train_scaled)
                loss = -mll(output, y_train)
                loss.backward()
                optimizer.step()
            
            # Set models to evaluation mode
            model_rbf.eval()
            likelihood_rbf.eval()
            model_matern.eval()
            likelihood_matern.eval()
            
            # Make predictions
            with torch.no_grad():
                # RBF predictions
                preds_rbf = likelihood_rbf(model_rbf(X_test_scaled))
                mean_rbf = preds_rbf.mean.numpy()
                std_rbf = preds_rbf.stddev.numpy()
                
                # Matérn predictions
                preds_matern = likelihood_matern(model_matern(X_test_scaled))
                mean_matern = preds_matern.mean.numpy()
                std_matern = preds_matern.stddev.numpy()
            
            # Calculate errors
            rbf_mse = np.mean((mean_rbf - y_test.numpy())**2)
            matern_mse = np.mean((mean_matern - y_test.numpy())**2)
            
            # Store results
            arch_results.append({
                'pilot_size': pilot_size,
                'rbf_mse': rbf_mse,
                'matern_mse': matern_mse,
                'rbf_mean': mean_rbf,
                'matern_mean': mean_matern,
                'rbf_std': std_rbf,
                'matern_std': std_matern,
                'true_values': y_test.numpy(),
                'train_sizes': X_test.numpy()
            })
        
        results[arch_name] = arch_results
    
    return results, architecture_data

def visualize_results(results, architecture_data):
    """Visualize the results of architecture generalization testing"""
    
    # 1. Create learning curves with predictions for each architecture
    for arch_name, arch_results in results.items():
        for result in arch_results:
            pilot_size = result['pilot_size']
            
            plt.figure(figsize=(14, 8))
            
            # Plot true learning curve
            plt.plot(result['train_sizes'], result['true_values'], 'k-', label='True learning curve', linewidth=3)
            
            # Plot pilot data
            pilot_indices = result['train_sizes'] <= pilot_size
            plt.scatter(result['train_sizes'][pilot_indices], result['true_values'][pilot_indices], 
                      c='r', s=80, zorder=10, label=f'Pilot data (n≤{pilot_size})')
            
            # Plot predictions
            plt.plot(result['train_sizes'], result['rbf_mean'], 'b--', label='APEx-GP (RBF)', linewidth=2)
            plt.fill_between(result['train_sizes'], result['rbf_mean'] - 2*result['rbf_std'], 
                           result['rbf_mean'] + 2*result['rbf_std'], alpha=0.1, color='b')
            
            plt.plot(result['train_sizes'], result['matern_mean'], 'g--', label='APEx-GP (Matérn)', linewidth=2)
            plt.fill_between(result['train_sizes'], result['matern_mean'] - 2*result['matern_std'], 
                           result['matern_mean'] + 2*result['matern_std'], alpha=0.1, color='g')
            
            plt.xscale('log')
            plt.xlabel('Training Set Size', fontsize=14)
            plt.ylabel('AUROC', fontsize=14)
            plt.title(f'APEx-GP Performance on {arch_name} Architecture (Pilot Size: {pilot_size})', fontsize=16)
            plt.ylim([0.5, 1.0])
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            plt.savefig(os.path.join(OUTPUT_DIR, f"{arch_name}_pilot{pilot_size}.png"), dpi=300)
            plt.close()
    
    # 2. Create MSE comparison across architectures and pilot sizes
    arch_names = list(results.keys())
    pilot_sizes = [r['pilot_size'] for r in results[arch_names[0]]]
    
    # Prepare data for visualization
    rbf_mse_data = np.zeros((len(arch_names), len(pilot_sizes)))
    matern_mse_data = np.zeros((len(arch_names), len(pilot_sizes)))
    
    for i, arch_name in enumerate(arch_names):
        for j, result in enumerate(results[arch_name]):
            rbf_mse_data[i, j] = result['rbf_mse']
            matern_mse_data[i, j] = result['matern_mse']
    
    # Plot RBF MSE by architecture and pilot size
    plt.figure(figsize=(14, 8))
    
    for i, arch_name in enumerate(arch_names):
        plt.plot(pilot_sizes, rbf_mse_data[i, :], 'o-', label=arch_name)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Pilot Dataset Size', fontsize=14)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
    plt.title('APEx-GP (RBF) Performance Across Different Architectures', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "rbf_mse_by_architecture.png"), dpi=300)
    plt.close()
    
    # Plot Matérn MSE by architecture and pilot size
    plt.figure(figsize=(14, 8))
    
    for i, arch_name in enumerate(arch_names):
        plt.plot(pilot_sizes, matern_mse_data[i, :], 'o-', label=arch_name)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Pilot Dataset Size', fontsize=14)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
    plt.title('APEx-GP (Matérn) Performance Across Different Architectures', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "matern_mse_by_architecture.png"), dpi=300)
    plt.close()
    
    # 3. Compare prediction at large dataset size (100,000)
    large_size_idx = -1  # Last index corresponds to largest size
    
    # Prepare data for visualization
    large_size_data = {
        'Architecture': [],
        'True_AUROC': [],
        'RBF_Predicted': [],
        'Matern_Predicted': [],
        'RBF_Error': [],
        'Matern_Error': []
    }
    
    for arch_name, arch_results in results.items():
        for pilot_size_idx, result in enumerate(arch_results):
            if pilot_size_idx == len(arch_results) - 1:  # Only for largest pilot size
                large_size_data['Architecture'].append(arch_name)
                large_size_data['True_AUROC'].append(result['true_values'][large_size_idx])
                large_size_data['RBF_Predicted'].append(result['rbf_mean'][large_size_idx])
                large_size_data['Matern_Predicted'].append(result['matern_mean'][large_size_idx])
                large_size_data['RBF_Error'].append(result['rbf_mean'][large_size_idx] - result['true_values'][large_size_idx])
                large_size_data['Matern_Error'].append(result['matern_mean'][large_size_idx] - result['true_values'][large_size_idx])
    
    # Convert to DataFrame
    large_size_df = pd.DataFrame(large_size_data)
    
    # Save to CSV
    large_size_df.to_csv(os.path.join(OUTPUT_DIR, "large_size_predictions.csv"), index=False)
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(large_size_data['Architecture']))
    width = 0.25
    
    plt.bar(x - width, large_size_data['True_AUROC'], width, label='True AUROC', color='gray')
    plt.bar(x, large_size_data['RBF_Predicted'], width, label='RBF Prediction', color='blue', alpha=0.7)
    plt.bar(x + width, large_size_data['Matern_Predicted'], width, label='Matérn Prediction', color='green', alpha=0.7)
    
    plt.xlabel('Architecture', fontsize=14)
    plt.ylabel('AUROC at 100,000 Training Examples', fontsize=14)
    plt.title('APEx-GP Predictions at Large Dataset Size', fontsize=16)
    plt.xticks(x, large_size_data['Architecture'])
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "large_size_comparison.png"), dpi=300)
    plt.close()
    
    # 4. Visualization of prediction error by architecture
    plt.figure(figsize=(14, 8))
    
    plt.bar(x - width/2, large_size_data['RBF_Error'], width, label='RBF Error', color='blue', alpha=0.7)
    plt.bar(x + width/2, large_size_data['Matern_Error'], width, label='Matérn Error', color='green', alpha=0.7)
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Architecture', fontsize=14)
    plt.ylabel('Prediction Error at 100,000 Training Examples', fontsize=14)
    plt.title('APEx-GP Prediction Error by Architecture', fontsize=16)
    plt.xticks(x, large_size_data['Architecture'])
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "prediction_error_by_architecture.png"), dpi=300)
    plt.close()

def main():
    print("Testing APEx-GP on different neural network architectures...")
    results, architecture_data = test_apex_gp_on_architectures()
    
    print("Visualizing results...")
    visualize_results(results, architecture_data)
    
    print(f"Analysis complete! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 