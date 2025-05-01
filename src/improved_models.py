import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
import sys
import os
import glob

# Add the src directory to the Python path
sys.path.append(os.path.abspath('.'))

# Import project modules
import models
import beta_priors
import matern_kernels
import combo_kernels
from reproduce_paper import load_encoded_data, train_and_evaluate_model

# Set torch float precision
torch.set_default_dtype(torch.float64)

def compare_models(X, y, model_classes, model_names):
    """Compare multiple GP models on the same dataset"""
    results = {}
    
    for model_class, model_name in zip(model_classes, model_names):
        print(f"\nTraining {model_name}...")
        model, likelihood, mse = train_and_evaluate_model(X, y, model_class, model_name)
        results[model_name] = {"model": model, "likelihood": likelihood, "mse": mse}
    
    # Compare MSE values
    print("\nModel Comparison (MSE):")
    for name, result in results.items():
        print(f"{name}: {result['mse']:.6f}")
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda k: results[k]["mse"])
    print(f"\nBest model: {best_model_name} with MSE {results[best_model_name]['mse']:.6f}")
    
    return results

def plot_model_comparison(X, y, results):
    """Plot comparison of multiple models"""
    plt.figure(figsize=(15, 8))
    
    # Plot data points
    plt.scatter(X.numpy(), y.numpy(), c='black', label='Data points')
    
    # Create test points for extrapolation
    max_size = X.max().item()
    X_extrapolate = torch.logspace(np.log10(X.min().item()), np.log10(max_size * 10), 100)
    
    # Plot each model's predictions
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (name, result) in enumerate(results.items()):
        model = result["model"]
        likelihood = result["likelihood"]
        
        model.eval()
        likelihood.eval()
        
        with torch.no_grad():
            predictions = likelihood(model(X_extrapolate))
            mean = predictions.mean.numpy()
            lower, upper = predictions.confidence_region()
            
            color = colors[i % len(colors)]
            plt.plot(X_extrapolate.numpy(), mean, color=color, label=f'{name} prediction')
            plt.fill_between(
                X_extrapolate.numpy(),
                lower.numpy(),
                upper.numpy(),
                alpha=0.1,
                color=color
            )
    
    plt.xscale('log')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Different GP Models for Learning Curve Extrapolation')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.ylim(0.5, 1.0)
    
    # Save the figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/model_comparison.png')
    plt.close()

def main():
    """Main function to test improved models"""
    print("Testing improved APEx-GP models...")
    
    try:
        # Load encoded data
        X, y = load_encoded_data()
        
        # Define models to compare
        model_classes = [
            # Original models
            models.GPPowerLaw,
            models.GPArctan,
            
            # Matern kernel models
            matern_kernels.GPMatern12,
            matern_kernels.GPMatern32,
            matern_kernels.GPMatern52,
            
            # Beta prior models
            beta_priors.GPPowerLawWithBetaPrior,
            beta_priors.GPArctanWithBetaPrior,
            
            # Combination kernel models
            lambda *args, **kwargs: combo_kernels.GPComboKernel(*args, kernel_type="rbf+matern", **kwargs),
            lambda *args, **kwargs: combo_kernels.GPComboKernel(*args, kernel_type="rbf+periodic", **kwargs)
        ]
        
        model_names = [
            # Original models
            "GPPowerLaw",
            "GPArctan",
            
            # Matern kernel models
            "GPMatern12",
            "GPMatern32",
            "GPMatern52",
            
            # Beta prior models
            "GPPowerLawWithBetaPrior",
            "GPArctanWithBetaPrior",
            
            # Combination kernel models
            "GPComboKernel_RBF+Matern",
            "GPComboKernel_RBF+Periodic"
        ]
        
        # Compare models
        results = compare_models(X, y, model_classes, model_names)
        
        # Plot comparison
        plot_model_comparison(X, y, results)
        
        print("\nImproved models testing completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()