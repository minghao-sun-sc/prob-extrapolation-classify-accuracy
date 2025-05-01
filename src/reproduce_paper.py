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
import priors  # For truncated_normal_uncertainty

# Set torch float precision
torch.set_default_dtype(torch.float64)

def load_encoded_data(dataset_name=None):
    """Load encoded dataset from the specified path"""
    # Path to encoded images
    encoded_path = 'D:/projects/prob-extrapolation-classify-accuracy/dataset/encoded_images/'
    
    # List available encoded data files
    encoded_files = glob.glob(os.path.join(encoded_path, '*.npz'))
    print(f"Available encoded datasets: {[os.path.basename(f) for f in encoded_files]}")
    
    # If dataset_name is provided, try to find a matching file
    if dataset_name and encoded_files:
        matching_files = [f for f in encoded_files if dataset_name.lower() in f.lower()]
        if matching_files:
            data_file = matching_files[0]
        else:
            data_file = encoded_files[0]  # Default to first file if no match
    elif encoded_files:
        data_file = encoded_files[0]  # Default to first file
    else:
        raise FileNotFoundError("No encoded datasets found in the specified directory")
    
    # Load the dataset
    data = np.load(data_file)
    print(f"Loaded dataset: {os.path.basename(data_file)}")
    print(f"Available keys: {data.files}")
    
    # Try to extract training sizes and scores
    if 'train_sizes' in data.files and 'train_scores' in data.files:
        X = torch.tensor(data['train_sizes'], dtype=torch.float64)
        y = torch.tensor(data['train_scores'], dtype=torch.float64)
        print(f"Data loaded successfully: {len(X)} data points")
        return X, y
    
    # If standard keys not found, try to find appropriate arrays
    print("Standard keys not found, examining data structure...")
    for key in data.files:
        print(f"{key}: shape {data[key].shape}, type {data[key].dtype}")
    
    # Try to find appropriate data
    potential_x = None
    potential_y = None
    
    for key in data.files:
        arr = data[key]
        # Look for arrays that could be training sizes or accuracies
        if len(arr.shape) == 1:
            if potential_x is None and arr.min() > 10:  # Likely to be training sizes
                potential_x = arr
            elif potential_y is None and 0 <= arr.min() <= arr.max() <= 1:  # Likely to be accuracies
                potential_y = arr
    
    if potential_x is not None and potential_y is not None:
        print("Found potential learning curve data")
        X = torch.tensor(potential_x, dtype=torch.float64)
        y = torch.tensor(potential_y, dtype=torch.float64)
        return X, y
    
    raise ValueError("Could not identify training sizes and scores in the dataset")

def train_and_evaluate_model(X, y, model_class, model_name, with_priors=True):
    """Train and evaluate a GP model on the given data"""
    # Split data for training and validation
    train_size = int(0.8 * len(X))
    indices = torch.randperm(len(X))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    # Sort by X values for better visualization
    sorted_indices = torch.argsort(X_train)
    X_train, y_train = X_train[sorted_indices], y_train[sorted_indices]
    
    sorted_indices = torch.argsort(X_test)
    X_test, y_test = X_test[sorted_indices], y_test[sorted_indices]
    
    # Initialize model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = model_class(X_train, y_train, likelihood, epsilon_min=0.05, with_priors=with_priors)
    
    # Train model
    likelihood, model, _ = models.train_gp(likelihood, model, X_train, y_train, max_iters=2000, lr=0.01)
    
    # Evaluate model
    model.eval()
    likelihood.eval()
    
    # Create test points for extrapolation
    max_size = X.max().item()
    X_extrapolate = torch.logspace(np.log10(X.min().item()), np.log10(max_size * 10), 100)
    
    with torch.no_grad():
        # Predictions on training data
        predictions_train = likelihood(model(X_train))
        mean_train = predictions_train.mean
        lower_train, upper_train = predictions_train.confidence_region()
        
        # Predictions on test data
        predictions_test = likelihood(model(X_test))
        mean_test = predictions_test.mean
        lower_test, upper_test = predictions_test.confidence_region()
        
        # Predictions for extrapolation
        predictions_extrapolate = likelihood(model(X_extrapolate))
        mean_extrapolate = predictions_extrapolate.mean.numpy()
        lower_extrapolate, upper_extrapolate = predictions_extrapolate.confidence_region()
        
        # Calculate MSE on test data
        mse = torch.mean((mean_test - y_test) ** 2).item()
        print(f"{model_name} - Test MSE: {mse:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.scatter(X_train.numpy(), y_train.numpy(), c='blue', label='Training data')
    
    # Plot test data
    plt.scatter(X_test.numpy(), y_test.numpy(), c='red', label='Test data')
    
    # Plot predictions
    plt.plot(X_extrapolate.numpy(), mean_extrapolate, 'k-', label=f'{model_name} prediction')
    plt.fill_between(
        X_extrapolate.numpy(),
        lower_extrapolate.numpy(),
        upper_extrapolate.numpy(),
        alpha=0.2,
        color='gray',
        label='95% confidence interval'
    )
    
    plt.xscale('log')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    plt.title(f'Learning Curve Extrapolation with {model_name}')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.ylim(0.5, 1.0)
    
    # Save the figure
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{model_name}_learning_curve.png')
    plt.close()
    
    return model, likelihood, mse

def main():
    """Main function to reproduce paper results"""
    print("Reproducing APEx-GP paper results...")
    
    try:
        # Load encoded data
        X, y = load_encoded_data()
        
        # Train and evaluate GPPowerLaw model (original from paper)
        power_law_model, power_law_likelihood, power_law_mse = train_and_evaluate_model(
            X, y, models.GPPowerLaw, "GPPowerLaw"
        )
        
        # Train and evaluate GPArctan model (original from paper)
        arctan_model, arctan_likelihood, arctan_mse = train_and_evaluate_model(
            X, y, models.GPArctan, "GPArctan"
        )
        
        print("\nReproduction completed successfully!")
        print(f"GPPowerLaw MSE: {power_law_mse:.6f}")
        print(f"GPArctan MSE: {arctan_mse:.6f}")
        
    except Exception as e:
        print(f"Error during reproduction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()