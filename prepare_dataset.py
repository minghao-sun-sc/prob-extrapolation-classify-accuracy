#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def ensure_directory(directory):
    """Ensure a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_learning_curve_data(dataset_name, input_csv_path, output_dir, sizes=None):
    """
    Generate learning curve data from classifier performance at different dataset sizes.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    input_csv_path : str
        Path to a CSV file with classifier results
        Expected format: CSV with at least 'train_size' and 'auroc' columns
        If this doesn't exist, we'll generate synthetic data
    output_dir : str
        Directory to save the output files
    sizes : list
        Optional list of training sizes to simulate
    """
    ensure_directory(output_dir)
    output_path = os.path.join(output_dir, f"{dataset_name}_performance.csv")
    
    if os.path.exists(input_csv_path):
        print(f"Loading data from {input_csv_path}")
        # Load existing performance data
        try:
            df = pd.read_csv(input_csv_path)
            required_columns = ['train_size', 'auroc']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain the following columns: {required_columns}")
            
            # Save to the expected format
            df[required_columns].to_csv(output_path, index=False)
            print(f"Learning curve data saved to {output_path}")
            return df
        except Exception as e:
            print(f"Error processing input file: {e}")
            print("Will generate synthetic data instead.")
    
    # Generate synthetic data if file doesn't exist or couldn't be processed
    if sizes is None:
        # Default dataset sizes
        sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    
    # Generate synthetic learning curve using a power law
    # y = a - b*x^(-c) where x is the dataset size
    a = 0.95  # asymptotic performance
    b = 0.3   # scale parameter
    c = 0.5   # rate parameter
    
    # Add some noise to make it realistic
    np.random.seed(42)
    auroc = a - b * np.power(np.array(sizes), -c) + np.random.normal(0, 0.01, len(sizes))
    
    # Ensure AUROC is between 0.5 and 1.0
    auroc = np.clip(auroc, 0.5, 0.99)
    
    # Create dataframe
    df = pd.DataFrame({'train_size': sizes, 'auroc': auroc})
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic learning curve data for {dataset_name}")
    print(f"Saved to {output_path}")
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.scatter(df['train_size'], df['auroc'])
    plt.xscale('log')
    plt.xlabel('Training Set Size')
    plt.ylabel('AUROC')
    plt.title(f'Learning Curve for {dataset_name}')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, f"{dataset_name}_learning_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curve plot saved to {plot_path}")
    
    return df

def create_simulation_data():
    """Create simulation data for multiple datasets."""
    output_dir = "prob-extrapolation-classify-accuracy/dataset"
    ensure_directory(output_dir)
    
    # Generate data for several datasets
    datasets = [
        ("ChestXRay14_Infiltration", None),
        ("ChestXRay14_Pneumonia", None),
        ("BUSI_Malignant", None),
        ("TMED2_AS", None),
        ("OASIS3_Alzheimer", None)
    ]
    
    for dataset_name, csv_path in datasets:
        # Different learning curves for each dataset
        if "ChestXRay14" in dataset_name:
            sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
            a, b, c = 0.92, 0.25, 0.4
        elif "BUSI" in dataset_name:
            sizes = [50, 100, 200, 500, 1000, 1500]
            a, b, c = 0.88, 0.3, 0.45
        elif "TMED2" in dataset_name:
            sizes = [100, 200, 500, 1000, 2000, 3000]
            a, b, c = 0.90, 0.28, 0.42
        elif "OASIS3" in dataset_name:
            sizes = [50, 100, 200, 300, 500, 1000]
            a, b, c = 0.87, 0.32, 0.48
        else:
            sizes = [100, 200, 500, 1000, 2000, 5000]
            a, b, c = 0.85, 0.3, 0.5
        
        # Generate synthetic learning curve
        np.random.seed(hash(dataset_name) % 1000)  # Different seed for each dataset
        auroc = a - b * np.power(np.array(sizes), -c) + np.random.normal(0, 0.01, len(sizes))
        auroc = np.clip(auroc, 0.5, 0.99)
        
        # Create and save dataframe
        df = pd.DataFrame({'train_size': sizes, 'auroc': auroc})
        output_path = os.path.join(output_dir, f"{dataset_name}_performance.csv")
        df.to_csv(output_path, index=False)
        
        # Plot the learning curve
        plt.figure(figsize=(10, 6))
        plt.scatter(df['train_size'], df['auroc'])
        plt.xscale('log')
        plt.xlabel('Training Set Size')
        plt.ylabel('AUROC')
        plt.title(f'Learning Curve for {dataset_name}')
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, f"{dataset_name}_learning_curve.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created simulation data for {dataset_name}")
    
    print(f"All simulation data created in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for classifier accuracy extrapolation')
    parser.add_argument('--dataset', type=str, default=None, help='Name of the dataset')
    parser.add_argument('--input', type=str, default=None, help='Path to input CSV with performance data')
    parser.add_argument('--output_dir', type=str, default='prob-extrapolation-classify-accuracy/dataset', help='Directory to save output files')
    parser.add_argument('--simulate', action='store_true', help='Create simulation data for multiple datasets')
    
    args = parser.parse_args()
    
    if args.simulate:
        create_simulation_data()
    elif args.dataset:
        generate_learning_curve_data(args.dataset, args.input, args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 