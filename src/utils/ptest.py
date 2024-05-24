import os
import json
import numpy as np
from scipy import stats

def load_results(output_dir, num_runs):
    accuracies = []
    for run in range(1, num_runs + 1):
        results_path = os.path.join(output_dir, f"{run:04d}", "results.pth")
        if os.path.exists(results_path):
            results = torch.load(results_path)
            accuracies.append(results["accuracies"][-1])
        else:
            print(f"Results file not found: {results_path}")
    return accuracies

def calculate_ttest(results_relu, results_abs):
    # Ensure that both inputs are lists of the same length
    if len(results_relu) != len(results_abs):
        raise ValueError("Both input lists must have the same length")

    # Convert to numpy arrays
    results_relu = np.array(results_relu)
    results_abs = np.array(results_abs)

    # Calculate t-statistic and p-value
    t_stat, p_value = stats.ttest_ind(results_relu, results_abs)

    return t_stat, p_value

if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Run t-test on activation function results")
    parser.add_argument("config_relu", type=str, help="Path to the config file for ReLU activation function")
    parser.add_argument("config_abs", type=str, help="Path to the config file for Abs activation function")
    args = parser.parse_args()

    # Load configurations
    with open(args.config_relu, "r") as f:
        config_relu = json.load(f)

    with open(args.config_abs, "r") as f:
        config_abs = json.load(f)

    # Load results
    results_relu = load_results(config_relu["output_dir"], config_relu["num_runs"])
    results_abs = load_results(config_abs["output_dir"], config_abs["num_runs"])

    # Perform t-test
    t_stat, p_value = calculate_ttest(results_relu, results_abs)
    print(f"t-statistic: {t_stat}, p-value: {p_value}")
