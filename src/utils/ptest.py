import os
import json
import numpy as np
from scipy import stats

def load_results(output_dir, num_runs):
    accuracies = []
    losses = []
    training_times = []
    for run in range(1, num_runs + 1):
        results_path = os.path.join(output_dir, f"{run:04d}", "results.pth")
        if os.path.exists(results_path):
            results = torch.load(results_path)
            accuracies.append(results["accuracies"][-1])
            losses.append(results["test_losses"][-1])
            training_times.append(results["training_time"])
        else:
            print(f"Results file not found: {results_path}")
    return accuracies, losses, training_times

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

def print_results(name, accuracies, losses, training_times):
    print(f"Test set results for {name}:")
    print(f"Final test losses: {['{:.4f}'.format(loss) for loss in losses]}")
    print(f"Final accuracies: {['{:.2f}'.format(acc) for acc in accuracies]}")
    print(f"Training times: {['{:.2f}'.format(time) for time in training_times]}")
    print(f"Average loss: {np.mean(losses):.4f}")
    print(f"Average accuracy: {np.mean(accuracies):.2f}%")
    print()

if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Run t-test on activation function results")
    parser.add_argument("config_a", type=str, help="Path to the config file for experiment A")
    parser.add_argument("config_b", type=str, help="Path to the config file for experiment B")
    args = parser.parse_args()

    # Load configurations
    with open(args.config_a, "r") as f:
        config_a = json.load(f)

    with open(args.config_b, "r") as f:
        config_b = json.load(f)

    # Load results
    results_a_accuracies, results_a_losses, results_a_times = load_results(config_a["output_dir"], config_a["num_runs"])
    results_b_accuracies, results_b_losses, results_b_times = load_results(config_b["output_dir"], config_b["num_runs"])

    # Perform t-test
    t_stat, p_value = calculate_ttest(results_a_accuracies, results_b_accuracies)

    # Print results
    print_results(f"{config_a['output_dir']}", results_a_accuracies, results_a_losses, results_a_times)
    print_results(f"{config_b['output_dir']}", results_b_accuracies, results_b_losses, results_b_times)

    print(f"t-statistic: {t_stat}, p-value: {p_value}")
