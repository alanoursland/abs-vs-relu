import os
import json
import numpy as np
from collections import defaultdict
import torch

def load_results(output_dir, num_runs):
    errors = []
    for run in range(1, num_runs + 1):
        results_path = os.path.join(output_dir, f"{run:04d}", "results.pth")
        if os.path.exists(results_path):
            results = torch.load(results_path)
            errors.append(results["errors"])
        else:
            print(f"Results file not found: {results_path}")
    return errors

def error_overlap_analysis(errors_relu, errors_abs):
    metrics = defaultdict(dict)

    for i, errors_r in enumerate(errors_relu):
        for j, errors_a in enumerate(errors_abs):
            common_errors = np.intersect1d(errors_r, errors_a)
            unique_errors_r = np.setdiff1d(errors_r, errors_a)
            unique_errors_a = np.setdiff1d(errors_a, errors_r)
            union_errors = np.union1d(errors_r, errors_a)

            metrics[f"Model_R_{i+1}_A_{j+1}"]["Common Error Rate"] = len(common_errors) / len(union_errors)
            metrics[f"Model_R_{i+1}_A_{j+1}"]["Unique Error Rate (ReLU)"] = len(unique_errors_r) / len(union_errors)
            metrics[f"Model_R_{i+1}_A_{j+1}"]["Unique Error Rate (Abs)"] = len(unique_errors_a) / len(union_errors)
            metrics[f"Model_R_{i+1}_A_{j+1}"]["Error Consistency"] = len(common_errors) / (len(errors_r) + len(errors_a) - len(common_errors))
            metrics[f"Model_R_{i+1}_A_{j+1}"]["Error Diversity Index"] = len(unique_errors_r) + len(unique_errors_a)

    return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Error Overlap Analysis on activation function results")
    parser.add_argument("config_relu", type=str, help="Path to the config file for ReLU activation function")
    parser.add_argument("config_abs", type=str, help="Path to the config file for Abs activation function")
    args = parser.parse_args()

    # Load configurations
    with open(args.config_relu, "r") as f:
        config_relu = json.load(f)

    with open(args.config_abs, "r") as f:
        config_abs = json.load(f)

    # Load results
    errors_relu = load_results(config_relu["output_dir"], config_relu["num_runs"])
    errors_abs = load_results(config_abs["output_dir"], config_abs["num_runs"])

    # Perform Error Overlap Analysis
    metrics = error_overlap_analysis(errors_relu, errors_abs)

    # Print metrics
    for key, value in metrics.items():
        print(f"Comparison: {key}")
        for metric, result in value.items():
            print(f"  {metric}: {result}")
