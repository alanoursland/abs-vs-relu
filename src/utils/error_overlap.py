import os
import json
import torch
from collections import defaultdict
import numpy as np

# Add the src directory to the Python path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.lenet import LeNet
from src.models.resnet18 import ResNet18
from src.data.mnist_loader import load_mnist
from src.data.cifar10_loader import load_cifar10
from src.data.cifar100_loader import load_cifar100
from src.config import Config


def load_model(config, run):
    activation_function = config.get_activation_function(config.activation_function)
    model_path = os.path.join(
        config.output_dir, f"{run:04d}", f"{config.dataset}_{config.model.lower()}_{config.activation.lower()}.pth"
    )
    if config.model == "lenet":
        model = LeNet(activation_function=activation_function)
    elif config.model == "resnet18" and config.dataset == "cifar10":
        model = ResNet18(10, activation_function=activation_function)
    elif config.model == "resnet18" and config.dataset == "cifar100":
        model = ResNet18(100, activation_function=activation_function)
    else:
        raise f"Unknown model type {config_a.model}"

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate_model(model, test_loader, device):
    model.to(device)
    incorrect_indices = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            incorrect_mask = pred.ne(target.view_as(pred)).squeeze()
            batch_base_index = batch_idx * test_loader.batch_size
            batch_incorrect_indices = incorrect_mask.nonzero(as_tuple=False)
            incorrect_indices.extend(
                (batch_base_index + batch_incorrect_indices).squeeze(1).tolist()
            )

    return incorrect_indices


def error_overlap_analysis(config_a, config_b, device):
    if config_a.dataset != config_b.dataset:
        raise "Cannot compare two different datasets"
    
    if config_a.dataset == "mnist":
        _, test_loader = load_mnist(batch_size=config_a.test_batch_size, cuda_device=device, use_gpu=True)
    elif config_a.dataset == "cifar10":
        _, test_loader = load_cifar10(batch_size=config_a.test_batch_size)
    elif config_a.dataset == "cifar100":
        _, test_loader = load_cifar100(batch_size=config_a.test_batch_size)
    else:
        raise f"Unknown dataset {config_a.dataset}"


    errors_a = []
    errors_b = []

    for run in range(1, config_a.num_runs + 1):
        model_a = load_model(config_a, run)
        errors_a.append(evaluate_model(model_a, test_loader, device))

        model_b = load_model(config_b, run)
        errors_b.append(evaluate_model(model_b, test_loader, device))

    metrics = {
        "A2A": defaultdict(list),
        "B2B": defaultdict(list),
        "A2B": defaultdict(list),
    }

    # Compare A to A
    for i, a0 in enumerate(errors_a):
        for j, b1 in enumerate(errors_a):
            # Only compare each pair once
            if i < j:
                common_errors = set(a0) & set(b1)
                unique_errors_a0 = set(a0) - set(b1)
                unique_errors_b1 = set(b1) - set(a0)
                sample_count = len(test_loader.dataset)
                common_count = len(common_errors)
                unique0_count = len(unique_errors_a0)
                unique1_count = len(unique_errors_b1)
                metrics["A2A"]["Common"].append(common_count / sample_count)
                metrics["A2A"]["Unique"].append(unique0_count / sample_count)
                metrics["A2A"]["Unique"].append(unique1_count / sample_count)
                metrics["A2A"]["Consistency"].append(common_count / (len(a0) + len(b1) - common_count))
                metrics["A2A"]["Diversity"].append(unique0_count + unique1_count)

    # Compare B to B
    for i, a0 in enumerate(errors_b):
        for j, b1 in enumerate(errors_b):
            # Only compare each pair once
            if i < j:
                common_errors = set(a0) & set(b1)
                unique_errors_a0 = set(a0) - set(b1)
                unique_errors_b1 = set(b1) - set(a0)
                sample_count = len(test_loader.dataset)
                common_count = len(common_errors)
                unique0_count = len(unique_errors_a0)
                unique1_count = len(unique_errors_b1)
                metrics["B2B"]["Common"].append(common_count / sample_count)
                metrics["B2B"]["Unique"].append(unique0_count / sample_count)
                metrics["B2B"]["Unique"].append(unique1_count / sample_count)
                metrics["B2B"]["Consistency"].append(common_count / (len(a0) + len(b1) - common_count))
                metrics["B2B"]["Diversity"].append(unique0_count + unique1_count)

    # Compare A to B
    for i, a0 in enumerate(errors_a):
        for j, b1 in enumerate(errors_b):
            # Compare all pairs
            common_errors = set(a0) & set(b1)
            unique_errors_a0 = set(a0) - set(b1)
            unique_errors_b1 = set(b1) - set(a0)
            sample_count = len(test_loader.dataset)
            common_count = len(common_errors)
            unique0_count = len(unique_errors_a0)
            unique1_count = len(unique_errors_b1)
            metrics["A2B"]["Common"].append(common_count / sample_count)
            metrics["A2B"]["Unique0"].append(unique0_count / sample_count)
            metrics["A2B"]["Unique1"].append(unique1_count / sample_count)
            metrics["A2B"]["Consistency"].append(common_count / (len(a0) + len(b1) - common_count))
            metrics["A2B"]["Diversity"].append(unique0_count + unique1_count)


    # Collect the data into a summary
    summary = {}
    for collection, collection_metrics in metrics.items():
        summary[collection] = {}
        for metric, values in collection_metrics.items():
            summary[collection][metric] = {
                "mean": np.mean(values),
                "stddev": np.std(values),
            }

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Error Overlap Analysis on activation function results")
    parser.add_argument("config_a", type=str, help="Path to the config file for Experiment A")
    parser.add_argument("config_b", type=str, help="Path to the config file for Experiment B")
    args = parser.parse_args()

    # Load configurations
    with open(args.config_a, "r") as f:
        config_a_dict = json.load(f)
    config_a = Config(config_a_dict)

    with open(args.config_b, "r") as f:
        config_b_dict = json.load(f)
    config_b = Config(config_b_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Perform Error Overlap Analysis
    metrics = error_overlap_analysis(config_a, config_b, device)

    # Print metrics
    for key, value in metrics.items():
        print(f"Comparison: {key}")
        for metric, result  in value .items():
            mean = result["mean"]
            stddev = result["stddev"]
            print(f"  {metric}: {mean:.4f} ± {stddev:.4f}")
