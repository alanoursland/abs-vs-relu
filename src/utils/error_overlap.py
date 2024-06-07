import os
import json
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add the src directory to the Python path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.lenet import LeNet
from src.data.mnist_loader import load_mnist
from src.config import Config


def load_model(model_path, activation_function):
    model = LeNet(activation_function=activation_function)
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
            incorrect_indices.extend(
                (batch_idx * test_loader.batch_size + incorrect_mask.nonzero(as_tuple=False)).squeeze().tolist()
            )

    return incorrect_indices


def error_overlap_analysis(config_a, config_b, device):
    _, test_loader = load_mnist(batch_size=config_a.test_batch_size, cuda_device=device, use_gpu=True)

    errors_a = []
    errors_b = []

    for run in range(1, config_a.num_runs + 1):
        model_path_a = os.path.join(
            config_a.output_dir, f"{run:04d}", f"mnist_{config_a.model.lower()}_{config_a.activation.lower()}.pth"
        )
        model_a = load_model(model_path_a, config_a.get_activation_function(config_a.activation_function))
        errors_a.append(evaluate_model(model_a, test_loader, device))

        model_path_b = os.path.join(
            config_b.output_dir, f"{run:04d}", f"mnist_{config_b.model.lower()}_{config_b.activation.lower()}.pth"
        )
        model_b = load_model(model_path_b, config_b.get_activation_function(config_b.activation_function))
        errors_b.append(evaluate_model(model_b, test_loader, device))

    metrics = defaultdict(dict)

    for i, errors_r in enumerate(errors_a):
        for j, errors_a in enumerate(errors_b):
            common_errors = set(errors_r) & set(errors_a)
            unique_errors_r = set(errors_r) - set(errors_a)
            unique_errors_a = set(errors_a) - set(errors_r)
            union_errors = set(errors_r) | set(errors_a)

            metrics[f"Model_R_{i+1}_A_{j+1}"]["Common Error Rate"] = len(common_errors) / len(test_loader.dataset)
            metrics[f"Model_R_{i+1}_A_{j+1}"]["Unique Error Rate (Model A)"] = len(unique_errors_r) / len(
                test_loader.dataset
            )
            metrics[f"Model_R_{i+1}_A_{j+1}"]["Unique Error Rate (Model B)"] = len(unique_errors_a) / len(
                test_loader.dataset
            )
            metrics[f"Model_R_{i+1}_A_{j+1}"]["Error Consistency"] = len(common_errors) / (
                len(errors_r) + len(errors_a) - len(common_errors)
            )
            metrics[f"Model_R_{i+1}_A_{j+1}"]["Error Diversity Index"] = len(unique_errors_r) + len(unique_errors_a)

    return metrics


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
        for metric, result in value.items():
            print(f"  {metric}: {result:.4f}")
