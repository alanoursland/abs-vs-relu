import os
import json
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add the src directory to the Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

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
            incorrect_indices.extend((batch_idx * test_loader.batch_size + incorrect_mask.nonzero(as_tuple=False)).squeeze().tolist())

    return incorrect_indices

def error_overlap_analysis(config_relu, config_abs, device):
    _, test_loader = load_mnist(batch_size=config_relu.test_batch_size, cuda_device=device, use_gpu=True)

    errors_relu = []
    errors_abs = []

    for run in range(1, config_relu.num_runs + 1):
        model_path_relu = os.path.join(config_relu.output_dir, f"{run:04d}", f"mnist_{config_relu.model.lower()}_{config_relu.activation.lower()}.pth")
        model_relu = load_model(model_path_relu, config_relu.activation)
        errors_relu.append(evaluate_model(model_relu, test_loader, device))

        model_path_abs = os.path.join(config_abs.output_dir, f"{run:04d}", f"mnist_{config_abs.model.lower()}_{config_abs.activation.lower()}.pth")
        model_abs = load_model(model_path_abs, config_abs.activation)
        errors_abs.append(evaluate_model(model_abs, test_loader, device))

    metrics = defaultdict(dict)

    for i, errors_r in enumerate(errors_relu):
        for j, errors_a in enumerate(errors_abs):
            common_errors = set(errors_r) & set(errors_a)
            unique_errors_r = set(errors_r) - set(errors_a)
            unique_errors_a = set(errors_a) - set(errors_r)
            union_errors = set(errors_r) | set(errors_a)

            metrics[f"Model_R_{i+1}_A_{j+1}"]["Common Error Rate"] = len(common_errors) / len(test_loader.dataset)
            metrics[f"Model_R_{i+1}_A_{j+1}"]["Unique Error Rate (ReLU)"] = len(unique_errors_r) / len(test_loader.dataset)
            metrics[f"Model_R_{i+1}_A_{j+1}"]["Unique Error Rate (Abs)"] = len(unique_errors_a) / len(test_loader.dataset)
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
        config_relu_dict = json.load(f)
    config_relu = Config(config_relu_dict)

    with open(args.config_abs, "r") as f:
        config_abs_dict = json.load(f)
    config_abs = Config(config_abs_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Perform Error Overlap Analysis
    metrics = error_overlap_analysis(config_relu, config_abs, device)

    # Print metrics
    for key, value in metrics.items():
        print(f"Comparison: {key}")
        for metric, result in value.items():
            print(f"  {metric}: {result:.4f}")