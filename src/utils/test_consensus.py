import os
import json
import torch
from collections import defaultdict
import numpy as np
import sys
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.lenet import LeNet
from src.models.resnet18 import ResNet18
from src.data.mnist_loader import load_mnist
from src.data.cifar10_loader import load_cifar10
from src.data.cifar100_loader import load_cifar100
from src.config import Config
from torch.nn.functional import softmax

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs, dim=0)


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
        raise f"Unknown model type {config.model}"

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_consensus(models, test_loader, device):
    for model in models:
        model.to(device)

    correct = 0
    total = 0
    losses = 0
    criterion = torch.nn.CrossEntropyLoss()

    ensemble_model = EnsembleModel(models).to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = ensemble_model(data) + 1e-6 # [n_models, batch_size, n_outputs]
            softmax_outputs = softmax(outputs, dim=2) # [n_models, batch_size, n_outputs]
            consensus_output = softmax_outputs.mean(0) # [batch_size, n_outputs]
            loss = criterion(consensus_output, target)
            losses += loss.item()
            pred = consensus_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = losses / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main(config_files):
    configs = []
    for config_file in config_files:
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        configs.append(Config(config_dict))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if configs[0].dataset == "mnist":
        _, test_loader = load_mnist(batch_size=configs[0].test_batch_size, cuda_device=device, use_gpu=True)
    elif configs[0].dataset == "cifar10":
        _, test_loader = load_cifar10(batch_size=configs[0].test_batch_size)
    elif configs[0].dataset == "cifar100":
        _, test_loader = load_cifar100(batch_size=configs[0].test_batch_size)
    else:
        raise f"Unknown dataset {configs[0].dataset}"

    models = []
    for config in configs:
        for run in range(1, config.num_runs + 1):
            model = load_model(config, run)
            models.append(model)

    avg_loss, accuracy = evaluate_consensus(models, test_loader, device)

    print(f"Configs used: {config_files}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Average accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Consensus Model Evaluation on activation function results")
    parser.add_argument("config_files", type=str, nargs='+', help="Paths to the config files for the experiments")
    args = parser.parse_args()

    main(args.config_files)
