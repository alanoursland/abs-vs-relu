import argparse
import json
import os
import sys
import torch

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.train_mnist import main as train_mnist_main
from src.training.train_cifar10 import main as train_cifar10_main
from src.training.train_cifar100 import main as train_cifar100_main
from src.training.train_imdb import main as train_imdb_main
from src.config import Config

def load_config(json_file):
    with open(json_file, "r") as f:
        config = json.load(f)
    return config


def main():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
        device = torch.device("cuda")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")


    parser = argparse.ArgumentParser(description="Run experiments with configurations from a JSON file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file")
    args = parser.parse_args()

    config_dict = load_config(args.config)
    config = Config(config_dict, cuda_device=device)

    torch.manual_seed(config.seed)
    if config.device.type == "cuda":
        torch.cuda.manual_seed(config.seed)


    # Choose the appropriate training function based on the dataset
    if config.dataset.lower() == 'mnist':
        train_function = train_mnist_main
    elif config.dataset.lower() == 'cifar10':
        train_function = train_cifar10_main
    elif config.dataset.lower() == 'cifar100':
        train_function = train_cifar100_main
    elif config.dataset.lower() == 'imdb':
        train_function = train_imdb_main
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    results_list = []
    for run in range(1, config.num_runs + 1):
        run_dir = os.path.join(config.output_dir, f"{run:04d}")
        os.makedirs(run_dir, exist_ok=True)
        config.run = run
        config.run_dir = run_dir
        print(f"Running experiment {run}/{config.num_runs}")
        results_list.append(train_function(config))

    test_losses = []
    test_accuracies = []
    training_times = []
    for results in results_list:
        test_losses.append(results["test_losses"][-1])
        test_accuracies.append(results["accuracies"][-1])
        training_times.append(results["training_time"])

    # Print summary of test dataset results
    print(f"Test set results for {config.dataset} {config.model} {config.activation_function}:")
    print("Final test losses:", [f"{loss:.4f}" for loss in test_losses])
    print("Final accuracies:", [f"{acc:.2f}" for acc in test_accuracies])
    print("Training times:", [f"{sec:.2f}" for sec in training_times])

    print(f"Average loss: {sum(test_losses)/len(test_losses):.4f}")
    print(f"Average accuracy: {sum(test_accuracies)/len(test_accuracies):.2f}%")

    # Print final loss and accuracy for each run in list format


if __name__ == "__main__":
    main()
