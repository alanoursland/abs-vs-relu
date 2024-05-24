import argparse
import json
import os
import sys
import torch

# Add the src directory to the Python path
print(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.train_mnist import main as train_mnist_main
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

    for run in range(1, config.num_runs + 1):
        run_dir = os.path.join(config.output_dir, f"{run:04d}")
        os.makedirs(run_dir, exist_ok=True)
        config.run = run
        config.run_dir = run_dir
        print(f"Running experiment {run}/{config.num_runs}")
        train_mnist_main(config)


if __name__ == "__main__":
    main()
