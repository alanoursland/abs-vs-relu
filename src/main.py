# src/main.py

import argparse
import json
import os
from src.training.train_mnist import main as train_mnist_main
from src.config import Config


def load_config(json_file):
    with open(json_file, "r") as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Run experiments with configurations from a JSON file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Override config values with those from the JSON file
    Config.activation_function = config.get("activation", Config.activation_function)
    Config.batch_size = config.get("batch_size", Config.batch_size)
    Config.test_batch_size = config.get("test_batch_size", Config.test_batch_size)
    Config.epochs = config.get("epochs", Config.epochs)
    Config.learning_rate = config.get("learning_rate", Config.learning_rate)
    Config.momentum = config.get("momentum", Config.momentum)
    Config.use_cuda = not config.get("no_cuda", False) and torch.cuda.is_available()
    Config.device = torch.device("cuda" if Config.use_cuda else "cpu")
    Config.seed = config.get("seed", Config.seed)
    Config.log_interval = config.get("log_interval", Config.log_interval)
    Config.save_model = config.get("save_model", Config.save_model)
    Config.output_dir = config.get("output_dir", Config.results_dir)
    Config.num_runs = config.get("num_runs", 1)

    for run in range(1, Config.num_runs + 1):
        run_dir = os.path.join(Config.output_dir, f"{run:04d}")
        os.makedirs(run_dir, exist_ok=True)
        Config.run_dir = run_dir
        print(f"Running experiment {run}/{Config.num_runs}")
        train_mnist_main(Config)


if __name__ == "__main__":
    main()
