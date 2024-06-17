# src/training/train_cifar100.py

# from https://github.com/kuangliu/pytorch-cifar

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

from models.resnet18 import ResNet18
from data.cifar100_loader import load_cifar100
from utils.visualization import plot_loss_curves
from training.train_utils import train, test_fast

def main(config):
    train_loader, test_loader = load_cifar100(batch_size=config.batch_size)

    activation_function = config.get_activation_function(config.activation_function)
    model = ResNet18(num_classes=100, activation_function=activation_function).to(config.device)

    # Get the entire test set in a single batch
    X_test, Y_test = next(iter(test_loader))

    # Move the data to GPU
    X_test = X_test.to(config.device)
    Y_test = Y_test.to(config.device)

    optimizer = optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    accuracies = []

    start_time = time.time()  # Record the start time

    for epoch in range(1, config.epochs + 1):
        train_loss = train(model, config.device, train_loader, optimizer, criterion, epoch, scheduler, config.log_interval)
        test_loss, accuracy = test_fast(model, X_test, Y_test, criterion, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

    end_time = time.time()  # Record the end time
    training_time = end_time - start_time  # Calculate the total training time

    if config.save_model:
        model_save_path = os.path.join(config.run_dir, f"cifar100_resnet18_{config.activation_function}.pth")
        torch.save(model.state_dict(), model_save_path)

    plot_title = f"Error {config.dataset} {config.model} {config.activation} {config.run}"
    plot_loss_curves(
        train_losses,
        test_losses,
        title=plot_title,
        save_path=os.path.join(config.run_dir, "loss_curves.png"),
        show_plot=False,
    )

    results = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "accuracies": accuracies,
        "training_time": training_time,
    }

    results_path = os.path.join(config.run_dir, "results.pth")
    torch.save(results, results_path)

    return results

# Test set results for experiments/CIFAR100_ResNet18_Abs:
# Final test losses: ['1.8746', '2.0900', '1.9631', '2.0767', '2.0798']
# Final accuracies: ['67.51', '65.58', '66.88', '65.08', '65.85']
# Training times: ['2281.48', '2281.90', '2279.20', '2279.65', '2281.14']
# Average loss: 2.0168
# Average accuracy: 66.18%

# Test set results for experiments/CIFAR100_ResNet18_ReLU:
# Final test losses: ['1.2627', '1.2602', '1.2797', '1.2823', '1.2784']
# Final accuracies: ['73.31', '73.66', '73.17', '72.73', '72.85']
# Training times: ['2271.07', '2265.05', '2259.17', '2259.63', '2365.16']
# Average loss: 1.2727
# Average accuracy: 73.14%

# t-statistic: -14.69754458730309, p-value: 4.512193075844569e-07

# Comparison: A2A
#   Common: 0.2515 ± 0.0056
#   Unique: 0.0867 ± 0.0074
#   Consistency: 0.5919 ± 0.0096
#   Diversity: 1734.0000 ± 47.7347
# Comparison: B2B
#   Common: 0.2027 ± 0.0021
#   Unique: 0.0658 ± 0.0031
#   Consistency: 0.6062 ± 0.0072
#   Diversity: 1317.0000 ± 31.8779
# Comparison: A2B
#   Common: 0.2096 ± 0.0029
#   Unique0: 0.1286 ± 0.0079
#   Unique1: 0.0589 ± 0.0019
#   Consistency: 0.5279 ± 0.0090
#   Diversity: 1875.3600 ± 66.0230

# Consensus models

# Configs used: ['configs/cifar100_abs.json']
# Average loss: 0.0004
# Average accuracy: 71.28%

# Configs used: ['configs/cifar100_relu.json']
# Average loss: 0.0004
# Average accuracy: 76.61%

# Average loss: 0.0004
# Average accuracy: 76.15%