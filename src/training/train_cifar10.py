# src/training/train_cifar10.py

# from https://github.com/kuangliu/pytorch-cifar

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

from models.resnet18 import ResNet18
from data.cifar10_loader import load_cifar10
from utils.visualization import plot_loss_curves
from training.train_utils import train, test_fast


def main(config):
    train_loader, test_loader = load_cifar10(batch_size=config.batch_size)

    activation_function = config.get_activation_function(config.activation_function)
    model = ResNet18(num_classes=10, activation_function=activation_function).to(config.device)

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
        train_loss = train(
            model, config.device, train_loader, optimizer, criterion, epoch, scheduler=scheduler, log_interval=config.log_interval
        )
        test_loss, accuracy = test_fast(model, X_test, Y_test, criterion, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

    end_time = time.time()  # Record the end time
    training_time = end_time - start_time  # Calculate the total training time

    if config.save_model:
        model_save_path = os.path.join(config.run_dir, f"cifar10_resnet18_{config.activation_function}.pth")
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


# Test set results for experiments/CIFAR10_ResNet18_Abs:
# Final test losses: ['0.4979', '0.4912', '0.5549', '0.5032', '0.5467']
# Final accuracies: ['90.36', '90.85', '89.94', '90.22', '89.94']
# Training times: ['2609.93', '2791.65', '2871.44', '2914.31', '2881.52']
# Average loss: 0.5188
# Average accuracy: 90.26%

# Test set results for experiments/CIFAR10_ResNet18_ReLU:
# Final test losses: ['0.3212', '0.3153', '0.3039', '0.3256', '0.3133']
# Final accuracies: ['92.70', '93.03', '93.22', '92.86', '93.34']
# Training times: ['2711.12', '2626.01', '2607.76', '2595.54', '2584.98']
# Average loss: 0.3159
# Average accuracy: 93.03%

# t-statistic: -13.551055294955578, p-value: 8.447693559533877e-07

# The difference in accuracies is statistically significant, indicating that our method performs worse than their method.

# A: Abs
# B: ReLU

# Comparison: A2A
#   Common: 0.0556 ± 0.0015
#   Unique: 0.0418 ± 0.0030
#   Consistency: 0.3994 ± 0.0098
#   Diversity: 836.0000 ± 27.5391
# Comparison: B2B
#   Common: 0.0429 ± 0.0015
#   Unique: 0.0268 ± 0.0019
#   Consistency: 0.4445 ± 0.0119
#   Diversity: 536.0000 ± 11.0272
# Comparison: A2B
#   Common: 0.0433 ± 0.0018
#   Unique0: 0.0541 ± 0.0032
#   Unique1: 0.0264 ± 0.0014
#   Consistency: 0.3494 ± 0.0141
#   Diversity: 805.6000 ± 28.9413

# Consensus models

# Configs used: ['configs/cifar10_abs.json']
# Average loss: 0.0002
# Average accuracy: 92.70%

# Configs used: ['configs/cifar10_relu.json']
# Average loss: 0.0002
# Average accuracy: 94.46%

# Configs used: ['configs/cifar10_abs.json', 'configs/cifar10_relu.json']
# Average loss: 0.0002
# Average accuracy: 94.32%