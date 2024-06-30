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
# Final test losses: ['1.9638', '1.9446', '2.0571', '2.0289', '2.0041']
# Final accuracies: ['67.12', '67.65', '66.18', '66.23', '65.80']
# Training times: ['2608.55', '2608.79', '2609.03', '2633.13', '2614.63']
# Average loss: 1.9997
# Average accuracy: 66.60%

# Test set results for experiments/CIFAR100_ResNet18_ReLU:
# Final test losses: ['1.2666', '1.2814', '1.2835', '1.3217', '1.2877']
# Final accuracies: ['72.95', '72.91', '73.04', '72.06', '73.23']
# Training times: ['2582.55', '2584.83', '2588.47', '2586.90', '2584.66']
# Average loss: 1.2882
# Average accuracy: 72.84%

# t-statistic: -15.744364010380329, p-value: 2.645716845600714e-07

# The difference in accuracies is statistically significant, indicating that our method performs worse than their method.

# A: Abs
# B: ReLU

# Comparison: A2A
#   Common: 0.2489 ± 0.0046
#   Unique: 0.0851 ± 0.0056
#   Consistency: 0.5938 ± 0.0072
#   Diversity: 1702.4000 ± 28.1929
# Comparison: B2B
#   Common: 0.2038 ± 0.0017
#   Unique: 0.0678 ± 0.0037
#   Consistency: 0.6004 ± 0.0073
#   Diversity: 1356.8000 ± 38.6311
# Comparison: A2B
#   Common: 0.2120 ± 0.0033
#   Unique0: 0.1220 ± 0.0062
#   Unique1: 0.0596 ± 0.0032
#   Consistency: 0.5386 ± 0.0094
#   Diversity: 1816.6000 ± 58.6583

# Consensus models

# Configs used: ['configs/cifar100_abs.json']
# Average loss: 0.0004
# Average accuracy: 71.92%

# Configs used: ['configs/cifar100_relu.json']
# Average loss: 0.0004
# Average accuracy: 76.55%

# Configs used: ['configs/cifar100_abs.json', 'configs/cifar100_relu.json']
# Average loss: 0.0004
# Average accuracy: 75.90%
