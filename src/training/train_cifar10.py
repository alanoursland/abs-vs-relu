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
from training.train_utils import train, test


def main(config):
    torch.manual_seed(config.seed)
    if config.device.type == "cuda":
        torch.cuda.manual_seed(config.seed)

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
            model, config.device, train_loader, optimizer, criterion, epoch, scheduler, config.log_interval
        )
        test_loss, accuracy = test(model, X_test, Y_test, criterion, epoch)
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
# Final test losses: ['0.5235', '0.4961', '0.5060', '0.5053', '0.5353']
# Final accuracies: ['90.47', '90.64', '90.62', '90.48', '90.10']
# Training times: ['2167.56', '2172.71', '2172.39', '2175.42', '2172.23']
# Average loss: 0.5132
# Average accuracy: 90.46%

# Test set results for experiments/CIFAR10_ResNet18_ReLU:
# Final test losses: ['0.3374', '0.3347', '0.3517', '0.3124', '0.3039']
# Final accuracies: ['92.67', '93.22', '92.46', '93.22', '93.33']
# Training times: ['2257.45', '2201.36', '2160.03', '2135.48', '2128.06']
# Average loss: 0.3280
# Average accuracy: 92.98%

# t-statistic: -12.65118997027362, p-value: 1.4317883786698362e-06

# The difference in accuracies is statistically significant, indicating that our method performs worse than their method.

# Comparison: A2A
#   Common: 0.0558 ± 0.0009
#   Unique: 0.0396 ± 0.0019
#   Consistency: 0.4132 ± 0.0088
#   Diversity: 792.2000 ± 22.0808
# Comparison: B2B
#   Common: 0.0427 ± 0.0019
#   Unique: 0.0275 ± 0.0030
#   Consistency: 0.4371 ± 0.0165
#   Diversity: 550.0000 ± 25.1515
# Comparison: A2B
#   Common: 0.0445 ± 0.0018
#   Unique0: 0.0508 ± 0.0028
#   Unique1: 0.0257 ± 0.0022
#   Consistency: 0.3682 ± 0.0148
#   Diversity: 764.8400 ± 30.6812

# Consensus models

# Configs used: ['configs/cifar10_abs.json']
# Average loss: 0.0002
# Average accuracy: 92.88%

# Configs used: ['configs/cifar10_relu.json']
# Average loss: 0.0002
# Average accuracy: 94.34%

# Configs used: ['configs/cifar10_abs.json', 'configs/cifar10_relu.json']
# Average loss: 0.0002
# Average accuracy: 94.13%
