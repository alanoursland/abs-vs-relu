# src/training/train_mnist.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

from models.lenet import LeNet
from data.mnist_loader import load_mnist
from utils.visualization import plot_loss_curves
from training.train_utils import train, test_fast

def main(config):
    train_loader, test_loader = load_mnist(batch_size=config.batch_size, cuda_device=config.device, use_gpu=True)

    activation_function = config.get_activation_function(config.activation_function)
    model = LeNet(activation_function=activation_function).to(config.device)

    # Get the entire test set in a single batch
    X_test, Y_test = next(iter(test_loader))

    # Move the data to GPU
    X_test = X_test.to(config.device)
    Y_test = Y_test.to(config.device)

    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    accuracies = []

    start_time = time.time()  # Record the start time

    for epoch in range(1, config.epochs + 1):
        train_loss = train(model, config.device, train_loader, optimizer, criterion, epoch, scheduler=None, log_interval=config.log_interval)
        test_loss, accuracy = test_fast(model, X_test, Y_test, criterion, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

    end_time = time.time()  # Record the end time
    training_time = end_time - start_time  # Calculate the total training time

    if config.save_model:
        model_save_path = os.path.join(config.run_dir, f"mnist_lenet_{config.activation_function}.pth")
        torch.save(model.state_dict(), model_save_path)

    plot_title = f"Error {config.dataset} {config.model} {config.activation} {config.run}"
    plot_loss_curves(
        train_losses, test_losses, title=plot_title, save_path=os.path.join(config.run_dir, "loss_curves.png"), show_plot=False
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

# Test set results for mnist lenet abs:
# Final test losses: ['0.0415', '0.0380', '0.0352', '0.0423', '0.0393']
# Final accuracies: ['98.75', '98.87', '98.97', '98.61', '98.64']
# Training times: ['20.43', '19.90', '20.03', '19.99', '20.01']
# Average loss: 0.0392
# Average accuracy: 98.77%

# Test set results for mnist lenet relu:
# Final test losses: ['0.0482', '0.0451', '0.0451', '0.0436', '0.0426']
# Final accuracies: ['98.49', '98.68', '98.56', '98.62', '98.55']
# Training times: ['19.44', '19.39', '19.70', '19.88', '19.72']
# Average loss: 0.0449
# Average accuracy: 98.58%

# t-statistic: 4.1216 
# p-value: 0.0033

# t-statistic: 2.4914324567779844
# p-value: 0.03743855894261221

# The difference in accuracies is statistically significant, indicating that our method performs better than their method.

# Comparison: A2A
#   Common: 0.0051 ± 0.0005
#   Unique: 0.0073 ± 0.0013
#   Consistency: 0.2598 ± 0.0267
#   Diversity: 145.0000 ± 13.5425
# Comparison: B2B
#   Common: 0.0066 ± 0.0004
#   Unique: 0.0076 ± 0.0007
#   Consistency: 0.3047 ± 0.0247
#   Diversity: 151.6000 ± 10.6790
# Comparison: A2B
#   Common: 0.0056 ± 0.0004
#   Unique0: 0.0067 ± 0.0012
#   Unique1: 0.0086 ± 0.0008
#   Consistency: 0.2687 ± 0.0220
#   Diversity: 153.1200 ± 12.9347
# Consensus models

# Configs used: ['configs/mnist_abs.json']
# Average loss: 0.0001
# Average accuracy: 99.29%

# Configs used: ['configs/mnist_relu.json']
# Average loss: 0.0001
# Average accuracy: 99.12%

# Configs used: ['configs/mnist_abs.json', 'configs/mnist_relu.json']
# Average loss: 0.0001
# Average accuracy: 99.26%