# src/training/train_mnist.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

from models.lenet import LeNet
from data.mnist_loader import load_mnist
from utils.visualization import plot_loss_curves


def train(model, device, train_loader, optimizer, criterion, epoch, log_interval=100):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # if batch_idx % log_interval == 0:
        #     print(
        #         f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
        #         f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
        #     )
    train_loss /= len(train_loader.dataset)
    return train_loss


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({accuracy:.2f}%)\n"
    )
    return test_loss, accuracy


def main(config):
    torch.manual_seed(config.seed)
    if config.device.type == "cuda":
        torch.cuda.manual_seed(config.seed)

    train_loader, test_loader = load_mnist(batch_size=config.batch_size, cuda_device=config.device, use_gpu=True)

    activation_function = config.get_activation_function(config.activation_function)
    model = LeNet(activation_function=activation_function).to(config.device)

    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    accuracies = []

    start_time = time.time()  # Record the start time

    for epoch in range(1, config.epochs + 1):
        train_loss = train(model, config.device, train_loader, optimizer, criterion, epoch, config.log_interval)
        test_loss, accuracy = test(model, config.device, test_loader, criterion)
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

# Test set results for cifar10 resnet18 abs:
# Final test losses: ['0.0007', '0.0007', '0.0007', '0.0007', '0.0007']
# Final accuracies: ['98.51', '98.70', '98.50', '98.60', '98.52']
# Training times: ['20.49', '20.49', '20.43', '20.48', '20.49']
# Average loss: 0.0007
# Average accuracy: 98.57%

# # Test set results for cifar10 resnet18 relu:
# Final test losses: ['0.0006', '0.0005', '0.0006', '0.0006', '0.0006']
# Final accuracies: ['98.76', '98.84', '98.75', '98.77', '98.67']
# Training times: ['21.99', '21.31', '21.22', '21.24', '21.14']
# Average loss: 0.0006
# Average accuracy: 98.76%

# t-statistic: 4.1216 
# p-value: 0.0033

# The difference in accuracies is statistically significant, indicating that our method performs better than their method.

# Comparison: A2A
#   Common: 0.0065 ± 0.0006
#   Unique: 0.0059 ± 0.0006
#   Consistency: 0.3580 ± 0.0362
#   Diversity: 117.6000 ± 9.3188
# Comparison: B2B
#   Common: 0.0120 ± 0.0005
#   Unique: 0.0024 ± 0.0006
#   Consistency: 0.7173 ± 0.0221
#   Diversity: 47.2000 ± 3.8158
# Comparison: A2B
#   Common: 0.0052 ± 0.0005
#   Unique0: 0.0072 ± 0.0007
#   Unique1: 0.0091 ± 0.0008
#   Consistency: 0.2440 ± 0.0265
#   Diversity: 162.8800 ± 11.5076

# Consensus models

# Configs used: ['configs/mnist_abs.json']
# Average loss: 0.0015
# Average accuracy: 99.07%

# Configs used: ['configs/mnist_relu.json']
# Average loss: 0.0015
# Average accuracy: 98.57%

# Configs used: ['configs/mnist_abs.json', 'configs/mnist_relu.json']
# Average loss: 0.0015
# Average accuracy: 99.10%