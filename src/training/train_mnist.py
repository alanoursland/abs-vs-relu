# src/training/train_mnist.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time

from models.lenet import LeNet
from data.mnist_loader import load_mnist
from utils.metrics import calculate_metrics
from utils.visualization import plot_loss_curves
from src.config import Config  # Import Config class


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

    # Print summary of test dataset results
    print(f"Test set results for {config.activation_function}:")
    print(f"Average loss: {sum(test_losses)/len(test_losses):.4f}")
    print(f"Average accuracy: {sum(accuracies)/len(accuracies):.2f}%")

    # Print final loss and accuracy for each run in list format
    print("Final test losses:", [f"{loss:.4f}" for loss in test_losses])
    print("Final accuracies:", [f"{acc:.2f}" for acc in accuracies])