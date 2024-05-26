# src/training/train_cifar10.py

# from https://github.com/kuangliu/pytorch-cifar

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

from models.resnet18 import ResNet18
from data.cifar10_loader import load_cifar10
from utils.metrics import calculate_metrics
from utils.visualization import plot_loss_curves
from src.config import Config  # Import Config class


def train(model, device, train_loader, optimizer,  scheduler, criterion,epoch, log_interval=100):
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
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
    if scheduler != None:
        scheduler.step()
    train_loss /= len(train_loader.dataset)
    return train_loss


def test(model, X_test, Y_test, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    sample_count = Y_test.size(0)
    with torch.no_grad():
        output = model(X_test)
        test_loss = criterion(output, Y_test).item() * sample_count
        pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
        correct += pred.eq(Y_test.view_as(pred)).sum().item()

    test_loss /= sample_count
    accuracy = 100.0 * correct / sample_count
    print(
        f"{epoch}: Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{sample_count} " f"({accuracy:.2f}%)"
    )
    return test_loss, accuracy


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
        train_loss = train(model, config.device, train_loader, optimizer, scheduler, criterion, epoch, config.log_interval)
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


# Test set results for cifar10 resnet18 abs:
# Final test losses: ['0.5235', '0.4961', '0.5060', '0.5053', '0.5353']
# Final accuracies: ['90.47', '90.64', '90.62', '90.48', '90.10']
# Training times: ['2167.56', '2172.71', '2172.39', '2175.42', '2172.23']
# Average loss: 0.5132
# Average accuracy: 90.46%

# Test set results for cifar10 resnet18 relu:
# Final test losses: ['0.3374', '0.3347', '0.3517', '0.3124', '0.3039']
# Final accuracies: ['92.67', '93.22', '92.46', '93.22', '93.33']
# Training times: ['2257.45', '2201.36', '2160.03', '2135.48', '2128.06']
# Average loss: 0.3280
# Average accuracy: 92.98%

# t-statistic: -12.65118997027362 (abs vs relu)
# p-value: 1.4317883786698362e-06
#The difference in accuracies is statistically significant, indicating that our method performs worse than their method.