import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

from models.winequality_mlp import WineQualityMLP
from data.winequality_loader import load_winequality
from utils.visualization import plot_loss_curves
from training.train_utils import train, test_regression

def main(config):
    train_loader, test_loader = load_winequality(batch_size=config.batch_size)

    activation_function = config.get_activation_function(config.activation_function)
    model = WineQualityMLP(activation_function=activation_function).to(config.device)

    # Get the entire test set in a single batch
    X_test, Y_test = next(iter(test_loader))

    # Move the data to GPU
    X_test = X_test.to(config.device)
    Y_test = Y_test.to(config.device)

    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []
    accuracies = []

    start_time = time.time()  # Record the start time

    for epoch in range(1, config.epochs + 1):
        train_loss = train(model, config.device, train_loader, optimizer, criterion, epoch, scheduler=None, log_interval=config.log_interval)
        test_loss = test_regression(model, X_test, Y_test, criterion, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(0.0)

    end_time = time.time()  # Record the end time
    training_time = end_time - start_time  # Calculate the total training time

    if config.save_model:
        model_save_path = os.path.join(config.run_dir, f"winequality_winequalitymlp_{config.activation_function}.pth")
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



# Test set results for winequality WineQualityMLP abs:
# Final test losses: ['1.0050', '0.9913', '0.9619', '0.9465', '0.9981']
# Final accuracies: ['0.00', '0.00', '0.00', '0.00', '0.00']
# Training times: ['0.39', '0.30', '0.29', '0.29', '0.28']
# Average loss: 0.9805

# Test set results for winequality WineQualityMLP relu:
# Final test losses: ['0.9634', '1.0342', '1.0300', '0.9888', '0.9724']
# Final accuracies: ['0.00', '0.00', '0.00', '0.00', '0.00']
# Training times: ['0.68', '0.58', '0.60', '0.60', '0.61']
# Average loss: 0.9978

