# src/training/train_imdb.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

from models.lstm import LSTMClassifier
from data.imdb_loader import load_imdb
from utils.visualization import plot_loss_curves
from training.train_utils import train, test


def main(config):
    train_loader, test_loader, tokenizer = load_imdb(batch_size=config.batch_size)

    vocab_size = len(tokenizer.vocab)
    embedding_dim = 768  # Choose an appropriate embedding dimension
    model = LSTMClassifier(
        vocab_size,
        config.hidden_size,
        config.num_layers,
        num_classes=2,
        embedding_dim=embedding_dim,
        activation=config.activation,
    ).to(config.device)

    # Get the entire test set in a single batch
    # X_test, Y_test = next(iter(test_loader))

    # # Move the data to GPU
    # X_test = X_test.to(config.device)
    # Y_test = Y_test.to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = None

    train_losses = []
    test_losses = []
    accuracies = []

    start_time = time.time()  # Record the start time

    for epoch in range(1, config.epochs + 1):
        train_loss = train(
            model, config.device, train_loader, optimizer, criterion, epoch, scheduler, config.log_interval
        )
        test_loss, accuracy = test(model, test_loader, criterion, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

    end_time = time.time()  # Record the end time
    training_time = end_time - start_time  # Calculate the total training time

    if config.save_model:
        model_save_path = os.path.join(config.run_dir, f"imdb_lstm.pth")
        torch.save(model.state_dict(), model_save_path)

    plot_title = f"Error {config.dataset} {config.model} {config.run}"
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


# All experiments are run for 10 epochs

# This is the torch library implementation for LSTM. It is included to validate my implementation.
# Test set results for experiments/IMDB_LSTM_ref:
# Final test losses: ['0.5230', '0.4916', '0.5270', '0.5054', '0.5304']
# Final accuracies: ['81.92', '79.84', '82.61', '81.42', '81.36']
# Training times: ['85.53', '84.88', '85.03', '85.25', '86.80']
# Average loss: 0.5155
# Average accuracy: 81.43%

# Test set results for imdb lstm relu:
# Final test losses: ['0.4997', '0.4867', '0.4858', '0.5282', '0.5007']
# Final accuracies: ['81.74', '81.57', '80.59', '81.66', '82.24']
# Training times: ['5716.51', '5675.51', '5667.90', '5722.77', '5703.18']
# Average loss: 0.5002
# Average accuracy: 81.56%
# vs reference: t-statistic: -0.2453780879552636, p-value: 0.8123412227561574

# Test set results for experiments/IMDB_LSTM_abs:
# Final test losses: ['0.4997', '0.4867', '0.4858', '0.5282', '0.5007']
# Final accuracies: ['81.74', '81.57', '80.59', '81.66', '82.24']
# Training times: ['5674.42', '5722.20', '5659.20', '5650.84', '5669.65']
# Average loss: 0.5002
# Average accuracy: 81.56%
# vs reference: t-statistic: -0.2453780879552636, p-value: 0.8123412227561574
