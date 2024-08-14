import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

from models.adult_mlp import AdultIncomeMLP
from data.adult_loader import load_adult
from utils.visualization import plot_loss_curves
from training.train_utils import train, test_fast

def main(config):
    train_loader, test_loader = load_adult(batch_size=config.batch_size)

    activation_function = config.get_activation_function(config.activation_function)
    model = AdultIncomeMLP(activation_function=activation_function).to(config.device)

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
        model_save_path = os.path.join(config.run_dir, f"adult_adultincomemlp_{config.activation_function}.pth")
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


# Test set results for experiments/Adult_Income_Abs:
# Final test losses: ['0.3232', '0.3262', '0.3222', '0.3283', '0.3230']
# Final accuracies: ['85.12', '84.68', '85.12', '84.75', '84.78']
# Training times: ['5.79', '5.67', '5.76', '5.84', '5.83']
# Average loss: 0.3246
# Average accuracy: 84.89%

# Test set results for experiments/Adult_Income_ReLU:
# Final test losses: ['0.3172', '0.3172', '0.3201', '0.3175', '0.3160']
# Final accuracies: ['85.31', '85.75', '85.37', '85.44', '85.34']
# Training times: ['5.75', '5.87', '5.63', '5.75', '5.87']
# Average loss: 0.3176
# Average accuracy: 85.44%

# t-statistic: -4.388757941937284, p-value: 0.002321402671931207

# The difference in accuracies is statistically significant, indicating that our method performs worse than their method.

# Comparison: A2A
#   Common: 0.1101 ± 0.0117
#   Unique: 0.0410 ± 0.0121
#   Consistency: 0.5797 ± 0.0990
#   Diversity: 533.6000 ± 156.2173
# Comparison: B2B
#   Common: 0.1276 ± 0.0030
#   Unique: 0.0179 ± 0.0027
#   Consistency: 0.7808 ± 0.0272
#   Diversity: 233.8000 ± 31.5620
# Comparison: A2B
#   Common: 0.1183 ± 0.0049
#   Unique0: 0.0328 ± 0.0057
#   Unique1: 0.0273 ± 0.0046
#   Consistency: 0.6649 ± 0.0469
#   Diversity: 390.9200 ± 65.8768

# Configs used: ['configs/adult_abs.json']
# Average loss: 0.0001
# Average accuracy: 85.54%

# Configs used: ['configs/adult_relu.json']
# Average loss: 0.0001
# Average accuracy: 85.57%

# Configs used: ['configs/adult_abs.json', 'configs/adult_relu.json']
# Average loss: 0.0001
# Average accuracy: 85.54%
