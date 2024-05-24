import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(
    train_losses, test_losses, title="Training and Test Loss (Log Scale)", save_path=None, show_plot=True
):
    """
    Plot training and test loss curves.

    Parameters:
    - train_losses (list of float): Training losses for each epoch.
    - test_losses (list of float): Test losses for each epoch.
    - save_path (str): Path to save the plot image (optional).
    - show_plot (bool): Whether to display the plot on the screen (default: True).
    """
    epochs = range(1, len(train_losses) + 1)
    log_train_losses = np.log(train_losses)
    log_test_losses = np.log(test_losses)

    plt.figure()
    plt.plot(epochs, log_train_losses, "bo-", label="Training loss (log scale)")
    plt.plot(epochs, log_test_losses, "ro-", label="Test loss (log scale)")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_activation_distribution(activation_values, title="Activation Distribution", save_path=None, show_plot=True):
    """
    Plot the distribution of activation values.

    Parameters:
    - activation_values (list of float): Activation values to plot.
    - title (str): Title of the plot.
    - save_path (str): Path to save the plot image (optional).
    - show_plot (bool): Whether to display the plot on the screen (default: True).
    """
    plt.figure()
    plt.hist(activation_values, bins=50, alpha=0.75)
    plt.title(title)
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Example usage
    train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
    test_losses = [0.6, 0.5, 0.4, 0.35, 0.3]
    plot_loss_curves(train_losses, test_losses, show_plot=True)

    activation_values = [0.1, 0.2, 0.3, 0.4, 0.5] * 100
    plot_activation_distribution(activation_values, show_plot=True)
