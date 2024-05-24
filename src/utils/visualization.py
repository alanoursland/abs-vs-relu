# src/utils/visualization.py

import matplotlib.pyplot as plt

def plot_loss_curves(train_losses, test_losses, save_path=None):
    """
    Plot training and test loss curves.
    
    Parameters:
    - train_losses (list of float): Training losses for each epoch.
    - test_losses (list of float): Test losses for each epoch.
    - save_path (str): Path to save the plot image (optional).
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, test_losses, 'ro-', label='Test loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_activation_distribution(activation_values, title='Activation Distribution', save_path=None):
    """
    Plot the distribution of activation values.
    
    Parameters:
    - activation_values (list of float): Activation values to plot.
    - title (str): Title of the plot.
    - save_path (str): Path to save the plot image (optional).
    """
    plt.figure()
    plt.hist(activation_values, bins=50, alpha=0.75)
    plt.title(title)
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Example usage
    train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
    test_losses = [0.6, 0.5, 0.4, 0.35, 0.3]
    plot_loss_curves(train_losses, test_losses)
    
    activation_values = [0.1, 0.2, 0.3, 0.4, 0.5] * 100
    plot_activation_distribution(activation_values)
