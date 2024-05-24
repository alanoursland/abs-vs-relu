# src/data/mnist_loader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_mnist(batch_size=64, download=True, data_dir="./datasets/MNIST"):
    """
    Load the MNIST dataset.

    Parameters:
    - batch_size (int): Number of samples per batch.
    - download (bool): Whether to download the dataset if not found locally.
    - data_dir (str): Directory where the dataset is stored.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the test set.
    """
    # Define the transformations for the training and test sets
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Load the training and test sets
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)

    # Create DataLoader objects for the training and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = load_mnist()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
