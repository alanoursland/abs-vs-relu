# src/data/cifar10_loader.py
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets, transforms


class GPUDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return self.data.size(0)


def load_cifar10(batch_size=128, download=True, data_dir="./datasets/CIFAR10"):
    """
    Load the CIFAR-10 dataset.

    Parameters:
    - batch_size (int): Number of samples per batch.
    - download (bool): Whether to download the dataset if not found locally.
    - data_dir (str): Directory where the dataset is stored.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the test set.
    """

    # Data augmentation and normalization

    mean = (0.4914, 0.4822, 0.4465)
    stddev = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, stddev), 
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),                 # Converts to tensor (scales pixel values to [0, 1])
        transforms.Normalize(mean, stddev),  # Normalize with mean and stddev
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=1)

    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = load_cifar10()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
