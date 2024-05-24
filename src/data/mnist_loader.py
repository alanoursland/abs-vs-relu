# src/data/mnist_loader.py
import torch
from torchvision import datasets, transforms
import torch.utils.data


class GPUDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


def load_mnist(batch_size=64, download=True, data_dir="./datasets/MNIST", cuda_device="cuda", use_gpu=False):
    """
    Load the MNIST dataset.

    Parameters:
    - batch_size (int): Number of samples per batch.
    - download (bool): Whether to download the dataset if not found locally.
    - data_dir (str): Directory where the dataset is stored.
    - use_gpu (bool): Whether to load the data onto the GPU.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the test set.
    """
    # Define the transformations for the training and test sets
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)

    if use_gpu:
        # Flatten the images
        X_train = train_dataset.data.float().unsqueeze(1).to(cuda_device)
        X_test = test_dataset.data.float().unsqueeze(1).to(cuda_device)

        # Get the labels
        Y_train = train_dataset.targets.to(cuda_device)
        Y_test = test_dataset.targets.to(cuda_device)

        # Create custom dataset objects using the GPU data
        train_dataset = GPUDataset(X_train, Y_train)
        test_dataset = GPUDataset(X_test, Y_test)

    # Create DataLoader objects for the training and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = load_mnist(use_gpu=True)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
