# src/data/winequality_loader.py
import os
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class WineQualityDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target

def download_winequality(data_dir):
    """Download the Wine Quality dataset if it is not already downloaded."""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    local_filename = os.path.join(data_dir, 'winequality-red.csv')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(local_filename):
        print("Downloading the Wine Quality dataset...")
        response = requests.get(url)
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        print("Download completed.")
    else:
        print("Wine Quality dataset already exists.")

def load_winequality(batch_size=64, data_dir="./datasets/WineQuality", test_size=0.2, random_state=42):
    """
    Load the Wine Quality dataset.

    Parameters:
    - batch_size (int): Number of samples per batch.
    - data_dir (str): Directory where the dataset is stored.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for shuffling the data.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the test set.
    """
    # Download the dataset if it doesn't exist
    download_winequality(data_dir)

    # Load the dataset
    data_path = f"{data_dir}/winequality-red.csv"
    df = pd.read_csv(data_path, delimiter=';')

    # Split features and target
    X = df.drop('quality', axis=1).values
    y = df['quality'].values

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create custom dataset objects
    train_dataset = WineQualityDataset(X_train, y_train)
    test_dataset = WineQualityDataset(X_test, y_test)

    # Create DataLoader objects for the training and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = load_winequality()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
