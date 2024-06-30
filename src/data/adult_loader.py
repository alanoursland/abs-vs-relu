# src/data/adult_loader.py
import os
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class AdultDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target

def download_adult(data_dir):
    """Download the Adult dataset if it is not already downloaded."""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    local_filename = os.path.join(data_dir, 'adult.data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(local_filename):
        print("Downloading the Adult dataset...")
        response = requests.get(url)
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        print("Download completed.")
    else:
        print("Adult dataset already exists.")

def load_adult(batch_size=64, data_dir="./datasets/Adult", test_size=0.2, random_state=42):
    """
    Load the UCI Adult dataset.

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
    download_adult(data_dir)

    # Column names for the dataset
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]

    # Load the dataset
    data_path = f"{data_dir}/adult.data"
    df = pd.read_csv(data_path, names=column_names, na_values=' ?', skipinitialspace=True)

    # Drop rows with missing values
    df = df.dropna()

    # Encode categorical features
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Split features and target
    X = df.drop('income', axis=1).values
    y = df['income'].values

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    # Create custom dataset objects
    train_dataset = AdultDataset(X_train, y_train)
    test_dataset = AdultDataset(X_test, y_test)

    # Create DataLoader objects for the training and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = load_adult()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
