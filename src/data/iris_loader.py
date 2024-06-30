# src/data/iris_loader.py
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class IrisDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target

def load_iris(batch_size=32, test_size=0.2, random_state=42):
    """
    Load the Iris dataset.

    Parameters:
    - batch_size (int): Number of samples per batch.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for shuffling the data.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the test set.
    """

    # Load the dataset
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    # Encode the target labels
    label_encoder = LabelEncoder()
    df["class"] = label_encoder.fit_transform(df["class"])

    # Split features and target
    X = df.drop("class", axis=1).values
    y = df["class"].values

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
    train_dataset = IrisDataset(X_train, y_train)
    test_dataset = IrisDataset(X_test, y_test)

    # Create DataLoader objects for the training and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = load_iris()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
