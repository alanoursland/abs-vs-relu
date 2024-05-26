# src/data/cifar10_loader.py
import torch
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image


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


def load_cifar10(batch_size=64, download=True, data_dir="./datasets/CIFAR10", cuda_device="cuda"):
    """
    Load the CIFAR-10 dataset.

    Parameters:
    - batch_size (int): Number of samples per batch.
    - download (bool): Whether to download the dataset if not found locally.
    - data_dir (str): Directory where the dataset is stored.
    - use_gpu (bool): Whether to load the data onto the GPU.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the test set.
    """

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=download)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=download)

    X_train = train_dataset.data.astype(np.float32) / 255
    X_train -= tuple(np.mean(X_train, axis=(0, 1, 2)))
    X_train /= tuple(np.std(X_train, axis=(0, 1, 2)))

   

    # the transforms.ToTensor() should convert the datasets to Tensors and apply the transpose((0, 3, 1, 2))
    # to convert from PIL HWC to tensor CHW.

    X_train = torch.from_numpy(train_dataset.data.transpose(0, 3, 1, 2)).float().to(cuda_device)
    Y_train = torch.tensor(train_dataset.targets).to(cuda_device)

    X_test = torch.from_numpy(test_dataset.data.transpose(0, 3, 1, 2)).float().to(cuda_device)
    Y_test = torch.tensor(test_dataset.targets).to(cuda_device)

    mean = X_train.mean(dim=(0, 2, 3))
    X_train -= mean[:, None, None]
    std = X_train.std(dim=(0, 2, 3))
    X_train /= std[:, None, None]
    X_test = (X_test - mean[:, None, None]) / std[:, None, None]

    # pad the training data. we will crop it later.
    X_train = torch.nn.functional.pad(X_train, (4, 4, 4, 4), mode="constant", value=0)

    # we need synthetic data to train ResNet on CIFAR-10.
    # define a custom collate function for the DataLoader to do batch transformations
    def transform_synthetic_collate(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0).to(cuda_device)  # Stack image tensors into a batch tensor
        labels = torch.tensor(labels).to(cuda_device) # convert the list of label ids into a tensor

        batch_size, channels, height, width = images.size()

        i = torch.randint(0, 8, (batch_size,), device=cuda_device)    # Random i values for each image
        j = torch.randint(0, 8, (batch_size,), device=cuda_device)    # Random j values for each image
        flip = torch.randint(0, 2, (batch_size,), device=cuda_device) # Random flip values (0 or 1)

        # Apply random crop (assuming 4-pixel padding already applied) using i and j        
        i_start = i.view(batch_size, 1, 1, 1).expand(batch_size, channels, 32, width)
        j_start = j.view(batch_size, 1, 1, 1).expand(batch_size, channels, 32, 32)

        # Create range tensors for the indices to gather
        i_range = torch.arange(32, device=cuda_device).view(1, 1, 32, 1).expand(batch_size, channels, -1, width)
        j_range = torch.arange(32, device=cuda_device).view(1, 1, 1, 32).expand(batch_size, channels, -1, -1)

        # Calculate the new indices for cropping
        i_indices = i_start + i_range
        j_indices = j_start + j_range

        # Adjust images tensor dimensions for correct gather operation
        images = images.gather(2, i_indices)
        images = images.gather(3, j_indices)

        # Apply flip along the width direction (CHW)
        images[flip==1] = images[flip==1].flip(3)

        # the tensors are already normalized
        return images, labels

    # Create custom dataset objects using the tensors on the GPU
    train_dataset = GPUDataset(X_train, Y_train)
    test_dataset = GPUDataset(X_test, Y_test)

    # Create DataLoader objects for the training and test sets
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=transform_synthetic_collate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = load_cifar10()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")




# X_train = train_dataset.data.astype(np.float32) / 255
# X_train -= tuple(np.mean(X_train, axis=(0, 1, 2)))
# X_train /= tuple(np.std(X_train, axis=(0, 1, 2)))

# # Create a list to store the synthetic dataset
# num_copies = 10
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
# ])

# X_train_transposed = X_train.transpose(0, 3, 1, 2)
# X_train_pil = torchvision.transforms.ToPILImage()(X_train_transposed)
# synthetic_dataset = [torch.tensor(X_train_transposed).float()]
# for _ in range(1, num_copies):
#     X_train_pil = [Image.fromarray(img.transpose(1, 2, 0)) for img in X_train_transposed]
#     # Convert the numpy array to a PIL Image
#     # Apply the transformations using transform_train
#     transformed_img = transform_train(X_train_pil)

#     # Convert the transformed PIL Image back to a numpy array
#     transformed_data = torch.tensor(transformed_img).float()

#     # Append the transformed copy to the synthetic dataset
#     synthetic_dataset.append(transformed_data)

# synthetic_dataset_tensor = torch.stack(synthetic_dataset)


# # the transforms.ToTensor() should convert the datasets to Tensors and apply the transpose((0, 3, 1, 2))
# # to convert from PIL HWC to tensor CHW.

# X_train = torch.stack(synthetic_dataset)
# Y_train = torch.tensor(train_dataset.targets).to(cuda_device)

# X_test = torch.from_numpy(test_dataset.data.transpose(0, 3, 1, 2)).float().unsqueeze(0).to(cuda_device)
# Y_test = torch.tensor(test_dataset.targets).to(cuda_device)
