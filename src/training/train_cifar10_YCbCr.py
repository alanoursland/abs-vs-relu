# src/training/train_cifar10_ycbcr.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import cv2
import numpy as np

from models.resnet18 import ResNet18
from data.cifar10_loader import load_cifar10
from utils.visualization import plot_loss_curves
from torchvision import datasets, transforms
from skimage.color import rgb2hed


class RGBToYCbCr:
    def __call__(self, img):
        return img.convert("YCbCr")

class CombinedColorSpace:
    def __call__(self, img):
        # Convert PIL image to numpy array
        img_np = np.array(img)
        
        # Convert RGB to HSV
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        # Convert RGB to YUV
        yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        
        # Convert RGB to LAB
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # Convert RGB to HED
        hed = rgb2hed(img_np)
        
        # Concatenate all color spaces along the channel dimension
        combined = np.concatenate((img_np, hsv, yuv, lab, hed), axis=-1)
        
        return combined

def train(model, device, train_loader, optimizer, scheduler, criterion, epoch, log_interval=100):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
    if scheduler is not None:
        scheduler.step()
    train_loss /= len(train_loader.dataset)
    return train_loss


def test(model, X_test, Y_test, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    sample_count = Y_test.size(0)
    with torch.no_grad():
        output = model(X_test)
        test_loss = criterion(output, Y_test).item() * sample_count
        pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
        correct += pred.eq(Y_test.view_as(pred)).sum().item()

    test_loss /= sample_count
    accuracy = 100.0 * correct / sample_count
    print(f"{epoch}: Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{sample_count} " f"({accuracy:.2f}%)")
    return test_loss, accuracy


def main(config):
    # Calculate the mean and variance of cifar-10 in YCbCr
    temp_transform = transforms.Compose([
            CombinedColorSpace(),
            transforms.ToTensor(),
        ])
    cifar10_dataset = datasets.CIFAR10(root="./datasets/CIFAR10", train=True, download=True, transform=temp_transform)
    loader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=len(cifar10_dataset), shuffle=False, num_workers=2)
    X_train = next(iter(loader))[0]
    mean = X_train.mean(dim=[0, 2, 3])
    stddev = X_train.std(dim=[0, 2, 3])
    X_train = loader = cifar10_dataset = None
    print(f"mean {mean} stddev {stddev}")

    # now the load the data for training
    transform_train = transforms.Compose([
        CombinedColorSpace(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), stddev.tolist()),
    ])

    transform_test = transforms.Compose([
        CombinedColorSpace(),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), stddev.tolist()),
    ])

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./datasets/CIFAR10', train=True, download=True, transform=transform_train), batch_size=config.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./datasets/CIFAR10', train=False, download=True, transform=transform_test), batch_size=50000, shuffle=False, num_workers=2)

    activation_function = config.get_activation_function(config.activation_function)
    model = ResNet18(num_classes=10, activation_function=activation_function, color_channels=15).to(config.device)

    # Get the entire test set in a single batch
    X_test, Y_test = next(iter(test_loader))

    # Move the data to GPU
    X_test = X_test.to(config.device)
    Y_test = Y_test.to(config.device)

    optimizer = optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    accuracies = []

    start_time = time.time()  # Record the start time

    for epoch in range(1, config.epochs + 1):
        train_loss = train(
            model, config.device, train_loader, optimizer, scheduler, criterion, epoch, scheduler=None, log_interval=config.log_interval
        )
        test_loss, accuracy = test(model, X_test, Y_test, criterion, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

    end_time = time.time()  # Record the end time
    training_time = end_time - start_time  # Calculate the total training time

    if config.save_model:
        model_save_path = os.path.join(config.run_dir, f"cifar10_resnet18_{config.activation_function}.pth")
        torch.save(model.state_dict(), model_save_path)

    plot_title = f"Error {config.dataset} {config.model} {config.activation} {config.run}"
    plot_loss_curves(
        train_losses,
        test_losses,
        title=plot_title,
        save_path=os.path.join(config.run_dir, "loss_curves.png"),
        show_plot=False,
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

# See: Gowda, Yuan. ColorNet: Investigating the Importance of Color Spaces for Image Classification. Computer Vision – ACCV 2018 (pp.581-596)

