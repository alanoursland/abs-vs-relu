# src/models/lenet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, activation_function=nn.ReLU):
        super(LeNet, self).__init__()
        self.activation = activation_function
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.activation()(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.activation()(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16 * 5 * 5)
        x = self.activation()(self.fc1(x))
        x = self.activation()(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # Example usage
    model = LeNet()
    print(model)
    # Create a random tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 1, 28, 28)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
