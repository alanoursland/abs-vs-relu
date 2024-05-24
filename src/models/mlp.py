# src/models/mlp.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, activation_function=nn.ReLU):
        super(MLP, self).__init__()
        self.activation = activation_function

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self.activation())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    # Example usage
    input_dim = 784  # For MNIST
    hidden_dim = 128
    output_dim = 10
    n_layers = 2
    batch_size = 32

    model = MLP(input_dim, hidden_dim, output_dim, n_layers)
    print(model)

    input_tensor = torch.randn(batch_size, input_dim)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
