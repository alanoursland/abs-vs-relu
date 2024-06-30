import torch.nn as nn

class IrisMLP(nn.Module):
    def __init__(self, activation_function=nn.ReLU):
        super(IrisMLP, self).__init__()
        self.layer1 = nn.Linear(4, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 3)
        self.activation = activation_function()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)  # No activation on the final layer, as it's a classification problem and we'll apply softmax/cross-entropy loss separately
        return x
