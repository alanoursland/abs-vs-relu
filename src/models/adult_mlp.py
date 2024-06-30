import torch.nn as nn

class AdultIncomeMLP(nn.Module):
    def __init__(self, activation_function=nn.ReLU):
        super(AdultIncomeMLP, self).__init__()
        self.layer1 = nn.Linear(14, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.activation = activation_function

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x
