# src/models/__init__.py

# Initialize the models package
from .lenet import LeNet
from .resnet18 import ResNet18
from .lstm import LSTM
from .adult_mlp import AdultIncomeMLP
from .winequality_mlp import WineQualityMLP
from .iris_mlp import IrisMLP

__all__ = ['LeNet', 'ResNet18', 'LSTM', 'AdultIncomeMLP', 'WineQualityMLP', 'IrisMLP']