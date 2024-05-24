# src/models/__init__.py

# Initialize the models package
from .lenet import LeNet
from .resnet18 import ResNet18
from .lstm import LSTM
from .mlp import MLP

__all__ = ['LeNet', 'ResNet18', 'LSTM', 'MLP']