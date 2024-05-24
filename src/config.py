# src/config.py

import torch
import torch.nn as nn

# Default configurations
class Config:
    # Training configurations
    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    learning_rate = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 100
    save_model = True
    
    # Model configurations
    activation_function = 'relu'
    
    # Device configurations
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Paths
    dataset_dir = './datasets/'
    model_save_dir = './models/'
    results_dir = './results/'

    # Activation functions mapping
    activation_functions = {
        'relu': nn.ReLU,
        'abs': lambda: torch.abs
    }

# Function to get activation function
def get_activation_function(name):
    return Config.activation_functions[name]

# Seed for reproducibility
torch.manual_seed(Config.seed)

if Config.use_cuda:
    torch.cuda.manual_seed(Config.seed)
