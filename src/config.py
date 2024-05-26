import torch
import torch.nn as nn

class Config:
    def __init__(self, config_dict=None, cuda_device="cuda"):
        if config_dict is None:
            config_dict = {}

        # description
        self.dataset = config_dict.get("dataset", "unknown_dataset")
        self.model = config_dict.get("model", "unknown_model")
        self.activation = config_dict.get("activation", "relu")

        # Training configurations
        self.batch_size = config_dict.get("batch_size", 64)
        self.test_batch_size = config_dict.get("test_batch_size", 1000)
        self.epochs = config_dict.get("epochs", 10)
        self.learning_rate = config_dict.get("learning_rate", 0.01)
        self.momentum = config_dict.get("momentum", 0.5)
        self.weight_decay = config_dict.get("weight_decay", 5e-4)
        self.seed = config_dict.get("seed", 1)
        self.log_interval = config_dict.get("log_interval", 100)
        self.save_model = config_dict.get("save_model", True)

        # Model configurations
        self.activation_function = config_dict.get("activation", 'relu')

        # Device configurations
        self.no_cuda = config_dict.get("no_cuda", False)
        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device(cuda_device if self.use_cuda else "cpu")

        # Paths
        self.dataset_dir = config_dict.get("dataset_dir", './datasets/')
        self.model_save_dir = config_dict.get("model_save_dir", './models/')
        self.results_dir = config_dict.get("results_dir", './results/')
        self.output_dir = config_dict.get("output_dir", './results/')
        self.num_runs = config_dict.get("num_runs", 5)
        self.run_dir = None

    @staticmethod
    def get_activation_function(name):
        activation_functions = {
            'relu': nn.ReLU,
            'abs': lambda: Abs()  # Ensure you have the Abs activation function defined somewhere
        }
        return activation_functions[name]()

# Define the Abs activation function if not already defined
class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)
