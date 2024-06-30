import numpy as np
import os
import torch
from models.lenet import LeNet
from data.mnist_loader import load_mnist

def load_pretrained_model(model, output_dir, experiment_run):
    model_path = os.path.join(output_dir, experiment_run, 'mnist_lenet_relu.pth')
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path))
        return model
    raise FileNotFoundError(f"No model weights found in {output_dir} for run {experiment_run}")

def evaluate_layer(model, x, layer_index):
    x = x.view(-1, 784)  # Flatten the input for MNIST
    y = model.partial(x, layer_index)
    return y

def layer_weighted_mean(x, y):
    # Calculate the inverse squared output of the node
    inverse_squared_output = 1 / (y ** 2)
    
    # Calculate the weighted mean of input features
    weighted_sum = torch.sum(x * inverse_squared_output.unsqueeze(-1), dim=0)
    normalization_factor = torch.sum(inverse_squared_output)
    weighted_mean = weighted_sum / normalization_factor
    
    return weighted_mean

def extract_patches(x, kernel_size, stride, padding):
    # Extract patches similar to how a convolution operation would
    x_unfold = F.unfold(x, kernel_size=kernel_size, stride=stride, padding=padding)
    return x_unfold

def interpret_node_outputs(model, x, layer_index, layer_type='fc1'):
    with torch.no_grad():
        node_outputs = evaluate_layer(model, x, layer_index)
        if 'conv' in layer_type:
            kernel_size, stride, padding = (5, 1, 2) if layer_type == 'conv1' else (5, 1, 0)
            patches = extract_patches(x, kernel_size=kernel_size, stride=stride, padding=padding)
            patches = patches.permute(0, 2, 1)  # Reorder for batch-wise operations
        else:
            x = x.view(-1, 1, 28, 28)
        interpreted_features = layer_weighted_mean(x, node_outputs)
    return interpreted_features

def main(config):
    _, test_loader = load_mnist(batch_size=config.batch_size, cuda_device=config.device, use_gpu=True)

    activation_function = config.get_activation_function(config.activation_function)
    model = LeNet(activation_function=activation_function).to(config.device)

    # Load pretrained model
    model = load_pretrained_model(model, config.output_dir, '0001')

    x, _ = next(iter(test_loader))
    interpreted_features = interpret_node_outputs(model, x, layer_index=2, layer_type="conv")
    print("Interpreted features:", interpreted_features)

