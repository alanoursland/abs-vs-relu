# src/utils/metrics.py

import torch

def calculate_metrics(output, target):
    """
    Calculate performance metrics.
    
    Parameters:
    - output (torch.Tensor): Model output logits.
    - target (torch.Tensor): Ground truth labels.
    
    Returns:
    - accuracy (float): Accuracy of the predictions.
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(target)
    return accuracy

if __name__ == "__main__":
    # Example usage
    output = torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
    target = torch.tensor([2, 0])
    accuracy = calculate_metrics(output, target)
    print(f"Accuracy: {accuracy}%")
