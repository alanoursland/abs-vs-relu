import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)

# Check for GPU availability
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
    device = torch.device("cuda")
else:
    print("CUDA is not available")
    device = torch.device("cpu")

# Generate 2D Gaussian cluster
def generate_gaussian_cluster(n_samples=1000):
    mean = torch.tensor([3.0, 2.0], device=device)
    cov = torch.tensor([[2.0, -1.0], [-1.0, 1.0]], device=device)
    dist = torch.distributions.MultivariateNormal(mean, cov)
    samples = dist.sample((n_samples,))
    return samples, mean, cov

# Calculate Mahalanobis distance
def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    inv_cov = torch.inverse(cov)
    return torch.sqrt(torch.einsum('...i,ij,...j->...', diff, inv_cov, diff))

# Modified LinearUnit for Abs model
class AbsLinearUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.abs(x)
        x = x ** 2
        x = self.linear2(x)
        return torch.sqrt(torch.relu(x))  # abs to ensure non-negative under sqrt

# Modified LinearUnit for ReLU model
class ReLULinearUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.offset = nn.Parameter(torch.randn(2))
        self.linear2 = nn.Linear(2, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = x + self.offset
        x = x ** 2
        x = self.linear2(x)
        return torch.sqrt(torch.relu(x))  # abs to ensure non-negative under sqrt

# Generate data
samples, mean, cov = generate_gaussian_cluster()
mahalanobis_dist = mahalanobis_distance(samples, mean, cov)

# Prepare data for training
X = samples
y = mahalanobis_dist.unsqueeze(1)

# Create and train models
abs_model = AbsLinearUnit().to(device)
relu_model = ReLULinearUnit().to(device)

def train_model(model, X, y, epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    return losses

abs_losses = train_model(abs_model, X, y, epochs=200)
relu_losses = train_model(relu_model, X, y, epochs=200)

# Move data to CPU for plotting
samples_cpu = samples.cpu().numpy()
abs_losses_cpu = np.array(abs_losses)
relu_losses_cpu = np.array(relu_losses)

range_min = -10
range_max = 10

def plot_model_parameters(ax, model, samples, color):
    if isinstance(model, AbsLinearUnit):
        W = model.linear1.weight.data.cpu().numpy()
    else:  # ReLULinearUnit
        W = model.linear1.weight.data.cpu().numpy()
        offset = model.offset.data.cpu().numpy()
    
    mean = samples.mean(axis=0).cpu().numpy()
    
    # Draw the W vectors
    scale = 2.0
    for i in range(W.shape[0]):
        ax.arrow(mean[0], mean[1], scale*W[i,0], scale*W[i,1], 
                 color=color, width=0.01, head_width=0.05, zorder=3)
        
        if isinstance(model, AbsLinearUnit):
            ax.arrow(mean[0], mean[1], -scale*W[i,0], -scale*W[i,1], 
                     color=color, width=0.01, head_width=0.05, zorder=3)
        else:  # ReLULinearUnit
            # Draw offset
            ax.arrow(mean[0]+scale*W[i,0], mean[1]+scale*W[i,1], 
                     offset[i]*scale*W[i,0], offset[i]*scale*W[i,1], 
                     color='green', width=0.01, head_width=0.05, zorder=3)

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot error curves (log scale)
ax1.semilogy(abs_losses_cpu, label='Abs')
ax1.semilogy(relu_losses_cpu, label='ReLU')
ax1.set_title('Error Curves (Log Scale)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Log(Loss)')
ax1.legend()

# Plot for Abs model
ax2.scatter(samples_cpu[:, 0], samples_cpu[:, 1], alpha=0.3, s=10)
plot_model_parameters(ax2, abs_model, samples, 'r')
ax2.set_title('Abs Model')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_xlim(range_min, range_max)
ax2.set_ylim(range_min, range_max)

# Plot for ReLU model
ax3.scatter(samples_cpu[:, 0], samples_cpu[:, 1], alpha=0.3, s=10)
plot_model_parameters(ax3, relu_model, samples, 'b')
ax3.set_title('ReLU Model')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_xlim(range_min, range_max)
ax3.set_ylim(range_min, range_max)

plt.tight_layout()
plt.savefig('gaussian_cluster_experiment2.png')
# plt.show()

# Calculate and print average errors
with torch.no_grad():
    abs_pred = abs_model(X)
    relu_pred = relu_model(X)
    
    abs_error = torch.mean(torch.abs(abs_pred - y)).item()
    relu_error = torch.mean(torch.abs(relu_pred - y)).item()

print(f"Average Abs Error: {abs_error:.4f}")
print(f"Average ReLU Error: {relu_error:.4f}")

# print("Abs model linear1 weight:", abs_model.linear1.weight.data.cpu().numpy())
# print("Abs model linear2 weight:", abs_model.linear2.weight.data.cpu().numpy())
# print("ReLU model linear1 weight:", relu_model.linear1.weight.data.cpu().numpy())
# print("ReLU model linear2 weight:", relu_model.linear2.weight.data.cpu().numpy())
# print("ReLU model offset:", relu_model.offset.data.cpu().numpy())