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

# Single linear unit with different activation functions
class LinearUnit(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x))

# Generate data
samples, mean, cov = generate_gaussian_cluster()
mahalanobis_dist = mahalanobis_distance(samples, mean, cov)

# Prepare data for training
X = samples
y = mahalanobis_dist.unsqueeze(1)

# Create and train models
abs_model = LinearUnit(torch.abs).to(device)
relu_model = LinearUnit(nn.ReLU()).to(device)

def train_model(model, X, y, epochs=1000):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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

abs_losses = train_model(abs_model, X, y, epochs=100)
relu_losses = train_model(relu_model, X, y, epochs=100)

# Move data to CPU for plotting
samples_cpu = samples.cpu().numpy()
abs_losses_cpu = np.array(abs_losses)
relu_losses_cpu = np.array(relu_losses)

range_min = -10
range_max = 10

def plot_model_parameters(ax, model, samples, color):
    W = model.linear.weight.data.cpu().numpy()[0]
    b = model.linear.bias.data.cpu().item()
    
    # Calculate the anchor point (projection of mean onto the line)
    mean = samples.mean(axis=0).cpu().numpy()
    t = -(W.dot(mean) + b) / (W.dot(W))
    anchor = mean + t * W
    
    # Calculate the line 0 = Wx + b
    x = np.linspace(range_min, range_max, 100)
    y = (-W[0] * x - b) / W[1]
    
    # Draw the separator line
    ax.plot(x, y, color=color, linestyle='--', linewidth=1, alpha=0.7)
    
    # Draw the anchor point
    ax.scatter(anchor[0], anchor[1], color=color, s=50, zorder=3)
    
    # Draw the W vector(s)
    scale = 2.0  # Adjust this to change the length of the arrow
    if isinstance(model.activation, torch.nn.ReLU):
        ax.arrow(anchor[0], anchor[1], scale*W[0], scale*W[1], 
                 color=color, width=0.01, head_width=0.05, zorder=3)
    else:  # Abs activation
        ax.arrow(anchor[0], anchor[1], scale*W[0], scale*W[1], 
                 color=color, width=0.01, head_width=0.05, zorder=3)
        ax.arrow(anchor[0], anchor[1], -scale*W[0], -scale*W[1], 
                 color=color, width=0.01, head_width=0.05, zorder=3)


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
plt.savefig('gaussian_cluster_experiment.png')
# plt.show()

# Calculate and print average errors
with torch.no_grad():
    abs_pred = abs_model(X)
    relu_pred = relu_model(X)
    
    abs_error = torch.mean(torch.abs(abs_pred - y)).item()
    relu_error = torch.mean(torch.abs(relu_pred - y)).item()

print(f"Average Abs Error: {abs_error:.4f}")
print(f"Average ReLU Error: {relu_error:.4f}")

print("Abs model weight:", abs_model.linear.weight.data.cpu().numpy())
print("Abs model bias:", abs_model.linear.bias.data.cpu().numpy())
print("ReLU model weight:", relu_model.linear.weight.data.cpu().numpy())
print("ReLU model bias:", relu_model.linear.bias.data.cpu().numpy())
