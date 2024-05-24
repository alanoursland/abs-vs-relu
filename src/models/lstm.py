# src/models/lstm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, activation_function=nn.ReLU):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation_function
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.activation()(out)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    # Example usage
    input_dim = 28
    hidden_dim = 128
    output_dim = 10
    n_layers = 2
    batch_size = 32
    seq_length = 28
    
    model = LSTM(input_dim, hidden_dim, output_dim, n_layers)
    print(model)
    
    input_tensor = torch.randn(batch_size, seq_length, input_dim)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
