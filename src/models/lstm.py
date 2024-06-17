# src/models/lstm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import typing


def sigmoid_or_exp_abs(x: torch.Tensor, activation: str = "sigmoid") -> torch.Tensor:
    if activation == "sigmoid":
        return torch.sigmoid(x)
    elif activation == "exp_abs":
        return torch.exp(-torch.abs(x))
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


def exp_abs(x):
    return torch.exp(-torch.abs(x))


def create_activation(activation: str = "sigmoid"):
    if activation == "sigmoid":
        return torch.sigmoid
    elif activation == "exp_abs":
        return exp_abs
    elif activation == "relu":
        return torch.relu
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, activation: str = "sigmoid"):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = create_activation(activation)

        # Input to Hidden weights and biases
        self.w_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.b_ih = nn.Parameter(torch.empty(4 * hidden_size))

        # Hidden to Hidden weights and biases
        self.w_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.b_hh = nn.Parameter(torch.empty(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # for weight in self.parameters():
        #     if len(weight.shape) >= 2:
        #         nn.init.orthogonal_(weight)
        #     else:
        #         nn.init.zeros_(weight)
        stdv = 1.0 / (self.hidden_size**0.5)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self, token_batch: torch.Tensor, hx: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]]
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        # batch_size, embedding_size = token_batch.size()
        # Unpack the hidden state and cell state
        hidden_state, cell_state = hx   # [batch_size, hidden_size]

        # Compute the gates and candidate cell state
        input_gates = torch.matmul(token_batch, self.w_ih.t()) + self.b_ih.unsqueeze(0)
        hidden_gates = torch.matmul(hidden_state, self.w_hh.t()) + self.b_hh.unsqueeze(0)  # [batch_size, layer_hidden_units]

        # Combine input and forget gate contributions
        all_gates = input_gates + hidden_gates  # [batch_size, layer_hidden_units]

        # Split into individual gates and candidate cell state
        input_gate, forget_gate, candidate_cell_state, output_gate = all_gates.chunk(4, -1)

        # Apply activations to the gates
        input_gate = self.activation(input_gate)
        forget_gate = self.activation(forget_gate)
        output_gate = self.activation(output_gate)
        candidate_cell_state = torch.tanh(candidate_cell_state)

        # Update cell state
        cell_state = forget_gate * cell_state + input_gate * candidate_cell_state  # [64, 512, 100]

        # Update hidden state
        hidden_state = output_gate * torch.tanh(cell_state)  # [64, 512, 100]

        return hidden_state, cell_state


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        activation: str = "sigmoid",
    ):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation

        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = LSTMCell(layer_input_size, hidden_size, activation)
            self.layers.append(cell)

    def forward(
        self, input_seq: torch.Tensor, hx: typing.Tuple[torch.Tensor, torch.Tensor]
    ) -> typing.Tuple[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]:

        batch_size, seq_len, embedding_size = input_seq.size()

        hidden_state, cell_state = hx
        # hidden_state is [num_layers, batch_size, hidden_size]
        # cell_state is [num_layers, batch_size, hidden_size]

        h_next = torch.empty(self.num_layers, batch_size, self.hidden_size, device=input_seq.device)
        c_next = torch.empty(self.num_layers, batch_size, self.hidden_size, device=input_seq.device)
        for layer_index in range(self.num_layers):
            layer_cell = self.layers[layer_index]
            layer_hidden_state = hidden_state[layer_index, :, :]  # [batch_size, hidden_size]
            layer_cell_state = cell_state[layer_index, :, :]  # [batch_size, hidden_size]

            next_input = []
            for token_index in range(seq_len):
                token = input_seq[:, token_index, :]
                layer_hidden_state, layer_cell_state = layer_cell(token, (layer_hidden_state, layer_cell_state))
                next_input.append(layer_hidden_state)

            input_seq = torch.stack(next_input, 1)
            h_next[layer_index] = layer_hidden_state
            c_next[layer_index] = layer_cell_state

        return input_seq, (h_next, c_next)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_classes, embedding_dim, activation="sigmoid"):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(embedding_dim, hidden_size, num_layers, activation=activation)
        # self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        # torch.autograd.set_detect_anomaly(True)

    def forward(self, x):
        embedded = self.embedding(x).float()

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(embedded, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

# All experiments are run for 10 epochs

# This is the torch library implementation for LSTM. It is included to validate my implementation.
# Test set results for experiments/IMDB_LSTM_ref:
# Final test losses: ['0.5230', '0.5230', '0.5230', '0.5230', '0.5230']
# Final accuracies: ['81.92', '81.92', '81.92', '81.92', '81.92']
# Training times: ['87.51', '89.52', '89.53', '89.15', '89.09']
# Average loss: 0.5230
# Average accuracy: 81.92%

