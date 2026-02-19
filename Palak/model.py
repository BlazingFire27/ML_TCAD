import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for silicon oxidation.

    Architecture: feedforward network with tanh activations.
    tanh is used (not ReLU) because we need smooth second derivatives
    for the physics loss (diffusion equation requires d2u/dx2).

    Input:  6 features [x, pres, o2, n2, temp, time] — normalized to [0,1]
    Output: 1 value    [log10(Y)] — log oxide concentration
    """

    def __init__(self, input_dim=6, hidden_dim=64, n_layers=4):
        super().__init__()

        layers = []

        # input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        # hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # output layer — no activation, raw regression output
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)