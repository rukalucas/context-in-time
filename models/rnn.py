"""RNN model with continuous-time dynamics."""

import torch
import torch.nn as nn
import numpy as np


class RNN(nn.Module):
    """
    Continuous-time RNN with exponential decay.
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        output_size: int = 2,
        tau: float = 100.0,  # ms
        dt: float = 20.0,  # ms
        activation: str = "elu",
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.tau = tau
        self.dt = dt
        self.noise_std = noise_std

        # Decay factor
        self.alpha = dt / tau

        # Activation function
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Layers
        self.w_in = nn.Linear(input_size, hidden_size, bias=True)
        self.w_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_out = nn.Linear(hidden_size, output_size, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights according to spec."""
        std = 1.0 / np.sqrt(self.hidden_size)
        nn.init.normal_(self.w_rec.weight, mean=0.0, std=std)

        # Input weights: Xavier/He initialization
        nn.init.xavier_uniform_(self.w_in.weight)

        # Output weights: Xavier/He initialization
        nn.init.xavier_uniform_(self.w_out.weight)

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the RNN for a single timestep.

        Args:
            input: [batch_size, input_size] - single timestep input
            hidden: [batch_size, hidden_size] - current hidden state

        Returns:
            output: [batch_size, output_size] - output at this timestep
            new_hidden: [batch_size, hidden_size] - updated hidden state
        """
        batch_size = input.shape[0]

        # Continuous-time dynamics: τ * dh/dt = -h + φ(i(t))
        # where i(t) = W_rec @ h(t) + W_in @ u(t) + b_in + ξ(t)
        # Discretized: h_{t+1} = (1 - α) * h_t + α * φ(i_t)
        # where α = dt/τ

        # Generate noise: ξ = σ_rec * N(0,1)
        noise = (
            torch.randn(batch_size, self.hidden_size, device=input.device)
            * self.noise_std
        )

        # Compute total input i(t)
        rec_input = self.w_rec(hidden)  # [batch_size, hidden_size]
        input_drive = self.w_in(input)  # [batch_size, hidden_size]
        total_input = rec_input + input_drive + noise

        # Update hidden state: h_{t+1} = (1 - α) * h_t + α * φ(i_t)
        new_hidden = (1 - self.alpha) * hidden + self.alpha * self.activation(
            total_input
        )

        # Compute output
        output = self.w_out(new_hidden)  # [batch_size, output_size]

        return output, new_hidden
