"""Modular RNN with multiple time constants and low-rank cross-connections."""

import torch
import torch.nn as nn
import numpy as np


class ModularRNN(nn.Module):
    """
    Modular continuous-time RNN with:
    - Two modules with different time constants
    - Full-rank self-recurrent connections within each module
    - Low-rank bidirectional connections between modules
    - Separate input and output weights for each module
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size_1: int = 64,  # Slow module
        hidden_size_2: int = 64,  # Fast module
        output_size: int = 2,
        tau_1: float = 300.0,  # Slow time constant (ms)
        tau_2: float = 100.0,  # Fast time constant (ms)
        dt: float = 10.0,  # ms
        cross_module_rank_pct: float = 0.1,  # Percentage for low-rank connections
        activation: str = 'elu',
        noise_std: float = 0.05,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size = hidden_size_1 + hidden_size_2  # Total for interface compatibility
        self.output_size = output_size
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.dt = dt
        self.noise_std = noise_std

        # Compute low-rank dimension
        self.cross_rank = max(1, int(cross_module_rank_pct * min(hidden_size_1, hidden_size_2)))

        # Decay factors for each module
        self.alpha_1 = dt / tau_1
        self.alpha_2 = dt / tau_2

        # Activation function
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Input weights (separate for each module)
        self.w_in_1 = nn.Linear(input_size, hidden_size_1, bias=True)
        self.w_in_2 = nn.Linear(input_size, hidden_size_2, bias=True)

        # Self-recurrent weights (full-rank)
        self.w_rec_1 = nn.Linear(hidden_size_1, hidden_size_1, bias=False)
        self.w_rec_2 = nn.Linear(hidden_size_2, hidden_size_2, bias=False)

        # Cross-module connections (low-rank: W = U @ V.T)
        # Module 1 -> Module 2: [hidden_2, hidden_1] = [hidden_2, rank] @ [rank, hidden_1]
        self.w_cross_12_V = nn.Linear(hidden_size_1, self.cross_rank, bias=False)  # V.T: [hidden_1, rank]
        self.w_cross_12_U = nn.Linear(self.cross_rank, hidden_size_2, bias=False)  # U.T: [rank, hidden_2]

        # Module 2 -> Module 1: [hidden_1, hidden_2] = [hidden_1, rank] @ [rank, hidden_2]
        self.w_cross_21_V = nn.Linear(hidden_size_2, self.cross_rank, bias=False)  # V.T: [hidden_2, rank]
        self.w_cross_21_U = nn.Linear(self.cross_rank, hidden_size_1, bias=False)  # U.T: [rank, hidden_1]

        # Output weights (separate for each module)
        self.w_out_1 = nn.Linear(hidden_size_1, output_size, bias=True)
        self.w_out_2 = nn.Linear(hidden_size_2, output_size, bias=False)  # No bias on second to avoid double bias

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights according to spec."""
        # Self-recurrent weights: scaled normal distribution
        std_1 = 1.0 / np.sqrt(self.hidden_size_1)
        std_2 = 1.0 / np.sqrt(self.hidden_size_2)
        nn.init.normal_(self.w_rec_1.weight, mean=0.0, std=std_1)
        nn.init.normal_(self.w_rec_2.weight, mean=0.0, std=std_2)

        # Input weights: Xavier uniform
        nn.init.xavier_uniform_(self.w_in_1.weight)
        nn.init.xavier_uniform_(self.w_in_2.weight)

        # Cross-module low-rank factors: conservative initialization
        # For W = U @ V.T, we want effective std ≈ 1/sqrt(hidden_from)
        # Use smaller std for each factor to account for the product
        std_cross_12 = 1.0 / np.sqrt(self.hidden_size_1 * self.cross_rank)
        std_cross_21 = 1.0 / np.sqrt(self.hidden_size_2 * self.cross_rank)
        nn.init.normal_(self.w_cross_12_U.weight, mean=0.0, std=std_cross_12)
        nn.init.normal_(self.w_cross_12_V.weight, mean=0.0, std=std_cross_12)
        nn.init.normal_(self.w_cross_21_U.weight, mean=0.0, std=std_cross_21)
        nn.init.normal_(self.w_cross_21_V.weight, mean=0.0, std=std_cross_21)

        # Output weights: Xavier uniform
        nn.init.xavier_uniform_(self.w_out_1.weight)
        nn.init.xavier_uniform_(self.w_out_2.weight)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the modular RNN for a single timestep.

        Args:
            input: [batch_size, input_size] - single timestep input
            hidden: [batch_size, hidden_size] - concatenated hidden state [h1; h2]

        Returns:
            output: [batch_size, output_size] - summed output from both modules
            new_hidden: [batch_size, hidden_size] - updated concatenated hidden state
        """
        batch_size = input.shape[0]

        # Split hidden state into module components
        h_1 = hidden[:, :self.hidden_size_1]  # [batch_size, hidden_size_1]
        h_2 = hidden[:, self.hidden_size_1:]  # [batch_size, hidden_size_2]

        # Generate noise for each module
        noise_1 = torch.randn(batch_size, self.hidden_size_1, device=input.device) * self.noise_std
        noise_2 = torch.randn(batch_size, self.hidden_size_2, device=input.device) * self.noise_std

        # Compute cross-module connections (low-rank)
        # W_cross_12 @ h_1 = U @ (V.T @ h_1) = U(V(h_1))
        cross_12 = self.w_cross_12_U(self.w_cross_12_V(h_1))  # Module 1 -> 2: [B, hidden_2]
        cross_21 = self.w_cross_21_U(self.w_cross_21_V(h_2))  # Module 2 -> 1: [B, hidden_1]

        # Compute total input for each module
        # Module 1: i_1 = W_rec_1 @ h_1 + W_in_1 @ u + W_cross_21 @ h_2 + noise_1
        rec_input_1 = self.w_rec_1(h_1)
        input_drive_1 = self.w_in_1(input)
        total_input_1 = rec_input_1 + input_drive_1 + cross_21 + noise_1

        # Module 2: i_2 = W_rec_2 @ h_2 + W_in_2 @ u + W_cross_12 @ h_1 + noise_2
        rec_input_2 = self.w_rec_2(h_2)
        input_drive_2 = self.w_in_2(input)
        total_input_2 = rec_input_2 + input_drive_2 + cross_12 + noise_2

        # Update hidden states with different time constants
        # h_{t+1} = (1 - α) * h_t + α * φ(i_t)
        new_h_1 = (1 - self.alpha_1) * h_1 + self.alpha_1 * self.activation(total_input_1)
        new_h_2 = (1 - self.alpha_2) * h_2 + self.alpha_2 * self.activation(total_input_2)

        # Concatenate for interface compatibility
        new_hidden = torch.cat([new_h_1, new_h_2], dim=1)

        # Compute output as sum of module outputs
        output = self.w_out_1(new_h_1) + self.w_out_2(new_h_2)

        return output, new_hidden
