"""Protocol for neural network model classes."""

from typing import Protocol
import torch


class Model(Protocol):
    """Protocol for neural network models.

    Models process a single timestep at a time. The caller manages
    the loop over timesteps.
    """

    input_size: int
    hidden_size: int
    output_size: int

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model for a single timestep.

        Args:
            input: [batch_size, input_size] - single timestep input
            hidden: [batch_size, hidden_size] - current hidden state

        Returns:
            output: [batch_size, output_size] - output at this timestep
            new_hidden: [batch_size, hidden_size] - updated hidden state
        """
        ...
