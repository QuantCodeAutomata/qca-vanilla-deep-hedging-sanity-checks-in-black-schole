"""
Critic network for Bellman deep hedging - approximates value function V^n.
"""
import torch
import torch.nn as nn
from typing import List


class BellmanCritic(nn.Module):
    """Critic network V^n(portfolio_state, market_state).

    Approximates the Bellman value function at iteration n.  Takes the
    concatenated portfolio+market feature vector as input and outputs a
    scalar value per sample.

    Args:
        state_dim: Dimensionality of the input state vector.
        n_layers: Total number of hidden layers.
        n_units: Number of units per hidden layer.
    """

    def __init__(self, state_dim: int, n_layers: int = 4, n_units: int = 64):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(state_dim, n_units), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_units, n_units), nn.Tanh()]
        layers.append(nn.Linear(n_units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate the value function at given states.

        Args:
            state: State tensor [N, state_dim].

        Returns:
            Value tensor [N].
        """
        return self.net(state).squeeze(-1)
