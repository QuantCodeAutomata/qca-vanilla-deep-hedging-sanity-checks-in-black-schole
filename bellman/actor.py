"""
Actor for Bellman deep hedging - computes optimal actions given value function.
"""
import torch
import torch.nn as nn
from typing import List


class BellmanActor(nn.Module):
    """Actor network for Bellman deep hedging.

    Maps the current portfolio+market state to hedge positions.  Architecture
    is a fully-connected network with Tanh activations and a linear output.

    An auxiliary OCE scalar ``y`` is stored as a learnable parameter so that
    it can be optimised jointly with the network weights during the actor step.

    Args:
        state_dim: Dimensionality of the input state vector.
        n_instruments: Number of hedge instruments (output neurons).
        n_layers: Total number of hidden layers.
        n_units: Number of units per hidden layer.
    """

    def __init__(self, state_dim: int, n_instruments: int = 1,
                 n_layers: int = 3, n_units: int = 64):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(state_dim, n_units), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_units, n_units), nn.Tanh()]
        layers.append(nn.Linear(n_units, n_instruments))
        self.net = nn.Sequential(*layers)
        # OCE auxiliary variable (optimised jointly with actor weights)
        self.y = nn.Parameter(torch.tensor(0.0))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute hedge action from current state.

        Args:
            state: State tensor [N, state_dim].

        Returns:
            Action tensor [N, n_instruments].
        """
        return self.net(state)
