"""
Bellman operator for deep hedging.

TV(Z^(t), s_t) = sup_a U_t[beta * V(Z^(t) + a'H^(t), s_{t+1})
                              + dZ_t + a'dH_t - c_t(a)]
"""
import torch
import torch.nn as nn
from typing import Callable, Optional


class BellmanOperator(nn.Module):
    """One-step Bellman operator for deep hedging.

    Given a value function V approximated by a critic network, computes TV,
    the discounted Bellman target that the actor optimises.

    Args:
        beta: Discount factor in (0, 1).
        risk_aversion: Exponential utility coefficient a > 0.
        cost_epsilon: Proportional transaction cost rate (0 = frictionless).
    """

    def __init__(self, beta: float = 0.99, risk_aversion: float = 1.0,
                 cost_epsilon: float = 0.0):
        super().__init__()
        assert 0 < beta < 1, "Discount factor beta must be in (0, 1)"
        self.beta = beta
        self.a = risk_aversion
        self.cost_epsilon = cost_epsilon

    def utility(self, x: torch.Tensor) -> torch.Tensor:
        """Exponential utility u(x) = (1 - exp(-a*x)) / a."""
        return (1.0 - torch.exp(-self.a * x)) / self.a

    def apply(self, V_func: Callable, portfolio_state: torch.Tensor,
              action: torch.Tensor, dZ: torch.Tensor, dH: torch.Tensor,
              next_state: torch.Tensor) -> torch.Tensor:
        """Compute one Bellman step.

        TV(Z, s) = beta * V(Z + a'H, s') + dZ + a'dH - c(a)

        Args:
            V_func: Callable critic (no grad tracked inside).
            portfolio_state: Current portfolio+market state [N, state_dim].
            action: Hedge action [N, n_instruments].
            dZ: Portfolio P&L from liability alone [N].
            dH: Hedge instrument gains [N, n_instruments].
            next_state: Next-period state [N, state_dim].

        Returns:
            Bellman target [N].
        """
        # Transaction cost
        cost = self.cost_epsilon * action.abs().sum(dim=-1)

        # One-step gain from hedging
        gain = (action * dH).sum(dim=-1) + dZ - cost

        # Discounted continuation value (no gradient through critic)
        with torch.no_grad():
            V_next = V_func(next_state)

        return self.beta * V_next + gain
