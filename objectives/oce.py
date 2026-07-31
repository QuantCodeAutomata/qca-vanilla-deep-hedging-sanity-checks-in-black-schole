# Custom — Context7 found no library equivalent for exact OCE/monetary utility formulation
import math

import torch
import torch.nn as nn
from typing import Optional


class OCELoss(nn.Module):
    """Optimal Certainty Equivalent (OCE) loss for deep hedging.

    U*(Z) = sup_{y} (1/N) sum_omega u(y + Z(omega)) - y

    For exponential utility u(x) = (1 - exp(-a*x)) / a,
    the OCE reduces to the negative entropic risk measure.
    """

    def __init__(self, risk_aversion: float = 1.0, cost_epsilon: float = 0.0):
        """
        Args:
            risk_aversion: Exponential utility risk-aversion coefficient a > 0.
            cost_epsilon: Proportional transaction cost rate (0 = frictionless).
        """
        super().__init__()
        self.a = risk_aversion
        self.cost_epsilon = cost_epsilon
        # Trainable OCE scalar y
        self.y = nn.Parameter(torch.tensor(0.0))

    def utility(self, x: torch.Tensor) -> torch.Tensor:
        """Exponential utility u(x) = (1 - exp(-a*x)) / a.

        Args:
            x: Input tensor of any shape.

        Returns:
            Utility values, same shape as x.
        """
        return (1.0 - torch.exp(-self.a * x)) / self.a

    def cost_fn(self, action: torch.Tensor) -> torch.Tensor:
        """Proportional transaction cost c(a) = cost_epsilon * sum |a_i|.

        Args:
            action: Hedge action tensor [..., n_instruments].

        Returns:
            Cost per path [...] (zero when cost_epsilon == 0 or action == 0).
        """
        return self.cost_epsilon * action.abs().sum(dim=-1)

    def forward(self, terminal_pnl: torch.Tensor) -> torch.Tensor:
        """Compute negative entropic risk measure (loss to minimise).

        Uses the log-sum-exp identity for numerical stability:

            U*(Z) = -(1/a) * log( E[exp(-a*Z)] )
                  = -(1/a) * (logsumexp(-a*Z, dim=0) - log(N))

        Args:
            terminal_pnl: Terminal P&L per path [N_paths].

        Returns:
            Negative utility (scalar to minimise).
        """
        N = terminal_pnl.shape[0]
        lse = torch.logsumexp(-self.a * terminal_pnl, dim=0) - math.log(N)
        U = -lse / self.a
        return -U

    def utility_value(self, terminal_pnl: torch.Tensor) -> torch.Tensor:
        """Compute U*(Z) = -(1/a)*log(E[exp(-aZ)]) (positive = good).

        Args:
            terminal_pnl: Terminal P&L per path [N_paths].

        Returns:
            Scalar entropic utility value.
        """
        with torch.no_grad():
            N = terminal_pnl.shape[0]
            lse = torch.logsumexp(-self.a * terminal_pnl, dim=0) - math.log(N)
            return -lse / self.a

    def marginal_price(
        self,
        pnl_with: torch.Tensor,
        pnl_without: torch.Tensor,
    ) -> float:
        """Compute utility-indifference marginal price p(D) = U*(Z) - U*(Z-D).

        Args:
            pnl_with: P&L paths that include the derivative payoff [N_paths].
            pnl_without: P&L paths without the derivative payoff [N_paths].

        Returns:
            Indifference marginal price (float).
        """
        return (self.utility_value(pnl_with) - self.utility_value(pnl_without)).item()
