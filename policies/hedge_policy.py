"""Neural network hedge policy and full deep-hedger module."""

import math

import torch
import torch.nn as nn
from typing import List


class HedgePolicy(nn.Module):
    """Neural network hedge policy a_t = a(s_t; theta).

    State s_t can include: time_to_maturity, log_moneyness, prev_hedge, [volatility].
    Output: hedge position (scalar per path, one per instrument).

    Architecture: fully-connected network with Tanh activations and a linear
    output layer (unbounded hedge positions).
    """

    def __init__(
        self,
        input_dim: int,
        n_instruments: int = 1,
        n_layers: int = 4,
        n_units: int = 32,
    ):
        """
        Args:
            input_dim: Dimensionality of the state vector s_t.
            n_instruments: Number of hedge instruments (output neurons).
            n_layers: Total number of hidden layers (first maps input_dim -> n_units).
            n_units: Number of units per hidden layer.
        """
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, n_units), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_units, n_units), nn.Tanh()]
        layers.append(nn.Linear(n_units, n_instruments))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute hedge action from state.

        Args:
            state: State tensor [N_paths, input_dim].

        Returns:
            Action tensor [N_paths, n_instruments].
        """
        return self.net(state)


class DeepHedger(nn.Module):
    """Full deep hedger combining a neural policy with the OCE objective.

    The hedger rolls forward through time, builds the terminal P&L, and
    computes the negative OCE loss for back-propagation.
    """

    def __init__(self, policy: HedgePolicy, risk_aversion: float = 1.0):
        """
        Args:
            policy: Trained or untrained HedgePolicy network.
            risk_aversion: Exponential utility risk-aversion coefficient a > 0.
        """
        super().__init__()
        self.policy = policy
        self.a = risk_aversion
        self.y = nn.Parameter(torch.tensor(0.0))  # OCE auxiliary variable

    def utility(self, x: torch.Tensor) -> torch.Tensor:
        """Exponential utility u(x) = (1 - exp(-a*x)) / a.

        Args:
            x: Input tensor of any shape.

        Returns:
            Utility values, same shape as x.
        """
        return (1.0 - torch.exp(-self.a * x)) / self.a

    def compute_pnl(
        self,
        paths: dict,
        liability_payoff: torch.Tensor,
        cost_epsilon: float = 0.0,
    ) -> torch.Tensor:
        """Compute terminal P&L for all paths by rolling through the hedge.

        At each step t the policy maps the current state to a hedge position.
        The cumulative trading gain minus transaction costs is added to the
        (negated) liability payoff to form the terminal P&L.

        State features: [time_to_maturity, log_spot, previous_hedge_position].

        Args:
            paths: Dictionary with:
                'S'  : Spot prices [N_paths, n_steps+1].
                'dH' : Discounted hedge-instrument gains [N_paths, n_steps].
                       dH[:, t] = gain per unit of hedge held over step t.
            liability_payoff: Short option payoff Z_T [N_paths] (negative = short).
            cost_epsilon: Proportional transaction cost rate.

        Returns:
            Terminal P&L per path [N_paths].
        """
        S = paths["S"]  # [N_paths, n_steps+1]
        N, T_plus_1 = S.shape
        T = T_plus_1 - 1
        dH = paths["dH"]  # [N_paths, n_steps]

        prev_hedge = torch.zeros(N, 1, device=S.device)
        total_gain = torch.zeros(N, device=S.device)

        for t in range(T):
            ttm = (T - t) / T  # time to maturity (normalised)
            log_m = torch.log(S[:, t]).unsqueeze(1)  # log spot (proxy for log-moneyness)
            state = torch.cat(
                [torch.full((N, 1), ttm, device=S.device), log_m, prev_hedge],
                dim=1,
            )  # [N_paths, 3]

            action = self.policy(state)  # [N_paths, 1]
            delta_action = action - prev_hedge

            # Transaction cost: epsilon * |delta_position| * S_t
            cost = cost_epsilon * delta_action.abs().squeeze(1) * S[:, t]

            # Gain from holding hedge position over step t
            gain = action.squeeze(1) * dH[:, t]
            total_gain = total_gain + gain - cost
            prev_hedge = action.detach()

        return liability_payoff + total_gain

    def forward(
        self,
        paths: dict,
        liability_payoff: torch.Tensor,
        cost_epsilon: float = 0.0,
    ) -> torch.Tensor:
        """Compute negative entropic risk measure (loss to minimise).

        Uses the numerically stable log-sum-exp identity:

            U*(Z) = -(1/a) * log( E[exp(-a*Z)] )
                  = -(1/a) * (logsumexp(-a*Z) - log(N))

        Args:
            paths: Path dictionary (see compute_pnl).
            liability_payoff: Short option payoff [N_paths].
            cost_epsilon: Proportional transaction cost rate.

        Returns:
            Negative utility scalar (to minimise).
        """
        pnl = self.compute_pnl(paths, liability_payoff, cost_epsilon)
        N = pnl.shape[0]
        lse = torch.logsumexp(-self.a * pnl, dim=0) - math.log(N)
        return lse / self.a
