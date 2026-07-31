"""
Inner OCE loss under a specific measure Q (as path weights).
Used in uncertainty-aware deep hedging (Experiment 4).
"""
import torch
import torch.nn as nn


class WeightedOCELoss(nn.Module):
    """OCE loss under measure Q represented as path weights.

    Computes the Optimised Certainty Equivalent under an arbitrary discrete
    probability measure Q, specified by per-path importance weights q_omega:

        U_Q(X) = sup_y { E^Q[u(y + X)] - y }
               = sup_y { sum_omega q_omega * u(y + X_omega) - y }

    For exponential utility this is equivalent to the negative entropic risk
    measure under Q.  The auxiliary scalar y participates in joint optimisation
    with the hedging policy.

    Args:
        risk_aversion: Coefficient a in u(x) = (1-exp(-a*x))/a  (a > 0).
    """

    def __init__(self, risk_aversion: float = 1.0):
        super().__init__()
        self.a = risk_aversion
        self.y = nn.Parameter(torch.tensor(0.0))

    def utility(self, x: torch.Tensor) -> torch.Tensor:
        """Exponential utility u(x) = (1 - exp(-a*x)) / a."""
        return (1.0 - torch.exp(-self.a * x)) / self.a

    def forward(self, terminal_pnl: torch.Tensor,
                q_weights: torch.Tensor) -> torch.Tensor:
        """Return negative entropic risk measure under Q (loss to minimise).

        Uses the log-sum-exp identity for numerical stability:

            U_Q(X) = -(1/a) * log( E^Q[exp(-a*X)] )
                   = -(1/a) * logsumexp(-a*X + log(q), dim=0)

        Args:
            terminal_pnl: Terminal P&L per path [N_paths].
            q_weights: Probability weights [N_paths]; non-negative, sums to 1.

        Returns:
            Scalar negative utility (loss to minimise).
        """
        log_q = torch.log(q_weights + 1e-40)
        log_sum = torch.logsumexp(-self.a * terminal_pnl + log_q, dim=0)
        U = -log_sum / self.a
        return -U

    def utility_value(self, terminal_pnl: torch.Tensor,
                      q_weights: torch.Tensor) -> torch.Tensor:
        """Evaluate U_Q(X) = -(1/a)*log(E^Q[exp(-aX)]) without a graph.

        Args:
            terminal_pnl: Terminal P&L per path [N_paths].
            q_weights: Probability weights [N_paths]; non-negative, sums to 1.

        Returns:
            Scalar tensor (positive = good means low entropic risk).
        """
        with torch.no_grad():
            log_q = torch.log(q_weights + 1e-40)
            log_sum = torch.logsumexp(-self.a * terminal_pnl + log_q, dim=0)
            return -log_sum / self.a
