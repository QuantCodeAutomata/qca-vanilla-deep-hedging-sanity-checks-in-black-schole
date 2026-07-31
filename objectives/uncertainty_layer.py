"""
Outer uncertainty utility layer W(f) for uncertainty-aware deep hedging.

W(f) = sup_e { sum_k mu_k * w(e + U_{Q_k}(f)) - e }

where w is the uncertainty utility function and U_{Q_k}(f) is the inner OCE
under measure Q_k.
"""
import torch
import torch.nn as nn
from typing import List


class UncertaintyLayer(nn.Module):
    """Outer uncertainty aggregation layer.

    Two aggregation modes are supported:

    * **Entropy** (``use_entropy=True``): uncertainty utility
      w(x) = (1 - exp(-lambda*x)) / lambda.  With lambda == risk_aversion,
      the objective analytically collapses to deep hedging under the mixture
      measure barQ = sum_k mu_k * Q_k.

    * **Linear** (``use_entropy=False``): W(f) = sum_k mu_k * U_{Q_k}(f),
      which is also equivalent to hedging under the mixture measure barQ.

    Args:
        uncertainty_aversion: lambda in the entropy utility (lambda > 0).
        use_entropy: If True use entropy utility, else use linear aggregation.
    """

    def __init__(self, uncertainty_aversion: float = 1.0, use_entropy: bool = True):
        super().__init__()
        self.lam = uncertainty_aversion
        self.use_entropy = use_entropy
        self.e = nn.Parameter(torch.tensor(0.0))  # outer OCE scalar

    def uncertainty_utility(self, x: torch.Tensor) -> torch.Tensor:
        """Uncertainty utility w(x) = (1 - exp(-lambda*x)) / lambda."""
        return (1.0 - torch.exp(-self.lam * x)) / self.lam

    def forward(self, inner_utilities: torch.Tensor,
                mu_weights: torch.Tensor) -> torch.Tensor:
        """Return negative W loss (to minimise).

        Args:
            inner_utilities: U_{Q_k}(X) values [M]; higher is better.
            mu_weights: Uncertainty weights [M]; non-negative, sums to 1.

        Returns:
            Scalar negative W loss = -W(f).
        """
        W = self.compute_W(inner_utilities, mu_weights)

        # Outer OCE envelope: sup_e { w(e + W) - e }
        w_val = self.uncertainty_utility(self.e + W)
        return -(w_val - self.e)

    def compute_W(self, inner_utilities: torch.Tensor,
                  mu_weights: torch.Tensor) -> torch.Tensor:
        """Compute W(f) value (positive = good) without recording a graph.

        Args:
            inner_utilities: U_{Q_k}(X) values [M].
            mu_weights: Uncertainty weights [M]; non-negative, sums to 1.

        Returns:
            Scalar tensor representing W(f).
        """
        with torch.no_grad():
            if self.use_entropy:
                # W = -(1/lam) * log(sum_k mu_k * exp(-lam * U_k))
                weighted_neg = -self.lam * inner_utilities
                log_sum = torch.logsumexp(
                    weighted_neg + torch.log(mu_weights + 1e-40), dim=0
                )
                return -log_sum / self.lam
            else:
                return (mu_weights * inner_utilities).sum()
