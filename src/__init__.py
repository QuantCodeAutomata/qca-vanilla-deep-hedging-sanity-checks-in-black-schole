"""Deep Hedging core infrastructure.

Top-level src package that re-exports the main components for convenience.
"""

from simulators import black_scholes, heston
from objectives import OCELoss
from policies import HedgePolicy, DeepHedger
from evaluation import (
    estimate_asymptotic_slope,
    bootstrap_slope_uncertainty,
    compute_learned_hedge_surface,
    plot_delta_comparison,
)

__all__ = [
    "black_scholes",
    "heston",
    "OCELoss",
    "HedgePolicy",
    "DeepHedger",
    "estimate_asymptotic_slope",
    "bootstrap_slope_uncertainty",
    "compute_learned_hedge_surface",
    "plot_delta_comparison",
]
