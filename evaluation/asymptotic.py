"""Asymptotic slope analysis for transaction cost scaling in deep hedging."""

import numpy as np
from scipy import stats
from typing import List, Tuple


def estimate_asymptotic_slope(
    epsilon_grid: List[float],
    pricing_corrections: List[float],
) -> Tuple[float, float, float]:
    """Estimate the asymptotic power-law slope in log-log space.

    Fits the model  p(epsilon) ~ C * epsilon^alpha  via ordinary least squares
    on log-transformed data:  log|p| ~ log(C) + alpha * log(epsilon).

    Args:
        epsilon_grid: Transaction cost levels (positive floats).
        pricing_corrections: Corresponding pricing corrections / utility values.
            Small corrections are handled by adding a regulariser before taking log.

    Returns:
        Tuple of (slope alpha, intercept log(C), Pearson r_value).
    """
    log_eps = np.log(epsilon_grid)
    log_corr = np.log(np.abs(pricing_corrections) + 1e-12)
    slope, intercept, r_value, _p_value, _se = stats.linregress(log_eps, log_corr)
    return slope, intercept, r_value


def bootstrap_slope_uncertainty(
    epsilon_grid: List[float],
    pricing_corrections: List[float],
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the asymptotic power-law slope.

    Resamples (epsilon, correction) pairs with replacement and refits the
    log-log regression on each bootstrap sample to estimate slope variability.

    Args:
        epsilon_grid: Transaction cost levels (positive floats).
        pricing_corrections: Corresponding pricing corrections / utility values.
        n_bootstrap: Number of bootstrap resamples.
        seed: Seed for the NumPy random number generator.

    Returns:
        Tuple of (slope_mean, slope_std) across bootstrap samples.
    """
    rng = np.random.default_rng(seed)
    n = len(epsilon_grid)
    log_eps = np.log(epsilon_grid)
    log_corr = np.log(np.abs(pricing_corrections) + 1e-12)

    slopes: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(set(idx.tolist())) < 2:
            # Need at least two distinct points to fit a line
            continue
        s, _intercept, _r, _p, _se = stats.linregress(log_eps[idx], log_corr[idx])
        slopes.append(s)

    return float(np.mean(slopes)), float(np.std(slopes))
