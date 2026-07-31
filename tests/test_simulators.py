"""
Tests for Black-Scholes and Heston simulators.
"""
import sys
sys.path.insert(0, '/workspace/project')

import pytest
import torch
import numpy as np

from simulators.black_scholes import (
    simulate_paths,
    call_price,
    call_delta,
    call_spread_delta,
    call_gamma,
)
from simulators.heston import simulate_paths as heston_simulate_paths


# ------------------------------------------------------------------ #
# Black-Scholes simulate_paths                                          #
# ------------------------------------------------------------------ #

def test_bs_paths_shape():
    """Spot path tensor has shape [N_paths, n_steps+1]."""
    N, T = 500, 21
    paths = simulate_paths(S0=100.0, sigma=0.2, r=0.0, T=1.0,
                           n_steps=T, N_paths=N, seed=0)
    assert paths["S"].shape == (N, T + 1)


def test_bs_paths_positive():
    """All spot prices remain strictly positive (log-Euler scheme)."""
    paths = simulate_paths(S0=100.0, sigma=0.2, r=0.0, T=1.0,
                           n_steps=21, N_paths=1000, seed=7)
    assert (paths["S"] > 0).all(), "Negative or zero spot prices detected."


def test_bs_martingale():
    """Under r=0, E[S_T / S_0] == 1 (risk-neutral martingale property).

    Test passes when the sample mean is within 3 standard errors of 1.
    """
    N = 50_000
    paths = simulate_paths(S0=100.0, sigma=0.2, r=0.0, T=1.0,
                           n_steps=21, N_paths=N, seed=99)
    ratio = paths["S"][:, -1] / paths["S"][:, 0]  # S_T / S_0
    mean = ratio.mean().item()
    std_err = ratio.std().item() / N ** 0.5
    assert abs(mean - 1.0) < 3 * std_err, (
        f"Martingale test failed: mean={mean:.4f}, 3*se={3*std_err:.4f}"
    )


# ------------------------------------------------------------------ #
# Black-Scholes analytical pricing                                      #
# ------------------------------------------------------------------ #

def test_bs_call_price_positive():
    """ATM Black-Scholes call price is strictly positive."""
    S = torch.tensor([100.0])
    price = call_price(S, K=100.0, sigma=0.2, r=0.0, T=1.0)
    assert price.item() > 0.0


def test_bs_delta_range():
    """Call delta lies in [0, 1] for all spot values."""
    S = torch.linspace(50.0, 150.0, 200)
    delta = call_delta(S, K=100.0, sigma=0.2, r=0.0, T=1.0)
    assert (delta >= 0.0).all(), "Delta below 0."
    assert (delta <= 1.0).all(), "Delta above 1."


def test_bs_call_spread_delta_range():
    """Call-spread delta lies in [0, 1] for all spot values."""
    S = torch.linspace(50.0, 150.0, 200)
    sp_delta = call_spread_delta(S, K1=95.0, K2=105.0, sigma=0.2, r=0.0, T=1.0)
    assert (sp_delta >= -1e-6).all(), "Call-spread delta below 0."
    assert (sp_delta <= 1.0 + 1e-6).all(), "Call-spread delta above 1."


def test_bs_call_price_lower_bound():
    """Call price >= intrinsic value max(S - K, 0) for all spots."""
    S = torch.tensor([110.0, 90.0, 100.0, 120.0, 80.0])
    K = 100.0
    price = call_price(S, K=K, sigma=0.2, r=0.0, T=1.0)
    intrinsic = torch.clamp(S - K, min=0.0)
    assert (price >= intrinsic - 1e-5).all(), (
        f"Call price below intrinsic: price={price}, intrinsic={intrinsic}"
    )


def test_bs_gamma_nonneg():
    """Call gamma is non-negative for all spot values and strictly positive away from expiry."""
    S = torch.linspace(50.0, 150.0, 100)
    gamma = call_gamma(S, K=100.0, sigma=0.2, r=0.0, T=1.0)
    assert (gamma >= 0.0).all(), "Gamma should be non-negative."
    assert (gamma > 0.0).any(), "Gamma should be strictly positive away from expiry."


def test_bs_gamma_zero_at_expiry():
    """Gamma is zero at T=0 (expiry)."""
    S = torch.tensor([95.0, 100.0, 105.0])
    gamma = call_gamma(S, K=100.0, sigma=0.2, r=0.0, T=0.0)
    assert (gamma == 0.0).all(), f"Expected zero gamma at expiry, got {gamma}."


# ------------------------------------------------------------------ #
# Heston simulate_paths                                                 #
# ------------------------------------------------------------------ #

def test_heston_paths_shape():
    """Heston spot and variance paths have correct shapes."""
    N, T = 300, 21
    paths = heston_simulate_paths(
        S0=100.0, V0=0.04, kappa=2.0, theta=0.04,
        xi=0.3, rho=-0.7, r=0.0, T=1.0,
        n_steps=T, N_paths=N, seed=0,
    )
    assert paths["S"].shape == (N, T + 1)
    assert paths["V"].shape == (N, T + 1)


def test_heston_paths_positive():
    """Heston spot prices remain strictly positive."""
    paths = heston_simulate_paths(
        S0=100.0, V0=0.04, kappa=2.0, theta=0.04,
        xi=0.3, rho=-0.7, r=0.0, T=1.0,
        n_steps=21, N_paths=1000, seed=1,
    )
    assert (paths["S"] > 0).all(), "Heston: negative or zero spot prices."


def test_heston_variance_positive():
    """Heston variance paths are non-negative after reflection at zero."""
    paths = heston_simulate_paths(
        S0=100.0, V0=0.04, kappa=2.0, theta=0.04,
        xi=0.3, rho=-0.7, r=0.0, T=1.0,
        n_steps=21, N_paths=1000, seed=2,
    )
    assert (paths["V"] >= 0.0).all(), "Heston: negative variance detected."
