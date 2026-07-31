"""
Tests for candidate measure construction.
"""
import sys
sys.path.insert(0, '/workspace/project')

import pytest
import torch
from measures.candidate_measures import (
    construct_bootstrap_measures,
    construct_perturbed_measures,
    uniform_uncertainty_weights,
)


# ------------------------------------------------------------------ #
# construct_bootstrap_measures                                           #
# ------------------------------------------------------------------ #

def test_bootstrap_shape():
    """Bootstrap weight matrix has shape [M, N_paths]."""
    M, N = 10, 500
    Q = construct_bootstrap_measures(N_paths=N, M=M, seed=0)
    assert Q.shape == (M, N), f"Expected ({M}, {N}), got {Q.shape}"


def test_bootstrap_measures_nonneg():
    """All bootstrap weights are non-negative."""
    Q = construct_bootstrap_measures(N_paths=1000, M=10, seed=42)
    assert (Q >= 0.0).all(), "Bootstrap weights contain negative values."


def test_bootstrap_measures_normalized():
    """Each bootstrap measure (row) sums to 1."""
    M, N = 10, 1000
    Q = construct_bootstrap_measures(N_paths=N, M=M, seed=42)
    row_sums = Q.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(M), atol=1e-5), (
        f"Row sums: {row_sums}"
    )


def test_bootstrap_measures_vary():
    """Different bootstrap measures are not identical (non-trivial resampling)."""
    M, N = 5, 200
    Q = construct_bootstrap_measures(N_paths=N, M=M, seed=7)
    # Not all rows identical
    diffs = (Q[0] - Q[1:]).abs().sum(dim=1)
    assert (diffs > 0).all(), "Bootstrap measures are unexpectedly identical."


def test_bootstrap_reproducible():
    """Bootstrap construction is reproducible with the same seed."""
    Q1 = construct_bootstrap_measures(N_paths=300, M=5, seed=99)
    Q2 = construct_bootstrap_measures(N_paths=300, M=5, seed=99)
    assert torch.allclose(Q1, Q2), "Bootstrap measures differ for the same seed."


# ------------------------------------------------------------------ #
# construct_perturbed_measures                                           #
# ------------------------------------------------------------------ #

def test_perturbed_measures_shape():
    """Perturbed measures have shape [M, N_paths]."""
    M, N = 8, 400
    base = torch.ones(N) / N
    Q = construct_perturbed_measures(base, M=M, perturbation_std=0.01, seed=0)
    assert Q.shape == (M, N), f"Expected ({M}, {N}), got {Q.shape}"


def test_perturbed_measures_nonneg():
    """Perturbed weights are non-negative (clamped)."""
    N = 200
    base = torch.ones(N) / N
    Q = construct_perturbed_measures(base, M=5, perturbation_std=0.05, seed=1)
    assert (Q >= 0.0).all(), "Perturbed weights contain negatives."


def test_perturbed_measures_normalized():
    """Each perturbed measure sums to 1."""
    N = 200
    M = 6
    base = torch.ones(N) / N
    Q = construct_perturbed_measures(base, M=M, perturbation_std=0.01, seed=2)
    row_sums = Q.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(M), atol=1e-5), (
        f"Row sums: {row_sums}"
    )


# ------------------------------------------------------------------ #
# uniform_uncertainty_weights                                            #
# ------------------------------------------------------------------ #

def test_uncertainty_weights_normalized():
    """Uniform uncertainty weights sum to 1."""
    for M in [1, 5, 10, 100]:
        mu = uniform_uncertainty_weights(M)
        assert torch.isclose(mu.sum(), torch.tensor(1.0), atol=1e-5), (
            f"M={M}: weights sum to {mu.sum().item():.6f}"
        )


def test_uncertainty_weights_uniform():
    """All uniform weights equal 1/M."""
    M = 8
    mu = uniform_uncertainty_weights(M)
    expected = torch.full((M,), 1.0 / M)
    assert torch.allclose(mu, expected, atol=1e-6), (
        f"Weights are not uniform: {mu}"
    )


def test_uncertainty_weights_shape():
    """Weights tensor has shape [M]."""
    M = 15
    mu = uniform_uncertainty_weights(M)
    assert mu.shape == (M,), f"Expected ({M},), got {mu.shape}"
