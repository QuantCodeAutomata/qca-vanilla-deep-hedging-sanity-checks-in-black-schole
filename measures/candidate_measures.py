"""
Candidate measure construction for uncertainty-aware deep hedging (Experiment 4).
Constructs M candidate equivalent measures Q_1,...,Q_M via bootstrap resampling.
"""
import torch
import numpy as np
from typing import List, Tuple


def construct_bootstrap_measures(N_paths: int, M: int,
                                  seed: int = 42) -> torch.Tensor:
    """Construct M candidate measures via bootstrap resampling.

    Each row of the returned weight matrix corresponds to one bootstrap
    resample: Q_k assigns weight proportional to the frequency of each path
    index in the k-th bootstrap draw (with replacement).

    Args:
        N_paths: Number of Monte Carlo paths (columns).
        M: Number of candidate measures (rows).
        seed: NumPy RNG seed for reproducibility.

    Returns:
        Q_weights: [M, N_paths] tensor; each row sums to 1 and is non-negative.
    """
    rng = np.random.default_rng(seed)
    Q_weights = torch.zeros(M, N_paths)
    for k in range(M):
        resample_idx = rng.integers(0, N_paths, size=N_paths)
        counts = np.bincount(resample_idx, minlength=N_paths)
        Q_weights[k] = torch.tensor(counts / N_paths, dtype=torch.float32)
    return Q_weights


def construct_perturbed_measures(base_Q: torch.Tensor, M: int,
                                  perturbation_std: float = 0.01,
                                  seed: int = 42) -> torch.Tensor:
    """Construct M measures by perturbing a base measure Q.

    Small Gaussian noise is added to each path weight and the result is
    projected back to the probability simplex by clamping and renormalising.

    Args:
        base_Q: Base probability vector [N_paths]; must sum to 1.
        M: Number of perturbed measures to generate.
        perturbation_std: Standard deviation of the Gaussian perturbation.
        seed: Torch manual seed.

    Returns:
        Q_weights: [M, N_paths] tensor; each row sums to 1 and is non-negative.
    """
    torch.manual_seed(seed)
    N_paths = base_Q.shape[0]
    perturbations = torch.randn(M, N_paths) * perturbation_std
    Q_perturbed = base_Q.unsqueeze(0) + perturbations
    Q_perturbed = torch.clamp(Q_perturbed, min=1e-10)
    Q_perturbed = Q_perturbed / Q_perturbed.sum(dim=1, keepdim=True)
    return Q_perturbed


def uniform_uncertainty_weights(M: int) -> torch.Tensor:
    """Return uniform uncertainty weights mu(Q_k) = 1/M for all k.

    Args:
        M: Number of candidate measures.

    Returns:
        mu: [M] tensor; each element equals 1/M.
    """
    return torch.ones(M) / M
