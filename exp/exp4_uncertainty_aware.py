"""
Experiment 4: Uncertainty-Aware Deep Hedging under a Distribution of Candidate Measures.

Compares:
1. Single-measure deep hedging (under one Q)
2. Bootstrapped deep hedging (linear W = average over bootstrap measures ≡ mixture measure)
3. Full uncertainty-aware deep hedging (entropy utility over measures)
4. Entropy special case: collapses to mixture measure when lambda == risk_aversion
"""
import os
import sys
import json

sys.path.insert(0, '/workspace/project')

import torch
import torch.nn as nn

from simulators.black_scholes import simulate_paths
from policies.hedge_policy import HedgePolicy, DeepHedger
from objectives.inner_oce import WeightedOCELoss
from measures.candidate_measures import construct_bootstrap_measures, uniform_uncertainty_weights

# --------------------------------------------------------------------- #
# Global constants                                                         #
# --------------------------------------------------------------------- #
_K: float = 100.0      # strike
_S0: float = 100.0     # initial spot
_R: float = 0.0        # risk-free rate (risk-neutral P&L)
_T: float = 1.0        # maturity (years)
_RISK_AVERSION: float = 1.0


# --------------------------------------------------------------------- #
# a) Path generation                                                       #
# --------------------------------------------------------------------- #

def prepare_synthetic_paths(
    N_paths: int = 20000,
    n_steps: int = 21,
    sigma: float = 0.2,
    seed: int = 42,
) -> dict:
    """Simulate GBM paths under the risk-neutral measure.

    Args:
        N_paths: Number of Monte Carlo paths.
        n_steps: Number of daily time steps.
        sigma: Constant Black-Scholes volatility.
        seed: Torch random seed.

    Returns:
        Dictionary with keys:
            'S'   : Spot paths [N_paths, n_steps+1].
            'dH'  : Absolute stock-price increments [N_paths, n_steps]
                    (returns per unit of stock held).
            'Z_T' : ATM call payoff max(S_T - K, 0) [N_paths].
            'dt'  : Scalar time step T / n_steps.
    """
    paths = simulate_paths(
        S0=_S0, sigma=sigma, r=_R, T=_T, n_steps=n_steps, N_paths=N_paths, seed=seed
    )
    S = paths["S"]  # [N, T+1]
    paths["dH"] = S[:, 1:] - S[:, :-1]  # [N, T]
    paths["Z_T"] = torch.clamp(S[:, -1] - _K, min=0.0)  # [N]
    return paths


# --------------------------------------------------------------------- #
# Helper: analytical inner OCE under measure Q                            #
# --------------------------------------------------------------------- #

def _inner_oce_analytical(pnl: torch.Tensor, q_weights: torch.Tensor,
                           risk_aversion: float = _RISK_AVERSION) -> torch.Tensor:
    """Compute U_Q(X) = -(1/a) log( E^Q[exp(-a X)] ) analytically.

    Uses log-sum-exp for numerical stability.

    Args:
        pnl: Terminal P&L [N_paths] (requires_grad may be True).
        q_weights: Probability weights [N_paths], non-negative, sums to 1.
        risk_aversion: Coefficient a > 0.

    Returns:
        Scalar tensor (positive = good).
    """
    a = risk_aversion
    log_q = torch.log(q_weights + 1e-40)  # [N]
    log_sum = torch.logsumexp(-a * pnl + log_q, dim=0)
    return -log_sum / a


# --------------------------------------------------------------------- #
# b) Single-measure training                                               #
# --------------------------------------------------------------------- #

def train_single_measure(
    paths: dict,
    Q_weights: torch.Tensor,
    n_epochs: int = 100,
    seed: int = 42,
) -> dict:
    """Standard deep hedging under one candidate measure Q.

    The OCE loss is weighted by the per-path probabilities Q_weights.

    Args:
        paths: Path dict from prepare_synthetic_paths().
        Q_weights: Probability vector [N_paths] for measure Q.
        n_epochs: Number of training epochs.
        seed: Torch manual seed.

    Returns:
        Dictionary {'utility': float, 'loss_history': list}.
    """
    torch.manual_seed(seed)
    policy = HedgePolicy(input_dim=3)
    hedger = DeepHedger(policy, risk_aversion=_RISK_AVERSION)
    loss_fn = WeightedOCELoss(risk_aversion=_RISK_AVERSION)

    optimizer = torch.optim.Adam(hedger.parameters(), lr=1e-3)

    loss_history = []
    liability = -paths["Z_T"]  # short call

    for _ in range(n_epochs):
        optimizer.zero_grad()
        pnl = hedger.compute_pnl(paths, liability)
        loss = loss_fn(pnl, Q_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hedger.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(loss.item())

    with torch.no_grad():
        pnl = hedger.compute_pnl(paths, liability)
        utility = loss_fn.utility_value(pnl, Q_weights)

    return {"utility": float(utility), "loss_history": loss_history}


# --------------------------------------------------------------------- #
# c) Bootstrapped (linear aggregation) training                           #
# --------------------------------------------------------------------- #

def train_bootstrapped(
    paths: dict,
    Q_bootstrap: torch.Tensor,
    mu_weights: torch.Tensor,
    n_epochs: int = 100,
    seed: int = 42,
) -> dict:
    """Deep hedging with linear uncertainty aggregation over bootstrap measures.

    W_linear(f) = sum_k mu_k U_{Q_k}(f) ≡ U_{barQ}(f)  where  barQ = sum_k mu_k Q_k.
    Training under the mixture measure barQ is therefore equivalent.

    Args:
        paths: Path dict from prepare_synthetic_paths().
        Q_bootstrap: Bootstrap measure weights [M, N_paths].
        mu_weights: Uncertainty weights [M], sums to 1.
        n_epochs: Number of training epochs.
        seed: Torch manual seed.

    Returns:
        Dictionary {'utility': float}.
    """
    # Mixture measure: barQ = sum_k mu_k * Q_k
    barQ: torch.Tensor = (mu_weights.unsqueeze(1) * Q_bootstrap).sum(dim=0)  # [N]
    result = train_single_measure(paths, barQ, n_epochs=n_epochs, seed=seed)
    return {"utility": result["utility"]}


# --------------------------------------------------------------------- #
# d) Full uncertainty-aware (entropy utility) training                    #
# --------------------------------------------------------------------- #

def train_uncertainty_aware(
    paths: dict,
    Q_bootstrap: torch.Tensor,
    mu_weights: torch.Tensor,
    n_epochs: int = 100,
    seed: int = 42,
) -> dict:
    """Deep hedging with entropic uncertainty aggregation over bootstrap measures.

    W_entropy(f) = -(1/lam) log( sum_k mu_k exp(-lam U_{Q_k}(f)) ).

    With lam == risk_aversion (here 1.0), this collapses analytically to
    U_{barQ}(f) — see verify_entropy_special_case.

    Args:
        paths: Path dict from prepare_synthetic_paths().
        Q_bootstrap: Bootstrap measure weights [M, N_paths].
        mu_weights: Uncertainty weights [M], sums to 1.
        n_epochs: Number of training epochs.
        seed: Torch manual seed.

    Returns:
        Dictionary {'utility': float}.
    """
    torch.manual_seed(seed)
    M = Q_bootstrap.shape[0]
    a = _RISK_AVERSION
    lam = a  # entropy aversion equals risk aversion

    policy = HedgePolicy(input_dim=3)
    hedger = DeepHedger(policy, risk_aversion=a)
    optimizer = torch.optim.Adam(hedger.parameters(), lr=1e-3)

    log_mu = torch.log(mu_weights + 1e-40)
    liability = -paths["Z_T"]

    for _ in range(n_epochs):
        optimizer.zero_grad()
        pnl = hedger.compute_pnl(paths, liability)  # [N]

        # Inner OCE under each Q_k (analytical formula)
        U_values = [
            _inner_oce_analytical(pnl, Q_bootstrap[k], a) for k in range(M)
        ]
        U_tensor = torch.stack(U_values)  # [M]

        # Entropy outer: W = -(1/lam) log( sum_k mu_k exp(-lam U_k) )
        log_sum = torch.logsumexp(-lam * U_tensor + log_mu, dim=0)
        W = -log_sum / lam

        loss = -W
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hedger.parameters(), max_norm=1.0)
        optimizer.step()

    with torch.no_grad():
        pnl = hedger.compute_pnl(paths, liability)
        U_values = [_inner_oce_analytical(pnl, Q_bootstrap[k], a) for k in range(M)]
        U_tensor = torch.stack(U_values)
        log_sum = torch.logsumexp(-lam * U_tensor + log_mu, dim=0)
        W = -log_sum / lam

    return {"utility": float(W)}


# --------------------------------------------------------------------- #
# e) Entropy special-case verification                                    #
# --------------------------------------------------------------------- #

def verify_entropy_special_case(
    paths: dict,
    Q_bootstrap: torch.Tensor,
    mu_weights: torch.Tensor,
    risk_aversion: float = 1.0,
    n_epochs: int = 100,
) -> dict:
    """Verify that entropy uncertainty-aware ≡ mixture-measure when lam == a.

    Trains two hedgers from the same seed:
    1. Under mixture measure barQ (standard OCE).
    2. Under entropy uncertainty aggregation with lam == risk_aversion.

    Analytically they have the same objective, so their converged utilities
    should be close.

    Args:
        paths: Path dict from prepare_synthetic_paths().
        Q_bootstrap: Bootstrap measure weights [M, N_paths].
        mu_weights: Uncertainty weights [M], sums to 1.
        risk_aversion: a (must equal lam for the special case to hold).
        n_epochs: Number of training epochs for each method.

    Returns:
        Dictionary {
            'utility_mixture': float,
            'utility_entropy': float,
            'match': bool  (True if |diff| < loose threshold),
        }.
    """
    # 1) Mixture measure barQ
    barQ: torch.Tensor = (mu_weights.unsqueeze(1) * Q_bootstrap).sum(dim=0)
    res_mix = train_single_measure(paths, barQ, n_epochs=n_epochs, seed=0)
    utility_mixture = res_mix["utility"]

    # 2) Entropy uncertainty-aware
    res_ent = train_uncertainty_aware(
        paths, Q_bootstrap, mu_weights, n_epochs=n_epochs, seed=0
    )
    utility_entropy = res_ent["utility"]

    threshold = max(0.05, 0.15 * abs(utility_mixture))
    match = abs(utility_mixture - utility_entropy) < threshold

    return {
        "utility_mixture": utility_mixture,
        "utility_entropy": utility_entropy,
        "match": bool(match),
    }


# --------------------------------------------------------------------- #
# f) Main                                                                  #
# --------------------------------------------------------------------- #

def main() -> dict:
    """Run all Experiment 4 methods and save results.

    Returns:
        Dictionary of results for all methods.
    """
    print("=== Experiment 4: Uncertainty-Aware Deep Hedging ===\n")

    # Data
    paths = prepare_synthetic_paths(N_paths=20000, n_steps=21, sigma=0.2, seed=42)
    N_paths = int(paths["Z_T"].shape[0])
    M = 10
    Q_bootstrap = construct_bootstrap_measures(N_paths=N_paths, M=M, seed=42)
    mu_weights = uniform_uncertainty_weights(M)

    print("1) Single-measure hedging (Q = first bootstrap measure)...")
    res_single = train_single_measure(paths, Q_bootstrap[0], n_epochs=100, seed=42)

    print("2) Bootstrapped hedging (linear W = mixture barQ)...")
    res_boot = train_bootstrapped(paths, Q_bootstrap, mu_weights, n_epochs=100, seed=42)

    print("3) Uncertainty-aware hedging (entropy utility)...")
    res_unc = train_uncertainty_aware(
        paths, Q_bootstrap, mu_weights, n_epochs=100, seed=42
    )

    print("4) Verifying entropy special case (lam=a => barQ)...")
    res_special = verify_entropy_special_case(
        paths, Q_bootstrap, mu_weights, risk_aversion=1.0, n_epochs=100
    )

    results = {
        "single_measure_utility": res_single["utility"],
        "bootstrapped_utility": res_boot["utility"],
        "uncertainty_aware_utility": res_unc["utility"],
        "entropy_special_case": res_special,
    }

    # Print comparison table
    print("\n=== Results Summary ===")
    print(f"  Single-measure utility   : {results['single_measure_utility']:.4f}")
    print(f"  Bootstrapped utility     : {results['bootstrapped_utility']:.4f}")
    print(f"  Uncertainty-aware utility: {results['uncertainty_aware_utility']:.4f}")
    sc = res_special
    print(
        f"  Entropy special case     : mixture={sc['utility_mixture']:.4f}, "
        f"entropy={sc['utility_entropy']:.4f}, match={sc['match']}"
    )

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/exp4_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nResults saved to results/exp4_results.json")

    # Optional: loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.plot(res_single["loss_history"], label="Single measure")
        plt.xlabel("Epoch")
        plt.ylabel("OCE loss")
        plt.title("Experiment 4 — Single-Measure Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/exp4_loss_history.png", dpi=100)
        plt.close()
        print("Loss curve saved to results/exp4_loss_history.png")
    except Exception:
        pass

    return results


if __name__ == "__main__":
    main()
