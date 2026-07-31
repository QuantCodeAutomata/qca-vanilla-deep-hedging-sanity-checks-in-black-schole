"""
Experiment 3: Statistical Arbitrage Diagnosis and Drift Removal via Path Reweighting

Demonstrates that naive deep hedging under P exploits statistical arbitrage,
then constructs a path-reweighted measure Q that removes drift exploitation.
"""

import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, "/workspace/project")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from simulators.black_scholes import simulate_paths as bs_simulate
from policies.hedge_policy import DeepHedger, HedgePolicy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

_LR = 1e-3
_BATCH_SIZE = 2048
_GRAD_CLIP = 1.0


# ---------------------------------------------------------------------------
# Path simulation with arbitrary drift
# ---------------------------------------------------------------------------

def simulate_paths_with_drift(
    mu: float = 0.1,
    sigma: float = 0.15,
    N_paths: int = 20_000,
    n_steps: int = 21,
    T: float = 0.0833,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate GBM paths with non-zero drift mu (real-world measure P).

    Args:
        mu:      Real-world drift (NOT zero-drift / risk-neutral).
        sigma:   Volatility.
        N_paths: Number of Monte Carlo paths.
        n_steps: Number of time steps.
        T:       Time horizon (years).
        seed:    Random seed.

    Returns:
        dict with keys:
            'S'    – spot paths (N_paths, n_steps+1).
            'dH'   – stock increments (N_paths, n_steps).
            'Z_T'  – ATM short-call terminal liability (N_paths,).
            'dt'   – time step size.
            'mu'   – drift used.
            'sigma'– volatility used.
    """
    raw = bs_simulate(
        S0=1.0, sigma=sigma, r=mu, T=T, n_steps=n_steps, N_paths=N_paths, seed=seed
    )
    S = raw["S"]
    dH = S[:, 1:] - S[:, :-1]
    K = 1.0
    Z_T = -torch.clamp(S[:, -1] - K, min=0.0)
    return {"S": S, "dH": dH, "Z_T": Z_T, "dt": raw["dt"], "mu": mu, "sigma": sigma}


# ---------------------------------------------------------------------------
# Empty-portfolio training under P
# ---------------------------------------------------------------------------

def _build_hedger(seed: int = 42, risk_aversion: float = 1.0) -> DeepHedger:
    torch.manual_seed(seed)
    policy = HedgePolicy(input_dim=3, n_instruments=1, n_layers=4, n_units=32)
    return DeepHedger(policy=policy, risk_aversion=risk_aversion)


def train_empty_portfolio(
    paths_P: dict,
    n_epochs: int = 100,
    risk_aversion: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a hedge policy on the EMPTY portfolio (Z_T = 0) under measure P.

    In a zero-drift (risk-neutral) world the optimal policy is to hold zero
    shares.  With positive drift mu > 0, the agent exploits the drift by
    holding a positive position – a statistical arbitrage.

    Args:
        paths_P:       Path dict from simulate_paths_with_drift.
        n_epochs:      Gradient epochs.
        risk_aversion: Exponential utility parameter a > 0.
        seed:          Random seed.

    Returns:
        dict with keys:
            'utility'         – OCE utility value at convergence (float).
            'y_star'          – Optimal OCE auxiliary variable y* (float).
            'optimal_policy'  – Trained DeepHedger.
            'terminal_gains'  – Hedging gains per path (tensor, N_paths).
    """
    S = paths_P["S"]
    dH = paths_P["dH"]
    N = S.shape[0]

    # Zero liability: the agent trades on drift alone
    Z_T_zero = torch.zeros(N)

    hedger = _build_hedger(seed, risk_aversion)
    optimizer = optim.Adam(hedger.parameters(), lr=_LR)

    for _ in range(n_epochs):
        hedger.train()
        idx = torch.randperm(N)[:_BATCH_SIZE]
        batch = {"S": S[idx], "dH": dH[idx]}
        loss = hedger(batch, Z_T_zero[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hedger.parameters(), _GRAD_CLIP)
        optimizer.step()

    hedger.eval()
    with torch.no_grad():
        terminal_gains = hedger.compute_pnl({"S": S, "dH": dH}, Z_T_zero)

    y_star = hedger.y.item()
    utility = -(hedger({"S": S, "dH": dH}, Z_T_zero).item())

    return {
        "utility": utility,
        "y_star": y_star,
        "optimal_policy": hedger,
        "terminal_gains": terminal_gains,
    }


# ---------------------------------------------------------------------------
# Q-measure construction
# ---------------------------------------------------------------------------

def compute_Q_weights(
    terminal_gains: torch.Tensor,
    y_star: float,
    risk_aversion: float = 1.0,
) -> torch.Tensor:
    """Compute path weights that define the Q measure (drift-free measure).

    Under the Q measure constructed from the optimal empty-portfolio policy a^0:
        dQ/dP(omega) ∝ u'(y* + sum_t a^0_t dH_t(omega))

    For exponential utility u(x) = (1-exp(-ax))/a:
        u'(x) = exp(-ax)

    So:  dQ/dP(omega) ∝ exp(-a * (y* + gain(omega)))

    The weights are normalised to sum to 1 (probability weights).

    Args:
        terminal_gains: Hedging gains per path (N_paths,) under the empty policy.
        y_star:         Optimal OCE auxiliary variable.
        risk_aversion:  Exponential utility parameter a > 0.

    Returns:
        q_weights: (N_paths,) non-negative probability weights summing to 1.
    """
    a = risk_aversion
    log_w = -a * (y_star + terminal_gains)        # log dQ/dP (un-normalised)
    log_w = log_w - log_w.max()                   # numerical stability
    w = torch.exp(log_w)
    return w / w.sum()


def verify_foc_identities(
    terminal_gains: torch.Tensor,
    y_star: float,
    risk_aversion: float = 1.0,
) -> dict[str, Any]:
    """Verify the first-order condition E^P[u'(y* + gains)] ≈ 1.

    At the optimum of the OCE objective:
        E^P[exp(-a * (y* + gains))] = 1

    Args:
        terminal_gains: Hedging gains from the optimal empty-portfolio policy.
        y_star:         Optimal OCE auxiliary variable.
        risk_aversion:  Exponential utility parameter a > 0.

    Returns:
        dict with keys:
            'E_u_prime'   – empirical E^P[u'(y* + gains)] (should be ≈ 1).
            'should_be_1' – True (label for the check).
    """
    a = risk_aversion
    u_prime = torch.exp(-a * (y_star + terminal_gains))
    E_u_prime = u_prime.mean().item()
    return {"E_u_prime": E_u_prime, "should_be_1": True}


# ---------------------------------------------------------------------------
# Q-weighted training
# ---------------------------------------------------------------------------

def train_under_Q(
    paths_P: dict,
    Q_weights: torch.Tensor,
    n_epochs: int = 100,
    risk_aversion: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Train the EMPTY portfolio under the reweighted Q measure.

    The OCE loss is computed as a Q-weighted expectation:
        L = -(sum_omega q_omega * u(y + gain(omega)) - y)

    Under Q (constructed to remove drift), the optimal strategy has zero
    utility gain: the agent can no longer exploit the drift.

    Args:
        paths_P:       Path dict from simulate_paths_with_drift.
        Q_weights:     (N_paths,) probability weights for Q (from compute_Q_weights).
        n_epochs:      Gradient epochs.
        risk_aversion: Exponential utility parameter a > 0.
        seed:          Random seed for network initialisation.

    Returns:
        dict with keys:
            'utility_Q' – Q-weighted OCE utility at convergence (should be ≈ 0).
    """
    S = paths_P["S"]
    dH = paths_P["dH"]
    N = S.shape[0]
    Z_T_zero = torch.zeros(N)
    q = Q_weights.detach()

    hedger = _build_hedger(seed, risk_aversion)
    optimizer = optim.Adam(hedger.parameters(), lr=_LR)
    a = risk_aversion

    for _ in range(n_epochs):
        hedger.train()
        # Q-weighting requires the full sample for E_Q[.] = Σ q_ω u(...)
        pnl_full = hedger.compute_pnl({"S": S, "dH": dH}, Z_T_zero)
        u_vals = hedger.utility(hedger.y + pnl_full)
        # Q-weighted OCE loss (negative = we minimise -OCE)
        loss = -(torch.dot(q, u_vals) - hedger.y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hedger.parameters(), _GRAD_CLIP)
        optimizer.step()

    hedger.eval()
    with torch.no_grad():
        pnl_full = hedger.compute_pnl({"S": S, "dH": dH}, Z_T_zero)
        u_vals = hedger.utility(hedger.y + pnl_full)
        utility_Q = (torch.dot(q, u_vals) - hedger.y).item()

    return {"utility_Q": utility_Q}


# ---------------------------------------------------------------------------
# Vol-mismatch toy example
# ---------------------------------------------------------------------------

def toy_vol_mismatch_example(
    realized_vol: float = 0.15,
    traded_vol: float = 0.20,
    N_paths: int = 5_000,
    n_steps: int = 21,
    seed: int = 42,
) -> dict[str, Any]:
    """Demonstrate exploitation of a volatility mismatch by the deep hedger.

    Paths are simulated with `realized_vol`, but the ATM call liability is
    priced at `traded_vol` (i.e., the option is overpriced relative to the
    realised dynamics).  The deep hedger under P will exploit this mismatch
    by holding a position that benefits from the vol gap.

    Args:
        realized_vol: Volatility used to simulate underlying paths.
        traded_vol:   Volatility used to price the short-call liability.
        N_paths:      Monte Carlo paths.
        n_steps:      Time steps.
        seed:         Random seed.

    Returns:
        dict with keys:
            'utility_P_mismatch' – utility under P with mismatch.
            'utility_P_matched'  – utility under P without mismatch (baseline).
            'utility_gap'        – difference (positive = exploitation detected).
    """
    T = 0.0833
    K = 1.0

    # Paths simulated with realized_vol
    raw_real = bs_simulate(
        S0=1.0, sigma=realized_vol, r=0.0, T=T, n_steps=n_steps, N_paths=N_paths, seed=seed
    )
    S_real = raw_real["S"]
    dH_real = S_real[:, 1:] - S_real[:, :-1]

    # Liability priced at TRADED vol (overpriced call)
    from simulators.black_scholes import call_price
    S_T = S_real[:, -1]
    call_payoff = torch.clamp(S_T - K, min=0.0)
    traded_premium = call_price(
        torch.tensor([1.0]), K=K, sigma=traded_vol, r=0.0, T=T
    ).item()
    # Agent receives the overpriced premium but pays realized payoff
    Z_T_mismatch = traded_premium - call_payoff

    # Train hedger on MISMATCHED liability
    hedger_mm = _build_hedger(seed=seed)
    optimizer = optim.Adam(hedger_mm.parameters(), lr=_LR)
    N = N_paths
    for _ in range(80):
        hedger_mm.train()
        idx = torch.randperm(N)[:_BATCH_SIZE]
        b = {"S": S_real[idx], "dH": dH_real[idx]}
        loss = hedger_mm(b, Z_T_mismatch[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hedger_mm.parameters(), _GRAD_CLIP)
        optimizer.step()
    hedger_mm.eval()
    with torch.no_grad():
        utility_mm = -(hedger_mm({"S": S_real, "dH": dH_real}, Z_T_mismatch).item())

    # Baseline: matched vol (no mismatch)
    matched_premium = call_price(
        torch.tensor([1.0]), K=K, sigma=realized_vol, r=0.0, T=T
    ).item()
    Z_T_matched = matched_premium - call_payoff

    hedger_base = _build_hedger(seed=seed + 1)
    optimizer = optim.Adam(hedger_base.parameters(), lr=_LR)
    for _ in range(80):
        hedger_base.train()
        idx = torch.randperm(N)[:_BATCH_SIZE]
        b = {"S": S_real[idx], "dH": dH_real[idx]}
        loss = hedger_base(b, Z_T_matched[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hedger_base.parameters(), _GRAD_CLIP)
        optimizer.step()
    hedger_base.eval()
    with torch.no_grad():
        utility_base = -(hedger_base({"S": S_real, "dH": dH_real}, Z_T_matched).item())

    return {
        "utility_P_mismatch": utility_mm,
        "utility_P_matched": utility_base,
        "utility_gap": utility_mm - utility_base,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_gains_distribution(
    gains_P: torch.Tensor,
    Q_weights: torch.Tensor,
    save_path: str,
) -> None:
    """Histogram of terminal gains under P and Q (reweighted)."""
    g = gains_P.detach().numpy()
    q = Q_weights.detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(g, bins=60, color="steelblue", edgecolor="none", density=True, alpha=0.8)
    axes[0].set_title("Terminal Gains Under P")
    axes[0].set_xlabel("Gain")
    axes[0].set_ylabel("Density")

    # Weighted histogram approximating Q
    axes[1].hist(g, bins=60, weights=q * len(q), color="darkorange",
                 edgecolor="none", density=True, alpha=0.8)
    axes[1].set_title("Terminal Gains Under Q (reweighted)")
    axes[1].set_xlabel("Gain")
    axes[1].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def _plot_utility_comparison(
    utility_P: float,
    utility_Q: float,
    utility_mismatch: float,
    save_path: str,
) -> None:
    labels = ["Under P\n(empty portfolio)", "Under Q\n(drift removed)", "Vol-mismatch\nexploit"]
    values = [utility_P, utility_Q, utility_mismatch]
    colors = ["steelblue", "darkorange", "firebrick"]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=colors, edgecolor="none")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.ylabel("OCE Utility")
    plt.title("Utility Comparison Across Scenarios")
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.0005 * np.sign(v),
                 f"{v:.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def _plot_q_weights(Q_weights: torch.Tensor, save_path: str) -> None:
    q = Q_weights.detach().numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(q, bins=60, color="purple", edgecolor="none", alpha=0.8)
    plt.xlabel("Q weight q_ω")
    plt.ylabel("Count")
    plt.title("Distribution of Path Weights dQ/dP")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict[str, Any]:
    """Run Experiment 3: stat-arb diagnosis and drift removal.

    Returns:
        Nested dict with all sub-experiment results.
    """
    t0 = time.time()

    print("Exp 3a: Simulating drifted paths (mu=0.10, sigma=0.15) …")
    paths_P = simulate_paths_with_drift(
        mu=0.1, sigma=0.15, N_paths=20_000, n_steps=21, T=0.0833, seed=42
    )
    dS_mean = (paths_P["S"][:, -1] - paths_P["S"][:, 0]).mean().item()
    print(f"  Mean terminal stock move: {dS_mean:.4f}  (positive = drift exploitable)")

    print("Exp 3b: Training empty portfolio under P …")
    train_result = train_empty_portfolio(paths_P, n_epochs=100, risk_aversion=1.0, seed=42)
    y_star = train_result["y_star"]
    utility_P = train_result["utility"]
    gains = train_result["terminal_gains"]
    print(f"  y* = {y_star:.5f},  utility under P = {utility_P:.5f}  "
          f"(positive = exploits drift)")

    print("Exp 3c: Verifying first-order condition …")
    foc = verify_foc_identities(gains, y_star, risk_aversion=1.0)
    print(f"  E^P[u'(y*+gains)] = {foc['E_u_prime']:.4f}  (should be ≈ 1.0)")

    print("Exp 3d: Computing Q weights …")
    Q_weights = compute_Q_weights(gains, y_star, risk_aversion=1.0)
    effective_sample = 1.0 / (Q_weights ** 2).sum().item()
    print(f"  Effective sample size: {effective_sample:.0f} / {len(Q_weights)}")

    print("Exp 3e: Training empty portfolio under Q (drift removed) …")
    q_result = train_under_Q(paths_P, Q_weights, n_epochs=100, risk_aversion=1.0, seed=0)
    utility_Q = q_result["utility_Q"]
    print(f"  Utility under Q = {utility_Q:.5f}  (should be ≈ 0)")

    print("Exp 3f: Volatility mismatch toy example …")
    mismatch_result = toy_vol_mismatch_example(
        realized_vol=0.15, traded_vol=0.20, N_paths=5_000, n_steps=21, seed=42
    )
    print(f"  Utility (mismatch) = {mismatch_result['utility_P_mismatch']:.5f}")
    print(f"  Utility (matched)  = {mismatch_result['utility_P_matched']:.5f}")
    print(f"  Gap                = {mismatch_result['utility_gap']:.5f}  "
          f"(positive = vol gap exploited)")

    # --- Plots ---
    _plot_gains_distribution(
        gains, Q_weights,
        os.path.join(RESULTS_DIR, "exp3_gains_distribution.png"),
    )
    _plot_q_weights(
        Q_weights,
        os.path.join(RESULTS_DIR, "exp3_q_weights.png"),
    )
    _plot_utility_comparison(
        utility_P, utility_Q, mismatch_result["utility_P_mismatch"],
        os.path.join(RESULTS_DIR, "exp3_utility_comparison.png"),
    )

    results: dict[str, Any] = {
        "drifted_paths": {
            "mu": paths_P["mu"],
            "sigma": paths_P["sigma"],
            "mean_terminal_move": dS_mean,
        },
        "train_under_P": {
            "y_star": y_star,
            "utility_P": utility_P,
        },
        "foc_check": {
            "E_u_prime": foc["E_u_prime"],
        },
        "q_measure": {
            "effective_sample_size": effective_sample,
        },
        "train_under_Q": {
            "utility_Q": utility_Q,
        },
        "vol_mismatch": {
            "realized_vol": 0.15,
            "traded_vol": 0.20,
            "utility_P_mismatch": mismatch_result["utility_P_mismatch"],
            "utility_P_matched": mismatch_result["utility_P_matched"],
            "utility_gap": mismatch_result["utility_gap"],
        },
        "elapsed_s": round(time.time() - t0, 1),
    }

    out_path = os.path.join(RESULTS_DIR, "exp3_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path} (elapsed {results['elapsed_s']} s)")

    return results


if __name__ == "__main__":
    main()
