"""
Experiment 1: Vanilla Deep Hedging Sanity Checks in Black-Scholes and Heston Models

Verifies that deep hedging reproduces theoretical pricing asymptotics and qualitative
hedge structure in Black-Scholes and Heston settings.

Key results:
- Black-Scholes asymptotic slope ~0.67
- Heston asymptotic slope ~0.71
- Call-spread delta comparison
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

from simulators.black_scholes import call_price, call_spread_delta, simulate_paths as bs_simulate
from simulators.heston import simulate_paths as heston_simulate
from objectives.oce import OCELoss
from policies.hedge_policy import DeepHedger, HedgePolicy
from evaluation.asymptotic import bootstrap_slope_uncertainty, estimate_asymptotic_slope
from evaluation.delta_surface import compute_learned_hedge_surface, plot_delta_comparison

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

_BS_SIGMA = 0.2
_BS_R = 0.0
_BS_S0 = 1.0
_BS_T = 0.0833
_BS_N_STEPS = 21
_BS_K = _BS_S0  # ATM

_HESTON_S0 = 1.0
_HESTON_V0 = 0.04
_HESTON_KAPPA = 2.0
_HESTON_THETA = 0.04
_HESTON_XI = 0.3
_HESTON_RHO = -0.7
_HESTON_R = 0.0
_HESTON_T = 0.0833
_HESTON_N_STEPS = 21
_HESTON_K = _HESTON_S0

_RISK_AVERSION = 1.0
_BATCH_SIZE = 2048
_LR = 1e-3
_GRAD_CLIP = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_paths_dict(S: torch.Tensor) -> dict:
    """Add hedge-instrument increments dH = dS to a raw path dict."""
    return {"S": S, "dH": S[:, 1:] - S[:, :-1]}


def _call_liability(S_T: torch.Tensor, K: float) -> torch.Tensor:
    """Short call terminal liability: Z_T = -max(S_T - K, 0)."""
    return -torch.clamp(S_T - K, min=0.0)


def _call_spread_liability(S_T: torch.Tensor, K1: float, K2: float) -> torch.Tensor:
    """Short bull-call-spread liability: Z_T = -[max(S_T-K1,0) - max(S_T-K2,0)]."""
    return -(torch.clamp(S_T - K1, min=0.0) - torch.clamp(S_T - K2, min=0.0))


def _build_hedger(seed: int, n_layers: int = 4, n_units: int = 32) -> DeepHedger:
    torch.manual_seed(seed)
    policy = HedgePolicy(input_dim=3, n_instruments=1, n_layers=n_layers, n_units=n_units)
    return DeepHedger(policy=policy, risk_aversion=_RISK_AVERSION)


def _train(
    hedger: DeepHedger,
    paths: dict,
    Z_T: torch.Tensor,
    n_epochs: int,
    cost_epsilon: float = 0.0,
) -> list[float]:
    """Mini-batch training loop. Returns list of per-epoch losses."""
    optimizer = optim.Adam(hedger.parameters(), lr=_LR)
    N = Z_T.shape[0]
    loss_history: list[float] = []

    for _ in range(n_epochs):
        hedger.train()
        idx = torch.randperm(N)[:_BATCH_SIZE]
        batch_paths = {"S": paths["S"][idx], "dH": paths["dH"][idx]}
        loss = hedger(batch_paths, Z_T[idx], cost_epsilon=cost_epsilon)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hedger.parameters(), _GRAD_CLIP)
        optimizer.step()
        loss_history.append(loss.item())

    return loss_history


def _oce_price(hedger: DeepHedger, paths: dict, Z_T: torch.Tensor,
               cost_epsilon: float = 0.0) -> float:
    """
    Evaluate the OCE implied price y* = self.y at convergence.

    y* captures the extra compensation the agent requires over the baseline
    (no premium received) to enter the short-option position; in the frictionless
    limit this is ~0, and it grows as ~epsilon^(2/3) with transaction costs.
    """
    hedger.eval()
    with torch.no_grad():
        hedger(paths, Z_T, cost_epsilon=cost_epsilon)
    return hedger.y.item()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_deep_hedge_bs(
    epsilon: float,
    N_paths: int = 20_000,
    n_epochs: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a deep hedger for an ATM short call in the Black-Scholes model.

    Uses the stock itself as the hedge instrument (dH_t = S_{t+1} - S_t).
    Transaction costs proportional to epsilon are applied at each rebalancing.

    Args:
        epsilon: Proportional transaction-cost rate.
        N_paths: Number of Monte Carlo training paths.
        n_epochs: Number of gradient-descent epochs.
        seed: Random seed for path simulation and network initialisation.

    Returns:
        dict with keys:
            'utility'      – OCE-implied y* at convergence (float).
            'bs_price'     – Analytical Black-Scholes ATM call price (float).
            'loss_history' – Per-epoch OCE loss values (list[float]).
    """
    raw = bs_simulate(
        S0=_BS_S0, sigma=_BS_SIGMA, r=_BS_R, T=_BS_T,
        n_steps=_BS_N_STEPS, N_paths=N_paths, seed=seed,
    )
    paths = _make_paths_dict(raw["S"])
    Z_T = _call_liability(raw["S"][:, -1], K=_BS_K)

    hedger = _build_hedger(seed)
    loss_history = _train(hedger, paths, Z_T, n_epochs, cost_epsilon=epsilon)

    bs_price_val = call_price(
        torch.tensor([_BS_S0]), K=_BS_K, sigma=_BS_SIGMA, r=_BS_R, T=_BS_T
    ).item()
    utility = _oce_price(hedger, paths, Z_T, cost_epsilon=epsilon)

    return {
        "utility": utility,
        "bs_price": bs_price_val,
        "loss_history": loss_history,
    }


def run_asymptotic_bs(
    epsilon_grid: list[float],
    N_paths: int = 20_000,
    n_epochs: int = 100,
) -> dict[str, Any]:
    """Asymptotic transaction-cost scaling experiment in Black-Scholes.

    For each epsilon in the grid, trains an independent deep hedger and records
    the OCE y* (the extra cost due to friction).  The log-log slope of y* vs
    epsilon should converge to 2/3 ≈ 0.67.

    Args:
        epsilon_grid: Transaction-cost rates to sweep.
        N_paths: Monte Carlo paths per training run.
        n_epochs: Gradient-descent epochs per training run.

    Returns:
        dict with keys:
            'epsilon_grid' – list of epsilon values.
            'corrections'  – corresponding y* values (the TC-driven price impact).
            'slope'        – log-log regression slope (target: ~0.67 for BS).
            'slope_std'    – bootstrap standard deviation of the slope.
    """
    corrections: list[float] = []
    for i, eps in enumerate(epsilon_grid):
        result = run_deep_hedge_bs(eps, N_paths=N_paths, n_epochs=n_epochs, seed=42 + i)
        corrections.append(abs(result["utility"]))

    # Guard against all-zero corrections (degenerate case)
    safe = [max(c, 1e-9) for c in corrections]
    slope, _, _ = estimate_asymptotic_slope(epsilon_grid, safe)
    slope_mean, slope_std = bootstrap_slope_uncertainty(epsilon_grid, safe)

    return {
        "epsilon_grid": epsilon_grid,
        "corrections": corrections,
        "slope": slope_mean,
        "slope_std": slope_std,
    }


def run_asymptotic_heston(
    epsilon_grid: list[float],
    N_paths: int = 20_000,
    n_epochs: int = 100,
) -> dict[str, Any]:
    """Asymptotic transaction-cost scaling experiment in the Heston model.

    Mirrors run_asymptotic_bs but simulates stochastic-volatility paths.
    The theoretical slope is slightly above the BS value (~0.71).

    Args:
        epsilon_grid: Transaction-cost rates to sweep.
        N_paths: Monte Carlo paths per training run.
        n_epochs: Gradient-descent epochs per training run.

    Returns:
        dict with keys:
            'epsilon_grid', 'corrections', 'slope', 'slope_std'.
    """
    corrections: list[float] = []
    for i, eps in enumerate(epsilon_grid):
        torch.manual_seed(42 + i)
        raw = heston_simulate(
            S0=_HESTON_S0, V0=_HESTON_V0, kappa=_HESTON_KAPPA, theta=_HESTON_THETA,
            xi=_HESTON_XI, rho=_HESTON_RHO, r=_HESTON_R, T=_HESTON_T,
            n_steps=_HESTON_N_STEPS, N_paths=N_paths, seed=42 + i,
        )
        paths = _make_paths_dict(raw["S"])
        Z_T = _call_liability(raw["S"][:, -1], K=_HESTON_K)

        hedger = _build_hedger(seed=42 + i)
        _train(hedger, paths, Z_T, n_epochs, cost_epsilon=eps)
        corrections.append(abs(_oce_price(hedger, paths, Z_T, cost_epsilon=eps)))

    safe = [max(c, 1e-9) for c in corrections]
    slope, _, _ = estimate_asymptotic_slope(epsilon_grid, safe)
    slope_mean, slope_std = bootstrap_slope_uncertainty(epsilon_grid, safe)

    return {
        "epsilon_grid": epsilon_grid,
        "corrections": corrections,
        "slope": slope_mean,
        "slope_std": slope_std,
    }


def run_call_spread_delta_bs(
    N_paths: int = 20_000,
    n_epochs: int = 150,
    seed: int = 42,
) -> dict[str, Any]:
    """Train deep hedger on a bull call spread and compare delta surfaces.

    Liability: short bull spread = -[max(S_T-K1,0) - max(S_T-K2,0)]
    with K1=0.95 (lower strike) and K2=1.05 (upper strike).

    Args:
        N_paths: Monte Carlo training paths.
        n_epochs: Gradient-descent epochs.
        seed: Random seed.

    Returns:
        dict with keys:
            'learned_surface'    – learned hedge surface (shape: [n_ttm, n_S]).
            'analytical_surface' – call-spread BS delta surface (same shape).
            'mse'                – mean squared error between surfaces.
            'S_grid'             – spot grid used (list).
            'ttm_grid'           – normalised time-to-maturity grid (list).
    """
    K1, K2 = 0.95, 1.05

    raw = bs_simulate(
        S0=_BS_S0, sigma=_BS_SIGMA, r=_BS_R, T=_BS_T,
        n_steps=_BS_N_STEPS, N_paths=N_paths, seed=seed,
    )
    paths = _make_paths_dict(raw["S"])
    Z_T = _call_spread_liability(raw["S"][:, -1], K1=K1, K2=K2)

    hedger = _build_hedger(seed, n_layers=4, n_units=64)
    _train(hedger, paths, Z_T, n_epochs, cost_epsilon=0.0)
    hedger.eval()

    # Evaluation grids
    S_grid = np.linspace(0.7, 1.3, 30)
    # Normalised TTM: 1 at start, close to 0 at expiry
    ttm_grid = np.linspace(1.0 / _BS_N_STEPS, 1.0, 15)

    learned = compute_learned_hedge_surface(hedger, S_grid, ttm_grid, device="cpu")

    # Analytical call-spread delta on same grid
    analytical = np.zeros_like(learned)
    for i, ttm_norm in enumerate(ttm_grid):
        ttm_years = ttm_norm * _BS_T
        S_t = torch.tensor(S_grid, dtype=torch.float32)
        analytical[i] = call_spread_delta(
            S_t, K1=K1, K2=K2, sigma=_BS_SIGMA, r=_BS_R, T=ttm_years
        ).numpy()

    mse = float(np.mean((learned - analytical) ** 2))

    return {
        "learned_surface": learned.tolist(),
        "analytical_surface": analytical.tolist(),
        "mse": mse,
        "S_grid": S_grid.tolist(),
        "ttm_grid": ttm_grid.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_asymptotic(results_bs: dict, results_heston: dict, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, res, label, target_slope, color in zip(
        axes,
        [results_bs, results_heston],
        ["Black-Scholes", "Heston"],
        [0.67, 0.71],
        ["steelblue", "darkorange"],
    ):
        eps = res["epsilon_grid"]
        corr = [max(c, 1e-12) for c in res["corrections"]]
        slope = res["slope"]
        slope_std = res["slope_std"]

        ax.loglog(eps, corr, "o-", color=color, label=f"Observed (slope={slope:.2f}±{slope_std:.2f})")

        # Reference line with target slope
        ref_c = corr[0] / (eps[0] ** target_slope)
        ref_line = [ref_c * e ** target_slope for e in eps]
        ax.loglog(eps, ref_line, "--", color="grey",
                  label=f"Target slope {target_slope:.2f}")

        ax.set_xlabel("Transaction cost ε")
        ax.set_ylabel("Pricing correction |y*|")
        ax.set_title(f"Asymptotic scaling – {label}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def _plot_loss(loss_bs: list[float], save_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(loss_bs)
    plt.xlabel("Epoch")
    plt.ylabel("OCE loss")
    plt.title("Training convergence (BS, ε=0.001)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def _plot_delta_surface(cs_results: dict, save_path: str) -> None:
    learned = np.array(cs_results["learned_surface"])
    analytical = np.array(cs_results["analytical_surface"])
    S_grid = np.array(cs_results["S_grid"])
    ttm_grid = np.array(cs_results["ttm_grid"])
    plot_delta_comparison(
        learned_surface=learned,
        analytical_surface=analytical,
        S_grid=S_grid,
        time_grid=ttm_grid,
        title=f"Call-Spread Delta Comparison (MSE={cs_results['mse']:.4f})",
        save_path=save_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict[str, Any]:
    """Run all Experiment 1 sub-tests and save figures / results.

    Returns:
        Nested dict with all sub-experiment results.
    """
    t0 = time.time()
    epsilon_grid = [0.001, 0.002, 0.005, 0.01]

    print("Exp 1a: Single BS deep hedge (ε=0.001) …")
    bs_single = run_deep_hedge_bs(epsilon=0.001, N_paths=20_000, n_epochs=100, seed=42)
    print(f"  y* = {bs_single['utility']:.5f},  BS price = {bs_single['bs_price']:.5f}")

    print("Exp 1b: Asymptotic scaling in Black-Scholes …")
    asym_bs = run_asymptotic_bs(epsilon_grid, N_paths=20_000, n_epochs=100)
    print(f"  slope = {asym_bs['slope']:.3f} ± {asym_bs['slope_std']:.3f}  (target ~0.67)")

    print("Exp 1c: Asymptotic scaling in Heston …")
    asym_heston = run_asymptotic_heston(epsilon_grid, N_paths=20_000, n_epochs=100)
    print(f"  slope = {asym_heston['slope']:.3f} ± {asym_heston['slope_std']:.3f}  (target ~0.71)")

    print("Exp 1d: Call-spread delta comparison (BS) …")
    cs_results = run_call_spread_delta_bs(N_paths=20_000, n_epochs=150, seed=42)
    print(f"  Delta surface MSE = {cs_results['mse']:.6f}")

    # --- Plots ---
    _plot_asymptotic(asym_bs, asym_heston,
                     os.path.join(RESULTS_DIR, "exp1_asymptotic_slopes.png"))
    _plot_loss(bs_single["loss_history"],
               os.path.join(RESULTS_DIR, "exp1_bs_loss.png"))
    _plot_delta_surface(cs_results,
                        os.path.join(RESULTS_DIR, "exp1_call_spread_delta.png"))

    results: dict[str, Any] = {
        "bs_single": {
            "utility": bs_single["utility"],
            "bs_price": bs_single["bs_price"],
        },
        "asymptotic_bs": {
            "epsilon_grid": asym_bs["epsilon_grid"],
            "corrections": asym_bs["corrections"],
            "slope": asym_bs["slope"],
            "slope_std": asym_bs["slope_std"],
        },
        "asymptotic_heston": {
            "epsilon_grid": asym_heston["epsilon_grid"],
            "corrections": asym_heston["corrections"],
            "slope": asym_heston["slope"],
            "slope_std": asym_heston["slope_std"],
        },
        "call_spread_delta": {
            "mse": cs_results["mse"],
            "S_grid": cs_results["S_grid"],
            "ttm_grid": cs_results["ttm_grid"],
        },
        "elapsed_s": round(time.time() - t0, 1),
    }

    out_path = os.path.join(RESULTS_DIR, "exp1_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path} (elapsed {results['elapsed_s']} s)")

    return results


if __name__ == "__main__":
    main()
