"""
Experiment 5: Deep Bellman Hedging with Value Iteration and Actor-Critic.

Implements Bellman-value formulation of deep hedging.
Tests value iteration from V^0=0 and actor-critic approximation.
Compares against vanilla deep hedging on a simple ATM call portfolio.
"""
import os
import sys
import json

sys.path.insert(0, '/workspace/project')

import torch
import torch.nn as nn

from simulators.black_scholes import (
    simulate_paths,
    call_price,
    call_delta,
    call_gamma,
)
from policies.hedge_policy import HedgePolicy, DeepHedger
from objectives.oce import OCELoss
from bellman.actor import BellmanActor
from bellman.critic import BellmanCritic
from bellman.value_iteration import ValueIterator

# --------------------------------------------------------------------- #
# Global constants                                                         #
# --------------------------------------------------------------------- #
_K: float = 100.0
_S0: float = 100.0
_R: float = 0.0
_T: float = 1.0        # 1-year maturity
_SIGMA: float = 0.2
_RISK_AVERSION: float = 1.0


# --------------------------------------------------------------------- #
# a) Daily transition data preparation                                     #
# --------------------------------------------------------------------- #

def prepare_daily_transitions(
    N_paths: int = 10000,
    n_steps: int = 21,
    sigma: float = 0.2,
    beta: float = 0.99,
    seed: int = 42,
) -> dict:
    """Simulate GBM paths and compute single-step Bellman transitions.

    Portfolio: short one ATM European call, hedged with the underlying stock.

    State features at step t: [ttm_normalised, log_moneyness, delta, gamma]
    where ttm = (n_steps - t) * dt / T (in [0, 1]).

    Transitions:
        dZ_t = C(S_t, T_t) - beta * C(S_{t+1}, T_{t+1})  (liability change)
        dH_t = beta * S_{t+1} - S_t                         (hedge gain)

    Args:
        N_paths: Number of Monte Carlo paths.
        n_steps: Number of discrete time steps (typically 21 for ~1 month).
        sigma: Black-Scholes volatility.
        beta: One-period discount factor.
        seed: Torch manual seed.

    Returns:
        Dictionary with:
            'states'     : [N_paths * n_steps, 4]
            'dZ'         : [N_paths * n_steps]
            'dH'         : [N_paths * n_steps, 1]
            'next_states': [N_paths * n_steps, 4]
            'beta'       : float
            'n_paths'    : int
            'n_steps'    : int
    """
    dt = _T / n_steps
    raw = simulate_paths(
        S0=_S0, sigma=sigma, r=_R, T=_T, n_steps=n_steps,
        N_paths=N_paths, seed=seed
    )
    S = raw["S"]  # [N, n_steps+1]

    all_states, all_next_states = [], []
    all_dZ, all_dH = [], []

    for t in range(n_steps):
        ttm_t = (n_steps - t) * dt        # absolute time-to-maturity at t
        ttm_t1 = (n_steps - t - 1) * dt   # absolute time-to-maturity at t+1

        S_t = S[:, t]       # [N]
        S_t1 = S[:, t + 1]  # [N]

        # Call option values
        C_t = call_price(S_t, K=_K, sigma=sigma, r=_R, T=ttm_t)
        C_t1 = call_price(S_t1, K=_K, sigma=sigma, r=_R, T=ttm_t1)

        # One-step transitions for short-call portfolio
        dZ_t = C_t - beta * C_t1            # [N]
        dH_t = (beta * S_t1 - S_t).unsqueeze(1)  # [N, 1]

        # State features: [ttm_norm, log_moneyness, delta, gamma]
        ttm_norm_t = ttm_t / _T  # normalised to [0, 1]
        ttm_vec_t = torch.full((N_paths,), ttm_norm_t)
        log_m_t = torch.log(S_t / _K)
        del_t = call_delta(S_t, K=_K, sigma=sigma, r=_R, T=ttm_t)
        gam_t = torch.clamp(
            call_gamma(S_t, K=_K, sigma=sigma, r=_R, T=ttm_t), max=5.0
        )
        state_t = torch.stack([ttm_vec_t, log_m_t, del_t, gam_t], dim=1)  # [N, 4]

        # Next-state features
        ttm_norm_t1 = ttm_t1 / _T
        ttm_vec_t1 = torch.full((N_paths,), ttm_norm_t1)
        log_m_t1 = torch.log(S_t1 / _K)
        del_t1 = call_delta(S_t1, K=_K, sigma=sigma, r=_R, T=ttm_t1)
        gam_t1 = torch.clamp(
            call_gamma(S_t1, K=_K, sigma=sigma, r=_R, T=ttm_t1), max=5.0
        )
        state_t1 = torch.stack([ttm_vec_t1, log_m_t1, del_t1, gam_t1], dim=1)  # [N, 4]

        all_states.append(state_t)
        all_next_states.append(state_t1)
        all_dZ.append(dZ_t)
        all_dH.append(dH_t)

    return {
        "states": torch.cat(all_states, dim=0),       # [N*T, 4]
        "dZ": torch.cat(all_dZ, dim=0),               # [N*T]
        "dH": torch.cat(all_dH, dim=0),               # [N*T, 1]
        "next_states": torch.cat(all_next_states, dim=0),  # [N*T, 4]
        "beta": beta,
        "n_paths": N_paths,
        "n_steps": n_steps,
    }


# --------------------------------------------------------------------- #
# b) Vanilla deep hedger                                                   #
# --------------------------------------------------------------------- #

def train_vanilla_hedger(
    paths: dict,
    n_epochs: int = 100,
    seed: int = 42,
) -> dict:
    """Standard (non-Bellman) deep hedging via OCE minimisation.

    Args:
        paths: Path dict from simulate_paths() augmented with 'dH' and 'Z_T'.
               If 'dH' is absent it is computed as S[:, 1:] - S[:, :-1].
               If 'Z_T' is absent it is computed as max(S_T - K, 0).
        n_epochs: Number of training epochs.
        seed: Torch manual seed.

    Returns:
        Dictionary {'final_utility': float, 'loss_history': list}.
    """
    torch.manual_seed(seed)

    S = paths["S"]  # [N, T+1]
    if "dH" not in paths:
        paths = dict(paths)
        paths["dH"] = S[:, 1:] - S[:, :-1]
    if "Z_T" not in paths:
        paths = dict(paths)
        paths["Z_T"] = torch.clamp(S[:, -1] - _K, min=0.0)

    liability = -paths["Z_T"]  # short call

    policy = HedgePolicy(input_dim=3)
    hedger = DeepHedger(policy, risk_aversion=_RISK_AVERSION)
    optimizer = torch.optim.Adam(hedger.parameters(), lr=1e-3)

    loss_history = []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = hedger.forward(paths, liability)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hedger.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(loss.item())

    # Evaluate final utility
    oce = OCELoss(risk_aversion=_RISK_AVERSION)
    with torch.no_grad():
        pnl = hedger.compute_pnl(paths, liability)
        final_utility = float(oce.utility_value(pnl))

    return {"final_utility": final_utility, "loss_history": loss_history}


# --------------------------------------------------------------------- #
# c) Bellman deep hedger                                                   #
# --------------------------------------------------------------------- #

def train_bellman_hedger(
    transitions: dict,
    n_iterations: int = 5,
    seed: int = 42,
) -> dict:
    """Deep Bellman hedging via actor-critic value iteration.

    Starts from V^0 ≈ 0 (random initialisation, quickly driven to near-zero)
    and applies n_iterations of (actor_step, critic_step).

    Args:
        transitions: Dict from prepare_daily_transitions().
        n_iterations: Number of Bellman iteration cycles.
        seed: Torch manual seed.

    Returns:
        Dictionary {
            'v_norm_history': list of ||V^n|| after each critic step,
            'actor_losses'  : list of final actor loss per iteration,
            'critic_losses' : list of final critic MSE per iteration,
        }.
    """
    torch.manual_seed(seed)
    beta = transitions.get("beta", 0.99)

    actor = BellmanActor(state_dim=4, n_instruments=1)
    critic = BellmanCritic(state_dim=4)
    vi = ValueIterator(
        actor, critic, beta=beta, risk_aversion=_RISK_AVERSION
    )

    states = transitions["states"]
    dZ = transitions["dZ"]
    dH = transitions["dH"]
    next_states = transitions["next_states"]

    history = vi.run(
        states, dZ, dH, next_states,
        n_iterations=n_iterations,
        n_actor_epochs=30,
        n_critic_epochs=30,
    )

    return {
        "v_norm_history": vi.v_values,
        "actor_losses": [h["actor_loss"] for h in history],
        "critic_losses": [h["critic_loss"] for h in history],
    }


# --------------------------------------------------------------------- #
# d) Comparison                                                             #
# --------------------------------------------------------------------- #

def compare_hedgers(vanilla_result: dict, bellman_result: dict) -> dict:
    """Compare vanilla and Bellman hedging performance metrics.

    Args:
        vanilla_result: Output of train_vanilla_hedger().
        bellman_result: Output of train_bellman_hedger().

    Returns:
        Dictionary with comparison statistics.
    """
    v_norms = bellman_result.get("v_norm_history", [])
    return {
        "vanilla_final_utility": vanilla_result.get("final_utility"),
        "bellman_v_norm_final": v_norms[-1] if v_norms else None,
        "bellman_actor_loss_final": bellman_result["actor_losses"][-1]
        if bellman_result["actor_losses"] else None,
        "bellman_critic_loss_final": bellman_result["critic_losses"][-1]
        if bellman_result["critic_losses"] else None,
        "bellman_v_norm_converged": v_norms[-1] < 50.0 if v_norms else False,
    }


# --------------------------------------------------------------------- #
# e) Main                                                                  #
# --------------------------------------------------------------------- #

def main() -> dict:
    """Run Experiment 5 and save results.

    Returns:
        Full results dictionary.
    """
    print("=== Experiment 5: Deep Bellman Hedging ===\n")

    # ---- Data ----
    print("Simulating GBM paths...")
    raw_paths = simulate_paths(
        S0=_S0, sigma=_SIGMA, r=_R, T=_T, n_steps=21, N_paths=10000, seed=42
    )

    print("Computing daily Bellman transitions...")
    transitions = prepare_daily_transitions(
        N_paths=10000, n_steps=21, sigma=_SIGMA, beta=0.99, seed=42
    )

    # ---- Vanilla hedger ----
    print("\n1) Training vanilla deep hedger...")
    vanilla = train_vanilla_hedger(raw_paths, n_epochs=100, seed=42)
    print(f"   Final utility: {vanilla['final_utility']:.4f}")

    # ---- Bellman hedger ----
    print("\n2) Training Bellman hedger (value iteration)...")
    bellman = train_bellman_hedger(transitions, n_iterations=5, seed=42)
    print(f"   V-norm history: {[f'{v:.4f}' for v in bellman['v_norm_history']]}")

    # ---- Comparison ----
    comparison = compare_hedgers(vanilla, bellman)

    results = {
        "vanilla": {"final_utility": vanilla["final_utility"]},
        "bellman": {
            "v_norm_history": bellman["v_norm_history"],
            "actor_losses": bellman["actor_losses"],
            "critic_losses": bellman["critic_losses"],
        },
        "comparison": comparison,
    }

    # ---- Save ----
    os.makedirs("results", exist_ok=True)
    with open("results/exp5_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nResults saved to results/exp5_results.json")

    # ---- Optional plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(vanilla["loss_history"])
        axes[0].set_title("Vanilla: OCE Loss")
        axes[0].set_xlabel("Epoch")

        axes[1].plot(bellman["actor_losses"], marker="o")
        axes[1].set_title("Bellman: Actor Loss")
        axes[1].set_xlabel("Iteration")

        axes[2].plot(bellman["v_norm_history"], marker="s")
        axes[2].set_title("Bellman: V-norm Convergence")
        axes[2].set_xlabel("Iteration")

        plt.tight_layout()
        plt.savefig("results/exp5_convergence.png", dpi=100)
        plt.close()
        print("Convergence curves saved to results/exp5_convergence.png")
    except Exception:
        pass

    return results


if __name__ == "__main__":
    main()
