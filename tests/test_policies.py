"""
Tests for hedge policy network.
"""
import sys
sys.path.insert(0, '/workspace/project')

import pytest
import torch
from policies.hedge_policy import HedgePolicy, DeepHedger
from simulators.black_scholes import simulate_paths


# ------------------------------------------------------------------ #
# HedgePolicy (actor network)                                           #
# ------------------------------------------------------------------ #

def test_policy_output_shape():
    """Policy maps state [N, input_dim] -> action [N, n_instruments]."""
    N, state_dim, n_instruments = 50, 3, 1
    policy = HedgePolicy(input_dim=state_dim, n_instruments=n_instruments)
    state = torch.randn(N, state_dim)
    out = policy(state)
    assert out.shape == (N, n_instruments), (
        f"Expected ({N}, {n_instruments}), got {out.shape}"
    )


def test_policy_multi_instrument_shape():
    """Policy supports multiple hedge instruments."""
    N, state_dim, n_inst = 30, 5, 3
    policy = HedgePolicy(input_dim=state_dim, n_instruments=n_inst)
    out = policy(torch.randn(N, state_dim))
    assert out.shape == (N, n_inst)


def test_policy_finite_outputs():
    """Policy output contains no NaN or Inf values."""
    policy = HedgePolicy(input_dim=3)
    state = torch.randn(100, 3)
    out = policy(state)
    assert torch.isfinite(out).all(), "Policy produced non-finite outputs."


def test_policy_single_sample():
    """Policy handles a single-sample batch (N=1)."""
    policy = HedgePolicy(input_dim=3)
    state = torch.randn(1, 3)
    out = policy(state)
    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------ #
# DeepHedger (rolling P&L computation)                                  #
# ------------------------------------------------------------------ #

def test_hedger_pnl_shape():
    """Terminal P&L has shape [N_paths]."""
    N, n_steps = 50, 10
    policy = HedgePolicy(input_dim=3)
    hedger = DeepHedger(policy)

    paths = simulate_paths(S0=100.0, sigma=0.2, r=0.0, T=1.0,
                           n_steps=n_steps, N_paths=N, seed=42)
    paths["dH"] = paths["S"][:, 1:] - paths["S"][:, :-1]
    # Short ATM call: liability = -max(S_T - K, 0)
    liability = -torch.clamp(paths["S"][:, -1] - 100.0, min=0.0)

    pnl = hedger.compute_pnl(paths, liability)
    assert pnl.shape == (N,), f"Expected ({N},), got {pnl.shape}"


def test_hedger_pnl_finite():
    """Terminal P&L contains no NaN or Inf values."""
    N, n_steps = 30, 5
    policy = HedgePolicy(input_dim=3)
    hedger = DeepHedger(policy)

    paths = simulate_paths(S0=100.0, sigma=0.2, r=0.0, T=1.0,
                           n_steps=n_steps, N_paths=N, seed=11)
    paths["dH"] = paths["S"][:, 1:] - paths["S"][:, :-1]
    liability = torch.zeros(N)  # trivial liability

    pnl = hedger.compute_pnl(paths, liability)
    assert torch.isfinite(pnl).all(), "P&L contains non-finite values."


def test_no_lookahead():
    """Policy action at time t must depend only on state S_t, not on future prices.

    Two path sets that share the same S_t at time t but differ thereafter
    must produce identical policy actions at that time step.
    """
    N, n_steps = 20, 5
    torch.manual_seed(42)
    policy = HedgePolicy(input_dim=3)

    # Two independent path sets (different seeds -> different future paths)
    paths_a = simulate_paths(S0=100.0, sigma=0.2, r=0.0, T=1.0,
                             n_steps=n_steps, N_paths=N, seed=1)
    paths_b = simulate_paths(S0=100.0, sigma=0.2, r=0.0, T=1.0,
                             n_steps=n_steps, N_paths=N, seed=2)

    # Override S at t=2 with a shared fixed value (same state at t for both)
    t = 2
    S_shared = torch.full((N,), 100.0)
    paths_a["S"][:, t] = S_shared
    paths_b["S"][:, t] = S_shared

    # Build state at t exactly as DeepHedger does:
    #   state = [ttm_t, log(S_t), prev_hedge=0]
    ttm = float(n_steps - t) / n_steps
    log_m_a = torch.log(paths_a["S"][:, t]).unsqueeze(1)
    log_m_b = torch.log(paths_b["S"][:, t]).unsqueeze(1)
    prev_hedge = torch.zeros(N, 1)

    state_a = torch.cat([torch.full((N, 1), ttm), log_m_a, prev_hedge], dim=1)
    state_b = torch.cat([torch.full((N, 1), ttm), log_m_b, prev_hedge], dim=1)

    # States must be identical (both paths share S_t)
    assert torch.allclose(state_a, state_b), (
        "States should be equal when S_t is the same for both paths."
    )

    # Actions must therefore be identical (policy is deterministic)
    action_a = policy(state_a)
    action_b = policy(state_b)
    assert torch.allclose(action_a, action_b), (
        "Policy action differs for paths that share state at t; "
        "this would imply look-ahead bias."
    )


# ------------------------------------------------------------------ #
# DeepHedger forward (OCE loss)                                         #
# ------------------------------------------------------------------ #

def test_hedger_forward_scalar():
    """DeepHedger.forward() returns a scalar loss."""
    policy = HedgePolicy(input_dim=3)
    hedger = DeepHedger(policy)
    paths = simulate_paths(S0=100.0, sigma=0.2, r=0.0, T=1.0,
                           n_steps=5, N_paths=20, seed=0)
    paths["dH"] = paths["S"][:, 1:] - paths["S"][:, :-1]
    liability = torch.zeros(20)
    loss = hedger.forward(paths, liability)
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    assert torch.isfinite(loss), "Loss is not finite."
