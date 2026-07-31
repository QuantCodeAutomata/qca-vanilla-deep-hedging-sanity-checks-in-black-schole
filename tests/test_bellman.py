"""
Tests for Bellman deep hedging components.
"""
import sys
sys.path.insert(0, '/workspace/project')

import pytest
import torch
from bellman.actor import BellmanActor
from bellman.critic import BellmanCritic
from bellman.operator import BellmanOperator
from bellman.value_iteration import ValueIterator


# ------------------------------------------------------------------ #
# BellmanCritic                                                          #
# ------------------------------------------------------------------ #

def test_critic_output_shape():
    """Critic maps [N, state_dim] -> [N]."""
    N, state_dim = 100, 4
    critic = BellmanCritic(state_dim=state_dim)
    states = torch.randn(N, state_dim)
    v = critic(states)
    assert v.shape == (N,), f"Expected ({N},), got {v.shape}"


def test_v0_is_zero():
    """Freshly initialised critic produces finite, bounded values (V^0 approx 0)."""
    torch.manual_seed(0)
    critic = BellmanCritic(state_dim=4)
    state = torch.zeros(10, 4)
    v = critic(state)
    assert v.shape == (10,)
    assert torch.isfinite(v).all(), "Initial critic V^0 produced non-finite values."
    # On zero input all layers collapse to biases; magnitude should be small
    assert (v.abs() < 10.0).all(), (
        f"Initial V^0 unexpectedly large on zero input: {v}"
    )


def test_critic_finite_on_random_input():
    """Critic output is finite on random state inputs."""
    critic = BellmanCritic(state_dim=4)
    states = torch.randn(50, 4) * 5.0
    v = critic(states)
    assert torch.isfinite(v).all(), "Critic produced non-finite values."


def test_critic_different_depths():
    """Critic can be constructed with different depths."""
    for n_layers in [2, 4, 6]:
        critic = BellmanCritic(state_dim=3, n_layers=n_layers, n_units=32)
        out = critic(torch.randn(10, 3))
        assert out.shape == (10,)
        assert torch.isfinite(out).all()


# ------------------------------------------------------------------ #
# BellmanActor                                                           #
# ------------------------------------------------------------------ #

def test_actor_output_finite():
    """Actor outputs contain no NaN or Inf values."""
    actor = BellmanActor(state_dim=4, n_instruments=1)
    state = torch.randn(50, 4)
    out = actor(state)
    assert torch.isfinite(out).all(), "Actor produced non-finite outputs."


def test_actor_output_shape():
    """Actor maps [N, state_dim] -> [N, n_instruments]."""
    N, state_dim, n_inst = 40, 4, 1
    actor = BellmanActor(state_dim=state_dim, n_instruments=n_inst)
    state = torch.randn(N, state_dim)
    out = actor(state)
    assert out.shape == (N, n_inst), f"Expected ({N}, {n_inst}), got {out.shape}"


def test_actor_has_y_parameter():
    """Actor exposes an OCE auxiliary parameter y."""
    actor = BellmanActor(state_dim=4)
    assert hasattr(actor, "y"), "Actor missing OCE parameter y."
    assert isinstance(actor.y, torch.nn.Parameter)
    assert actor.y.shape == torch.Size([])


# ------------------------------------------------------------------ #
# BellmanOperator                                                        #
# ------------------------------------------------------------------ #

def test_bellman_operator_discount():
    """BellmanOperator requires beta strictly in (0, 1)."""
    with pytest.raises(AssertionError):
        BellmanOperator(beta=1.0)

    with pytest.raises(AssertionError):
        BellmanOperator(beta=0.0)

    # Valid beta should construct without error
    op = BellmanOperator(beta=0.99)
    assert op.beta == 0.99


def test_bellman_operator_apply_shape():
    """Bellman operator apply() returns target [N]."""
    N, state_dim = 30, 4
    actor = BellmanActor(state_dim=state_dim)
    critic = BellmanCritic(state_dim=state_dim)
    op = BellmanOperator(beta=0.99)

    states = torch.randn(N, state_dim)
    actions = actor(states)
    dZ = torch.randn(N)
    dH = torch.randn(N, 1)
    next_states = torch.randn(N, state_dim)

    target = op.apply(critic, states, actions, dZ, dH, next_states)
    assert target.shape == (N,), f"Expected ({N},), got {target.shape}"
    assert torch.isfinite(target).all()


# ------------------------------------------------------------------ #
# ValueIterator                                                           #
# ------------------------------------------------------------------ #

def test_value_iteration_bounded():
    """V-norm stays bounded after a few iterations with small gains.

    With zero-mean, low-variance gains, the Bellman value V should stay
    close to zero throughout training.
    """
    torch.manual_seed(0)
    N_total = 200  # N_paths * n_steps
    state_dim = 4

    actor = BellmanActor(state_dim=state_dim)
    critic = BellmanCritic(state_dim=state_dim)
    vi = ValueIterator(actor, critic, beta=0.99, risk_aversion=1.0)

    states = torch.randn(N_total, state_dim)
    dZ = torch.randn(N_total) * 0.1
    dH = torch.randn(N_total, 1) * 0.1
    next_states = torch.randn(N_total, state_dim)

    history = vi.run(
        states, dZ, dH, next_states,
        n_iterations=2,
        n_actor_epochs=10,
        n_critic_epochs=10,
    )

    assert len(history) == 2, f"Expected 2 history entries, got {len(history)}"
    for entry in history:
        assert torch.isfinite(torch.tensor(entry["actor_loss"]))
        assert torch.isfinite(torch.tensor(entry["critic_loss"]))

    # V-norm should be bounded (not exploding)
    assert all(v < 50.0 for v in vi.v_values), (
        f"V-norm unexpectedly large: {vi.v_values}"
    )


def test_value_iteration_history_length():
    """run() returns one entry per iteration."""
    torch.manual_seed(1)
    actor = BellmanActor(state_dim=3)
    critic = BellmanCritic(state_dim=3)
    vi = ValueIterator(actor, critic, beta=0.95)

    N = 50
    history = vi.run(
        torch.randn(N, 3),
        torch.randn(N) * 0.05,
        torch.randn(N, 1) * 0.05,
        torch.randn(N, 3),
        n_iterations=3,
        n_actor_epochs=5,
        n_critic_epochs=5,
    )
    assert len(history) == 3
    assert len(vi.v_values) == 3
