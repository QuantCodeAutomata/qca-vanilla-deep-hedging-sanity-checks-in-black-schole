"""
Tests for OCE/monetary utility objective.
"""
import sys
sys.path.insert(0, '/workspace/project')

import pytest
import torch
from objectives.oce import OCELoss


# ------------------------------------------------------------------ #
# Utility function properties                                           #
# ------------------------------------------------------------------ #

def test_oce_zero_pnl():
    """Utility value is 0 for zero P&L when y=0 (fresh initialisation)."""
    loss = OCELoss(risk_aversion=1.0)
    pnl = torch.zeros(100)
    # u(0+0) = (1 - exp(0)) / 1 = 0;  E[u] - y = 0 - 0 = 0
    uval = loss.utility_value(pnl)
    assert abs(float(uval)) < 1e-5, f"Expected ~0, got {float(uval):.2e}"


def test_oce_utility_range():
    """Exponential utility is bounded above by 1/a."""
    a = 1.0
    loss = OCELoss(risk_aversion=a)
    x = torch.linspace(-20.0, 20.0, 2000)
    u = loss.utility(x)
    assert (u <= 1.0 / a + 1e-6).all(), "Utility exceeded its upper bound 1/a."


def test_utility_monotone():
    """Exponential utility is strictly increasing."""
    loss = OCELoss(risk_aversion=1.0)
    x1 = torch.tensor([-2.0, 0.0, 1.0, 5.0])
    x2 = x1 + 0.5
    assert (loss.utility(x2) > loss.utility(x1)).all(), (
        "Utility should be strictly increasing."
    )


# ------------------------------------------------------------------ #
# Transaction cost                                                       #
# ------------------------------------------------------------------ #

def test_oce_cost_zero():
    """c_t(0) = 0: zero action incurs zero transaction cost."""
    loss = OCELoss(risk_aversion=1.0, cost_epsilon=0.5)
    action = torch.zeros(5, 1)
    cost = loss.cost_fn(action)
    assert (cost == 0.0).all(), f"Expected zero cost for zero action, got {cost}."


def test_oce_cost_proportional():
    """Transaction cost scales linearly with action size and cost_epsilon."""
    eps = 0.02
    loss = OCELoss(risk_aversion=1.0, cost_epsilon=eps)
    # 3 paths, 2 instruments
    action = torch.tensor([[1.0, 2.0], [0.5, 0.0], [0.0, 0.0]])
    cost = loss.cost_fn(action)
    expected = torch.tensor([3.0 * eps, 0.5 * eps, 0.0])
    assert torch.allclose(cost, expected, atol=1e-6), (
        f"Cost mismatch: {cost} vs {expected}"
    )


# ------------------------------------------------------------------ #
# OCE loss / forward                                                    #
# ------------------------------------------------------------------ #

def test_oce_loss_is_negative_utility():
    """forward(pnl) == -utility_value(pnl) at any fixed y."""
    loss = OCELoss(risk_aversion=1.0)
    torch.manual_seed(0)
    pnl = torch.randn(200)
    fwd = loss.forward(pnl).detach()
    uval = loss.utility_value(pnl).detach()
    assert torch.isclose(fwd, -uval, atol=1e-5), (
        f"forward={fwd:.4f}, -utility_value={-uval:.4f}"
    )


def test_oce_forward_is_scalar():
    """forward() returns a scalar (0-dimensional tensor)."""
    loss = OCELoss()
    pnl = torch.randn(50)
    out = loss.forward(pnl)
    assert out.shape == torch.Size([]), f"Expected scalar, got shape {out.shape}"


def test_oce_different_risk_aversion():
    """Higher risk aversion assigns lower utility to the same risky P&L."""
    torch.manual_seed(1)
    pnl = torch.randn(500) * 2.0  # volatile P&L
    loss_low = OCELoss(risk_aversion=0.5)
    loss_high = OCELoss(risk_aversion=5.0)
    u_low = float(loss_low.utility_value(pnl))
    u_high = float(loss_high.utility_value(pnl))
    assert u_low >= u_high, (
        f"Higher risk aversion should give lower utility; "
        f"a=0.5 -> {u_low:.4f}, a=5.0 -> {u_high:.4f}"
    )
