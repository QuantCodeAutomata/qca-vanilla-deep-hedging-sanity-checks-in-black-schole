"""Heston stochastic-volatility path simulator."""

import torch


def simulate_paths(
    S0: float,
    V0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    r: float,
    T: float,
    n_steps: int,
    N_paths: int,
    seed: int | None = None,
) -> dict:
    """Simulate Heston model paths via Euler-Maruyama with variance reflection.

    Variance process (absorbed / reflected at zero):
        V_{t+1} = max(V_t + kappa*(theta - V_t)*dt + xi*sqrt(V_t*dt)*Z_v, 0)

    Spot process (log-Euler):
        S_{t+1} = S_t * exp((r - 0.5*V_t)*dt + sqrt(V_t*dt)*(rho*Z_v + sqrt(1-rho^2)*Z_s))

    where Z_v, Z_s ~ N(0,1) independent, yielding instantaneous correlation rho
    between Brownian motions driving S and V.

    Args:
        S0: Initial spot price.
        V0: Initial variance (V0 = sigma0^2).
        kappa: Mean-reversion speed of variance.
        theta: Long-run variance.
        xi: Volatility of variance (vol-of-vol).
        rho: Correlation between spot and variance Brownians.
        r: Risk-free rate.
        T: Time to maturity (in years).
        n_steps: Number of time steps.
        N_paths: Number of Monte Carlo paths.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Dictionary with keys:
            'S'       : Spot paths [N_paths, n_steps+1].
            'V'       : Variance paths [N_paths, n_steps+1].
            'dt'      : Scalar time step T / n_steps.
            'n_steps' : Number of steps.
    """
    if seed is not None:
        torch.manual_seed(seed)

    dt = T / n_steps
    sqrt_1_minus_rho2 = (1.0 - rho ** 2) ** 0.5

    # Pre-allocate path tensors
    S = torch.empty(N_paths, n_steps + 1)
    V = torch.empty(N_paths, n_steps + 1)

    S[:, 0] = S0
    V[:, 0] = V0

    for t in range(n_steps):
        V_t = V[:, t]  # [N_paths]

        # Correlated Brownian increments
        Z_v = torch.randn(N_paths)
        Z_s = torch.randn(N_paths)

        sqrt_Vdt = torch.clamp(V_t, min=0.0).sqrt() * (dt ** 0.5)

        # Variance update with reflection
        V_next = V_t + kappa * (theta - V_t) * dt + xi * sqrt_Vdt * Z_v
        V[:, t + 1] = torch.clamp(V_next, min=0.0)

        # Spot update (log-Euler using V_t from current step)
        W_s = rho * Z_v + sqrt_1_minus_rho2 * Z_s  # correlated Brownian for spot
        log_increment = (r - 0.5 * V_t) * dt + sqrt_Vdt * W_s
        S[:, t + 1] = S[:, t] * torch.exp(log_increment)

    return {"S": S, "V": V, "dt": dt, "n_steps": n_steps}
