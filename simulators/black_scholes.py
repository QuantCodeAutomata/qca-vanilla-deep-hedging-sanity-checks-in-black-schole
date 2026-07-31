"""Black-Scholes path simulator and analytical pricing utilities."""

import torch
from torch.distributions import Normal

_NORMAL = Normal(0.0, 1.0)


def simulate_paths(
    S0: float,
    sigma: float,
    r: float,
    T: float,
    n_steps: int,
    N_paths: int,
    seed: int | None = None,
) -> dict:
    """Simulate Black-Scholes spot paths via Euler-Maruyama (log-Euler exact scheme).

    S_{t+1} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z),  Z ~ N(0,1)

    Args:
        S0: Initial spot price.
        sigma: Constant volatility.
        r: Risk-free rate.
        T: Time to maturity (in years).
        n_steps: Number of time steps.
        N_paths: Number of Monte Carlo paths.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Dictionary with keys:
            'S'       : Spot paths [N_paths, n_steps+1].
            'dt'      : Scalar time step T / n_steps.
            'n_steps' : Number of steps.
    """
    if seed is not None:
        torch.manual_seed(seed)

    dt = T / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * dt ** 0.5

    # Draw all Brownian increments at once: [N_paths, n_steps]
    Z = torch.randn(N_paths, n_steps)
    log_increments = drift + diffusion * Z  # [N_paths, n_steps]

    # Cumulative log-returns from t=0; prepend zero for S0
    log_S0 = torch.log(torch.tensor(S0, dtype=torch.float32))
    cum_log = torch.cat(
        [torch.zeros(N_paths, 1), torch.cumsum(log_increments, dim=1)], dim=1
    )  # [N_paths, n_steps+1]

    S = torch.exp(log_S0 + cum_log)  # [N_paths, n_steps+1]

    return {"S": S, "dt": dt, "n_steps": n_steps}


# ---------------------------------------------------------------------------
# Analytical Black-Scholes pricing
# ---------------------------------------------------------------------------

def _d1_d2(
    S: torch.Tensor,
    K: float | torch.Tensor,
    sigma: float,
    r: float,
    T: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute d1 and d2 for the Black-Scholes formula.

    Handles T == 0 gracefully by clamping sqrt(T) away from zero.
    """
    T_tensor = torch.as_tensor(T, dtype=S.dtype, device=S.device)
    K_tensor = torch.as_tensor(K, dtype=S.dtype, device=S.device)
    sqrt_T = torch.clamp(T_tensor, min=1e-8) ** 0.5
    d1 = (torch.log(S / K_tensor) + (r + 0.5 * sigma ** 2) * T_tensor) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def call_price(
    S: torch.Tensor,
    K: float | torch.Tensor,
    sigma: float,
    r: float,
    T: float | torch.Tensor,
) -> torch.Tensor:
    """Black-Scholes call option price.

    Args:
        S: Spot price tensor (any shape).
        K: Strike price (scalar or tensor broadcastable to S).
        sigma: Constant volatility.
        r: Risk-free rate.
        T: Time to maturity (scalar or tensor broadcastable to S).

    Returns:
        Call price tensor with the same shape as S.
    """
    T_tensor = torch.as_tensor(T, dtype=S.dtype, device=S.device)
    K_tensor = torch.as_tensor(K, dtype=S.dtype, device=S.device)
    d1, d2 = _d1_d2(S, K, sigma, r, T)
    discount = torch.exp(-r * T_tensor)
    price = S * _NORMAL.cdf(d1) - K_tensor * discount * _NORMAL.cdf(d2)
    # At expiry (T==0) use intrinsic value
    intrinsic = torch.clamp(S - K_tensor, min=0.0)
    return torch.where(T_tensor <= 0.0, intrinsic, price)


def call_delta(
    S: torch.Tensor,
    K: float | torch.Tensor,
    sigma: float,
    r: float,
    T: float | torch.Tensor,
) -> torch.Tensor:
    """Analytical Black-Scholes call delta: dC/dS = N(d1).

    Args:
        S: Spot price tensor (any shape).
        K: Strike price (scalar or tensor broadcastable to S).
        sigma: Constant volatility.
        r: Risk-free rate.
        T: Time to maturity (scalar or tensor broadcastable to S).

    Returns:
        Delta tensor with the same shape as S.
    """
    T_tensor = torch.as_tensor(T, dtype=S.dtype, device=S.device)
    K_tensor = torch.as_tensor(K, dtype=S.dtype, device=S.device)
    d1, _ = _d1_d2(S, K, sigma, r, T)
    delta = _NORMAL.cdf(d1)
    # At expiry: delta = 1 if S > K, else 0
    intrinsic_delta = (S > K_tensor).to(S.dtype)
    return torch.where(T_tensor <= 0.0, intrinsic_delta, delta)


def call_spread_delta(
    S: torch.Tensor,
    K1: float | torch.Tensor,
    K2: float | torch.Tensor,
    sigma: float,
    r: float,
    T: float | torch.Tensor,
) -> torch.Tensor:
    """Delta of a call spread (long K1-call, short K2-call): delta(K1) - delta(K2).

    Args:
        S: Spot price tensor (any shape).
        K1: Lower strike (call-spread long leg).
        K2: Upper strike (call-spread short leg).
        sigma: Constant volatility.
        r: Risk-free rate.
        T: Time to maturity (scalar or tensor broadcastable to S).

    Returns:
        Call-spread delta tensor with the same shape as S.
    """
    return call_delta(S, K1, sigma, r, T) - call_delta(S, K2, sigma, r, T)


def call_gamma(
    S: torch.Tensor,
    K: float | torch.Tensor,
    sigma: float,
    r: float,
    T: float | torch.Tensor,
) -> torch.Tensor:
    """Analytical Black-Scholes call gamma: d^2C/dS^2 = N'(d1) / (S * sigma * sqrt(T)).

    Returns zero at expiry (T == 0) to avoid division by zero.

    Args:
        S: Spot price tensor (any shape).
        K: Strike price (scalar or tensor broadcastable to S).
        sigma: Constant volatility.
        r: Risk-free rate.
        T: Time to maturity (scalar or tensor broadcastable to S).

    Returns:
        Gamma tensor with the same shape as S.
    """
    T_tensor = torch.as_tensor(T, dtype=S.dtype, device=S.device)
    d1, _ = _d1_d2(S, K, sigma, r, T)
    pdf_d1 = torch.exp(_NORMAL.log_prob(d1))  # N'(d1)
    sqrt_T = torch.clamp(T_tensor, min=1e-8) ** 0.5
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    return torch.where(T_tensor <= 0.0, torch.zeros_like(gamma), gamma)
