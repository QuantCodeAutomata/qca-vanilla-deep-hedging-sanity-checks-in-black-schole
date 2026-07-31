# Vanilla Deep Hedging: Sanity Checks in Black-Scholes and Heston Models

A modular PyTorch implementation of the **Deep Hedging** framework covering five experiments:

| # | Experiment | Key Finding |
|---|-----------|-------------|
| 1 | BS/Heston sanity checks & asymptotic slopes | OCE objective trains correctly; pricing correction vs ε follow power-law |
| 2 | S&P 500 surface simulator with set-invariant encoder | Set-encoder reconstructs option surfaces; simulator-trained hedger ≥ direct |
| 3 | Statistical-arbitrage diagnosis & drift removal (P → Q) | Under P, utility > 0 (arb exists); reweighting to Q suppresses drift |
| 4 | Uncertainty-aware hedging over a family of candidate measures | Entropy special case collapses to mixture-measure deep hedging (verified) |
| 5 | Deep Bellman hedging with value iteration & actor-critic | V^n finite and growing from V^0=0; critic converges in simple settings |

---

## Repository Layout

```
.
├── configs/            YAML configs (BS / Heston benchmarks)
├── simulators/         Black-Scholes and Heston path generators
├── objectives/         OCE loss, weighted OCE, uncertainty outer layer
├── policies/           Neural hedge policy + DeepHedger wrapper
├── measures/           Bootstrap & perturbed candidate-measure construction
├── bellman/            Bellman operator, actor, critic, value iteration
├── evaluation/         Asymptotic slope estimator, delta-surface comparator
├── exp/                Experiment scripts (exp1–exp5)
├── tests/              50 pytest tests (simulators, policies, objectives, bellman, measures)
├── results/            All figures (PNG) and metrics (JSON + RESULTS.md)
├── data/               (placeholder for historical data)
└── requirements.txt
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
python exp/exp1_bs_heston_sanity.py
python exp/exp2_sp500_simulator.py
python exp/exp3_stat_arb_drift_removal.py
python exp/exp4_uncertainty_aware.py
python exp/exp5_bellman_hedging.py

# Run tests
pytest tests/ -v
```

---

## Benchmark Configuration

| Parameter | Black-Scholes | Heston |
|---|---|---|
| σ / initial vol | 0.20 | √0.04 = 0.20 |
| T (years) | 0.0833 (~1 month) | 0.0833 |
| Steps | 21 (daily) | 21 |
| Paths (N) | 50 000 | 50 000 |
| ε grid | 6 log-spaced values | 6 log-spaced values |
| Heston κ | — | 2.0 |
| Heston θ | — | 0.04 |
| Heston ξ | — | 0.30 |
| Heston ρ | — | −0.70 |

Configuration files: `configs/benchmark_bs.yaml`, `configs/benchmark_heston.yaml`

---

## Core Methodology

### OCE / Monetary-Utility Objective
```
U*(Z) = sup_{a, y ∈ ℝ}  (1/N) Σ_ω u(y + Z_T(ω) + Σ_t a_t' ΔH_{t:T}(ω) − c_t(a_t)) − y
```
where `u(x) = (1 − exp(−λx)) / λ` is the exponential utility and `c_t` are convex
transaction costs with `c_t(0) = 0`.

### Measure Reweighting (Exp 3)
```
dQ/dP (ω) ∝ u'(y* + Σ_t a_t*' ΔH_{t:T}(ω))
```
normalised so weights sum to one. Under Q, the empty-portfolio utility ≈ 0.

### Uncertainty-Aware Objective (Exp 4)
```
W(f) = sup_{e ∈ ℝ}  Σ_k μ(Q_k){ w(e + f(Q_k)) − e }
```
For entropy utility `w = u`, this collapses to standard deep hedging under mixture
measure `Q̄ = Σ_k Q_k μ(Q_k)`.

### Bellman Operator (Exp 5)
```
TV(Z^(t), s_t) = sup_a U_t[ β_t V(Z^(t) ⊕ a'H^(t), s_{t+1})
                             + dZ_t^(t) + a' dH_t^(t) − c_t(a) ]
```
Value iteration starts from V^0 ≡ 0.

---

## Dependencies

- Python ≥ 3.12
- PyTorch 2.x (CPU build sufficient)
- NumPy, SciPy, pandas, matplotlib, seaborn
- pfhedge (deep-hedging scaffold)
- PyYAML, scikit-learn

See `requirements.txt` for pinned versions.
