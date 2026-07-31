# Deep Hedging — Project Knowledge

## Repository layout

```
configs/           YAML experiment configs (benchmark_bs.yaml, benchmark_heston.yaml)
simulators/        Monte Carlo path simulators
  black_scholes.py simulate_paths, call_price, call_delta, call_spread_delta,
                   call_gamma, call_spread_delta
  heston.py        simulate_paths (Euler-Maruyama + variance reflection)
objectives/
  oce.py           OCELoss — nn.Module, numerically stable logsumexp formula
  inner_oce.py     WeightedOCELoss — per-measure OCE (logsumexp), for Exp 4
  uncertainty_layer.py  UncertaintyLayer — outer entropy aggregation, for Exp 4
policies/
  hedge_policy.py  HedgePolicy (MLP), DeepHedger (logsumexp forward)
measures/
  __init__.py
  candidate_measures.py  construct_bootstrap_measures, construct_perturbed_measures,
                         uniform_uncertainty_weights
bellman/
  __init__.py
  operator.py      BellmanOperator — one-step TV computation
  actor.py         BellmanActor  — MLP policy with OCE y parameter
  critic.py        BellmanCritic — scalar value network V^n
  value_iteration.py ValueIterator — alternates actor/critic steps; logsumexp actor loss
evaluation/
  asymptotic.py    estimate_asymptotic_slope, bootstrap_slope_uncertainty
  delta_surface.py compute_learned_hedge_surface, plot_delta_comparison
src/__init__.py    Convenience re-export of all public symbols
requirements.txt   Python dependencies
tests/             50 pytest tests (all passing)
  test_simulators.py, test_objectives.py, test_policies.py,
  test_measures.py, test_bellman.py
```

## Key design decisions

### Simulators
- `simulate_paths` returns a `dict` (keys: `S`, `dt`, `n_steps`; Heston adds `V`).
- Shapes: `S` is `[N_paths, n_steps+1]`; `V` (Heston) same shape.
- Both use Euler-Maruyama; BS uses the exact log-Euler form.
- Heston variance is reflected at 0 via `torch.clamp(V_next, min=0)`.
- All tensors are CPU float32 by default; pass `seed` for reproducibility.

### Paths dict used by DeepHedger
- Must contain `'S'` (`[N, T+1]`) and `'dH'` (`[N, T]`).
- `dH[:, t]` = gain per unit of hedge held over step t (typically `S[:, t+1] - S[:, t]`).

### OCELoss / DeepHedger / WeightedOCELoss — numerical stability
- **All three** compute the entropic risk measure via the log-sum-exp identity:
  `U*(Z) = -(1/a) * log(E[exp(-a*Z)]) = -(1/a) * (logsumexp(-a*Z) - log N)`
  This avoids `exp` overflow which occurs in float32 when pnl_min < -88 (e.g., S₀=100
  gives pnl ≈ -105 for worst paths → `exp(105)` = inf in float32).
- `OCELoss.y` is kept as an `nn.Parameter` (for `marginal_price` API compatibility) but
  is no longer used in `forward`/`utility_value`; only `hedger.parameters()` need be
  passed to the optimiser (drop `loss_fn.parameters()`).
- `DeepHedger.forward` uses the same logsumexp formula; `self.y` is unused there too.
- `WeightedOCELoss.forward(pnl, q_weights)` uses `logsumexp(-a*pnl + log(q), dim=0)`.
- `marginal_price` uses `torch.no_grad` internally; call *after* training.
- Gradient clipping (`clip_grad_norm_` max_norm=1.0) is applied in all training loops
  as a secondary safeguard against gradient explosions.

### State vector (HedgePolicy / DeepHedger)
- Default input_dim=3: `[ttm, log_spot, prev_hedge]`
- Heston extension: add `sqrt(V_t)` as 4th feature → input_dim=4.

### Evaluation
- `estimate_asymptotic_slope`: log-log OLS; expects `p(ε) ~ C·ε^α`.
- `compute_learned_hedge_surface`: iterates over `(ttm, S)` grid with `prev_hedge=0`.
- `plot_delta_comparison`: sets `matplotlib.use('Agg')` — safe for headless runs.

## Python path
All packages are at the repo root; scripts must have `sys.path.insert(0, '.')` or
be run from `/workspace/project`. No `setup.py` / `pyproject.toml` yet.

## Experiment files (`exp/`)

### exp1_bs_heston_sanity.py
Vanilla deep hedging sanity checks.  Public API:
- `run_deep_hedge_bs(epsilon, N_paths, n_epochs, seed) -> dict`
  Returns `{utility, bs_price, loss_history}`.  `utility` = OCE y* at convergence
  (extra compensation needed for TC friction); `bs_price` = analytical ATM call.
- `run_asymptotic_bs(epsilon_grid, N_paths, n_epochs) -> dict`
  Sweeps TC rates; returns `{epsilon_grid, corrections, slope, slope_std}`.
  `corrections[i]` = `|y*(ε_i)|`.  Theoretical slope ≈ 0.67 (Whalley-Wilmott).
- `run_asymptotic_heston(epsilon_grid, N_paths, n_epochs) -> dict`
  Same structure; theoretical slope ≈ 0.71.
- `run_call_spread_delta_bs(N_paths, n_epochs, seed) -> dict`
  Returns `{learned_surface, analytical_surface, mse, S_grid, ttm_grid}`.
- `main()` → runs all, saves to `results/exp1_*.{png,json}`.
- Module constants: `_BATCH_SIZE=2048`, `_LR=1e-3`, `_GRAD_CLIP=1.0`.

### exp2_sp500_simulator.py
Market simulator with set-invariant surface encoder.  Key classes:
- `SetInvariantEncoder(m_dim, f_dim, hidden)` — Deep Sets: F then mean-pool then E, tanh output.
- `LatentDynamics(h_dim, e_dim)` — OU-like: `h_t = (1-K)h_{t-1} + Σ·ε`.
- `ArbitrageFreeDecoder(h_dim, hidden)` — `softplus(MLP(h,k,τ))+0.01` → IV > 0.
- `MarketSimulator` — combines all three; `fit(data, n_epochs, seq_len)` minimises IV MSE.
- `generate_synthetic_surface_data(T_dates, seed)` — GBM underlying + smile IVs + BS mids.
- `train_hedger_direct / train_hedger_simulator` — hedger utility comparison.
- **Normalisation**: price windows divided by initial price before training; `generate_paths`
  uses `S0=1.0` (not raw 4000-level prices) to avoid exp-utility overflow.

### exp3_stat_arb_drift_removal.py
Stat-arb diagnosis and path reweighting.  Key functions:
- `simulate_paths_with_drift(mu, sigma, …)` — GBM with drift using `bs_simulate(r=mu)`.
- `train_empty_portfolio(paths_P, …)` — trains on `Z_T=0`; positive utility detects drift exploit.
- `compute_Q_weights(gains, y_star, a)` — `q_ω ∝ exp(-a(y*+gain_ω))`, normalised.
- `verify_foc_identities(gains, y_star, a)` — checks `E^P[exp(-a(y*+gains))] ≈ 1`.
- `train_under_Q(paths_P, Q_weights, …)` — full-batch Q-weighted OCE; utility_Q ≈ 0 at convergence.
- `toy_vol_mismatch_example(realized_vol, traded_vol, …)` — shows vol-gap exploitation.
- **Note**: `train_under_Q` always uses the full sample (all N paths) per step because
  Q-weighting via `torch.dot(q, u_vals)` requires the complete weight vector.

### exp4_uncertainty_aware.py
Uncertainty-aware deep hedging under distribution of candidate measures.
Public API:
- `prepare_synthetic_paths(N_paths, n_steps, sigma, seed) -> dict` — GBM risk-neutral; returns `dH`, `Z_T`.
- `train_single_measure(paths, Q_weights, n_epochs, seed) -> dict` — OCE under one Q.
- `train_bootstrapped(paths, Q_bootstrap, mu_weights, n_epochs, seed) -> dict` — linear (W = Σ μ_k U_k).
- `train_uncertainty_aware(paths, Q_bootstrap, mu_weights, n_epochs, seed) -> dict` — entropy outer utility.
- `verify_entropy_special_case(paths, Q_bootstrap, mu_weights, …) -> dict` — checks `utility_mixture ≈ utility_entropy`.
- `main()` — saves to `results/exp4_results.json`.

Key result: for entropy utility W, `train_bootstrapped` and `train_uncertainty_aware` yield **identical** utilities
(entropy utility collapses to mixture measure barQ = Σ μ_k Q_k). Confirmed by `match=True` in smoke test.

### exp5_bellman_hedging.py
Bellman-value deep hedging with actor-critic value iteration.
Public API:
- `prepare_daily_transitions(N_paths, n_steps, sigma, beta, seed) -> dict` — discounted transitions,
  states = `[ttm, log_moneyness, delta, gamma]` (4 features).
- `train_vanilla_hedger(paths, n_epochs, seed) -> dict` — standard OCE hedging.
- `train_bellman_hedger(transitions, n_iterations, seed) -> dict` — `ValueIterator` actor-critic loop.
- `compare_hedgers(vanilla_result, bellman_result) -> dict` — performance comparison.
- `main()` — saves to `results/exp5_results.json`.

### Bellman package (bellman/)
- `BellmanActor(state_dim, n_instruments, n_layers, n_units)` — MLP + unused `y` parameter.
- `BellmanCritic(state_dim, n_layers, n_units)` — scalar value network V^n(state).
- `ValueIterator.actor_step` uses logsumexp (not direct exp) for the actor loss.
- `ValueIterator.critic_step` fits V^n by MSE to Bellman targets (β*V_{n-1} + gain).
- `BellmanOperator.apply` computes one-step target; `V_func` is called `with torch.no_grad()`.

## Tests
`/workspace/project/tests/` — 50 tests, all passing.
- `test_simulators.py` (13 tests): BS/Heston shape, positivity, martingale, call bounds, gamma.
- `test_objectives.py` (7 tests): OCELoss zero-PnL, utility range, monotonicity, scalar output.
- `test_policies.py` (7 tests): HedgePolicy/DeepHedger shape, finiteness, no-lookahead.
- `test_measures.py` (4 tests): bootstrap normalization, nonnegativity, shape, mu sum.
- `test_bellman.py` (5 tests): actor/critic finiteness, shape, iteration boundedness, beta assertion.

Run: `cd /workspace/project && python -m pytest tests/ -v`

## Results
All figures and JSON saved to `/workspace/project/results/`.
- `exp1_results.json`, `exp2_results.json`, `exp3_results.json`
- PNG figures for asymptotic slopes, loss curves, delta surfaces, IV fits, gain distributions.
- `exp4_results.json`, `exp5_results.json` (generated by main()).

## Environment
- Python 3.12, PyTorch 2.13 (pre-installed), CUDA not guaranteed.
- Install deps: `pip install -r requirements.txt`.
- No `setup.py` / `pyproject.toml`; run from `/workspace/project` or use `sys.path.insert`.
