# Barrier functions

**Audience:** User

Answer a simple question: **“If I pick a profit target and a stop, how often would a simulated path hit the target first?”**

A **barrier** is just a price line that ends the story when touched — take-profit (bank a win) or stop-loss (cut a loss). mtdata walks many imaginary paths and counts the outcomes. Use this *after* you have a market picture and a sense of typical movement — not as a standalone “buy” signal.

Beginner walkthrough: [SAMPLE-TRADE.md](SAMPLE-TRADE.md) (one pair). Grid search lives in [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md).

**Dense terms:** [Barrier](GLOSSARY.md#barrier) · [Monte Carlo](GLOSSARY.md#monte-carlo-simulation) · [GBM](GLOSSARY.md#gbm-geometric-brownian-motion) · [Edge](GLOSSARY.md#edge) · [Kelly](GLOSSARY.md#kelly-criterion) · [prob_tp_first](GLOSSARY.md#prob_tp_first) · [grid_style](GLOSSARY.md#grid_style)

**Related:** [Glossary](GLOSSARY.md) · [Sample trade](SAMPLE-TRADE.md) · [CLI](CLI.md) · [Uncertainty](forecast/UNCERTAINTY.md) · [Volatility](forecast/VOLATILITY.md)

---

## Quick start

### 1) Probability for one TP/SL pair

Percent barriers are expressed in percent (for example, `take_profit: 0.40` with `unit: pct` means **0.40%**, not 40%). The same KV dialect is accepted by `forecast_barrier_prob --barrier` and `labels_triple_barrier --barrier` (commas or spaces between pairs, or JSON). `ticks` is the broker trade tick/point (0.1 pip on typical 5-digit FX), not conventional FX pips; use `unit=pips` for forex pip distances.

```bash
mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 --direction long --barrier 'kind=tp_sl,unit=pct,take_profit=0.40,stop_loss=0.60' --json
mtdata-cli labels_triple_barrier EURUSD --timeframe H1 --barrier 'kind=tp_sl,unit=pct,take_profit=0.40,stop_loss=0.60' --json
```

JSON is equivalent:

```bash
mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 --direction long --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.40,"stop_loss":0.60}' --json
```

Look for `prob_tp_first`, `prob_sl_first`, `prob_no_hit`, and
`probability_edge` in the output. The optimizer uses `edge` for its scored
objective; the single-pair probability response does not alias that field.

**Defaults**: `--method mc_gbm_bb`, `--horizon 12`, `--direction long`. The
default uses a Brownian-bridge correction so single-barrier touches between
simulated bar closes count. For a TP/SL pair, the bridge samples each barrier
independently; `bridge_joint_first_passage: false` marks the resulting
same-bar ordering as an approximation. Methods that evaluate only simulated
closes report `intra_bar_hit_detection: simulated_bar_close` and emit an
undercount warning. Close-only methods such as `mc_gbm` cannot observe dual
intrabar hits, so a non-default `same_bar_policy` is rejected with guidance to
use `mc_gbm_bb`. The default `sl_first` is recorded as
`same_bar_policy_applied=false` / `same_bar_policy_reason=close_only_path`.
Compact output keeps Wilson intervals, `n_sims`, `history_bars_used`, and a
concise `history_window` so simulation precision is visible. These intervals
measure finite Monte Carlo noise under the fitted path model; they do not
measure uncertainty about real-world market edge.

### 2) Search for “good” TP/SL levels

```bash
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --method hmm_mc --mode pct --grid-style volatility --objective edge --json
```

---

## What these numbers mean

You will see three probabilities on a take-profit / stop-loss query:

| Field | Plain idea |
|-------|------------|
| `prob_tp_first` | Share of paths that touched the **win** line first. |
| `prob_sl_first` | Share of paths that touched the **loss** line first. |
| `prob_no_hit` | Share of paths that hit **neither** before time ran out. |

With the default `same_bar_policy=sl_first` (or `tp_first`), `prob_tp_first + prob_sl_first + prob_no_hit` is about 1. Under `neutral`, same-bar ties sit in `prob_same_bar` / `prob_unresolved` instead of being assigned to TP or SL, so use `prob_tp_first + prob_sl_first + prob_same_bar + prob_no_hit`. A large `prob_no_hit` usually means the lines are far away or the horizon is short. These are simulation counts, not a broker guarantee.

**Deeper detail** (methods, grids, Kelly / EV objectives) starts below.

---

## Core Concepts

### What is a Barrier?

A barrier is a price level that, when touched, triggers an event:
- **Take-Profit (TP)**: Profit-taking level
- **Stop-Loss (SL)**: Risk-limiting level

### The Barrier Problem

Given:
- Current price: `S₀`
- TP level: `B_tp`
- SL level: `B_sl`
- Time horizon: `T` (number of bars)
- Trade direction: long/short

We want to compute:
- `P(TP first)`: Probability TP is hit before SL
- `P(SL first)`: Probability SL is hit before TP
- `P(no hit)`: Probability neither is hit by time T
- `E[time to hit]`: Expected time until resolution

### Which tool models what

Four tools compute barrier outcomes, and they deliberately make different
execution assumptions. Numbers are not interchangeable between them; pick the one
that matches your question.

| Tool | Question it answers | Price path | Barrier touch test |
|------|--------------------|------------|--------------------|
| `forecast_barrier_prob` | What are the odds for one TP/SL pair from here? | Simulated forward paths | Simulated bar closes, or a Brownian bridge for intra-bar hits |
| `forecast_barrier_optimize` | Which TP/SL pair looks best from here? | Simulated forward paths | Same as above, across a candidate grid |
| `labels_triple_barrier` | What would this rule have labeled historically? | Actual history | Raw intrabar high/low by default, or closes only |
| `strategy_validate` | What would trading this rule have returned historically? | Actual history | Raw intrabar high/low, non-overlapping trades |

The two forecast tools measure TP/SL distances from the executable **entry**
quote but score simulated paths against the executable **exit** quote (bid for
longs, ask for shorts), reported as `simulation_reference_price`. A barrier placed
inside the current spread is therefore already resolved before the first
simulated bar; both tools reject that geometry rather than reporting a
probability the path matrix cannot observe. Use `--as-of`, `--start`, or `--end`
for candle-close-anchored research with no spread embedded in the geometry.

The two historical tools anchor on the completed bar close and model no spread in
the barrier geometry, so their hit rates are systematically more optimistic than a
live-quoted forecast at the same distances. `strategy_validate` additionally
enforces one position at a time, which makes its trade count far lower than the
label count from `labels_triple_barrier` over the same history.

Both `labels_triple_barrier` and the forecast tools expose `same_bar_policy` for
bars that touch both barriers, but only when intrabar extremes are in play.
Close-only labeling cannot produce a both-touched bar, so the label output reports
`labeling_spec.same_bar_policy_applied: false` in that mode.

---

## Available Methods & Algorithms

### 1. Monte Carlo GBM (`mc_gbm`)

**Algorithm**: Geometric Brownian Motion simulation

**Mathematical Model**:
```
Δ log S = mean(historical log-return) + σ Z
S_{t+1} = S_t * exp(Δ log S)
```
where:
- the drift is the **mean log-return** (no extra −½σ² Itô term; that term is already inside the historical mean)
- `σ` = sample standard deviation of log-returns
- `Z` = standard normal draw (antithetic pairs by default in `forecast_barrier_prob`)

**How it works**:
1. Calibrate μ and σ from historical log-returns
2. Generate N random paths using the formula above (with antithetic variance reduction by default for more stable estimates)
3. Count paths that hit TP first vs SL first

Antithetic paths come in mirrored pairs, so they are not `n_sims` independent
draws and a plain `p(1-p)/n` interval would be wrong in both directions: too
narrow for `prob_no_hit`, too wide for the first-hit probabilities. Each reported
Wilson interval is therefore sized by that outcome's own design effect, and the
resulting effective counts are echoed under
`probability_effective_sample_size` with `probability_ci_basis:
"wilson_design_effect_adjusted"`. `forecast_barrier_optimize` draws independent
paths and reports `wilson_independent_paths`.

**Strengths**:
- Fast and stable
- Minimal assumptions
- Works with limited history

**Weaknesses**:
- Assumes constant volatility
- Ignores fat tails
- No regime awareness

**When to use**:
- Short horizons (< 20 bars)
- Stable market conditions
- Quick screening of many TP/SL combinations
- Baseline for comparison

**Real-life example**:
> **Scenario**: EURUSD scalp on 5-minute chart, horizon = 12 bars (1 hour)
>
> **Why GBM**: Short timeframe, relatively stable intraday volatility, need fast results for live trading
>
> **Command**:
> ```bash
> mtdata-cli forecast_barrier_prob EURUSD --timeframe M5 --horizon 12 --method mc_gbm --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.2,"stop_loss":0.15}'
> ```

---

### 2. Brownian Bridge GBM (`mc_gbm_bb`)

**Algorithm**: GBM with Brownian Bridge correction

**Mathematical Model**:
The Brownian Bridge is a Brownian motion conditioned to return to a specific
value at time T. For a single barrier, it improves short-horizon touch
estimates by accounting for the path between endpoints. TP and SL bridge hits
are sampled independently, so it does not provide an exact joint two-barrier
first-passage ordering.

**Key insight**: If we know the log-price at the ends of a bar, the probability of hitting a log-barrier `b = ln B` between those closes is:
```
P(hit | X_t, X_{t+1}) = exp(-2 * (b - X_t)(b - X_{t+1}) / σ²)
```
where `X = ln S` and `σ` is the per-bar log-return standard deviation (Δt = 1 bar).

**How it works**:
1. Generate GBM paths as normal
2. Apply bridge correction between each time step
3. Detect hits that occur "between" discrete simulation points

**Strengths**:
- Improves single-barrier touch detection for short horizons
- Captures single-barrier intra-bar touches

**Weaknesses**:
- Slightly slower than plain GBM
- Requires more history for stable σ
- Tight TP/SL can overstate same-bar ties because their bridge hits are independent

**When to use**:
- Very short horizons (< 10 bars)
- Tight TP/SL (near current price)
- Scalping strategies
- When precision matters more than speed

**Real-life example**:
> **Scenario**: Crypto scalping on 1-minute BTC chart, TP = 0.1%, SL = 0.08%, horizon = 6 bars
>
> **Why Bridge**: Tiny barriers, very short horizon - need to capture hits that occur between 1-minute marks
>
> **Command**:
> ```bash
> mtdata-cli forecast_barrier_prob BTCUSD --timeframe M1 --horizon 6 --method mc_gbm_bb --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.1,"stop_loss":0.08}'
> ```

---

### 3. HMM Monte Carlo (`hmm_mc`)

**Algorithm**: Hidden Markov Model with regime-switching

**Mathematical Model**:
```
State s_t ∈ {1, 2, ..., K} (hidden)
Return r_t ~ N(μ_{s_t}, σ_{s_t})
Transition: P(s_t = j | s_{t-1} = i) = A_{ij}
```

**How it works**:
1. Fit a Gaussian HMM to log-returns (Baum-Welch)
2. Use its transition matrix and filtered final state distribution
3. Simulate state paths via Markov chain
4. Sample returns conditional on current state
5. Build price paths

If fitting cannot support the requested number of states, barrier probability
and optimization outputs expose `sim_meta.requested_n_states` and
`sim_meta.fitted_n_states`. Optimization also emits an HMM state-collapse
warning and reports `effective_method: single_regime_gaussian_mc` when every
simulation batch fits only one state.

**Strengths**:
- Captures volatility regimes (low/high vol)
- Adapts to changing market conditions
- More realistic than constant-volatility models
- Handles trending vs ranging markets

**Weaknesses**:
- Requires more history (200+ bars)
- Computationally intensive
- Needs careful initialization

**When to use**:
- Medium to long horizons (10-100 bars)
- Clearly regime-switching markets
- Swing/position trading
- When volatility clustering is evident

**Real-life example**:
> **Scenario**: GBPUSD swing trade on 4H chart, horizon = 48 bars (8 days)
>
> **Why HMM**: Markets show clear regimes - trending days vs consolidation periods
>
> **Command**:
> ```bash
> mtdata-cli forecast_barrier_prob GBPUSD --timeframe H4 --horizon 48 --method hmm_mc --barrier '{"kind":"tp_sl","unit":"pct","take_profit":1.5,"stop_loss":1.0}' --params "n_states=2 n_sims=5000"
> ```

---

### 4. GARCH Monte Carlo (`garch`)

**Algorithm**: Generalized Autoregressive Conditional Heteroskedasticity

**Mathematical Model**:
```
r_t = μ + ε_t
ε_t = σ_t * z_t, z_t ~ N(0,1)
σ_t² = ω + α * ε_{t-1}² + β * σ_{t-1}²
```

**How it works**:
1. Fit GARCH(1,1) to scaled returns
2. Forecast volatility path conditional on history
3. Simulate returns using time-varying σ_t
4. Build price paths

**Strengths**:
- Explicit volatility modeling
- Captures volatility clustering
- Widely studied and validated
- Good for risk forecasting

**Weaknesses**:
- Requires `arch` package
- Needs substantial history (100+ bars)
- Assumes normal innovations
- Sensitive to parameter initialization

**When to use**:
- Volatility clustering is strong (high autocorrelation in squared returns)
- Risk management focus
- Options pricing or volatility trading
- When you need volatility forecasts, not just prices

**Real-life example**:
> **Scenario**: SPY options hedge on daily chart, horizon = 20 days
>
> **Why GARCH**: Volatility clustering is well-documented in equity indices; need accurate vol for options
>
> **Command**:
> ```bash
> mtdata-cli forecast_barrier_prob SPY --timeframe D1 --horizon 20 --method garch --barrier '{"kind":"tp_sl","unit":"pct","take_profit":3.0,"stop_loss":2.0}' --params "p=1 q=1"
> ```

---

### 5. Bootstrap (`bootstrap`)

**Algorithm**: Circular block bootstrap

**How it works**:
1. Compute historical returns
2. Sample contiguous blocks of returns (size = `block_size`)
3. Concatenate blocks to create synthetic paths
4. No parametric assumptions

**Strengths**:
- Non-parametric (no distribution assumptions)
- Preserves autocorrelation structure
- Captures fat tails and skew
- Works with any return distribution

**Weaknesses**:
- Limited to historical patterns
- Cannot extrapolate beyond observed volatility
- Requires long history for good coverage

**When to use**:
- Non-normal returns with complex patterns
- When you suspect model misspecification
- Validation of parametric models
- Markets with structural breaks

**Real-life example**:
> **Scenario**: Emerging market FX (e.g., USDTRY), horizon = 24 bars
>
> **Why Bootstrap**: Returns have fat tails, skew, and jumps that parametric models miss
>
> **Command**:
> ```bash
> mtdata-cli forecast_barrier_prob USDTRY --timeframe H1 --horizon 24 --method bootstrap --barrier '{"kind":"tp_sl","unit":"pct","take_profit":2.0,"stop_loss":1.5}' --params "block_size=5"
> ```

---

### 6. Heston Model (`heston`)

**Algorithm**: Stochastic volatility (Heston 1993)

**Mathematical Model**:
```
dS_t = μS_t dt + √v_t S_t dW_t^1
dv_t = κ(θ - v_t) dt + ξ√v_t dW_t^2
dW_t^1 dW_t^2 = ρ dt
```

**Parameters**:
- `κ`: mean reversion speed
- `θ`: long-term variance
- `ξ`: volatility of volatility
- `ρ`: correlation between price and vol
- `v₀`: initial variance

**Strengths**:
- Captures leverage effects (ρ)
- Stochastic volatility
- Closed-form option pricing available
- More realistic than GBM

**Weaknesses**:
- Complex parameter estimation
- Requires many assumptions
- Computationally expensive
- Hard to calibrate

**When to use**:
- Equity options trading
- When leverage effect is important
- Long-dated derivatives
- Academic/research contexts

**Real-life example**:
> **Scenario**: AAPL options strategy, horizon = 60 days
>
> **Why Heston**: Need stochastic vol + leverage effect for accurate option pricing
>
> **Command**:
> ```bash
> mtdata-cli forecast_barrier_prob AAPL --timeframe D1 --horizon 60 --method heston --barrier '{"kind":"tp_sl","unit":"pct","take_profit":10.0,"stop_loss":5.0}' --params "kappa=2.0 theta=0.04 xi=0.3 rho=-0.5"
>
> `kappa` / `theta` / `xi` / `v0` overrides use the **annual** QuantLib/literature
> convention (θ=0.04 ⇒ 20% annual vol) and are converted to the per-bar clock.
> Omit them to use per-bar auto-calibration. The payload `sim_meta.param_time_unit`
> reports which family was used.
> ```

---

### 7. Jump Diffusion (`jump_diffusion`)

**Algorithm**: Merton jump-diffusion

**Mathematical Model**:
```
dS_t = (μ - λk)S_t dt + σS_t dW_t + S_t dJ_t
J_t: compound Poisson process with log-normal jumps
```

**Parameters**:
- `jump_lambda`: jump intensity (frequency)
- `jump_mu`: mean jump size
- `jump_sigma`: jump size volatility
- `jump_threshold`: outlier threshold for calibration

**Strengths**:
- Captures sudden price jumps
- Models tail events
- Better for crisis periods
- Accounts for news events

**Weaknesses**:
- Difficult to calibrate reliably
- Many parameters
- Overfits easily
- Requires long history

**When to use**:
- Earnings announcements
- Major news events
- Cryptocurrency markets
- Crisis periods

**Real-life example**:
> **Scenario**: Trading around FOMC announcement, horizon = 4 hours
>
> **Why Jump Diffusion**: Expecting discrete price jumps from news, not continuous movement
>
> **Command**:
> ```bash
> mtdata-cli forecast_barrier_prob EURUSD --timeframe M15 --horizon 16 --method jump_diffusion --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.5,"stop_loss":0.3}' --params "jump_lambda=0.5 jump_mu=0.001 jump_sigma=0.002"
> ```

---

### 8. Automatic Selection (`auto`)

**Algorithm**: Heuristic method selector based on data characteristics

**Decision Tree**:
```
Note: `garch` requires the `arch` package; auto falls back to `heston` if it is not available.
1. Insufficient history (< 10 bars or < 30 returns)?
   → mc_gbm_bb if horizon ≤ 12, else mc_gbm

2. Crypto or short timeframe (≤ 15 min)?
   → If heavy tails (kurtosis > 2, jump_ratio > 4): jump_diffusion
   → Otherwise continue

3. Heavy tails (kurtosis > 3.5, jump_ratio > 5)?
   → jump_diffusion

4. Volatility regime change (vol_ratio > 1.6 or < 0.65)?
   → hmm_mc (if enough history)

5. Strong volatility clustering (vol_corr > 0.3, r2 size ≥ 150)?
   → garch (if available) or heston

6. Moderate volatility clustering (vol_corr > 0.15)?
   → heston

7. Non-normal but no clear patterns (skew > 0.7, not jumpy)?
   → bootstrap

8. Short horizon and mild tails?
   → mc_gbm_bb

9. Default:
   → mc_gbm
```

**Strengths**:
- No need to manually select method
- Adapts to data characteristics
- Prevents obvious mistakes

**Weaknesses**:
- Black box (but reports reason)
- May not be optimal for all cases
- Still needs sufficient history

**When to use**:
- Initial exploration
- Production automation
- When unsure about market characteristics
- Multi-asset screening

**Real-life example**:
> **Scenario**: Screening 20 currency pairs for trading opportunities
>
> **Why Auto**: Don't want to manually analyze each pair's characteristics
>
> **Command**:
> ```bash
> mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 --method auto --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.5,"stop_loss":0.3}'
>
> # Returns: method_used: "hmm_mc", auto_reason: "auto: regime shift (volatility change)"
> ```

---

## Optimization: Finding Optimal TP/SL

### Grid Search (`forecast_barrier_optimize`)

Searches combinations of TP/SL to maximize an objective function.

With `method=ensemble`, every member simulator evaluates the same TP/SL grid.
The optimizer aggregates each candidate's metrics by `ensemble_agg` and optional
`ensemble_weights`, then selects the best aggregate candidate. Member metrics
for that exact TP/SL pair are returned under `best.member_metrics`; the ensemble
does not combine statistics from different member-specific optima.

**Search Profiles**: Control the compute budget with `--search-profile`:

| Profile | n_sims | n_trials | Grid density | Refine |
|---------|--------|----------|-------------|--------|
| `fast` | 1,200 | 24 | 4 × 4 | off |
| `medium` (default) | 4,000 | 63 | 7 × 9 | off |
| `long` | 10,000 | 600 | 41 × 51 | on |

Use `--search-profile fast` for the fast profile. Explicit `params` values such
as `n_sims` and `tp_steps` override the profile values. Parameter names are
strict and method-aware: unknown keys fail before history is fetched. The
documented `sims` alias remains accepted. Pass command options such as
`return_grid` at the top level (`--return-grid false`), not inside `params`.

**Grid Styles**:

#### 1. Fixed Grid (`grid_style=fixed`)
```
TP: [tp_min, tp_max] with tp_steps points
SL: [sl_min, sl_max] with sl_steps points
Total combinations: tp_steps × sl_steps
```

**Use case**: General purpose, known reasonable ranges

**Example**:
```bash
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --method hmm_mc --mode pct --params "tp_min=0.25 tp_max=1.5 tp_steps=7 sl_min=0.25 sl_max=2.5 sl_steps=9"
```

---

#### 2. Volatility Grid (`grid_style=volatility`)
Scales TP/SL based on recent volatility

**Algorithm**:
```
vol_per_bar = std(returns[-vol_window:])
vol_horizon = vol_per_bar * sqrt(horizon)
vol_pct = vol_horizon * 100

tp_start = max(vol_floor_pct, vol_pct * vol_min_mult)
tp_end = max(tp_start * 1.1, vol_pct * vol_max_mult)
sl_start = max(vol_floor_pct, vol_pct * vol_min_mult * 0.8)
sl_end = sl_start * vol_sl_multiplier
```

**Use case**: Adapts to current market volatility

Use `vol_sl_multiplier` to control the volatility-grid stop-loss multiplier.

**Example**:
```bash
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --method hmm_mc --mode pct --grid-style volatility --params "vol_window=250 vol_min_mult=0.5 vol_max_mult=4.0 vol_sl_multiplier=1.8"
```

---

#### 3. Ratio Grid (`grid_style=ratio`)
Fixes risk/reward ratio

**Algorithm**:
```
For each SL in [sl_min, sl_max]:
  For each ratio in [ratio_min, ratio_max]:
    TP = SL * ratio
```

**Use case**: When you want specific risk/reward profiles

**Example**:
```bash
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --method hmm_mc --mode pct --grid-style ratio --params "ratio_min=1.5 ratio_max=3.0 ratio_steps=5 sl_min=0.3 sl_max=1.0 sl_steps=5"
```

---

#### 4. Preset Grid (`grid_style=preset`)
Pre-configured ranges for trading styles

**Presets**:
- `scalp`: TP 0.08-0.60%, SL 0.20-1.20%
- `intraday`: TP 0.25-1.50%, SL 0.25-2.50%
- `swing`: TP 0.60-3.50%, SL 0.50-4.50%
- `position`: TP 1.00-8.00%, SL 0.75-6.00%

**Use case**: Quick setup for standard trading styles

Note:
- `--preset` implies `--grid-style preset` when `--grid-style` is omitted. Explicit `--grid-style fixed|volatility|ratio` still rejects `--preset`.
- Presets are stored in percentage terms.
- In `mode=ticks`, the optimizer converts those preset percentages to tick-size distances using the current reference price.
- That means named presets in tick mode are not portable across different price levels.

**Example**:
```bash
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --method hmm_mc --mode pct --grid-style preset --preset swing
```

---

### Objectives

Choose what to optimize. Each objective answers a different trading question:

#### Quick Reference Table

| Objective | Formula | Best For |
|-----------|---------|----------|
| `edge` | `P(win) - P(loss)` | General purpose, consistent win rate |
| `prob_tp_first` | `P(win)` | Maximize win rate only |
| `prob_resolve` | `1 - P(no hit)` | Ensure trades complete |
| `kelly` | `P(tp_first) - P(sl_first)/net_RR` | Unconditional Kelly edge proxy |
| `kelly_cond` | Kelly on resolved trades only | Position sizing (ignoring timeouts) |
| `ev` | `P(tp_first)*net_TP - P(sl_first)*net_SL + P(no_hit)*timeout_MTM` | Maximize full-path profit per trade |
| `ev_cond` | EV on resolved trades only | Profit per trade (ignoring timeouts) |
| `ev_per_bar` | `EV / mean_time_in_trade_all_paths` | Fast trades, capital turnover |
| `profit_factor` | Sum of winning realized payoffs / sum of losing realized payoffs, over resolved paths only | Risk/reward ratio focus |
| `min_loss_prob` | Minimize `P(loss)` | Capital preservation |
| `utility` | `mean(log1p(path_payoff * unit_to_return))`; `unit_to_return` is `0.01` for pct, `pip_size / entry` for pips, and `tick_size / entry` for ticks | Risk-averse trading |

#### Detailed Descriptions

**`edge`** — *"How often do I win vs lose?"*
- Measures raw probability advantage
- Edge = 0.20 means you win 20% more often than you lose
- **Use when:** You want consistent winners, regardless of payoff size
- **Limitation:** Ignores how much you win/lose

**`ev` (Expected Value)** — *"What's my average profit per trade?"*
- Accounts for both probability AND payoff size
- Includes the horizon mark-to-market payoff for paths that hit neither barrier
- Same-bar ties contribute zero gross payoff when `same_bar_policy=neutral`; configured round-trip costs still apply
- When spread, slippage, or commission inputs are supplied, EV is net of those costs
- EV = 0.15% means you expect to gain 0.15% per trade on average
- **Use when:** Payoff asymmetry matters (e.g., small wins, big losses)
- **Limitation:** Favorable timeout drift can dominate the barrier-hit edge; inspect
  `ev_unresolved`, `prob_no_hit`, and `phantom_profit_risk`

**`ev_per_bar`** — *"What's my profit per unit of time?"*
- Normalizes EV by how long trades take
- Uses unconditional time-in-trade, so unresolved paths count at full horizon
- Favors fast trades over slow ones with same total EV
- **Use when:** Capital turnover matters (reinvesting profits)
- **Limitation:** May favor trades with higher transaction costs

**`kelly`** — *"How strong is the barrier edge for sizing?"*
- An unconditional two-barrier Kelly edge proxy; timeout probability shrinks the value
- Uses net reward/risk when trading costs are supplied
- Do not use the raw value directly as a lot or capital fraction
- `kelly_cond` is the standard two-outcome Kelly fraction after conditioning on
  TP/SL resolution; use fractional Kelly and an independent risk cap for sizing
- **Use when:** Comparing barrier candidates with the same timeout semantics
- **Limitation:** Neither variant models horizon mark-to-market outcomes; default
  `ev` does, so Kelly and EV use different timeout payoff assumptions

**`prob_resolve`** — *"Will my trade actually close?"*
- Probability that TP or SL is hit within the horizon
- High value means trades complete; low value means many expire
- **Use when:** You need trades to close (e.g., day trading)
- **Limitation:** High resolve doesn't mean profitable

**`profit_factor`** — *"What's my gains-to-losses ratio?"*
- Profit factor > 1 means profitable; > 2 is very good
- Summed from each simulated path's realized payoff rather than from
  `P(win) * net_TP / P(loss) * net_SL`, so it reflects gap-aware stop fills and
  the `same_bar_policy` you chose instead of assuming every stop fills exactly at
  the barrier
- Uses net wins/net losses when trading costs are supplied
- Paths that hit neither barrier are excluded from both the numerator and the
  denominator
- Returns `null` with a `profit_factor_note` when a candidate produced no
  simulated losses, since the ratio is undefined
- Because it ignores timeouts, this objective (like `min_loss_prob`) applies a
  default `min_prob_resolve` floor of 0.20 so that a candidate resolving 2% of
  the time cannot win on a lossless handful of paths. Pass
  `--params "min_prob_resolve=0"` to opt out; the value in effect is reported at
  `constraints_applied.min_prob_resolve` alongside
  `constraints_applied.min_prob_resolve_source`.
- **Use when:** Comparing strategies by risk/reward
- **Limitation:** Doesn't account for trade frequency

**`min_loss_prob`** — *"How do I avoid losses?"*
- Minimizes probability of stop-loss being hit
- **Use when:** Capital preservation is priority
- **Limitation:** May result in tiny TP or trades that never resolve

**`utility`** — *"What's my risk-adjusted outcome?"*
- Logarithmic utility penalizes large losses more than it rewards equivalent gains
- Naturally avoids bets that could wipe you out
- **Use when:** Risk-averse trading, avoiding ruin
- **Limitation:** More theoretical than practical

Notes:
- `_cond` variants (e.g., `ev_cond`, `kelly_cond`) calculate metrics only on trades that resolved (hit TP or SL), ignoring timeouts, and remain cost-adjusted when trading costs are supplied.
- `ev_per_bar` uses unconditional time-in-trade (`t_hit_resolve_mean_all`), counting unresolved paths at horizon expiry.
- `t_hit_tp_median` and `t_hit_sl_median` are conditional medians for paths that hit that barrier; `t_hit_tp_median_cond` and `t_hit_sl_median_cond` are aliases.
- `edge_vs_breakeven` compares the win rate conditional on TP/SL resolution with the matching two-outcome break-even rate. Full-path EV is kept separate because it includes timeout mark-to-market.

#### Execution-aware stop gaps

Set `gap_aware_stops=true` in `params` to value an SL at the first simulated
crossing price when that crossing is worse than the requested stop. TP remains
valued at the target, matching a resting limit-order premise. The result adds
`realized_loss_mean` and `gap_aware_stops`. This is particularly useful with
`jump_diffusion`, bootstrap, and other models capable of discrete gap moves.

---

### Two-Stage Refinement

**Problem**: Coarse grid may miss optimal values

**Solution**:
1. Run coarse grid search
2. Zoom around best result
3. Run fine grid in zoomed region

**Parameters**:
- `refine=true`: Enable refinement
- `refine_radius=0.3`: Search within ±30% of best
- `refine_steps=5`: Points per dimension in refinement

**Example**:
```bash
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --method hmm_mc --mode pct --params "tp_min=0.25 tp_max=1.5 tp_steps=5 sl_min=0.25 sl_max=2.5 sl_steps=5 refine=true refine_radius=0.35 refine_steps=7"
```

---

### Constraints

Filter candidates before ranking:

| Constraint | Description | Example |
|------------|-------------|---------|
| `min_prob_win` | Minimum tie-adjusted TP-first probability | `0.5` (50%) |
| `max_prob_no_hit` | Maximum no-hit probability | `0.2` (20%) |
| `min_prob_resolve` | Minimum probability that TP or SL is hit | `0.3` (30%) |
| `min_barrier_pct` / `min_barrier_pips` / `min_barrier_ticks` | Minimum TP and SL distance, in the unit matching `mode` | `2.0` |
| `min_barrier_multiplier` | Minimum TP and SL distance as a multiple of round-trip cost | `2.0` (default) |
| `max_median_time` | Maximum median bars to resolution, counting resolved paths only | `10` |

The minimum barrier distance is `max(min_barrier_<unit>, min_barrier_multiplier ×
round-trip cost)`, where round-trip cost is the modeled total of spread,
slippage, and commission. Pass the `min_barrier_*` key that matches your `mode`:
`min_barrier_pips` with `mode=pct` is rejected rather than ignored, because a
floor in the wrong unit cannot be compared to the barrier grid. The resolved
values are echoed under `constraints_applied`.

`max_median_time` is evaluated against the median over resolved paths only, so a
candidate that resolves rarely but quickly can satisfy it. Pair it with
`min_prob_resolve` or `max_prob_no_hit` when you need trades to actually close.

**Example**:
```bash
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --method hmm_mc --mode pct --params "tp_min=0.5 tp_max=2.0 tp_steps=5 sl_min=0.5 sl_max=2.0 sl_steps=5 min_prob_win=0.55 max_prob_no_hit=0.15 max_median_time=8"
```

---

### Statistical Robustness

`forecast_barrier_optimize` can attach extra diagnostics for Monte Carlo quality. This is useful when you want to check whether the selected TP/SL pair is stable enough to trust for research or discretionary trading review.

Enable it with `statistical_robustness=true` in `params` or with the Python keyword arguments.

**What it currently adds:**
- Minimum simulation guidance based on the requested CI width
- Convergence diagnostics for the Monte Carlo success estimate
- Cross-seed stability checks for the selected barrier pair
- Optional bootstrap uncertainty for `prob_win`, `prob_loss`, `ev`, `edge`, `kelly`, and related metrics
- Optional power analysis
- Optional local sensitivity analysis around the selected `tp` and `sl`
- Drift-stress scenarios (enabled with statistical robustness by default)
- Independent-seed post-selection evaluation and full-search selection stability
- Optional held-out walk-forward validation with `enable_oos_validation=true`

Every optimizer run also reports the selected candidate's EV standard error and
a Bonferroni-adjusted simulation-precision interval over the searched candidate
set. This is a convergence diagnostic only: changing `n_sims` can tighten the
interval, but cannot create evidence of market edge, so it does not control the
trade gate. Economic viability, costs, unresolved-path behavior, and explicit
validation diagnostics remain separate decisions.

**Important:** These checks do not make a trade valid by themselves. They are quality controls on the simulation and on the optimizer output.

#### Selection bias and the trade gate

The metrics on `best` are the maximum over every candidate the search evaluated,
so they are optimistically biased: some of the winning value is Monte Carlo luck
that happened to favor that TP/SL pair. Two diagnostics measure that bias, and
both can now close the trade gate:

- `post_selection_evaluation` re-scores the selected pair on seeds the search
  never saw. Its `optimism` field is the in-sample value minus the independent
  estimate. If the independent EV is negative, `trade_gate_passed` becomes
  `false` with the flag `post_selection_ev_negative`, because the reported
  positive EV came from selection rather than edge.
- `walk_forward_oos` (enable with `enable_oos_validation=true`) re-runs the
  search on truncated history and measures what the chosen pair actually did next.
  Negative `mean_realized_ev` sets the flag `walk_forward_ev_negative` and also
  closes the gate.

`cross_seed_stability` reports two distinct things, which are easy to conflate:

- `metrics` holds the seed-to-seed spread of the **selected pair, held fixed**, so
  it measures estimator noise. `metric_basis` records this.
- `selection_stability` records how often each seed's own search picked the same
  pair. A low `selected_pair_frequency` means the grid has many
  near-indistinguishable candidates, not that the metrics are noisy.

For signed metrics such as `ev`, `edge`, and `kelly`, a coefficient of variation
is meaningless when values straddle zero. Those metrics report `cv: null` and are
listed under `sign_unstable_metrics` with a reason: the sign is not reproducible.
That is a statement about missing edge, not about needing a larger `n_sims`.

#### Python Example

```python
from mtdata.forecast.barriers_optimization import forecast_barrier_optimize

result = forecast_barrier_optimize(
    symbol="EURUSD",
    timeframe="H1",
    horizon=12,
    method="hmm_mc",
    mode="pct",
    grid_style="volatility",
    objective="ev",
    statistical_robustness=True,
    target_ci_width=0.05,
    n_seeds_stability=3,
    enable_bootstrap=True,
    n_bootstrap=500,
    enable_power_analysis=True,
    enable_sensitivity_analysis=True,
    sensitivity_params=["tp", "sl"],
    params={"n_sims": 5000, "seed": 42},
)
```

#### CLI Example

```bash
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --method hmm_mc --mode pct --grid-style volatility --objective ev --params "statistical_robustness=true,target_ci_width=0.05,n_seeds_stability=3,enable_bootstrap=true,n_bootstrap=500,enable_power_analysis=true,n_sims=5000,seed=42" --json
```

Use Python when you want to pass `sensitivity_params` explicitly.

#### Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `statistical_robustness` | `False` | Enables the extra diagnostics block |
| `target_ci_width` | `0.05` | Requested width used for minimum-simulation guidance |
| `n_seeds_stability` | `3` | Number of seed re-runs requested for the selected barrier pair |
| `enable_convergence_check` | `True` | Adds a non-overlapping batch-means precision diagnostic |
| `enable_bootstrap` | `False` | Adds `bootstrap_uncertainty` |
| `n_bootstrap` | `200` | Number of bootstrap resamples |
| `enable_power_analysis` | `False` | Adds `power_analysis` |
| `power_effect_size` | `0.05` | Effect size assumed in the power calculation |
| `enable_sensitivity_analysis` | `False` | Adds local `tp`/`sl` sensitivity checks |
| `sensitivity_params` | `["tp", "sl"]` | Parameters varied in the sensitivity pass |
| `enable_drift_stress` | `True` with robustness | Stress the selected pair over scaled historical drift |
| `drift_stress_multipliers` | `[0, 0.5, 1, 1.5]` | Drift scenarios to evaluate |
| `enable_oos_validation` | `False` | Enable held-out walk-forward selection and realized validation |
| `oos_folds` | `5` | Number of walk-forward folds |
| `oos_n_sims` | `1000` | Maximum simulation paths per fold |
| `oos_holdout_bars` | `250` | Trailing history reserved for folds |

#### Output Shape

When enabled, the result can include:

```json
{
  "min_sims_recommended": 1537,
  "statistical_robustness": {
    "minimum_simulations": {
      "recommended": 1537,
      "used": 5000,
      "target_ci_width": 0.05,
      "confidence": 0.95
    },
    "convergence_diagnostic": {
      "converged": true,
      "current_estimate": 0.55,
      "window_mean": 0.548,
      "window_std": 0.008,
      "max_change": 0.006
    },
    "cross_seed_stability": {
      "stable": true,
      "n_seeds": 3,
      "seeds_attempted": 3,
      "seeds_succeeded": 3,
      "metrics": {
        "prob_win": {
          "mean": 0.55,
          "std": 0.02,
          "cv": 0.036
        }
      }
    },
    "bootstrap_uncertainty": {
      "prob_win": {
        "mean": 0.55,
        "std": 0.018,
        "ci_low": 0.51,
        "ci_high": 0.59
      }
    },
    "power_analysis": {
      "power": 0.92,
      "min_n_for_80_power": 3200
    },
    "sensitivity_analysis": {
      "tp": {
        "success": true,
        "values_tested": 5
      }
    }
  }
}
```

Only the sections you enable are returned. `cross_seed_stability` may also report an error if fewer than two seed re-runs succeed.

#### Practical Use

- For a quick research pass, start with `target_ci_width=0.10`, `n_seeds_stability=2`, and leave bootstrap off.
- For a fuller review, use `target_ci_width=0.05`, `n_seeds_stability=3`, and turn on bootstrap.
- Treat a wide batch-means CI, unstable cross-seed candidate selection, or poor held-out EV as a reason to increase `n_sims`, simplify the setup, or reject it.
- Sensitivity analysis is most useful after you already have a candidate worth inspecting; it is not a replacement for the main grid search.

Walk-forward validation supports fixed, ratio, and preset grids. Volatility
grids are rejected in this mode so holdout volatility cannot leak into candidate
generation. Each fold calibrates only on bars available before that fold,
selects from the original request grid, and scores the following realized close
path. This remains a close-only research validation; it is not an execution
backtest with bid/ask or intrabar OHLC ordering.

---

## Real-Life Trading Scenarios

### Scenario 1: Scalping EURUSD

**Context**: 5-minute chart, quick trades, tight spreads

**Requirements**:
- Fast execution
- High win rate
- Short holding time

**Method Selection**:
```bash
# Use auto or mc_gbm_bb for precision
mtdata-cli forecast_barrier_optimize EURUSD --timeframe M5 --horizon 6 --method mc_gbm_bb --mode pct --grid-style preset --preset scalp --objective kelly_cond --params "refine=true refine_radius=0.25"
```

**Interpretation**:
- Look for `kelly_cond > 0` (positive edge after costs)
- `prob_resolve > 0.7` (most trades close out)
- `t_hit_resolve_median < 4` (fast resolution)

---

### Scenario 2: Swing Trading Gold

**Context**: 4H chart, multi-day holds, volatility spikes

**Requirements**:
- Handle regime changes
- Account for volatility clustering
- Wider stops

**Method Selection**:
```bash
# HMM captures regimes, volatility grid adapts to current vol
mtdata-cli forecast_barrier_optimize XAUUSD --timeframe H4 --horizon 60 --method hmm_mc --mode pct --grid-style volatility --params "vol_window=300 vol_min_mult=0.8 vol_max_mult=3.0 min_prob_win=0.5" --objective edge
```

**Interpretation**:
- Look for `edge > 0.1` (10% advantage)
- Check `prob_no_hit` (avoid if too high)
- Verify `time_to_tp_seconds.median` aligns with swing timeframe

---

### Scenario 3: Post-Earnings Options

**Context**: AAPL daily chart, jump risk from earnings

**Requirements**:
- Model jumps
- Short horizon (1-5 days)
- High volatility expected

**Method Selection**:
```bash
# Jump diffusion captures earnings moves
mtdata-cli forecast_barrier_optimize AAPL --timeframe D1 --horizon 5 --method jump_diffusion --mode pct --params "jump_lambda=0.3 tp_min=3.0 tp_max=10.0 tp_steps=5 sl_min=2.0 sl_max=6.0 sl_steps=5" --objective ev
```

**Interpretation**:
- Focus on `ev` (expected value)
- Check `prob_same_bar` (should be low)
- Verify jump parameters via `model_summary`

---

### Scenario 4: Range-Bound Market

**Context**: EURUSD in tight range, mean reversion

**Requirements**:
- High probability of no hit
- Small barriers
- Quick resolution if hit

**Method Selection**:
```bash
# Bootstrap preserves range characteristics
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --method bootstrap --mode pct --params "tp_min=0.2 tp_max=0.6 tp_steps=5 sl_min=0.2 sl_max=0.6 sl_steps=5 max_prob_no_hit=0.3" --objective prob_resolve
```

**Interpretation**:
- High `prob_resolve` (trades close out)
- Low `prob_no_hit`
- Balanced `prob_win` and `prob_loss`

---

### Scenario 5: Multi-Asset Screening

**Context**: Finding best setups across 20 pairs

**Requirements**:
- Automated selection
- Consistent comparison
- Fast processing

**Method Selection**:
```bash
# Auto method adapts to each pair
for pair in EURUSD GBPUSD USDJPY AUDUSD NZDUSD USDCAD USDCHF; do
  mtdata-cli forecast_barrier_optimize $pair --timeframe H1 --horizon 12 --method auto --mode pct --grid-style volatility --objective edge --top-k 1
done
```

**Interpretation**:
- Rank pairs by `edge` from `best` output
- Check `auto_reason` to understand method choice
- Filter by `prob_resolve > 0.6`

---

## Output Interpretation

### TP/SL Probability Query

```json
{
  "success": true,
  "method": "hmm_mc",
  "prob_tp_first": 0.62,
  "prob_sl_first": 0.28,
  "prob_no_hit": 0.10,
  "probability_edge": 0.34,
  "time_to_tp_bars": {"mean": 6.2, "median": 5.0},
  "time_to_sl_bars": {"mean": 4.8, "median": 4.0}
}
```

**Interpretation**:
- `prob_tp_first = 0.62`: 62% chance TP hits first
- `prob_sl_first = 0.28`: 28% chance SL hits first
- `prob_no_hit = 0.10`: 10% chance neither hits
- `probability_edge = 0.34`: 34 percentage-point probability bias (TP prob - SL prob)
- **Decision**: A positive value favors TP-first outcomes, but is not expected
  value and does not account for payoff asymmetry or trading costs

---

### Optimization Output

```json
{
  "best": {
    "tp": 0.85,
    "sl": 0.62,
    "rr": 1.37,
    "prob_win": 0.58,
    "prob_loss": 0.32,
    "prob_no_hit": 0.10,
    "prob_resolve": 0.90,
    "edge": 0.26,
    "kelly": 0.34,
    "kelly_cond": 0.47,
    "ev": 0.22,
    "ev_cond": 0.32,
    "ev_per_bar": 0.03,
    "profit_factor": 2.45,
    "utility": 0.18,
    "t_hit_resolve_mean": 6.8,
    "t_hit_resolve_median": 6.0
  },
  "auto_reason": "auto: regime shift (volatility change)"
}
```

**Interpretation**:
- **TP/SL**: 0.85% / 0.62% (RR = 1.37)
- **Win prob**: 58% (conditional on resolution: 58/(58+32) = 64%)
- **Edge**: 26% raw (TP prob − SL prob)
- **Resolve prob**: 90% of paths hit TP/SL within the horizon
- **Kelly**: 34% (raw), 47% conditional on resolution
- **EV**: 0.22% per trade; **EV per bar**: 0.03% based on unconditional time in trade
- **Profit factor**: 2.45 (wins/losses ratio)
- **Utility**: 0.18 (risk‑averse log utility)
- **Resolve time**: mean 6.8 bars, median 6 bars
- **Method used**: HMM (detected regime shift)

**Decision**: This is a good setup - positive edge, reasonable RR, fast resolution.

---

## Common Pitfalls & Solutions

### Pitfall 1: Insufficient History

**Problem**: Method fails or gives unreliable results

**Solution**:
```bash
# Check history length first
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 500

# If insufficient, use mc_gbm (or mc_gbm_bb for short horizons)
mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 10 --method mc_gbm --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.2,"stop_loss":0.15}'
```

---

### Pitfall 2: Overfitting to Recent Volatility

**Problem**: Optimizer picks TP/SL that won't generalize

**Solution**:
```bash
# Use longer vol_window for stable estimates
--params "vol_window=500"

# Or bootstrap (non-parametric)
--method bootstrap

# Validate with rolling backtest
```

---

### Pitfall 3: Ignoring No-Hit Probability

**Problem**: High edge but many trades never close

**Solution**:
```bash
# Add constraint
--params "max_prob_no_hit=0.2"

# Or optimize for resolve probability
--objective prob_resolve
```

---

### Pitfall 4: Wrong Method for Market Regime

**Problem**: Using GBM in highly regime-switching market

**Solution**:
```bash
# Use auto to select appropriate method
--method auto

# Or manually check for regimes
mtdata-cli regime_detect EURUSD --timeframe H1 --method hmm --params "n_states=3"
```

---

### Pitfall 5: Not Accounting for Slippage/Spread

**Problem**: Paper trading results don't match reality

**Solution**:
- On a live request, the optimizer automatically applies a valid current bid/ask spread when no spread parameter is supplied.
- Supply commission and slippage assumptions explicitly, including `0` when that is intentional. Historical requests also require an explicit spread assumption.
- Top-level `--commission-bps-per-side` and `--slippage-bps` values are per fill side and are normalized to a two-sided round trip before optimization. Low-level `params` cost fields are already normalized totals.
- The `trading_costs.complete` and `missing_assumptions` fields describe the effective model. An incomplete model can support research output, but cannot pass the trade gate.
- `spread_pct`, `spread_bps`, and `spread_pips` are equivalent representations
  of the effective normalized spread; non-FX instruments report pips as null.
  The same rule applies to commission and slippage fields. Original user values
  remain under `trading_costs.explicit_inputs`, while live bid/ask fallback is
  identified by `spread_source=live_bid_ask`.
- When live entry and exit quote sides differ, the spread is already embedded
  by re-anchoring simulated paths to the executable exit quote. It remains in
  `cost_per_trade` for reporting but is excluded from `payoff_deduction`, with
  `spread_embedded_in_path_geometry=true`, so it is not charged twice.
- `tp_ticks` / `sl_ticks` are `trade_tick_size` distances only (broker tick/point, not FX pips). Use `unit=pips` for conventional forex pips. They do **not** subtract spread.

```bash
# Tick distances (no automatic spread adjustment)
mtdata-cli forecast_barrier_prob EURUSD --timeframe M5 --horizon 12 --method hmm_mc --barrier '{"kind":"tp_sl","unit":"ticks","take_profit":20,"stop_loss":15}'
# Complete explicit cost model (pip/bps inputs on forecast_barrier_optimize)
mtdata-cli forecast_barrier_optimize EURUSD --timeframe M5 --horizon 12 --params "spread_pips=1.2,slippage_pips=0.3,commission_bps=0"
```

---

## Advanced Tips

### 1. Multi-Horizon Analysis

Run multiple horizons to understand time dynamics:

```bash
for H in 6 12 24 48; do
  echo "Horizon $H bars:"
  mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon $H --method hmm_mc --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.5,"stop_loss":0.3}' --json | jq '{horizon: .horizon, probability_edge: .probability_edge, prob_resolve: (.prob_tp_first + .prob_sl_first)}'
done
```

---

### 2. Regime-Conditional Analysis

Detect regimes first, then optimize per regime:

```bash
# Detect current regime
mtdata-cli regime_detect EURUSD --timeframe H1 --method hmm --params "n_states=2"

# If high-vol regime, use wider barriers
# If low-vol regime, use tighter barriers
```

---

### 3. Denoising for Cleaner Signals

```bash
# Apply denoising before simulation
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --method hmm_mc --mode pct --grid-style preset --preset swing --denoise lowpass_fft --denoise-params "cutoff_ratio=0.1,causality=zero_phase"
```

---

### 4. Parameter Sensitivity

Test stability of results:

```bash
# Vary n_sims
for N in 1000 2000 5000 10000; do
  mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 --method hmm_mc --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.5,"stop_loss":0.3}' --params "n_sims=$N" --json | jq '.prob_tp_first'
done
```

---

### 5. Cross-Validation

Compare methods on same data:

```bash
for METHOD in mc_gbm hmm_mc bootstrap; do
  echo "Method: $METHOD"
  mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 --method $METHOD --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.5,"stop_loss":0.3}' --json | jq '{method: .method, probability_edge: .probability_edge, prob_resolve: (.prob_tp_first + .prob_sl_first)}'
done
```

---

## Performance Considerations

### Speed vs Accuracy Trade-off

| Method | Speed | Accuracy | History Needed |
|--------|-------|----------|----------------|
| mc_gbm | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Minimal |
| mc_gbm_bb | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Minimal |
| hmm_mc | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 200+ bars |
| garch | ⭐⭐ | ⭐⭐⭐⭐ | 100+ bars |
| bootstrap | ⭐⭐⭐ | ⭐⭐⭐⭐ | 400+ bars |
| heston | ⭐ | ⭐⭐⭐⭐ | 200+ bars |
| jump_diffusion | ⭐ | ⭐⭐⭐⭐ | 300+ bars |

### Optimization Speed

- **Fixed grid**: `O(S × T × S)` where S=sl_steps, T=tp_steps
- **Volatility grid**: Similar, but adaptive ranges
- **Refinement**: Adds ~2× computation
- **n_sims**: Linear scaling (2× sims = 2× time)

**Tips for speed**:
1. Start with `mc_gbm` for screening
2. Use `detail="compact"` to limit grid output
3. Set `top_k` to limit evaluations
4. Reduce `n_sims` for initial runs (1000), increase for final (5000+)

---

## Summary Decision Tree

```
Start with: --method auto --horizon [your_horizon]

If horizon < 10:
  → Consider mc_gbm_bb for precision

If horizon > 50:
  → Consider hmm_mc or garch for regimes

If crypto/jumpy:
  → Consider jump_diffusion

If non-normal/fat tails:
  → Consider bootstrap

If optimizing:
  → Use grid-style volatility for adaptive ranges
  → Use refine=true for precision
  → Set constraints (min_prob_win, max_prob_no_hit)

Validate:
  → Check prob_no_hit < 0.2
  → Check probability_edge > 0 for probability responses, or best.edge > 0 for optimizer responses
  → Check median time fits your style
  → Run cross-validation across methods
```

---

## Quick Reference

### Most Common Commands

**Quick check**:
```bash
mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 --method auto --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.5,"stop_loss":0.3}'
```

**Find optimal**:
```bash
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --method hmm_mc --mode pct --grid-style volatility --params "refine=true" --objective edge
```

**Closed-form check**:
```bash
mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 --method closed_form --direction long --barrier '{"kind":"single_price","level":1.1000}'
```

For a current request, barrier distance starts at the executable entry side
(ask for a long, bid for a short). Simulated exit-price paths start at the
opposite executable side (bid for a long, ask for a short), so the current
spread is present before a TP or SL can be reached. The response discloses both
anchors. The closed-form single-barrier calculation remains entry-side anchored.
`last_price_close` retains the model-data anchor. Requests with
`--as-of`, `--start`, or `--end` remain candle-close-anchored historical
research and are labeled `analysis_mode=historical_research`.

---

## Further Reading

- [FORECAST.md](FORECAST.md) - General forecasting documentation
- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) - Example trading workflows
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Troubleshooting

---

*Last updated: 2026-01-01*
