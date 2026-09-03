# Glossary

**Audience:** User

Plain-language definitions for the trading and forecasting terms you will see in mtdata docs and tool output. Each entry aims for **enough to follow the examples**, not a textbook — plus **where it shows up in mtdata**.

**Tip:** Skim [Basic trading concepts](#basic-trading-concepts-start-here) if you are new. Use the [quick find](#quick-find) table when a command or doc drops an acronym (BOCPD, Kelly, ADF, …).

**Related:** [README](../README.md) · [Sample trade](SAMPLE-TRADE.md) · [Forecasting](FORECAST.md) · [Docs index](README.md)

---

## Quick find

Jump straight to dense concepts that show up in tools and docs:

| Concept | Plain idea | Jump |
|---------|------------|------|
| **BOCPD** | “Did the market just change regime?” (change-point prob) | [BOCPD](#change-point-detection-bocpd) |
| **HMM / GMM / MS-AR / PELT** | Label or segment market “modes” | [HMM](#hidden-markov-model-hmm) · [GMM](#gaussian-mixture-gmm) · [MS-AR](#markov-switching-ar-ms-ar) · [PELT](#pelt-change-point-detection) |
| **Kelly** | How large a bet maximizes long-run growth (often use a fraction) | [Kelly](#kelly-criterion) |
| **Edge / EV** | Win-rate advantage vs dollar expectation | [Edge](#edge) · [EV](#ev-expected-value) |
| **VaR / CVaR** | Statistical loss thresholds for open risk | [VaR](#var-value-at-risk) · [CVaR](#cvar-conditional-var--expected-shortfall) |
| **EWMA / GARCH / HAR-RV** | How much price tends to move | [EWMA](#ewma-exponentially-weighted-moving-average) · [GARCH](#garch-generalized-autoregressive-conditional-heteroskedasticity) · [HAR-RV](#har-rv-heterogeneous-autoregressive-realized-volatility) |
| **Conformal intervals** | Bands from past forecast errors (not a normal-curve assumption) | [Conformal](#conformal-intervals) |
| **GBM / Monte Carlo** | Random simulated price paths | [Monte Carlo](#monte-carlo-simulation) · [GBM](#gbm-geometric-brownian-motion) |
| **ADF / KPSS** | Is the series stationary enough to model? | [Stationarity tests](#stationarity-tests-adf-kpss-phillipsperron) |
| **Granger / cointegration** | Predictive lead-lag vs shared long-run levels | [Granger](#granger-causality) · [Cointegration](#cointegration) |
| **POC / VAH / VAL** | Volume-by-price fair value | [Volume profile](#volume-profile-poc-vah-val) |
| **LTTB** | Downsample a series while keeping shape | [LTTB](#lttb-largest-triangle-three-buckets) |
| **Dry-run / guardrails** | Preview orders; cap symbols, size, and risk | [Dry-run](#dry-run) · [Guardrails](#trade-guardrails) |
| **Margin / leverage** | Collateral vs borrowed size | [Margin](#margin-and-leverage) |
| **Pending vs market** | Wait for a price vs trade now | [Market vs pending](#market-vs-pending-orders) |
| **Point vs pip** | Broker tick vs FX convention | [Point vs pip](#point-vs-pip) |
| **TOON / MCP** | Default CLI presentation; agent tool protocol | [TOON](#toon) · [MCP](#mcp-model-context-protocol) |
| **Heston / QuantLib** | Stochastic vol & option pricing | [Heston](#heston-model) · [QuantLib](#quantlib) |

---

## Basic trading concepts (start here)

If you are new to trading, this section is enough to follow the walkthroughs in this repo.

### Long vs Short
- **Long**: you profit if price goes **up** (buy now, sell later).
- **Short**: you profit if price goes **down** (sell now, buy back later).

### TP/SL (Take Profit / Stop Loss)
- **Take Profit (TP)**: close a position when price reaches your profit target.
- **Stop Loss (SL)**: close a position to cap the loss if price moves against you.

### Bid/Ask and Spread
- **Ask**: the price you pay to buy.
- **Bid**: the price you receive when you sell.
- **Spread**: `ask - bid` (a cost you pay to enter/exit). See also: [Spread](#spread).

### Pip (and Pipette)
- A **pip** is a standard unit of FX price movement (often `0.0001`, or `0.01` for JPY pairs). See also: [Pip](#pip).
- Some brokers quote an extra digit (“pipettes”): `1 pip = 10 pipettes`.

### Timeframe and Candles (Bars)
- A **timeframe** (e.g., `M5`, `H1`, `D1`) controls how data is grouped.
- A **candle/bar** summarizes that period (open/high/low/close and volume).

### Lot Size
Forex position sizes are often expressed in **lots** (see: [Lot Size](#lot-size)).

### Margin and leverage
- **Margin** is collateral set aside so you can hold a position larger than your cash.
- **Leverage** is how large that position can be relative to margin (for example 1:30).
- Higher leverage makes both gains and losses faster. Demo first.

See also: [Balance, equity, free margin](#balance-equity-and-free-margin).

### Balance, equity, and free margin
- **Balance**: cash result of *closed* trades.
- **Equity**: balance plus the floating profit or loss on *open* trades.
- **Free margin**: equity minus margin already in use — room for new positions.

Risk tools that say “% of equity” use the equity number, not the balance.

### Market vs pending orders
- **Market** (`BUY` / `SELL`): trade as soon as possible at the current bid or ask.
- **Pending**: wait until price reaches a level you chose.
  - **Limit**: buy cheaper than now, or sell richer than now.
  - **Stop**: buy if price breaks up, or sell if it breaks down.

### Point vs pip
A **point** is the broker’s smallest price increment (`symbols_describe` → `point`).
A **pip** is a *convention* (often 10 points on 5-digit FX quotes). Deviation and
many MT5 distances are in **points**. When a doc says “pips,” it is the human
unit; when a flag says points, it is the broker unit.

### Magic number
An integer stamp you put on orders so you can filter “this strategy’s trades”
later (`--magic`). It is a label, not a luck charm.

### Deviation (slippage tolerance)
On a market order, how many **points** the fill may differ from the quote you
saw. Larger deviation → more likely to fill in a fast market; smaller → more
likely to be rejected.

### GTC vs expiration
- **GTC** (“good till cancelled”): a pending order stays until you cancel it or
  it fills.
- **Expiration**: the pending order dies at a time you set.

### Commission vs spread
The **spread** is the bid/ask gap — a cost you pay on every round trip.
A **commission** is an extra broker fee on top. Both matter for small targets.

---

## Core Concepts

### Time Series
A sequence of data points indexed by time. In trading, this is typically OHLCV data (Open, High, Low, Close, Volume) at regular intervals.

**Example:** 500 hourly candles of EURUSD form a time series.

### Horizon
How many bars into the future a forecast predicts. Unless you pass `--as-of` or a historical range, those bars start at the open of the current forming bar — not the next closed bar.

**Example:** `--horizon 12` with H1 timeframe means "predict the next 12 hours," including the hour that is still forming.

### Lookback
How many historical bars a model uses to make its prediction.

**Example:** `--lookback 500` tells the model to learn from the last 500 candles.

---

## Forecasting Methods

### Theta Method
A simple, robust forecasting technique that decomposes a time series into trend and seasonality components.

**When to use:** As a baseline for short-to-medium horizons. Often surprisingly accurate.

**Example:**
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta
```

### ARIMA (AutoRegressive Integrated Moving Average)
A classical statistical model that predicts future values based on:
- **AR (AutoRegressive):** Past values of the series
- **I (Integrated):** Differencing to make data stationary
- **MA (Moving Average):** Past forecast errors

**When to use:** Short-term forecasting when data shows clear autocorrelation patterns.

**Interpretation:** ARIMA works best on *stationary* data (constant mean/variance). Raw prices aren't stationary, but returns usually are.

### ETS (Error, Trend, Seasonality)
Exponential smoothing that weighs recent observations more heavily than older ones.

**When to use:** Data with clear seasonal patterns (e.g., higher volatility during market opens).

### Monte Carlo Simulation
Generates thousands of possible future price paths by randomly sampling from a model of returns (often GBM or a richer process). Essential for **ranges**, **barrier hit odds**, and risk sizing — not for a single “the” price.

**When to use:** Barrier tools (`forecast_barrier_prob` / `forecast_barrier_optimize`), path simulation methods (`mc_gbm`, `hmm_mc`, …).

**Example:**
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method mc_gbm --params "n_sims=2000"
```

**Interpretation:** Percentile bands and hit rates across paths — not a guaranteed path.

### GBM (Geometric Brownian Motion)
A standard random-walk-with-drift model for prices: continuous paths, log-returns roughly normal, constant vol in the basic form. mtdata’s `mc_gbm` draws many GBM paths; variants add fat tails, jumps, or stochastic vol.

**Limitation:** Real markets jump, cluster volatility, and switch regimes — so treat GBM as a **baseline simulator**, not truth.

**In mtdata:** `mc_gbm` (and related barrier methods). See [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md).

### Barrier simulation methods (quick map)
| Method key | Plain idea |
|------------|------------|
| `mc_gbm` | Classic GBM Monte Carlo |
| `hmm_mc` | Simulate paths that can **switch regimes** (HMM) |
| `garch` | Paths with **clustered** volatility |
| `heston` | **Stochastic volatility** (vol itself wanders) |
| `jump_diffusion` | Continuous moves **plus jumps** |
| `bootstrap` | Resample historical returns |
| `auto` / `ensemble` | Pick or blend methods from diagnostics |

Defaults and when-to-use details: [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md).

### Chronos / foundation models
**Foundation models** (Chronos, Chronos-Bolt, TimesFM, TimesFM 3, …) are pre-trained on huge collections of series, then applied to your symbol with little or no task-specific training — analogous to large language models for text.

**When to use:** Strong baselines without hand-tuning ARIMA orders; needs optional deps (`chronos-forecasting`, `torch`; TimesFM via extra). TimesFM 3.0 (`timesfm3`) adds covariate support; its default weights are non-commercial / non-production — use `timesfm` (2.5) when that restriction matters.

**Example:**
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 --library pretrained --method chronos2
```

**In mtdata:** `forecast_list_methods --json` shows what your install can run. See [FORECAST.md](FORECAST.md) and [forecast/METHODS.md](forecast/METHODS.md).

### Ensemble (forecast)
Combines several methods (average, RMSE-weighted, stacking, …) so no single model owns the call. Useful when methods disagree or you want robustness over one hero model.

**In mtdata:** `--method ensemble` on forecast tools; see [forecast/FORECAST_GENERATE.md](forecast/FORECAST_GENERATE.md).

---

## Uncertainty & Intervals

### Confidence Interval
A range around the forecast indicating uncertainty. A 95% CI means "if model assumptions hold, the true value will fall within this range 95% of the time."

**Caution:** Financial data often violates model assumptions (fat tails, regime changes), so intervals may be too narrow.

### Conformal Intervals
A distribution-free method to create prediction intervals. Instead of assuming a bell curve, it uses historical forecast errors to calibrate the band width.

**How it works:**
1. Run a small rolling backtest
2. Collect actual errors at each horizon step
3. Use error quantiles to set band width

**When to use:** When you don't trust model-based intervals. Conformal intervals are empirically calibrated.

**Example:**
```bash
mtdata-cli forecast_conformal_intervals EURUSD --timeframe H1 --method theta --horizon 12 --steps 25 --ci-alpha 0.1
```

---

## Regime Detection

### Regime
The current "behavior mode" of the market. Different regimes require different strategies.

**Common regimes:**
| Regime | Characteristics | Strategy Implication |
|--------|-----------------|---------------------|
| **Low Volatility / Ranging** | Price oscillates in a narrow band | Mean reversion works well |
| **Trending** | Price moves directionally | Trend-following works well |
| **High Volatility / Crisis** | Large, unpredictable swings | Reduce position sizes |

### Hidden Markov Model (HMM)
Assumes the market flips between a few **hidden states** you never observe directly. The implementation fits the selected one-dimensional target (returns by default, or price), then infers each state's mean, volatility, and transition probabilities.

**Friendly picture:** Like labeling weather as Calm / Stormy without a perfect sensor — you only see rain intensity, and the model estimates the label.

In `regime_detect`, `hmm` is a **Gaussian HMM**. Prefer [filtered inference](#filtered-vs-smoothed-inference) for live use; smoothed is retrospective. **`gmm` is not an HMM** (no transitions).

**Output:** State IDs (0, 1, …), probabilities, and per-state σ / μ — you name the states by reading those stats (the model will not say “bull”).

**Example:**
```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --method hmm --params "n_states=2"
```

**Interpretation:**
- State 0: Low volatility (σ = 0.0003) → quiet / ranging
- State 1: High volatility (σ = 0.0008) → active / stressed

**In mtdata:** `regime_detect --method hmm`; barrier sim `hmm_mc`. See [forecast/REGIMES.md](forecast/REGIMES.md).

### Change-Point Detection (BOCPD)
**Bayesian Online Change-Point Detection** answers a simpler question than HMM: *“Did the market’s statistical behavior just change?”* — not *“which named regime are we in?”*

Think of it as a smoke alarm for structure: it raises a **transition probability** (0–1) that the recent bars no longer match the recent past (mean/vol shifts, breaks). High probability → stand down, shrink size, or retrain — not “buy” or “sell” by itself.

**Output (typical):** `latest_transition_probability`, `transition_summary.max_transition_probability`, change-point counts, and (at full detail) a series of probabilities.

**When to use:**
- Gate trades after a suspected breakout or news shock
- Invalidate stale forecasts when the series “feels different”
- Cross-check with PELT or HMM labels

**Example:**
```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --method bocpd --threshold 0.5
```

Omit `--threshold` when you want automatic asset/timeframe calibration; a fixed number (including `0.5`) is treated as fixed.

**In mtdata:** `regime_detect --method bocpd`. Deep dive: [forecast/REGIMES.md](forecast/REGIMES.md).

### PELT (change-point detection)
**Pruned Exact Linear Time** segments a series at structural breaks (from the `ruptures` library). Unlike BOCPD’s online “probability right now,” PELT is often used to **cut history into regimes** for retrospective analysis.

**When to use:** Segment history before fitting forecasts; confirm breaks BOCPD flagged.

**Example:**
```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --method pelt --params "model=rbf penalty=auto min_size=5"
```

**In mtdata:** `regime_detect --method pelt`. See [forecast/REGIMES.md](forecast/REGIMES.md).

### Markov-Switching AR (MS-AR)
Combines **regime switches** with an **autoregressive** model: each state has its own AR dynamics (how past returns map to the next bar), not only a mean/vol label.

**When to use:** When autocorrelation as well as volatility changes by regime; research-style workflows.

**Example:**
```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --method ms_ar --params "n_states=2 order=1"
```

**In mtdata:** `regime_detect --method ms_ar`. Defaults to filtered inference (live-oriented); `inference=smoothed` is retrospective.

### Gaussian Mixture (GMM)
Soft clustering of the selected one-dimensional target (returns by default, or price) into Gaussian components **without** a Markov transition matrix. Volatility is each component's fitted sigma; it is not a separate input feature. Bars are assigned by similarity to the components, not by “how long we stay in a state.”

**Do not** read GMM output as an HMM filter — no persistence model is assumed.

**In mtdata:** `regime_detect --method gmm`.

### Clustering (regimes)
Distance-based grouping (e.g. K-means style) on return/volatility features. Descriptive exploration when parametric HMM/MS-AR fits are unstable.

**In mtdata:** `regime_detect --method clustering`.

### Filtered vs smoothed inference
| Mode | Uses data… | Good for |
|------|------------|----------|
| **Filtered** (default for HMM/MS-AR) | Up through each bar | Live / causal decisions |
| **Smoothed** | The whole analysis window, including later bars | Retrospective segmentation only |

Using smoothed probabilities for “what should I do *now*” leaks future information.

---

## Volatility

### Volatility
A measure of how much price typically moves. Higher volatility = larger price swings.

**Types:**
| Type | Definition |
|------|------------|
| **Realized** | How much price *did* move (historical) |
| **Forecasted** | How much we expect it *will* move |
| **Implied** | Market's expectation (from options prices) |

### EWMA (Exponentially Weighted Moving Average)
A volatility estimator that gives more weight to recent observations — a fast “how wild has it been lately?” gauge.

**Parameter:** `lambda` / `lambda_` near **0.94** is a common RiskMetrics-style default. Higher = slower adaptation; lower = reacts faster.

**Example:**
```bash
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method ewma --params "lambda_=0.94"
```

**Interpretation:** If output is `volatility_per_bar: 0.0006`, expect hourly returns to have ~0.06% standard deviation.

**In mtdata:** `forecast_volatility_estimate --method ewma`. See [forecast/VOLATILITY.md](forecast/VOLATILITY.md).

### GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
Models **volatility clustering** — quiet periods follow quiet periods, wild follows wild. Variants add fat tails (`garch_t`), asymmetry (`egarch`, `gjr_garch`), or long memory (`figarch`).

**When to use:** When big moves cluster; slower than EWMA but richer for multi-step vol forecasts and some barrier sims.

**In mtdata:** `forecast_volatility_estimate` GARCH family methods; barrier method `garch`. See [forecast/VOLATILITY.md](forecast/VOLATILITY.md).

### HAR-RV (Heterogeneous AutoRegressive Realized Volatility)
Uses realized volatility at **multiple horizons** (e.g. daily / weekly / monthly components) to forecast future vol — “slow and fast traders” intuition.

**When to use:** When you can build RV from denser bars (e.g. M5) for an H1 decision horizon.

**In mtdata:** `forecast_volatility_estimate --method har_rv`. See [forecast/VOLATILITY.md](forecast/VOLATILITY.md).

---

## Barrier Analysis

### Barrier
A price level that triggers an event when touched:
- **Take Profit (TP):** Close position at profit target
- **Stop Loss (SL):** Close position to limit loss

### Triple-Barrier Labeling
A method to label historical data based on which barrier was hit first:
- **+1 (Win):** TP hit first
- **-1 (Loss):** SL hit first
- **0 (Neutral):** Neither hit within horizon

For triple-barrier labeling in `high_low` mode, `same_bar_policy` controls a
bar that touches both TP and SL. It defaults to conservative `sl_first`;
`tp_first` and `neutral` are explicit alternatives.

**Use case:** Creating labels for machine learning models.

---

## Barrier Optimization Objectives

When optimizing TP/SL levels, you must choose what to maximize. Each objective answers a different trading question.

### Edge
**What it measures:** The raw probability advantage of winning vs losing.

**Formula:** `Edge = P(TP first) - P(SL first)`

**Example:** Edge = 0.20 means you win 20% more often than you lose.

**When to use:**
- General-purpose objective for consistent advantage
- When you want high win rates regardless of payoff size
- Good starting point for most strategies

**Limitation:** Ignores how much you win/lose. A setup with 60% wins but tiny TP and large SL may have positive edge but negative EV.

---

### Kelly Criterion
**What it measures:** The stake size (as a **fraction of bankroll**) that maximizes long-run growth if your win odds and payoff ratio are correct.

**Intuition:** If you bet too small, you under-compound. If you bet too large, a string of losses can crush you. Kelly is the theoretical “sweet spot” — and it is **aggressive**.

**Simplified form (even-money-style barrier framing):**
`Kelly ≈ P(win) − P(loss) / (TP/SL)` when TP/SL are percent distances (net reward/risk).

**Example:** Kelly = 0.25 → full Kelly would risk **25% of equity** per trade. Most people use **fractional Kelly** (e.g. ¼ or ½ Kelly → 6–12% in that example, still large for many accounts).

**When to use:**
- Compare TP/SL grids on a growth-oriented objective (`objective=kelly`)
- Cross-check with `trade_risk_analyze --sizing '{"method":"kelly",...}'` when you have win-rate / avg win / avg loss

**Limitation:** Wrong inputs → wrong size. Full Kelly has large drawdowns. Prefer fractional Kelly and hard risk caps (see guardrails).

**Interpretation of barrier output:**
- `kelly`: from all simulated paths
- `kelly_cond`: only paths that hit TP or SL (resolved)

**In mtdata:** barrier optimize/prob metrics; `trade_risk_analyze` Kelly sizing; journal-derived win stats via `trade_journal_analyze`. See [TRADING_RISK.md](TRADING_RISK.md) and [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md).

### Fixed-fraction sizing
Risk a **fixed percent of equity** per trade (e.g. 1% from entry to stop). Simpler and more common than full Kelly; often combined with a Kelly *cap*.

**In mtdata:** `trade_risk_analyze` accepts a nested fixed-fraction or Kelly `sizing` specification.

---

### EV (Expected Value)
**What it measures:** The average profit per trade, accounting for both probability and payoff size.

**Formula:** `EV = P(win) × TP - P(loss) × SL`

**Example:** EV = 0.15 means you expect to gain 0.15% per trade on average.

**When to use:**
- When payoff asymmetry matters (small wins, big losses or vice versa)
- Comparing setups with different TP/SL ratios
- Maximizing total profit over many trades

**Variants:**
- `ev`: Based on all paths (includes no-hit scenarios as 0)
- `ev_cond`: Based only on resolved paths (ignores trades that never hit TP or SL)

---

### EV Per Bar
**What it measures:** Expected value normalized by how long the trade takes.

**Formula:** `EV_per_bar = EV / mean_time_to_resolution`

**Example:** Setup A has EV=0.30 but takes 20 bars; Setup B has EV=0.15 but takes 5 bars. Setup B has higher EV per bar (0.03 vs 0.015).

**When to use:**
- When capital turnover matters (you want fast trades to reinvest)
- Comparing setups across different timeframes
- Scalping and high-frequency strategies

**Limitation:** Favors fast trades, which may have higher transaction costs.

---

### Prob Resolve
**What it measures:** The probability that the trade closes (hits either TP or SL) within the horizon.

**Formula:** `Prob_resolve = P(TP first) + P(SL first) = 1 - P(no hit)`

**Example:** Prob_resolve = 0.85 means 85% of trades close before the time limit.

**When to use:**
- When you want trades to actually complete
- Avoiding setups where most trades expire without hitting targets
- Day trading where you must close by end of session

**Limitation:** High resolve probability doesn't mean profitable—could resolve at a loss.

---

### Profit Factor
**What it measures:** Ratio of expected gains to expected losses.

**Formula:** `Profit_factor = (P(win) × TP) / (P(loss) × SL)`

**Example:** Profit factor = 2.0 means you expect $2 in wins for every $1 in losses.

**When to use:**
- When you want a risk/reward focused metric
- Comparing profitability of different strategies
- Common in backtesting reports

**Interpretation:**
- Profit factor > 1: Profitable
- Profit factor > 2: Very good
- Profit factor < 1: Losing money

---

### Min Loss Prob
**What it measures:** Minimizes the probability of losing (SL hit first).

**Formula:** Minimize `P(SL first)`

**Example:** Setup with min_loss_prob = 0.15 means only 15% chance of hitting stop loss.

**When to use:**
- Capital preservation is priority
- Conservative trading styles
- When you can't afford drawdowns (e.g., trading client money)

**Limitation:** May result in very small TP targets or trades that rarely resolve.

---

### Utility (Log Utility)
**What it measures:** Risk-adjusted expected value using logarithmic utility (diminishing returns on gains).

**Formula:** `Utility = P(win) × log(1 + TP) + P(loss) × log(1 - SL)`

**Example:** Log utility penalizes large losses more than it rewards equivalent gains.

**When to use:**
- Risk-averse traders
- When you want to avoid ruin (log utility naturally avoids bets that could wipe you out)
- Academic/theoretical contexts

**Interpretation:** Higher utility = better risk-adjusted outcome. Negative utility means the trade is worse than doing nothing.

---

## Objective Selection Guide

| Your Priority | Use Objective | Why |
|---------------|---------------|-----|
| Win rate | `edge` | Maximizes probability advantage |
| Total profit | `ev` | Accounts for both probability and payoff |
| Fast trades | `ev_per_bar` | Optimizes profit per unit time |
| Position sizing | `kelly` | Tells you how much to bet |
| Avoid losses | `min_loss_prob` | Minimizes chance of stop-loss hit |
| Trade completion | `prob_resolve` | Ensures trades actually close |
| Risk-adjusted | `utility` | Penalizes large losses |
| Gross profit ratio | `profit_factor` | Gains vs losses ratio |

---

## Backtesting Metrics

### MAE (Mean Absolute Error)
**What it measures:** Average size of forecast errors, ignoring direction.

**Formula:** `MAE = average(|actual - predicted|)`

**Example:** MAE = 0.0015 means forecasts are off by 0.15% on average.

**Interpretation:**
- Lower is better
- Easy to interpret (same units as forecast)
- Doesn't penalize large errors more than small ones

---

### RMSE (Root Mean Squared Error)
**What it measures:** Average size of forecast errors, penalizing large errors more.

**Formula:** `RMSE = sqrt(average((actual - predicted)²))`

**Example:** RMSE = 0.0020 with MAE = 0.0015 suggests some large outlier errors.

**Interpretation:**
- Lower is better
- Always ≥ MAE (equality means all errors are same size)
- RMSE much larger than MAE → outliers are a problem

---

### Directional Accuracy
**What it measures:** How often the forecast predicts the correct terminal
horizon direction (up/down) from the forecast anchor. Backtest path-directional
accuracy is a separate diagnostic for step-to-step shape agreement.

**Example:** Directional accuracy = 0.58 means 58% of forecasts got the direction right.

**Interpretation:**
- 0.50 is the balanced binary-chance reference when realized directions are
  non-flat; flat outcomes and class imbalance change the empirical baseline
- 0.55+ is generally useful for trading
- High directional accuracy with high MAE: right direction, wrong magnitude

---

## Monte Carlo Simulation Terms

### n_sims (Number of Simulations)
How many random price paths to generate. More simulations = more stable results but slower.

**Guidelines:**
- Quick checks: 1,000
- Normal use: 2,000-5,000
- Publication/final decisions: 10,000+

---

### seed
Random number generator seed. Set this for reproducible results.

**Example:** `--params "seed=42"` ensures the same random paths each time.

---

### Drift (μ)
The expected average return per period. In GBM simulations, drift represents the trend component.

---

### Diffusion (σ)
The volatility component of price movement. Represents random fluctuation around the drift.

---

## Probability Terms

### prob_tp_first
Probability that take-profit is hit before stop-loss.

**Example:** prob_tp_first = 0.62 means 62% of simulated paths hit TP first.

---

### prob_sl_first
Probability that stop-loss is hit before take-profit.

**Example:** prob_sl_first = 0.28 means 28% of simulated paths hit SL first.

---

### prob_no_hit
Probability that neither TP nor SL is hit within the horizon.

**Example:** prob_no_hit = 0.10 means 10% of paths expire without hitting either barrier.

**Important:** High prob_no_hit means your barriers may be too far or horizon too short.

---

### prob_resolve
Probability that the trade completes (hits TP or SL).

**Formula:** prob_resolve = prob_tp_first + prob_sl_first = 1 - prob_no_hit

---

## Grid Optimization Terms

### grid_style
How to generate the TP/SL combinations to test:

| Style | Description | Use When |
|-------|-------------|----------|
| `fixed` | Regular grid from min to max | You know the range to search |
| `volatility` | Scales barriers to current volatility | Adaptive to market conditions |
| `ratio` | Varies risk/reward ratios | You want specific RR profiles |
| `preset` | Pre-configured for trading styles | Quick setup for scalp/swing/position |

---

### refine
Two-stage optimization: coarse search, then fine search around the best result.

**Example:** `--refine true --refine-radius 0.3` searches ±30% around the best initial candidate.

---

### top_k
Number of best candidates to return from optimization.

**Example:** `--top-k 5` returns the top 5 TP/SL combinations instead of just the best.

---

## Forecast Parameters

### Quantity
The target variable for a forecast: `price` (raw closing prices), `return` (log returns), or `volatility` (predicted variance). Most users want `price`; use `return` for stationarity or `volatility` for risk sizing.

### CI Alpha
Miscoverage rate for confidence intervals. `--ci-alpha 0.1` produces a 90% interval; `--ci-alpha 0.05` produces 95%.

### Library
Selects which forecasting backend to use: `native` (built-in), `statsforecast`, `sktime`, `mlforecast`, or `pretrained` (foundation models like Chronos).

### As-Of Date
The `--as-of` parameter lets you generate a retrospective forecast as if running at a past point in time. Useful for comparison and auditing.

---

## Signal Processing

### Denoising
Removing random fluctuations ("noise") to reveal the underlying trend ("signal").

**Trade-off:** More smoothing = clearer trend but more lag (delay).

**Common methods:**
| Method | Best For |
|--------|----------|
| `ema` | General smoothing |
| `median` | Spike removal |
| `kalman` | Adaptive filtering |
| `kalman_robust` | Adaptive filtering that down-weights spikes |
| `kama` | Adaptive moving average (fast in trends, slow in chop) |
| `supersmoother` | Low-lag causal smoother for bar data |
| `preaverage` | Microstructure noise on increments |
| `wavelet` | Frequency-based separation |

**Example:**
```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 500 --denoise ema --denoise-params "alpha=0.2"
```

### Stationarity
A statistical property where mean and variance don't change over time. Many forecasting models assume stationarity.

**Problem:** Raw prices are *not* stationary (they wander and trend).
**Solution:** Often model **returns** (or other transforms) instead of raw price.

### Stationarity tests (ADF, KPSS, Phillips–Perron)
**Plain idea:** three homework-style tests that ask “is this series stable enough to model, or is it still wandering?”

Three classical checks for whether a series looks stationary enough to model:

| Test | Null hypothesis (plain English) | Rule of thumb |
|------|----------------------------------|---------------|
| **ADF** (Augmented Dickey–Fuller) | “There *is* a unit root” (non-stationary) | Reject null → evidence *for* stationarity |
| **PP** (Phillips–Perron) | Same idea as ADF, different correction | Same interpretation as ADF |
| **KPSS** | “The series *is* stationary” | Reject null → evidence *against* stationarity |

mtdata combines them into a short `conclusion`: `stationary`, `non_stationary`, `mixed`, or `inconclusive`.

**In mtdata:** `stationarity_test`. See [TIME_SERIES_DIAGNOSTICS.md](TIME_SERIES_DIAGNOSTICS.md).

### Log return
`log(price_t / price_{t-1})` — a standard transform for modeling and correlation. Closer to stationary than raw prices; additive over time.

### Autocorrelation / seasonality (periodogram)
**Autocorrelation** measures how much today’s value relates to lags of itself. A **periodogram** looks for cyclic strength by frequency. Together they suggest candidate seasonal periods (e.g. 24 bars on H1 ≈ daily).

**In mtdata:** `seasonality_detect`. Confirm on held-out history; peaks are not proof of a tradable calendar edge.

### Outlier scores (MAD, IQR, z-score)
Ways to flag extreme bars:

| Method | Idea |
|--------|------|
| **z-score** | Distance from mean in standard deviations (sensitive to extremes) |
| **IQR** | Outside the interquartile range fence |
| **MAD** | Median absolute deviation — more robust to outliers |

**In mtdata:** `outliers_detect`.

### Causal vs non-causal filters
A **causal** filter uses only past (and present) data — safe for live trading and honest backtests. A **non-causal** (zero-phase) filter uses future bars to smooth, which looks great on charts but **leaks** future information.

**In mtdata:** See [DENOISING.md](DENOISING.md); `denoise_list_methods` reports causality.

### LTTB (Largest Triangle Three Buckets)
A **downsampling** algorithm that keeps a small number of points while preserving visual shape (peaks/valleys). It does **not** smooth values — it drops rows.

**In mtdata:** `--simplify lttb` on data tools. See [SIMPLIFICATION.md](SIMPLIFICATION.md).

### Kalman filter
A recursive estimator that blends a simple process model with noisy observations — often used as an **adaptive smoother** for price or indicators.

**In mtdata:** `--denoise kalman` or `--denoise kalman_robust`. See [DENOISING.md](DENOISING.md).

### SuperSmoother
John Ehlers’ two-pole smoother: a causal low-pass tuned for bar data, with less lag than a simple moving average at a similar cutoff.

**In mtdata:** `--denoise supersmoother`. See [DENOISING.md](DENOISING.md).

### KAMA (Kaufman Adaptive Moving Average)
A moving average that speeds up when price is trending (high efficiency ratio) and slows down in chop.

**In mtdata:** `--denoise kama` (also an indicator). See [DENOISING.md](DENOISING.md) and [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md).

### Pre-averaging
A high-frequency trick: smooth *increments* with a short kernel, then rebuild the price path. Helps when bounce and spread dominate the last few ticks.

**In mtdata:** `--denoise preaverage`. See [DENOISING.md](DENOISING.md).

### Wavelet (denoise / regimes)
Decomposes a series into frequency bands (detail vs trend). Used for **denoising** (drop noisy bands) and, in some regime modes, **energy-profile** style state labels.

**In mtdata:** denoise `wavelet`; regime method `wavelet` where enabled. See [DENOISING.md](DENOISING.md) and [forecast/REGIMES.md](forecast/REGIMES.md).

---

## Multi-asset relationships

### Correlation
How two series move **together** (usually on returns). High correlation ≠ “A causes B”; it is co-movement over the window.

**In mtdata:** `correlation_matrix`. See [CAUSAL_DISCOVERY.md](CAUSAL_DISCOVERY.md).

### Cross-correlation (lead / lag)
Correlation at different time lags: does A tend to move **before** B?

**In mtdata:** `cross_correlation`.

### Granger causality
A **predictive** test: does past of A improve forecasts of B beyond B’s own past? This is **not** philosophical causality — only lead-lag predictability in-sample.

**In mtdata:** `causal_discover_signals`. See [CAUSAL_DISCOVERY.md](CAUSAL_DISCOVERY.md).

### Cointegration
Two (or more) series can wander individually yet keep a **stable long-run relationship** (a mean-reverting spread). Classic pairs-trading intuition.

| Flavor | Idea |
|--------|------|
| **Engle–Granger** | Pairwise: regress one on the other, test residual stationarity |
| **Johansen** | Multivariate: how many cointegrating relations in a basket |

**In mtdata:** `cointegration_test` (`engle_granger` pairs or `johansen`). Prefer level/log-level transforms as documented in the tool.

---

## Technical Indicators

### Moving Average
The average price over the last N bars. Smooths out short-term fluctuations.

- **SMA:** Simple Moving Average (equal weights)
- **EMA:** Exponential Moving Average (recent bars weighted more)

### RSI (Relative Strength Index)
A momentum oscillator measuring speed and change of price movements. Range: 0-100.

**Interpretation:**
- RSI > 70: Potentially overbought (consider selling)
- RSI < 30: Potentially oversold (consider buying)
- RSI = 50: Neutral

### MACD (Moving Average Convergence Divergence)
Shows relationship between two moving averages.

**Components:**
- MACD line: Fast EMA - Slow EMA
- Signal line: EMA of MACD line
- Histogram: MACD - Signal

**Interpretation:** When MACD crosses above Signal, bullish momentum. Below, bearish momentum.

### ATR (Average True Range)
Measures volatility by averaging the true range (high-low including gaps) over N bars.

**Use case:** Setting stop-loss distance. A common rule is `SL = 2 × ATR`.

---

## Pattern Recognition

### Candlestick Patterns
Visual patterns formed by one or more candles that historically precede certain price moves.

**Examples:**
- **Engulfing:** Large candle completely covers previous candle (reversal signal)
- **Doji:** Open ≈ Close (indecision)
- **Hammer:** Small body with long lower wick (potential bottom)

### Chart Patterns
Larger-scale geometric shapes formed over multiple candles.

**Examples:**
- **Head and Shoulders:** Three peaks, middle highest (reversal)
- **Double Top/Bottom:** Two peaks/troughs at similar level
- **Triangle:** Converging trendlines (breakout imminent)

---

## Data & Execution

### OHLCV
Standard candle data format:
- **O**pen: First price of the period
- **H**igh: Highest price
- **L**ow: Lowest price
- **C**lose: Last price
- **V**olume: Trading activity

### Market Depth (DOM)
Order book showing pending buy/sell orders at various price levels.

### Slippage
Difference between expected execution price and actual fill price. Occurs due to market movement or insufficient liquidity.

**Example:** You place a market buy at 1.1750, but get filled at 1.1752. Slippage = 2 pips.

**Why it matters:** Slippage reduces profits and increases losses. Account for it when backtesting.

### Pip
Smallest price increment for a currency pair. For most pairs: 0.0001. For JPY pairs: 0.01.

**Example:** EURUSD moves from 1.1750 to 1.1775 = 25 pips.

**Note:** Some brokers show "pipettes" (5 decimal places), where 1 pip = 10 pipettes.

### Spread
Difference between bid (sell) and ask (buy) price. This is a transaction cost.

**Example:** Bid 1.1748, Ask 1.1750 → Spread = 2 pips.

**Impact:** You start every trade 1 spread behind. Tight spreads are crucial for scalping.

### Lot Size
Standard position size in forex:
- **Standard lot:** 100,000 units
- **Mini lot:** 10,000 units
- **Micro lot:** 1,000 units

**Example:** 0.1 lots of EURUSD = 10,000 EUR notional.

### Risk/Reward Ratio (RR)
Ratio of potential profit to potential loss.

**Formula:** RR = TP distance / SL distance

**Example:** TP = 50 pips, SL = 25 pips → RR = 2.0 (you risk 25 to make 50).

**Rule of thumb:** Higher RR means you can be wrong more often and still profit.

### Drawdown
Peak-to-trough decline in account equity.

**Example:** Account peaks at $10,000, drops to $8,500 → Drawdown = 15%.

**Maximum drawdown:** Largest historical drawdown. Key risk metric.

### VaR (Value at Risk)
The loss your positions are not expected to exceed over a holding period at a given confidence level.

**Example:** One-bar 95% VaR = $120 means that, 95% of the time, you should not lose more than $120 over the next bar.

**In mtdata:** Estimated for open positions by `trade_var_cvar_calculate` (historical or Gaussian). See [TRADING_RISK.md](TRADING_RISK.md).

### CVaR (Conditional VaR / Expected Shortfall)
The average loss in the worst cases beyond the VaR threshold — a measure of **tail severity**. Also called **Expected Shortfall (ES)**.

**Example:** If the 95% VaR is $120, the 95% CVaR is the average loss across the worst 5% of outcomes, which is ≥ $120.

**In mtdata:** Returned alongside VaR by `trade_var_cvar_calculate`; portfolio tools may say ES. See [TRADING_RISK.md](TRADING_RISK.md) and [ADVANCED_ANALYTICS.md](ADVANCED_ANALYTICS.md).

### Sharpe ratio
Return per unit of volatility (often excess return / σ). Higher is “more reward per risk unit” under strong assumptions; sensitive to sample and path.

**In mtdata:** Appears in some backtest / strategy validation summaries. See [forecast/BACKTESTING.md](forecast/BACKTESTING.md).

### Dry-run
A **preview** of a trading request: validation and routing checks **without** sending the order to MT5. Clean dry-runs are necessary but not sufficient (broker margin/stops still apply live).

**In mtdata:** `trade_place`, `trade_modify`, and `trade_close` default to preview mode (`--dry-run true`). Set `--dry-run false` explicitly for live execution. See [TRADING_SAFETY.md](TRADING_SAFETY.md).

### Trade guardrails
Optional environment limits: allowed symbols, max volume, max risk % of equity, blocklists, etc. Defense-in-depth so a bad agent or typo cannot empty an account.

**In mtdata:** `MTDATA_TRADE_*` variables in [ENV_VARS.md](ENV_VARS.md#trade-guardrails); behavior in [TRADING_SAFETY.md](TRADING_SAFETY.md).

### MCP (Model Context Protocol)
A standard way for AI assistants (and other clients) to call tools on a server. mtdata exposes its research/trading tools over MCP transports (`mtdata-stdio`, `mtdata-sse`, …).

**In mtdata:** [SETUP.md](SETUP.md#mcp-server) · [DEPLOYMENT.md](DEPLOYMENT.md).

### TOON
mtdata’s default **human-readable compact** CLI presentation (tabular-ish text with a schema hint). Use `--json` for scripts and agents.

### NDJSON
**Newline-delimited JSON** — one JSON object per line. The `mtdata-cli shell` batch mode prints this so a script can read each command’s result separately.

### SAX / PAA
Ways to squash a long series into a short *symbolic* or averaged sketch (used by some `--simplify` modes). They are for compact comparison, not for pricing an order. See [SIMPLIFICATION.md](SIMPLIFICATION.md).

### Markout
How price moved *after* your fill, at a few delays (1s, 5s, …). Positive markout usually means the fill aged well; it is a research metric, not a broker fee.

### Implementation shortfall
The gap between the price when you *decided* and the price you *got*. Related to [slippage](#slippage), with more timing context. See [ADVANCED_ANALYTICS.md](ADVANCED_ANALYTICS.md).

**In mtdata:** [CLI.md](CLI.md) · [OUTPUT.md](OUTPUT.md).

### Microstructure
Tick-level structure of the market: spreads, quote updates, short-horizon liquidity — “how the tape behaves,” not daily chart patterns.

**In mtdata:** `market_microstructure_analyze`. See [ADVANCED_ANALYTICS.md](ADVANCED_ANALYTICS.md).

### Execution quality
How fills and latency compare to what you intended (slippage, adverse selection, etc.).

**In mtdata:** `trade_execution_quality`. See [ADVANCED_ANALYTICS.md](ADVANCED_ANALYTICS.md).

### Relative strength
How one symbol’s performance compares to peers or a basket over a window (not the RSI indicator).

**In mtdata:** `market_relative_strength`. See [ADVANCED_ANALYTICS.md](ADVANCED_ANALYTICS.md).

---

## External Tools and Techniques

### Finviz
A financial visualization platform providing fundamental data, stock screening, insider trading activity, analyst ratings, and market news for US equities. Data is delayed 15–20 minutes.

**In mtdata:** Finviz is an internal adapter for `news`, `calendar`, `equity_profile`, `screener`, and `asset_performance`. See [FINVIZ.md](FINVIZ.md).

### QuantLib
**Plain idea:** a calculator library for option-style prices. mtdata uses it locally; you do not need a live options feed for `options_barrier_price`.

An open-source C++ library (with Python bindings) for quantitative finance, providing pricing engines for exotic options, yield curves, and calibration routines.

**In mtdata:** Used for barrier option pricing (`options_barrier_price`) and Heston model calibration (`options_heston_calibrate`). See [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md).

### Heston Model
**Plain idea:** volatility itself wanders (it is not a fixed number), which matters for option-style prices.

A stochastic volatility model where the asset price and its variance follow correlated stochastic processes. Characterized by five parameters: v0 (initial variance), kappa (mean reversion speed), theta (long-run variance), sigma (vol of vol), and rho (correlation).

**When to use:** Pricing barrier options and exotic derivatives where constant-volatility (Black-Scholes) assumptions are inadequate.

### Optuna
A Bayesian hyperparameter optimization framework supporting TPE, CMA-ES, and random sampling with pruning (early stopping of unpromising trials).

**In mtdata:** Used by `forecast_tune_optuna` for automated parameter tuning. See [FORECAST.md](FORECAST.md).

### Barrier Option
A financial derivative whose payoff depends on whether the underlying asset's price reaches a specified barrier level. Types include knock-in (activated when barrier is hit) and knock-out (extinguished when barrier is hit).

**In mtdata:** Barrier analysis is central to TP/SL optimization. See [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md).

### Support and Resistance
Price levels where buying pressure (support) or selling pressure (resistance) tends to concentrate, causing price to pause or reverse.

**In mtdata:** Detected via `support_resistance_levels` and the Web API `/api/support-resistance` endpoint. See [WEB_API.md](WEB_API.md) and [LEVELS.md](LEVELS.md).

### Pivot Points
Formula-derived price levels (a central pivot plus resistances R1–R3 and supports S1–S3) computed from the prior bar's high, low, and close. Common methods include classic, Fibonacci, Camarilla, Woodie, and DeMark.

**In mtdata:** Computed by `pivot_compute_points`. See [LEVELS.md](LEVELS.md).

### Volume Profile (POC, VAH, VAL)
A distribution of traded volume across price rather than time. The **Point of Control (POC)** is the most-traded price; the **Value Area** (bounded by **VAH** and **VAL**) is the price range that contains a chosen share of volume (70% by default).

**In mtdata:** Computed by `volume_profile_levels` from bounded ticks or an M1-bar approximation. See [LEVELS.md](LEVELS.md).

### Confluence
A price zone where several independent methods (pivots, support/resistance, Fibonacci, volume profile) agree, raising the odds of a reaction.

**In mtdata:** Ranked by `confluence_levels`; default `min_source_families=2` requires independent agreement. See [LEVELS.md](LEVELS.md).

### Fundamental Analysis
Evaluating a security by examining its intrinsic value through financial statements, earnings, revenue, P/E ratios, and other economic data — as opposed to technical analysis which focuses on price/volume patterns.

**In mtdata:** The Finviz commands provide fundamental data for US equities. See [FINVIZ.md](FINVIZ.md).

### Temporal Analysis
Analyzing how a symbol's behavior varies across time dimensions — day of week, hour of day, month of year — to identify recurring seasonal patterns.

**In mtdata:** The `temporal_analyze` command groups returns by time dimension. See [TEMPORAL.md](TEMPORAL.md).

---

## See also

| Topic | Doc |
|-------|-----|
| Forecasting stages & methods | [FORECAST.md](FORECAST.md) · [forecast/METHODS.md](forecast/METHODS.md) |
| Regimes & change points | [forecast/REGIMES.md](forecast/REGIMES.md) |
| Barriers, edge, Kelly objectives | [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) |
| VaR, sizing, stress | [TRADING_RISK.md](TRADING_RISK.md) |
| Live orders, dry-run | [TRADING_SAFETY.md](TRADING_SAFETY.md) |
| Stationarity, outliers, vol cone | [TIME_SERIES_DIAGNOSTICS.md](TIME_SERIES_DIAGNOSTICS.md) |
| Correlation, Granger, cointegration | [CAUSAL_DISCOVERY.md](CAUSAL_DISCOVERY.md) |
| Indicators & denoise | [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) · [DENOISING.md](DENOISING.md) |
| Levels (pivots, POC, confluence) | [LEVELS.md](LEVELS.md) |
| Full doc map | [docs/README.md](README.md) |
