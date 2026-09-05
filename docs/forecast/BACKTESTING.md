# Backtesting

**Audience:** User

Ask “**did this method work historically on this symbol and timeframe?**” before you trust a live forecast. Covers rolling-origin evaluation, metrics, and parameter search.

**Dense terms:** [MAE](../GLOSSARY.md#mae-mean-absolute-error) · [RMSE](../GLOSSARY.md#rmse-root-mean-squared-error) · [Directional accuracy](../GLOSSARY.md#directional-accuracy) · [Horizon](../GLOSSARY.md#horizon) · [Lookback](../GLOSSARY.md#lookback) · [Sharpe](../GLOSSARY.md#sharpe-ratio)

**Related:** [Glossary](../GLOSSARY.md) · [Forecasting](../FORECAST.md) · [forecast_generate](FORECAST_GENERATE.md) · [Volatility](VOLATILITY.md) · [Methods](METHODS.md)

---

## Key concepts

### What is backtesting?

Backtesting answers: *"How well would this forecast method have performed on past data?"*

Instead of testing on the same data used for training (overfitting), backtesting:
1. Picks historical "anchor" points
2. At each anchor, generates a forecast using only data available at that time
3. Compares the forecast to what actually happened
4. Aggregates error metrics across all test points

### Rolling-Origin Backtest

The standard backtesting approach in mtdata:

```
Timeline: [----history----][forecast horizon]
                          ^
                       anchor
```

**Parameters:**
- **steps**: Number of anchor points to test
- **spacing**: Bars between anchor points; when `steps > 1`, this must be at least `horizon`
- **horizon**: How far ahead each forecast predicts

**Example:** `steps=20, spacing=12, horizon=12` creates 20 test points, each 12 bars apart, each forecasting 12 bars ahead.

### Exact explicit anchors

Use `--anchors` when an experiment must evaluate a preregistered timestamp grid
instead of letting the rolling scheduler work backward from the latest bar. Pass
1–200 strictly increasing UTC ISO timestamps, either as separate values or as a
JSON array:

```bash
mtdata-cli forecast_backtest_run BTCUSD --timeframe H1 --horizon 24 \
  --lookback 720 --start 2022-06-01 --end 2022-12-31 \
  --anchors '["2022-07-04T00:00:00Z","2022-07-11T00:00:00Z"]' \
  --methods mlf_lightgbm --quantity return --detail full
```

An anchor timestamp is the candle's **open time**. With an H1 anchor at
`00:00Z`, that input candle completes at `01:00Z`; only then is its forecast
actionable. The built-in trade simulation enters at the next candle open, also
`01:00Z` when the feed is continuous.

Explicit mode resolves each timestamp to that exact MT5 candle open. A missing
or duplicate bar, insufficient prior history, incomplete future horizon,
target timestamp that differs from the forecast's market-calendar projection,
or overlapping validation window fails the entire request before any model is
fit. It never substitutes a nearby candle or drops an origin. `--start` is the
history floor, not the first scored origin, and `--end` must include every
realized target candle. `--steps` and `--spacing` remain rolling-mode settings
and do not alter an explicit list.

Successful output records the canonical list in
`backtest_plan.requested_anchors` and `resolved_anchors`, with
`anchor_resolution=exact_bar_open` and
`target_resolution=forecast_calendar_projection_exact`. Full detail also
includes each forecast's `actual_timestamps`; preserve those fields when
matching results across methods or horizons.

---

## Quick Start

### Compare Forecasting Methods

```bash
mtdata-cli forecast_backtest_run EURUSD --timeframe H1 --horizon 12 --methods "theta sf_autoarima analog" --steps 20 --spacing 12
```

### Single Method with Custom Parameters

```bash
mtdata-cli forecast_backtest_run EURUSD --timeframe H1 --horizon 12 --methods ses --params "alpha=0.3" --steps 30
```

### Volatility Backtest

```bash
mtdata-cli forecast_backtest_run EURUSD --timeframe H1 --horizon 12 --quantity volatility --methods "ewma parkinson garch" --steps 20
```

---

## Command Reference

```bash
mtdata-cli forecast_backtest_run <SYMBOL> [OPTIONS]
```

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | (required) | Trading symbol (e.g., EURUSD) |
| `--timeframe` | H1 | Candle timeframe |
| `--horizon` | 12 | Bars to forecast at each anchor |
| `--steps` | 5 | Number of test anchors |
| `--spacing` | 20 | Bars between anchors; must be `>= --horizon` when `--steps > 1` |
| `--anchors` | unset | Exact UTC candle-open timestamps (1–200); replaces rolling anchor selection |
| `--methods` | auto | Space or comma-separated method names |
| `--lookback` | unset | Expanding: all available history up to each anchor. Pass `--lookback N` for a fixed N-bar window at every origin. HAR-RV rejects this option and uses `params.days` plus `params.rv_timeframe`. |

### Method Parameters

| Parameter | Description |
|-----------|-------------|
| `--params` | Parameters applied to all methods (JSON or `k=v`) |
| `--params-per-method` | Per-method parameters: `{"fourier_ols": {"seasonality": 24, "terms": 3}}` |

**Example with per-method params:**
```bash
mtdata-cli forecast_backtest_run EURUSD --horizon 12 --methods "fourier_ols arima" --params-per-method '{"fourier_ols": {"seasonality": 24, "terms": 3}, "arima": {"p": 2, "d": 1, "q": 2}}'
```

### Quantity

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--quantity` | `price`, `return`, `volatility` | What to forecast |

Notes:
- `return` uses **log returns** (`ln(close_t / close_{t-1})`), which is often more stationary than prices.
- `volatility` backtests compare predicted volatility vs realized volatility; use volatility methods like `ewma`, `garch`, `har_rv`.
- Volatility methods can own method-specific fit windows. For HAR-RV, omit
  `--lookback` and set `--params-per-method
  '{"har_rv":{"days":120,"rv_timeframe":"M5"}}'`. Full-detail
  `training_window` and `training_bars_used` describe the prepared or available
  per-anchor window; they are not proof of the rows mathematically consumed by
  range, kernel, GARCH, proxy, or HAR-RV estimators. Use
  `input_evidence.source`, `input_evidence.returns`, and
  `input_evidence.transformed_input` for exact fit-input evidence.

**Examples:**
```bash
# Forecast returns instead of prices
mtdata-cli forecast_backtest_run EURUSD --quantity return

# Backtest volatility methods
mtdata-cli forecast_backtest_run EURUSD --quantity volatility --methods "ewma garch"
```

### Trade Simulation

The built-in strategy is a forecast-target exit heuristic, not a TP/SL
backtest. A long exits at the first realized bar at or above the terminal
forecast price; a short exits at the first bar at or below it. If the target is
never reached, the trade exits at the forecast horizon. There is no stop-loss,
so losing trades remain open to the horizon. In return mode, the equivalent
rule is applied to cumulative log returns.

Each forecast is formed after its anchor bar completes. A non-flat signal enters
at the next bar's open (`signal_timing=completed_bar_close`,
`execution_timing=next_bar_open`), so overnight and weekend gaps are included in
the simulated return. Compact results with trading metrics also expose this as
machine-readable `execution_policy`:
`entry=next_bar_open`,
`exit=first_close_reaching_terminal_forecast_else_horizon`,
`target_fill=forecast_target`, `marketable_at_entry_fill=entry_open`,
`horizon_fill=horizon_close`,
`stop_loss=none`. The final anchor is used only when that next open and the
full realized horizon are available.

If the next open has already crossed the terminal target, the target is
marketable immediately and the simulation exits at that opening price. This
models price improvement without crediting a gap that occurred before entry;
the gross trade return is zero before configured costs. The same rule applies
to long and short positions and to price and cumulative-return targets.

Every detail level includes `analysis_time_window`. It records requested
`start`/`end` cutoffs, the effective source-history bounds, first and last
anchors, and the first/last evaluated target bar in UTC. These timestamps use
candle open times; `input_bar_policy=closed_bars_only` states that each anchor's
inputs were available only after that candle completed. Keep this block with
persisted scores so results from different historical periods are not compared
as though they covered the same sample.

Consequently, `win_rate`, Sharpe, drawdown, and return metrics describe this
specific take-profit-only heuristic. They are not directly comparable to a
hold-to-horizon or dual-barrier strategy.

Annualized metrics use the median number of bars in complete observed sessions
for session-limited instruments. Full metric output includes `bars_per_year`
and `annualization_basis`; a basis ending in `assumed_24h` identifies the
timestamp-free generic fallback.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--slippage-bps` | 0.0 | Execution slippage in basis points per side (not a complete transaction-cost model) |
| `--spread-bps` | unset | Explicit round-trip spread in basis points. Omit to leave spread unmodeled. |
| `--commission-bps-per-side` | unset | Explicit commission in basis points per side, deducted twice per round-trip. Omit to leave commission unmodeled. |
| `--trade-threshold` | 0.0 | Minimum expected return to trigger a trade |

Slippage alone is not a complete transaction-cost model. Trading metrics are
gross unless you also pass `--spread-bps` and `--commission-bps-per-side`
(explicit `0` is a modeled zero). `cost_assumptions.spread_and_commission`
reports `modeled` only when both are supplied.

**Example with trading costs:**
```bash
# 2 bps slippage per side, 1 bp round-trip spread, 0 commission
mtdata-cli forecast_backtest_run EURUSD --horizon 12 --methods theta --slippage-bps 2 --spread-bps 1 --commission-bps-per-side 0 --trade-threshold 0.0005
```

### Preprocessing Options

| Parameter | Description |
|-----------|-------------|
| `--denoise` | Denoising method (e.g., `ema`, `kalman`) |
| `--denoise-params` | Denoising parameters |
| `--features` | Feature engineering spec |
| `--dimred` | Dimensionality-reduction method and parameters as JSON |

Dimred methods supported by the forecasting pipeline include `pca` and
`selectkbest` (requires `scikit-learn`). t-SNE is analysis-only because it
cannot transform forecast prediction rows after fitting.

Feature-bearing backtests require a method whose catalog row reports both
`supports_historical_exog=true` and `supports_future_exog=true`. Observed
features are lagged one bar. For horizons greater than one, explicitly choose
`observed_future_policy=carry_forward`; this freezes the latest observed value
over the forecast horizon and is not a known-future indicator path.

Use full detail to audit the selected columns and actual fit/predict consumption:

```bash
mtdata-cli forecast_backtest_run EURUSD --timeframe H1 --horizon 12 \
  --methods mlf_lightgbm --detail full \
  --features '{"indicators":"rsi(14),roc(12)","future_covariates":["hour","dow"],"observed_future_policy":"carry_forward"}'
```

Require `complete_success=true`, per-anchor `feature_usage.status=consumed`,
and `feature_usage.anchors_verified` equal to the planned anchor count. Run raw
or univariate controls in separate commands without `--features`.

**Example with denoising:**
```bash
mtdata-cli forecast_backtest_run EURUSD --horizon 12 --methods theta --denoise ema --denoise-params "alpha=0.2"
```

---

## Understanding Output

### Aggregate Metrics

```json
{
  "units": {
    "forecast_error": "price",
    "avg_mae": "price",
    "avg_rmse": "price",
    "avg_directional_accuracy": "fraction"
  },
  "directional_accuracy_reference": {
    "value": 0.5,
    "basis": "balanced_binary_chance"
  },
  "ranked_methods": [
    {
      "method": "theta",
      "success": true,
      "avg_mae": 0.00142,
      "avg_rmse": 0.00186,
      "avg_directional_accuracy": 0.583,
      "win_rate": 0.625,
      "successful_tests": 20,
      "num_tests": 20
    }
  ]
}
```

| Metric | Description | Good Value |
|--------|-------------|------------|
| `avg_mae` | Mean Absolute Error (average) | Lower is better |
| `avg_rmse` | Root Mean Squared Error (average) | Lower is better |
| `avg_directional_accuracy` | Fraction of terminal horizon moves called in the correct direction from the anchor price | > 0.55 |
| `avg_path_directional_accuracy` | Diagnostic agreement between forecast and realized step-to-step path directions | context-dependent |
| `win_rate` | % of profitable forecast-target/horizon trades | > 0.50 |
| `successful_tests` | Tests that completed without error | = num_tests |

MAE and RMSE use the forecast target's native space: symbol price for
`quantity=price`, log return for `quantity=return`, and return fraction for
`quantity=volatility`. The 0.5 directional reference assumes balanced, non-flat
binary outcomes; flat observations and class imbalance change the empirical
baseline, so a below-reference result is not automatically invertible.

### Trading Performance Metrics

When `slippage-bps`, `spread-bps`, `commission-bps-per-side`, or
`trade-threshold` is set:

```json
{
  "metrics": {
    "avg_return_per_trade": 0.00082,
    "win_rate": 0.625,
    "sharpe_ratio": 1.45,
    "max_drawdown": 0.034,
    "calmar_ratio": 2.12,
    "cumulative_return": 0.0164,
    "annual_return": 0.087,
    "num_trades": 20,
    "trades_per_year": 365
  }
}
```

| Metric | Description | Good Value |
|--------|-------------|------------|
| `sharpe_ratio` | Risk-adjusted return under the built-in exit heuristic | > 1.0 |
| `max_drawdown` | Largest peak-to-trough decline | < 0.10 (10%) |
| `calmar_ratio` | Annual return / max drawdown | > 1.0 |
| `cumulative_return` | Total return over test period | > 0 |
| `win_rate` | Fraction of profitable forecast-target/horizon trades | > 0.50 |

### Per-Anchor Details

Use `detail=full` to include individual test results:

```json
{
  "details": [
    {
      "anchor": "2025-12-15 14:00",
      "success": true,
      "mae": 0.00128,
      "rmse": 0.00165,
      "directional_accuracy": 0.636,
      "params_used": {"season_length": 24},
      "forecast": [1.0542, 1.0545, ...],
      "actual": [1.0540, 1.0548, ...],
      "entry_price": 1.0538,
      "exit_price": 1.0552,
      "expected_return": 0.00094,
      "position": "long",
      "trade_return": 0.00133
    }
  ]
}
```

`params_used` contains the effective parameters returned by the underlying
forecaster after defaults and normalization. It is included for successful
price, return, and volatility anchors at full detail and can vary by anchor.

For volatility backtests, every full-detail row deep-copies the evidence
created by the estimator. This includes `input_evidence`, GARCH
`fit_diagnostics`, denoise application metadata, proxy trust flags, and HAR-RV
`daily_rv`, `daily_rv_quality`, and final-boundary evidence. Structured failed
anchors retain the same available evidence in full detail. A proxy forecast
marked `trust_level=unusable` or `history_policy_ok=false` is not scored.
Compact, standard, and summary output keeps bounded error information but omits
the full evidence blocks.

---

## Method Comparison

### Default Methods

If `--methods` is not specified, the backtest uses available classical methods:
- `naive`, `drift`, `seasonal_naive`, `theta`, `fourier_ols`
- Plus `sf_autoarima`, `sf_theta` if statsforecast is installed

### Comparing Categories

**Fast baselines:**
```bash
mtdata-cli forecast_backtest_run EURUSD --horizon 12 --methods "naive drift theta seasonal_naive" --steps 30
```

**Statistical models:**
```bash
mtdata-cli forecast_backtest_run EURUSD --horizon 12 --methods "sf_autoarima sf_autoets sf_theta" --steps 30
```

**ML models:**
```bash
mtdata-cli forecast_backtest_run EURUSD --horizon 12 --methods "mlf_lightgbm mlf_rf" --steps 20
```

**Foundation models:**
```bash
mtdata-cli forecast_backtest_run EURUSD --horizon 24 --methods "chronos2 chronos_bolt" --steps 15
```

---

## Parameter Optimization

### Genetic Search (`forecast_tune_genetic`)

Automatically find optimal parameters for a forecasting method:

```bash
mtdata-cli forecast_tune_genetic EURUSD --timeframe H1 --methods fourier_ols --horizon 12 --steps 30 --spacing 12 --metric avg_rmse --mode auto --population 20 --generations 10
```

### Genetic Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--methods` / `--method` | `fourier_ols` | One or more methods to optimize |
| `--metric` | `avg_rmse` | Metric to optimize |
| `--mode` | `auto` | Uses the metric's standard direction; `min` or `max` explicitly overrides it |
| `--lookback` | unset | Optional fixed training bars at each rolling-origin anchor. Omit for the method default (native theta/fourier_ols: 300 bars). |
| `--steps` | 5 | Rolling-origin anchors evaluated for every candidate |
| `--spacing` | 20 | Bars between anchors; must be at least the horizon when steps is greater than 1 |
| `--slippage-bps` | `0` | Execution slippage per side; always disclosed in tuning output |
| `--spread-bps` | unset | Round-trip spread in basis points. Required with `--commission-bps-per-side` when optimizing a trading metric. |
| `--commission-bps-per-side` | unset | Commission per side in basis points. Required with `--spread-bps` when optimizing a trading metric. Pass `0` for a zero-commission assumption. |
| `--trade-threshold` | `0` | Minimum expected return required to enter a simulated trade |
| `--population` | 12 | Population size per generation (minimum 2) |
| `--generations` | 10 | Number of generations |
| `--max-search-time-seconds` | (none) | Optional wall-clock limit; returns the best completed candidate and partial-search counts |
| `--crossover-rate` | 0.6 | Probability of crossover |
| `--mutation-rate` | 0.3 | Probability of mutation |
| `--seed` | 42 | Random seed for reproducibility |

### Available Metrics

| Metric | Mode | Description |
|--------|------|-------------|
| `avg_mae` | min | Minimize mean absolute error |
| `avg_rmse` | min | Minimize root mean squared error |
| `avg_directional_accuracy` | max | Maximize direction accuracy |
| `win_rate` | max | Maximize profitable trades |
| `max_drawdown` | min | Minimize peak-to-trough loss magnitude |
| `sharpe_ratio` | max | Maximize risk-adjusted return |
| `calmar_ratio` | max | Maximize return/drawdown ratio |
| `annual_return` | max | Maximize annualized return |
| `avg_return_per_trade` | max | Maximize mean simulated trade return |
| `avg_win_loss_ratio` | max | Maximize average win relative to average loss |
| `kelly_fraction` | max | Maximize the estimated Kelly fraction |
| `half_kelly_fraction` | max | Maximize the more conservative half-Kelly fraction |

`sharpe_ratio`, `calmar_ratio`, and `annual_return` require at least 30
rolling-origin anchors. Requests with fewer steps fail before the search starts
instead of producing an annualized score from an undersized sample.

Accuracy searches may use fewer than 30 anchors for cheap smoke tests, but the
winner is labeled `selection_status: exploratory`, `selection_reliability: low`,
and `deployment_eligible: false`. Use at least 30 anchors before treating tuned
parameters as selection evidence. A zero-phase denoiser always makes every
winner and configuration hint research-only because it uses future observations.

### Custom Search Space

Define which parameters to search:

```bash
mtdata-cli forecast_tune_genetic EURUSD --methods fourier_ols --search-space '{"seasonality": {"type": "int", "min": 12, "max": 48}}'
```

**Search space format:**
```json
{
  "param_name": {
    "type": "int" | "float" | "categorical",
    "min": 0,
    "max": 100,
    "log": false,          // For float: use log scale
    "choices": [...]       // For categorical
  }
}
```

### Default Search Spaces

Each method has sensible defaults. Examples:

| Method | Parameters Searched |
|--------|-------------------|
| `theta` | none (the canonical native Theta model fits its own smoothing parameters) |
| `arima` | p (0-3), d (0-2), q (0-3) |
| `fourier_ols` | seasonality (8-96), terms (1-6), trend (true/false) |
| `sf_autoarima` | seasonality, stepwise, d, D |
| `mlf_lightgbm` | n_estimators, learning_rate, num_leaves, max_depth |

---

## Practical Examples

### Example 1: Find Best Method for Scalping

```bash
# Short horizon, tight spacing
mtdata-cli forecast_backtest_run EURUSD --timeframe M5 --horizon 6 --methods "naive theta fourier_ols sf_autoarima" --steps 50 --spacing 12 --slippage-bps 1 --spread-bps 1 --commission-bps-per-side 0 --trade-threshold 0.0003
```

**What to look for:**
- Highest `win_rate` with positive `avg_return_per_trade`
- Low `max_drawdown`
- `sharpe_ratio` > 1.0

### Example 2: Optimize Fourier Seasonality for Swing Trading

```bash
# Step 1: Find an optimal Fourier period
mtdata-cli forecast_tune_genetic EURUSD --timeframe H4 --methods fourier_ols --horizon 48 --steps 30 --spacing 48 --metric sharpe_ratio --mode auto --slippage-bps 2 --spread-bps 1 --commission-bps-per-side 0 --population 20 --generations 15

# Step 2: Backtest with optimal params
mtdata-cli forecast_backtest_run EURUSD --timeframe H4 --horizon 48 --methods fourier_ols --params "seasonality=48 terms=3" --steps 50 --spacing 48 --slippage-bps 2 --spread-bps 1 --commission-bps-per-side 0
```

### Example 3: Compare Volatility Methods

```bash
mtdata-cli forecast_backtest_run EURUSD --timeframe H1 --horizon 12 --quantity volatility --methods "ewma parkinson garch har_rv" --steps 30 --spacing 24
```

**Output interpretation:**
- `forecast_sigma`: Predicted volatility
- `realized_sigma`: Actual volatility that occurred
- `mae`: Error between forecast and realized

### Example 4: Robust Testing with Denoising

```bash
# Test if denoising improves accuracy
mtdata-cli forecast_backtest_run EURUSD --horizon 12 --methods theta --steps 30 --denoise ema --denoise-params "alpha=0.3"

# Compare to non-denoised
mtdata-cli forecast_backtest_run EURUSD --horizon 12 --methods theta --steps 30
```

### Example 5: Walk-Forward Optimization

Simulate real-world model updates:

```bash
# Period 1: Optimize on first 6 months
mtdata-cli forecast_tune_genetic EURUSD --methods fourier_ols --horizon 12 --steps 50 --spacing 24 --metric avg_rmse

# Record best params, then test on next 3 months with those params
mtdata-cli forecast_backtest_run EURUSD --horizon 12 --methods fourier_ols --params "seasonality=24 terms=3" --steps 30 --spacing 24

# Repeat: re-optimize, test out-of-sample
```

---

## Interpreting Results

### Good Results Checklist

✅ `avg_rmse` is small relative to price volatility
✅ `avg_directional_accuracy` > 0.55 (better than random)
✅ `win_rate` > 0.50 with positive `avg_return_per_trade`
✅ `sharpe_ratio` > 1.0
✅ `max_drawdown` < 10-15%
✅ Results consistent across different `spacing` values

### Warning Signs

⚠️ Very high accuracy on backtests but poor live results → overfitting
⚠️ `successful_tests` << `num_tests` → method fails frequently
⚠️ `avg_rmse` much larger than `avg_mae` → outlier errors
⚠️ `max_drawdown` > 20% → high risk
⚠️ Results vary wildly with small parameter changes → unstable
⚠️ `history_sample_ok=false` or `forecast_reliability=low` → one or more
anchors trained on fewer than the method's recommended history bars; increase
the available history or the explicit `--lookback` before relying on the result

### Avoiding Overfitting

1. **Use enough test points:** start with `steps` ≥ 20 for accuracy checks;
   trading metrics (`win_rate`, Sharpe, drawdown, Kelly, and annualized
   returns) require at least 30 observed trades, so `steps` must be ≥ 30
   before the search starts
2. **Test across timeframes:** Method should work on H1, H4, D1
3. **Test across symbols:** Don't optimize for a single pair
4. **Out-of-sample validation:** Reserve recent data for final test
5. **Realistic costs:** Include `slippage-bps`, `spread-bps`,
   `commission-bps-per-side`, and `trade-threshold`. Trading-metric searches
   require the spread and commission fields even when they are zero.

---

## Performance Tips

### Speed Optimization

1. **Reduce steps for initial screening:**
   ```bash
   --steps 10 --spacing 30  # Quick check
   --steps 50 --spacing 12  # Full validation
   ```

2. **Use fast methods first:**
   - `naive`, `theta`, `seasonal_naive` are instant
   - `sf_autoarima`, `chronos2` are slower

3. **Limit genetic search:**
   ```bash
   --population 15 --generations 8  # Quick
   --population 30 --generations 20 # Thorough
   ```

### Parallelization

Run multiple backtests in parallel (different terminals):

```bash
# Terminal 1
mtdata-cli forecast_backtest_run EURUSD --methods theta --steps 30

# Terminal 2
mtdata-cli forecast_backtest_run GBPUSD --methods theta --steps 30
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Compare methods | `mtdata-cli forecast_backtest_run EURUSD --methods "theta arima analog" --steps 20` |
| With trading costs | `--slippage-bps 2 --spread-bps 1 --commission-bps-per-side 0 --trade-threshold 0.0005` |
| Volatility backtest | `--quantity volatility --methods "ewma garch"` |
| With denoising | `--denoise ema --denoise-params "alpha=0.2"` |
| Optimize params | `mtdata-cli forecast_tune_genetic EURUSD --methods fourier_ols --metric avg_rmse` |
| JSON output | `--json` |

---

## See Also

- [GLOSSARY.md](../GLOSSARY.md) — MAE, RMSE, Sharpe ratio definitions
- [FORECAST.md](../FORECAST.md) — Forecasting methods overview
- [FORECAST_GENERATE.md](FORECAST_GENERATE.md) — Forecast generation options
- [DENOISING.md](../DENOISING.md) — Preprocessing options
