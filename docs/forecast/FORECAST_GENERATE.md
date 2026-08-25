# `forecast_generate` reference

**Audience:** Operator

Deep dive into the main price-path command: next **N bars**, many methods, optional indicators/denoise, and quantity modes. Start with [FORECAST.md](../FORECAST.md) for the big picture; use this page for flags and pipeline detail.

**Related:** [Forecasting](../FORECAST.md) · [Methods](METHODS.md) · [Indicators](../TECHNICAL_INDICATORS.md) · [Denoising](../DENOISING.md) · [Barriers](../BARRIER_FUNCTIONS.md)

---

## Basic usage

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta
```

**Output:**
```
forecast[12]{time,bar_state,value}:
    "2026-01-01 18:00",forming,1.17569
    "2026-01-01 19:00",future,1.17570
    ...
```

Forecast row `time` is the target bar's **open timestamp**, while `value` is
the predicted **bar close**. `bar_state` is `forming` when that target bar is
already in progress, `future` before it opens, and `closed` when its wall-clock
interval has elapsed. With `--as-of`, these states and `last_bar_complete` are
evaluated at the replay timestamp rather than the current wall clock. This is
independent of the closed-bars-only input policy.

For intraday symbols with an unambiguous NYSE/Nasdaq suffix, the horizon counts
available exchange-session bars rather than elapsed wall-clock intervals. The
projector learns recurring broker bar-open slots from the fetched history and
applies New York holidays, early closes, and daylight-saving transitions. If
the history is too short to learn those slots, it uses the regular 09:30–16:00
exchange grid and identifies that fallback in `calendar_treatment`.

For price and return forecasts, `last_price_source=candle_close` identifies the
forecast anchor and `price_basis` identifies the broker chart series
(`bid`, `last_trade`, or `broker_chart_price`), matching `data_fetch_candles`.
This is a historical candle close, not a live executable bid or ask.

For `analog` forecasts, compact output retains concise `component_status` and
`ensemble_metrics` summaries. Raw analog paths, per-timeframe diagnostics, and
component diagnostic blobs are available with `--detail standard` or
`--detail full`.

---

## Parameters

### Required
| Parameter | Description |
|-----------|-------------|
| `symbol` | Trading symbol (positional argument) |

### Method Selection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--library` | `native` | Method library: native, statsforecast, sktime, mlforecast, pretrained |
| `--method` | `theta` | Method name within the library |
| `--params` | — | Method-specific parameters (JSON or `key=value`) |
| `--model-cache` | `reuse` | Trainable-model policy: `reuse` loads or persists an artifact, `ephemeral` trains without model-store reads/writes, and `require_existing` fails on a cache miss |
| `--model-id` | — | Reuse a compatible stored model artifact instead of training a new one |
| `--async-mode` | `false` | Submit trainable methods to a persistent task runtime and return a task ID |

Method parameters are validated against the selected method before history is
fetched. Misspelled or unsupported keys return `unknown_parameter` with
`unknown_keys`, `valid_keys`, and close-name suggestions. Methods shown with no
parameters in [METHODS.md](METHODS.md), such as `naive` and `theta`, reject a
non-empty `--params` mapping.

### Window
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timeframe` | `H1` | Candle timeframe |
| `--horizon` | 12 | Available bars to forecast; closed equity-session intervals do not count |
| `--lookback` | auto | Historical bars to use. Native theta/fourier_ols default to 300 bars. For analog forecasts this is a hard upper bound: `search_depth` is reduced to fit, or the request fails if `window_size` and `horizon` cannot fit. |
| `--as-of` | now | Reference time (for backtesting) |
| `--start` / `--end` | — | Bounded training range; use this range style instead of `--as-of` |

For `analog`, an explicit lookback must contain at least
`2 × window_size + horizon` bars. A smaller request returns
`analog_lookback_too_small` with `minimum_lookback_bars`. Otherwise the
effective `search_depth` is capped at
`lookback - (2 × window_size + horizon - 1)`. Without `--lookback`, the method
fetches its configured search depth plus the required window/horizon overhead.

### Target
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--quantity` | `price` | What to forecast: price, return, volatility |
| `--target-spec` | — | Optional structured target transformation or aggregation settings |

### Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--detail` | `compact` | Output detail: compact, standard, summary, or full |

### Uncertainty
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ci-alpha` | 0 | Point forecast by default; pass 0.05 for a 95% interval when the selected method supports native intervals |

Not every method can produce a native interval. The default native Theta call is
a point forecast; request an interval explicitly only when the selected method
supports it. When a requested interval is
unavailable, the result is explicitly `signal_status: not_actionable` and keeps
any model drift under `point_estimate_direction` instead of publishing a
directional claim. Use `forecast_conformal_intervals` to calibrate empirical
bands for point-only methods such as the native Theta fallback.

When price intervals are available, `direction` is published only when the
horizon interval excludes the last observed price. The neutral threshold is
also scaled from recent absolute bar returns and the forecast horizon, with a
minimum effect size of 0.05 percentage points.

For trainable library methods, the default `--model-cache reuse` may write a
new reusable artifact when no compatible model exists. Use `--model-cache
ephemeral` for a fresh evaluation that leaves the model store unchanged, or
`--model-cache require_existing` to prevent implicit training. Background
training (`--async-mode`) always persists its artifact and therefore requires
the default `reuse` policy.

### Pipeline
| Parameter | Description |
|-----------|-------------|
| `--denoise` | Denoising method (ema, kalman, etc.) |
| `--features` | Feature specification |
| `--dimred` | Dimensionality-reduction method name or JSON specification |

---

## Quantity (`--quantity`)

`forecast_generate` can model different target quantities:

- `price` (default): forecasts the future **close price** (output includes `forecast_price`).
- `return`: forecasts **log returns** (`ln(close_t / close_{t-1})`). Output includes `forecast_return` and a reconstructed `forecast_price` path when possible.
- `volatility`: routes to the volatility forecasters (same family as `forecast_volatility_estimate`). When using `--quantity volatility`, set `--method` to a volatility method (e.g., `ewma`, `garch`).

Examples:
```bash
# Price forecast (default)
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --quantity price

# Return forecast (log returns) + reconstructed price path
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --quantity return

# Volatility forecast (recommended alternative: use forecast_volatility_estimate)
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --quantity volatility --method ewma
```

---

## Dimensionality Reduction (`--dimred`)

Dimensionality reduction (dimred) compresses the feature matrix when you provide many inputs (for example via `--features`). This is most useful for ML-style methods that consume multiple features.

Supported dimred methods in the forecasting pipeline:
- `pca` — Principal Component Analysis (`n_components`)
- `tsne` — t-SNE (`n_components` is typically 2 or 3)
- `selectkbest` — keep top-K features (`k`)

Examples:
```bash
mtdata-cli forecast_generate EURUSD --horizon 12 --method mlf_lightgbm --features '{"include":["close","volume"]}' --dimred '{"method":"pca","params":{"n_components":5}}'
```

Tip: the Web UI exposes a broader method list via `GET /api/dimred/methods` (for example: `svd` (TruncatedSVD), `umap`, `isomap`), depending on what is installed (see [../WEB_API.md](../WEB_API.md)).

---

## Model Libraries

### Native (`--library native`)

Built-in implementations with minimal dependencies.

```bash
mtdata-cli forecast_generate EURUSD --library native --method theta
mtdata-cli forecast_generate EURUSD --library native --method arima
mtdata-cli forecast_generate EURUSD --library native --method mc_gbm
mtdata-cli forecast_generate EURUSD --library native --method analog
```

**Available models:**
```bash
mtdata-cli forecast_list_library_models native
```

### StatsForecast (`--library statsforecast`)

Fast statistical models from Nixtla.

```bash
mtdata-cli forecast_generate EURUSD --library statsforecast --method AutoARIMA
mtdata-cli forecast_generate EURUSD --library statsforecast --method AutoETS
```

**Requires:** `pip install statsforecast`

**Note:** Use capitalized class names (AutoARIMA, not autoarima). Or use native wrappers (`sf_autoarima`).

### Pretrained (`--library pretrained`)

Foundation models pre-trained on large time series datasets.

On the supported Python 3.14 install path:
- `chronos2` and `chronos_bolt` are part of the package-index install path
- `timesfm` uses the package-index 2.x release via its dedicated extra

```bash
mtdata-cli forecast_generate EURUSD --library pretrained --method chronos2
mtdata-cli forecast_generate EURUSD --library pretrained --method chronos_bolt
mtdata-cli forecast_generate EURUSD --library pretrained --method timesfm
```

Tip: `mtdata-cli forecast_list_library_models pretrained` shows requirements for your current environment.

**Dependencies (by model):**
- `chronos2` / `chronos_bolt`: `chronos-forecasting`, `torch`
- `timesfm`: `timesfm`, `torch` (install with `pip install -e .[forecast-timesfm]`)

**Parameters:**
- Common: `context_length`, `quantiles`
- Chronos: `model_name`, `device_map`
- TimesFM: `device`, `model_class`

### sktime (`--library sktime`)

Scikit-learn style time series forecasters.

```bash
mtdata-cli forecast_generate EURUSD --library sktime --method ThetaForecaster
mtdata-cli forecast_generate EURUSD --library sktime --method NaiveForecaster --params "strategy=last sp=24"
```

**Requires:** `pip install sktime`

### MLForecast (`--library mlforecast`)

Machine learning models with lag features.

```bash
mtdata-cli forecast_generate EURUSD --library mlforecast --method LGBMRegressor
```

**Requires:** `pip install mlforecast lightgbm`

---

## Common Models

### Classical

| Model | Description | Example Params |
|-------|-------------|----------------|
| `theta` | Theta decomposition | — |
| `naive` | Last value repeated | — |
| `ses` | Simple exponential smoothing | `alpha=0.3` |
| `holt` | Double exponential smoothing | `damped=true` |
| `arima` | ARIMA(p,d,q) | `p=2 d=1 q=2` |
| `sarima` | Seasonal ARIMA | `seasonality=24` |

### Simulation

| Model | Description | Example Params |
|-------|-------------|----------------|
| `mc_gbm` | Monte Carlo GBM | `n_sims=2000 seed=42` |
| `hmm_mc` | HMM-based Monte Carlo | `n_states=2 n_sims=1000` |

### Pattern-Based

| Model | Description | Example Params |
|-------|-------------|----------------|
| `analog` | Historical pattern matching | `window_size=64 top_k=20` |
| `ensemble` | Combine multiple methods | `{"methods":["theta","naive"],"mode":"rmse_weighted"}` |

### Foundation

| Model | Description | Example Params |
|-------|-------------|----------------|
| `chronos2` | Amazon Chronos-II | `context_length=512` |
| `chronos_bolt` | Fast Chronos variant | `context_length=256` |
| `timesfm` | TimesFM (foundation model adapter) | `context_length=512` |

---

## Examples

### Basic Forecast
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta
```

### With Confidence Intervals
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method arima --ci-alpha 0.1 --json
```

### Monte Carlo Simulation
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method mc_gbm --params "n_sims=3000 seed=7"
```

### Foundation Model
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 --library pretrained --method chronos2 --params "context_length=512"
```

### Analog Forecasting
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method analog --params "window_size=64 search_depth=5000 top_k=20"
```

### With Denoising
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta --denoise ema
```

### Ensemble

`ensemble` combines multiple base methods. Common `--params` keys:
- `methods` (list): component methods to run
- `mode` (str): `average`, `rmse_weighted`, or `stacking`
- `weights` (list): manual weights (only used when `mode=average`)
- `cv_points` (int): walk-forward anchors used for `rmse_weighted`/`stacking` weighting
- `method_params` (dict): per-method parameter overrides
- `expose_components` (bool): include component forecasts in the JSON output

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method ensemble --params '{"methods":["theta","naive","arima"],"mode":"average"}'

# RMSE-weighted blend (weights inferred from walk-forward CV)
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method ensemble --params '{"methods":["theta","naive","fourier_ols"],"mode":"rmse_weighted","cv_points":12}'
```

---

## Output Fields

| Field | Description |
|-------|-------------|
| `forecast` | Compact rows with target-bar `time`, `bar_state`, and `value` |
| `data_window` | Closed-history anchor plus forecast timestamp/value semantics and first target-bar state |
| `forecast_price` | Predicted price values |
| `forecast_return` | Predicted return values when `quantity=return` |
| `lower_price` | Lower confidence bound for price forecasts (if available) |
| `upper_price` | Upper confidence bound for price forecasts (if available) |
| `forecast_vs_last_price` | Horizon move, volatility-aware threshold, and confirmed or suppressed direction metadata |
| `signal_status` | `not_actionable` when uncertainty cannot confirm a directional point estimate |
| `trend` | Detected trend direction (if available) |
| `method` | Method used |
| `params_used` | Actual parameters applied |

---

## Quick Reference

| Task | Command |
|------|---------|
| List methods | `mtdata-cli forecast_list_methods` |
| List library models | `mtdata-cli forecast_list_library_models native` |
| Basic forecast | `mtdata-cli forecast_generate EURUSD --method theta --horizon 12` |
| With CI | `mtdata-cli forecast_generate EURUSD --method analog --ci-alpha 0.1` |
| Foundation method | `mtdata-cli forecast_generate EURUSD --library pretrained --method chronos2` |
| JSON output | `mtdata-cli forecast_generate EURUSD --method theta --json` |

---

## See Also

- [../FORECAST.md](../FORECAST.md) — Overview
- [../DENOISING.md](../DENOISING.md) — Preprocessing
- [VOLATILITY.md](VOLATILITY.md) — Volatility forecasting
- [UNCERTAINTY.md](UNCERTAINTY.md) — Confidence intervals
