# Forecast methods reference

**Audience:** Operator

Catalog of `--method` keys for `forecast_generate` (and related tune/backtest tools): categories, libraries, default `--params`, and dependencies.

Plain-language method families: [Theta](../GLOSSARY.md#theta-method) · [ARIMA](../GLOSSARY.md#arima-autoregressive-integrated-moving-average) · [ETS](../GLOSSARY.md#ets-error-trend-seasonality) · [Monte Carlo / GBM](../GLOSSARY.md#monte-carlo-simulation) · [Foundation models](../GLOSSARY.md#chronos--foundation-models) · [Ensemble](../GLOSSARY.md#ensemble-forecast).

> **Your install is the source of truth.** Optional extras change what is available:
> ```bash
> mtdata-cli forecast_list_methods --json
> mtdata-cli forecast_list_methods --library statsforecast --json
> ```
> The catalog includes unavailable methods by default and marks them with
> `available=false` plus the missing requirement. Pass
> `--show-unavailable false` for installed methods only. The default page is
> 20 methods; use `--limit 100` when you need the complete catalog.
> Defaults below match the code; `"auto"` parameters are estimated from data unless you override them.

**Related:** [Forecasting concepts](../FORECAST.md) · [Parameter optimization](../FORECAST.md#parameter-optimization) · [forecast_generate](FORECAST_GENERATE.md) · [Glossary](../GLOSSARY.md)

---

## Method keys at a glance

```
Baselines/classical : naive, drift, seasonal_naive, theta, fourier_ols
Exp. smoothing/ARIMA: ses, holt, holt_winters_add, holt_winters_mul, ets, arima, sarima
Monte Carlo         : mc_gbm, hmm_mc
Analog / ensemble   : analog, ensemble
Machine learning    : mlforecast, mlf_rf, mlf_lightgbm
Foundation (pretr.) : chronos2, chronos_bolt, timesfm, timesfm3
Neural              : nhits, nbeatsx, tft, patchtst
StatsForecast       : statsforecast + sf_* family (sf_autoarima, sf_autoets, sf_theta, …)
Sktime              : sktime + skt_theta, skt_naive, skt_autoets
```

The `sf_*` and `skt_*` families are convenience aliases that pin a specific model within the StatsForecast / sktime libraries (e.g. `sf_ets` → AutoETS). Browse them with `forecast_list_methods --library statsforecast`.

---

## Categories, libraries & dependencies

| Category | Methods | Library | Optional dependency |
|----------|---------|---------|---------------------|
| Baseline / classical | `naive`, `drift`, `seasonal_naive`, `theta`, `fourier_ols` | native | none (core stack) |
| Exp. smoothing / ARIMA | `ses`, `holt`, `holt_winters_add`, `holt_winters_mul`, `ets`, `arima`, `sarima` | native | `statsmodels` |
| Monte Carlo | `mc_gbm`, `hmm_mc` | native | none (core stack) |
| Analog | `analog` | native | none (scipy/numpy/pandas) |
| Ensemble | `ensemble` | native | none |
| Machine learning | `mlforecast`, `mlf_rf`, `mlf_lightgbm` | mlforecast | `mlforecast` + `scikit-learn` / `lightgbm` |
| Foundation | `chronos2`, `chronos_bolt`, `timesfm`, `timesfm3` | pretrained | `chronos` / `timesfm` (+ `torch`) |
| Neural | `nhits`, `nbeatsx`, `tft`, `patchtst` | neuralforecast | `neuralforecast` (+ `torch`) |
| StatsForecast | `statsforecast`, `sf_*` | statsforecast | `statsforecast` |
| Sktime | `sktime`, `skt_*` | sktime | `sktime` |

`native` methods ship with the lean core install. Everything else follows the dependency groups in [SETUP.md](../SETUP.md) / [README](../../README.md#quick-start). Use `forecast_list_library_models --library <name>` to enumerate the concrete models within a library-backed method.

---

## Baseline & classical (`native`)

Fast, dependency-free references and solid intraday baselines.

| Method | Key params (default) | Notes |
|--------|----------------------|-------|
| `naive` | — | Last value carried forward |
| `drift` | — | Last value plus average drift |
| `seasonal_naive` | requires `seasonality` | Repeats the last seasonal cycle |
| `theta` | — (no user-settable method parameters) | Robust, fast general-purpose baseline; fitted alpha is reported in `params_used` |
| `fourier_ols` | `terms=auto` (≈`min(3, m/2)`), `trend=true` | OLS on Fourier terms; good for periodic series |

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta
mtdata-cli forecast_generate EURUSD --method fourier_ols --params "terms=4,trend=true"
```

---

## Exponential smoothing & ARIMA (`statsmodels`)

| Method | Key params (default) | Notes |
|--------|----------------------|-------|
| `ses` | `alpha=auto` | Simple exponential smoothing |
| `holt` | `damped=false`; `alpha`,`beta`=auto | Linear-trend smoothing |
| `holt_winters_add` | `damped=false`; `alpha`,`beta`,`gamma`=auto; requires `seasonality` | Additive seasonality |
| `holt_winters_mul` | `damped=false`; `alpha`,`beta`,`gamma`=auto; requires `seasonality` | Multiplicative seasonality |
| `ets` | `trend=add`, `seasonal=auto`, `damped=false`; smoothing params auto | State-space ETS |
| `arima` | `(p,d,q)=(1,1,1)`, `trend=c` when omitted | Non-seasonal ARIMA |
| `sarima` | `(p,d,q)=(1,1,1)`; seasonal order defaults to `(0,1,1,m)` when `seasonality > 1` | Seasonal ARIMA |

```bash
mtdata-cli forecast_generate EURUSD --method ets --params "trend=add,damped=true"
```

---

## Monte Carlo simulation (`native`)

Simulation methods return distributional forecasts (paths + intervals), which pair naturally with the barrier tools.

| Method | Key params (default) | Notes |
|--------|----------------------|-------|
| `mc_gbm` | `n_sims=500`, `seed=42`, `mu=auto`, `sigma=auto`, `ci_alpha=0.05` | Geometric Brownian motion paths |
| `hmm_mc` | `n_states=2`, `n_sims=500`, `seed=42`, `ci_alpha=0.05` | Hidden-Markov regime-switched simulation |

```bash
mtdata-cli forecast_generate EURUSD --method mc_gbm --params "n_sims=2000,seed=7"
```

Set `seed` explicitly for reproducible simulation runs.

---

## Analog & ensemble (`native`)

| Method | Key params (default) | Notes |
|--------|----------------------|-------|
| `analog` | `window_size=64`, `search_depth=5000`, `top_k=20`, `metric=euclidean`, `min_separation=window_size/4`, `ci_alpha=0.05` | Nearest-neighbour analogs from history |
| `ensemble` | combines multiple methods | Blends member forecasts |

```bash
mtdata-cli forecast_generate EURUSD --method analog --params "window_size=96,top_k=30"
```

---

## Machine learning (`mlforecast`)

Gradient-boosted / tree models over lagged features. All MLForecast methods default their lags to `1..min(30, seasonality or 24)` when `lags` is omitted.

| Method | Key params (default) | Notes |
|--------|----------------------|-------|
| `mlforecast` | `model` (required), `lags=auto` | Generic MLForecast wrapper; pick the underlying model |
| `mlf_rf` | `n_estimators=200`, `max_depth=None`, `lags=auto` | Random-forest regressor |
| `mlf_lightgbm` | `n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`, `max_depth=-1`, `lags=auto` | LightGBM regressor |

```bash
mtdata-cli forecast_generate EURUSD --method mlf_lightgbm --params "n_estimators=400,learning_rate=0.03"
```

---

## Foundation models (`pretrained`)

Zero-shot pretrained forecasters. These download model weights on first use and benefit from a GPU (see [ENV_VARS.md § Forecasting & GPU](../ENV_VARS.md#forecasting--gpu)). Heavyweight fits are best run through the [async training / model store](../FORECAST.md#background-training--model-store).

| Method | Key params (default) | Notes |
|--------|----------------------|-------|
| `chronos2` | `model_name=amazon/chronos-2` | Amazon Chronos-2 |
| `chronos_bolt` | `model_name=amazon/chronos-bolt-base` | Faster Chronos-Bolt |
| `timesfm` | model defaults | Google TimesFM 2.5 (Apache-2.0 weights). Install via `pip install -e ".[forecast-timesfm]"` |
| `timesfm3` | `model_name=google/timesfm-3.0-pytorch` | Google TimesFM 3.0. Same extra as `timesfm`. **Weights are non-commercial / non-production**; use `timesfm` when that restriction matters. |

```bash
mtdata-cli forecast_generate EURUSD --library pretrained --method chronos2 --horizon 24
mtdata-cli forecast_generate EURUSD --library pretrained --method timesfm3 --horizon 24
```

---

## Neural (`neuralforecast`)

Deep models via NeuralForecast. Not installed by `requirements.txt` or any extra today — install manually with `pip install neuralforecast torch` where the dependency stack resolves. On Windows Python 3.14 this currently fails because `ray` has no Windows cp314 wheels.

| Method | Key params (default) | Notes |
|--------|----------------------|-------|
| `nhits` | `input_size=auto`, `max_steps=50`, `batch_size=32` | N-HiTS |
| `nbeatsx` | `input_size=auto`, `max_steps=50`, `batch_size=32` | N-BEATSx |
| `tft` | `input_size=auto`, `max_steps=50`, `batch_size=32` | Temporal Fusion Transformer |
| `patchtst` | `input_size=auto`, `max_steps=50`, `batch_size=32` | PatchTST |

---

## StatsForecast & sktime families

| Method | Key params (default) | Notes |
|--------|----------------------|-------|
| `statsforecast` | `model_name=autoarima`, `season_length=auto` | Generic StatsForecast wrapper; pick the model |
| `sf_*` | fixed model per alias (e.g. `sf_autoets`, `sf_theta`, `sf_garch`) | `season_length=auto` |
| `sktime` | `estimator=ThetaForecaster`, `estimator_params` pass-through | Generic sktime wrapper |
| `skt_theta` / `skt_naive` / `skt_autoets` | fixed estimator per alias | Seasonality injected as `sp` when supported |

```bash
mtdata-cli forecast_generate EURUSD --method sf_autoets --params "season_length=24"
mtdata-cli forecast_list_methods --library statsforecast --limit 100 --json  # full sf_* catalog
```

---

## See Also

- [FORECAST.md](../FORECAST.md) — Concepts, workflow, reproducibility, and libraries
- [FORECAST.md § Parameter Optimization](../FORECAST.md#parameter-optimization) — `forecast_tune_genetic`, `forecast_tune_optuna`, `forecast_optimize_hints`
- [BACKTESTING.md](BACKTESTING.md) — Rolling backtests and metric selection
- [UNCERTAINTY.md](UNCERTAINTY.md) — Confidence and conformal intervals

StatsForecast aliases and library discovery expose the selected installed constructor's parameters, including required values such as SES `alpha`. Missing required parameters fail before fitting. `NaNModel` is excluded because it is a diagnostic placeholder; `SklearnModel` is excluded because it requires a Python estimator object. Use the supported `mlforecast` models for regressor forecasting.
