# forecast/ — Forecasting Engine

Price forecasting pipeline: preprocessing → method selection → execution → post-processing. 43 files across `forecast/` and `forecast/methods/`.

## FILE MAP

### Engine Pipeline

| File | Lines | Purpose |
|------|-------|---------|
| `forecast_engine.py` | 1867 | Main orchestrator: prep → run → post-process |
| `forecast_preprocessing.py` | 673 | Data cleaning, normalization, feature extraction |
| `forecast_validation.py` | 541 | Input validation and sanity checks |
| `forecast_methods.py` | 281 | Method dispatch helpers |
| `forecast.py` | 164 | High-level forecast convenience functions |
| `use_cases/` | — | Forecast use-case orchestration (`compact`, `generate`, `backtest`, `barriers`, `tune`, `sktime_index`) |
| `requests.py` | 408 | Forecast request models |

### Method Registry

| File | Purpose |
|------|---------|
| `interface.py` | `ForecastMethod` ABC + `ForecastResult` dataclass |
| `forecast_registry.py` | Maps method names → implementations |
| `capabilities.py` | Forecast library/model capability helpers |

### Barrier Analysis

| File | Lines | Purpose |
|------|-------|---------|
| `barriers_shared.py` | 869 | Shared barrier types and helpers |
| `barriers_probabilities.py` | 847 | TP/SL hit probability calculation |
| `barriers_optimization.py` | 3785 | Barrier level optimization |

### Specialized Engines

| File | Lines | Purpose |
|------|-------|---------|
| `volatility.py` | 1633 | Volatility estimation (GARCH, realized, etc.) |
| `monte_carlo.py` | 920 | Monte Carlo price simulation |
| `backtest.py` | 2201 | Rolling forecast backtesting |
| `tune.py` | 1351 | Hyperparameter tuning (Optuna) |
| `quantlib_tools.py` | 617 | QuantLib barrier pricing, Heston calibration |

### Shared

| File | Purpose |
|------|---------|
| `common.py` | Time normalization utilities (**DO NOT re-normalize**) |
| `exceptions.py` | Forecast-specific exceptions |
| `job_store.py` | Durable SQLite-backed training task registry (`MTDATA_FORECAST_JOBS_DB`) |
| `task_manager.py` | Durable training runtime: light thread workers, heavy subprocess workers, cancellation, sweeper |
| `target_builder.py` | Forecast target construction |

### methods/ — Model Implementations

| File | Lines | Models | Dep Group |
|------|-------|--------|-----------|
| `classical.py` | — | Theta, naive, seasonal naive | core |
| `ets_arima.py` | 527 | ETS, ARIMA, auto-ARIMA | forecast-classical |
| `statsforecast.py` | — | StatsForecast wrappers | forecast-classical |
| `sktime.py` | — | sktime model wrappers | forecast-classical |
| `mlforecast.py` | — | LightGBM via mlforecast | forecast-classical |
| `neural.py` | — | Neural network models | forecast-foundation |
| `pretrained.py` | — | Chronos, TimesFM 2.5, TimesFM 3 | forecast-foundation |
| `pretrained_helpers.py` | — | Pretrained model utilities | forecast-foundation |
| `analog.py` | — | Analog/pattern-matching forecast | core |
| `monte_carlo.py` | — | MC-specific forecast method | core |

## HOW TO ADD A FORECAST METHOD

1. Create class in `methods/` implementing `ForecastMethod` ABC from `interface.py`
2. Implement `name`, `category`, `required_packages`, `forecast()` method
3. Register in `forecast_registry.py` mapping name → class
4. If new dependency needed, add to appropriate group in `pyproject.toml`

## ANTI-PATTERNS

- **Never** re-normalize already-normalized time data — see `common.py` comment.
- **Never** import all methods eagerly — some have heavy deps (torch, QuantLib). Use lazy imports.
- Methods in `forecast-foundation` group require GPU-capable torch — guard with try/except on import.
- Prefer `forecast_train` + `forecast_task_wait`/`forecast_task_status` for explicit model training; trainable methods now run through the durable task runtime rather than ad-hoc sync execution.
