# Forecasting guide

**Audience:** User

Predict **where price might go**, **how wide the range is**, and **how much it might move** — then check the idea on recent history before you act.

A **forecast** here is an estimate for the next few candles (bars), not a promise. Start with method `theta`. Heavier models and training jobs come later on this page.

Forecasts are **estimates**, not guarantees. Pair point forecasts with uncertainty, barriers, and backtests.

**Dense terms:** [Horizon](GLOSSARY.md#horizon) · [Lookback](GLOSSARY.md#lookback) · [Theta](GLOSSARY.md#theta-method) · [ARIMA](GLOSSARY.md#arima-autoregressive-integrated-moving-average) · [Monte Carlo](GLOSSARY.md#monte-carlo-simulation) · [Foundation models](GLOSSARY.md#chronos--foundation-models) · [Conformal](GLOSSARY.md#conformal-intervals) · [MAE / RMSE](GLOSSARY.md#mae-mean-absolute-error)

**Related:** [Glossary](GLOSSARY.md) · [Methods reference](forecast/METHODS.md) · [Volatility](forecast/VOLATILITY.md) · [Uncertainty](forecast/UNCERTAINTY.md) · [Barriers](BARRIER_FUNCTIONS.md)

---

## Key concepts

| Term | Meaning |
|------|---------|
| **Horizon** | How far ahead (in bars), counted from the open of the current forming bar unless you pass `--as-of` or a historical range. On session-limited equities, closed-session hours do not count. |
| **Lookback** | How much history the model sees. More can help, but costs time. |
| **Quantity** | Often price, returns, or other targets depending on method and flags. |

**Confidence vs reality**

1. Read intervals, not only the midline
2. Validate with [backtests](forecast/BACKTESTING.md) before trading ideas
3. Size risk from **volatility**, not a single point forecast

---

## Price forecasting (`forecast_generate`)

### Basic usage

```bash
# Fast, reliable baseline
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta

# Structured output for scripts / agents
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta --json
```

### Choosing a method

```bash
mtdata-cli forecast_list_methods
mtdata-cli forecast_list_methods --json   # source of truth for *your* install
mtdata-cli forecast_list_methods --supports-training true
```

Availability depends on extras you installed:

- Supported foundation options on the Python 3.14 path include Chronos, Chronos-Bolt, and TimesFM (TimesFM via opt-in extra).
- NeuralForecast methods (`nhits`, `tft`, `patchtst`, `nbeatsx`) need a manual `neuralforecast` + `torch` setup. On Windows Python 3.14 they do not resolve because `ray` (a NeuralForecast dependency) has no Windows cp314 wheels.
- Always trust `forecast_list_methods --json` over static docs for what runs locally.
- The unfiltered default returns the full catalog. Use `--profile quickstart`
  when you only want the small native baseline set.

Full per-method keys, defaults, and dependencies: [forecast/METHODS.md](forecast/METHODS.md).

| Category | Models | When to Use |
|----------|--------|-------------|
| **Classical** | `theta`, `naive`, `drift`, `ses`, `holt`, `arima` | Fast baselines, short horizons |
| **Seasonal** | `seasonal_naive`, `ets`, `holt_winters_add`, `holt_winters_mul`, `sarima`, `fourier_ols` | Data with recurring patterns |
| **Statistical** | `sf_autoarima`, `sf_autoets`, `sf_autotheta` | Auto-tuning, medium horizons |
| **ML-Based** | `mlf_lightgbm`, `mlf_rf` | Non-linear patterns, feature engineering |
| **Neural** | `nhits`, `tft`, `patchtst`, `nbeatsx` | Deep learning, long horizons; manual `neuralforecast` install required |
| **Foundation** | `chronos2`, `chronos_bolt`, `timesfm`, `timesfm3` | Pretrained models (optional deps) |
| **Simulation** | `mc_gbm`, `hmm_mc` | Risk sizing, barrier analysis |
| **Ensemble** | `ensemble` | Combine multiple models |

**Ensemble note:** `ensemble` supports advanced modes (`average`, `rmse_weighted`, `stacking`). See [forecast/FORECAST_GENERATE.md](forecast/FORECAST_GENERATE.md) for parameters and examples.

---

## Recommended workflow

Treat forecast tools as **stages**, each answering one question:

| Stage | Question | Tool |
|-------|----------|------|
| 1. Discover | Which methods are available here? | `forecast_list_methods` |
| 2. Forecast | What is the point forecast? | `forecast_generate` |
| 3. Uncertainty | How wide is the plausible range? | `forecast_conformal_intervals` |
| 4. Trade levels | What TP/SL levels fit this horizon? | `forecast_barrier_optimize` |
| 5. Probability check | How likely is a specific TP/SL pair? | `forecast_barrier_prob` |
| 6. Validation | Did this method work historically? | `forecast_backtest_run` |
| 7. Tuning | Can parameters improve validation metrics? | `forecast_tune_optuna` or `forecast_tune_genetic` |

Keep `--symbol`, `--timeframe`, `--horizon`, and `--method` aligned across stages unless you are deliberately comparing alternatives.

```bash
mtdata-cli forecast_list_methods
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta
mtdata-cli forecast_conformal_intervals EURUSD --timeframe H1 --horizon 12 --method theta
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --direction long
mtdata-cli forecast_backtest_run EURUSD --timeframe H1 --horizon 12 --methods theta --steps 20 --spacing 12
```

### Reproducibility notes

Extended broker equity sessions use weekday-specific observed clock slots with
a standard 24/5 weekend boundary. Friday closing slots and Sunday reopening
are retained; partial history edge dates do not establish recurring closures.
Cash-exchange holidays and early closes apply only to regular equity schedules.
Extended-session broker holidays are unknown, so their projected calendar is
an estimate and the horizon note identifies that limitation.

Defaults vary by method and can change over time. For any result you want to compare later, make the run **self-describing**:

- Save the exact command, including `--symbol`, `--timeframe`, `--horizon`, `--lookback`, `--method`, `--library`, and `--params`.
- Prefer `--json` for stored results so downstream scripts do not depend on text formatting.
- Set important method parameters explicitly instead of relying on implicit defaults.
- Use `forecast_list_methods --json` to confirm which methods are available in the current environment.

Example:

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --library native --method arima --lookback 500 --params "p=2 d=1 q=2" --json
```

---

### Classical Models

**Theta Method** — Decomposes trend and curvature. Robust baseline.
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta
```

**ARIMA** — Models autocorrelation in the data.
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method arima --params "p=2 d=1 q=2"
```

**ETS** — Exponential smoothing with optional trend/seasonality.
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 --method ets --params "seasonality=24"
```

### Foundation Models

Pre-trained deep learning models that work without tuning.

**Chronos 2** — Amazon's foundation model for time series.
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 --library pretrained --method chronos2
```

*Requires: `pip install chronos-forecasting torch`*

**Chronos-Bolt** — Faster Chronos variant.
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 --library pretrained --method chronos_bolt
```

**TimesFM** — Google's foundation model. `timesfm` is TimesFM 2.5 (Apache-2.0 weights, the production option). `timesfm3` is TimesFM 3.0; its default weights are non-commercial and non-production.
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 --library pretrained --method timesfm
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 --library pretrained --method timesfm3
```

*Requires: `pip install -e ".[forecast-timesfm]"` (TimesFM 3.x, which still runs 2.5).*

GPU-backed forecast calls run in a short-lived child process by default
(`MTDATA_FORECAST_PROCESS_ISOLATION=gpu`). This lets the child exit after
inference so CUDA context memory is returned instead of staying reserved by a
long MCP server session. Set `MTDATA_FORECAST_PROCESS_ISOLATION=all` to isolate
every forecast tool call, or `off` to keep the previous in-process behavior.

### Monte Carlo Simulation

Generates thousands of possible future paths instead of a single forecast.

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method mc_gbm --params "n_sims=2000 seed=42" --ci-alpha 0.05
```

**Output includes:**
- Point forecast (median of simulations)
- A 95% lower/upper simulation band requested by `--ci-alpha 0.05`
- Useful for risk sizing and barrier analysis

### Analog Forecasting

Finds historical windows similar to the current pattern and averages what happened next.

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method analog --params "window_size=64 top_k=20"
```

**Parameters:**
- `window_size`: Pattern length to match (default: 64 bars)
- `search_depth`: How far back to search (default: 5000 bars)
- `top_k`: Number of similar patterns to average (default: 20)
- `metric`: Initial search metric (euclidean, cosine, correlation); use `refine_metric=dtw` for DTW re-ranking

---

## Model Libraries

mtdata supports multiple forecasting libraries. Use `--library` to select:

| Library | Description | Example Models |
|---------|-------------|----------------|
| `native` | Built-in implementations | theta, naive, mc_gbm, analog |
| `statsforecast` | Nixtla's fast statistical models | AutoARIMA, AutoETS, Theta |
| `sktime` | Scikit-learn style time series | Various forecasters |
| `mlforecast` | ML models with lag features | LightGBM, RandomForest |
| `pretrained` | Foundation models | Chronos, Chronos-Bolt, TimesFM |

**List models in a library:**
```bash
mtdata-cli forecast_list_library_models native
mtdata-cli forecast_list_library_models statsforecast
mtdata-cli forecast_list_library_models sktime
mtdata-cli forecast_list_library_models pretrained
```

The sktime catalog writes a versioned class-name index. Later one-shot
forecasts using an exact catalog name (for example `NaiveForecaster`) reuse
that index instead of importing every module under `sktime.forecasting`.
Registered aliases such as `skt_naive` and exact dotted estimator paths also
resolve directly.

---

## Backtesting

Validate forecast accuracy with rolling-origin backtests.

```bash
mtdata-cli forecast_backtest_run EURUSD --timeframe H1 --horizon 12 --methods "theta sf_autoarima analog" --steps 20 --spacing 12
```

**Parameters:**
- `--steps`: Number of historical test points
- `--spacing`: Bars between test points
- `--methods`: Space-separated list of models to compare

**Output includes:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Directional accuracy

See **[BACKTESTING.md](forecast/BACKTESTING.md)** for complete guide including parameter optimization.

Replayable analytics use the same time-window contract as point forecasts.
`forecast_conformal_intervals`, both tuning tools, `forecast_optimize_hints`, and
the barrier probability/optimization tools accept either `--as-of` or a
`--start`/`--end` range. Historical runs use the final eligible candle close as
their reference and report `analysis_time_window`; they never mix in a live tick.

---

## Adding Features

For a different forecast target, pass `--target-spec column=volume` or, for
example, `--target-spec column=close,transform=log`. These outputs use
`quantity=custom`, with `target_quantity`, `target_units`, and the base column
and transformation in `target`. Each forecast row keeps its time, bar state,
value, and any target-scale bounds. Broker price precision is not applied to
transformed values.

The standard history loader's `volume` alias uses `tick_volume`, a count of
Bid updates rather than lots or shares. The target reports this source and
unit; unattributed input volume remains explicitly unspecified. Custom targets
do not carry a price-currency label.

Custom-target conformal calibration is unsupported. If a method cannot produce
intervals for your target, use `forecast_list_methods --supports-ci true` to
choose an interval-capable method while keeping the same target and history
window. A price conformal forecast does not calibrate a custom volume or
log-price forecast.

### Technical Indicators

Use an audited feature-consuming method and pass indicators as one compact
string. Observed indicators are lagged by one bar; horizons longer than one bar
must state how their future values are supplied:

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --method mlf_lightgbm \
  --features '{"indicators":"rsi(14),roc(12)","observed_future_policy":"carry_forward"}'
```

Use `forecast_list_methods --detail full` and require both
`supports_historical_exog` and `supports_future_exog`. Backtests reject an
entire feature-bearing run when any selected method lacks either capability;
run univariate controls separately without `--features`.

### Denoising

Smooth data before forecasting:
```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta --denoise '{"method":"ema","params":{"alpha":0.2}}'
```

See [DENOISING.md](DENOISING.md) for available filters.

---

## Submodule Documentation

- **[BACKTESTING.md](forecast/BACKTESTING.md)** — Rolling backtests and parameter optimization
- **[FORECAST_GENERATE.md](forecast/FORECAST_GENERATE.md)** — Detailed `forecast_generate` reference
- **[VOLATILITY.md](forecast/VOLATILITY.md)** — Volatility forecasting methods
- **[REGIMES.md](forecast/REGIMES.md)** — Regime and change-point detection
- **[UNCERTAINTY.md](forecast/UNCERTAINTY.md)** — Confidence and conformal intervals
- **[PATTERN_SEARCH.md](forecast/PATTERN_SEARCH.md)** — Pattern detection and analog search

## Parameter Optimization

Three tools are available for automated tuning and configuration search:

All three accept `--lookback` as the fixed training bars available at every
rolling-origin anchor. When it is omitted, candidate backtests use the
expanding roughly 400-bar default.

The default five-anchor accuracy searches are inexpensive exploratory runs.
Results below 30 anchors are explicitly low-reliability and deployment-ineligible;
use `--steps 30` or more for model-selection evidence. Zero-phase denoising is
always labeled research-only throughout tuning and configuration hints.

### Genetic Algorithm (`forecast_tune_genetic`)

Evolutionary search through parameter space. Good for discrete/mixed search spaces.

```bash
mtdata-cli forecast_tune_genetic EURUSD --methods fourier_ols --horizon 12 --metric avg_rmse --mode auto --population 20 --generations 10 --max-search-time-seconds 300
```

Population and generations are each capped at 100. When the optional wall-clock
limit is reached, a run with at least one valid evaluation returns its best
completed candidate and marks `timed_out: true`, with completed and planned
evaluation counts. A timeout before any valid result is a failed search.

See [BACKTESTING.md](forecast/BACKTESTING.md) for full parameters and examples.

### Optuna (`forecast_tune_optuna`)

Bayesian optimization with TPE, CMA-ES, or random sampling. Supports parallel trials and persistent study storage. Each trial is an atomic rolling backtest, so trial pruning is not exposed.

```bash
mtdata-cli forecast_tune_optuna EURUSD --methods fourier_ols --horizon 12 --metric avg_rmse --mode auto --n-trials 40 --sampler tpe --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--methods` | `fourier_ols` | Forecast methods to optimize |
| `--lookback` | expanding | Optional fixed training window at each anchor |
| `--n-trials` | 40 | Number of optimization trials |
| `--sampler` | `tpe` | Sampling algorithm: `tpe`, `random`, `cmaes` |
| `--timeout` | (none) | Max wall-clock seconds |
| `--n-jobs` | 1 | Parallel trial workers |
| `--study-name` | (auto) | Name for resumable study |
| `--storage` | (none) | DB URL for persistence (e.g., `sqlite:///study.db`); URL credentials are redacted from results and shell-batch logs |
| `--seed` | 42 | Random seed |

*Requires: `pip install optuna`*

### Configuration Search (`forecast_optimize_hints`)

Broader than single-method tuning: `forecast_optimize_hints` runs a genetic search across **timeframes, methods, and method-specific parameters at once**, returning the top-N configurations ranked by forecast accuracy (`avg_rmse`) by default. Pass `--fitness-metric composite` for a multi-metric trading-fitness ranking. Composite fitness needs at least 30 rolling-origin anchors (`--steps 30`) and complete cost inputs so each candidate can produce a comparable trade sample. The default `--steps 5` is valid for `avg_rmse` and is rejected for `composite`. Pass `--lookback` to tune a fixed rolling window that matches `forecast_generate`. Use it to answer *"which timeframe/method/params should I even start from?"* before drilling in with `forecast_tune_genetic` / `forecast_tune_optuna`.

```bash
mtdata-cli forecast_optimize_hints EURUSD --timeframes H1 H4 D1 --methods theta ets arima --horizon 12 --steps 30 --top-n 5 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timeframes` | `H1 H4 D1 W1` | Timeframes to search (space- or comma-separated) |
| `--methods` | fast classical baselines | Methods to search; neural/foundation methods must be requested explicitly and may initialize or download large models |
| `--horizon` | 12 | Bars forecast after each backtest anchor |
| `--steps` | 5 | Rolling-origin backtest anchors per candidate; composite fitness requires 30 |
| `--lookback` | unset | Optional fixed training window matching `forecast_generate` |
| `--population` / `--generations` | 8 / 5 | Genetic search population and generation counts |
| `--fitness-metric` | `avg_rmse` | Objective; `composite` is an explicit trading-metric mode that requires `--steps 30` (or greater) |
| `--top-n` | 5 | Number of ranked configurations to return |

---

## Background Training & Model Store

Heavyweight methods (neural / foundation models, large `mlforecast` runs) can take minutes to fit. mtdata exposes a small task-and-cache layer so those fits happen once and are reused.

```bash
# One-shot commands wait for training and return the stored model_id.
mtdata-cli forecast_train EURUSD --timeframe H1 --method mlf_rf --horizon 24

# Start a long-lived CLI session to submit background tasks.
mtdata-cli shell

# Then submit and observe the task from that shell.
forecast_train EURUSD --timeframe H1 --method nhits --horizon 24
# Pin a reproducible historical anchor (or use --start/--end for a range).
forecast_train EURUSD --timeframe H1 --method nhits --horizon 24 --as-of "2026-01-15T12:00:00Z" --lookback 500
forecast_task_status <task_id> --json
forecast_task_wait <task_id> --timeout-seconds 120 --json
forecast_task_list --json

# Cancel if needed.
forecast_task_cancel <task_id>
forecast_task_cancel_all --dry-run false  # pending and running tasks
```

One-shot `mtdata-cli forecast_train ...` commands and stdin shell batches wait
for a terminal task state and return the stored `model_id`; this keeps the
in-process worker alive without bypassing the task runtime. Interactive shell,
MCP, and Web API calls remain background submissions by default.
`mtdata-cli forecast_generate ... --async-mode true` requires one of those
persistent processes. A `forecast_task_wait` deadline returns
`success: false`, `status: "timeout"`, and preserves the live task state in
`task_status` so automation does not treat an unfinished model as usable.
Failed and cancelled terminal tasks also make `forecast_task_wait` unsuccessful,
with `forecast_training_failed` or `forecast_training_cancelled` as the stable
error code. Successfully completed tasks always report terminal progress `1.0`.
`--as-of` cannot be combined with `--start`/`--end`. The submitted window is
returned as `training_window` and stored with the completed model alongside the
observed training context.

Once training completes, the model is persisted under a key derived from method,
symbol, timeframe, horizon, seasonality, exogenous-input shape, preprocessing, and
training parameters. The observed history start/end are freshness metadata, not
part of that identity. Live `forecast_generate` calls reuse the latest matching
artifact for at most one resolved seasonal cycle only when the method can refresh
its fitted history from the newly supplied bars (currently MLForecast adapters).
They report `model_staleness_bars`; methods without safe live
history refresh retrain instead of forecasting from a stale cutoff. Historical
`as_of` calls always require the artifact's exact training anchor to prevent
look-ahead reuse.

Training and generation share the same supported horizon range of 1–500 bars.
`forecast_models_list` keeps the training cutoff, horizon, and expiration in its
default rows so you can review freshness before choosing a model. A compatible
cache artifact can still be old; creation time and training cutoff are distinct.
`forecast_models_list --detail full` exposes the stored
`compatibility_fingerprint`, a `request_compatibility_status`, and a replayable
`reuse_request` containing the model ID and `model_cache: require_existing`.
Legacy artifacts without identity metadata report `unknown`; artifacts whose
request is no longer valid report `unusable`. Store-file format health is reported
separately as `store_compatibility_status`. If a supplied model ID differs from
the requested horizon, timeframe, target, preprocessing, exogenous-input shape,
or training parameters, `forecast_generate` returns
`forecast_model_incompatible` with per-dimension stored and requested values.

Default pickle-based model artifacts use a versioned envelope that records the
Python, mtdata, and observed scientific-library versions. Compatibility is
checked before unpickling. Legacy or runtime-mismatched artifacts are rejected
and retrained through the normal cache-miss path; custom method-owned artifact
formats remain responsible for their own compatibility checks.

```bash
mtdata-cli forecast_models_list --json
mtdata-cli forecast_models_list --limit 50 --json  # Larger explicit page
mtdata-cli forecast_models_delete "nhits/EURUSD_H1/abc123"  # preview only
mtdata-cli forecast_models_delete "nhits/EURUSD_H1/abc123" --dry-run false --confirm-model-id "nhits/EURUSD_H1/abc123"
mtdata-cli forecast_models_cleanup --json          # preview stale/expired
```

Single-model deletion is also preview-first. The default command reports the
method, symbol/timeframe data scope, creation and last-use times, age, and disk
size without changing the store. Permanent deletion requires both
`--dry-run false` and the same complete ID in `--confirm-model-id`; a missing or
mismatched confirmation fails without mutation. Confirmed deletion removes the
artifact permanently, so it cannot be recovered from the model store.

Cleanup is deterministic and batch-safe: `--limit` caps both the preview and
the apply scope, while `--offset` selects the same model-ID page in either
mode. Re-run the preview with identical filters before setting
`--dry-run false`.

The compact listing defaults to ten models and returns reusable model IDs plus
pagination when another page exists. Incompatibility status and its reason are
retained only for affected rows. Use `--detail full` for method counts, dates,
sizes, and store diagnostics, or increase `--limit` when browsing a larger store.
Concrete library aliases such as `sf_naive` and `skt_naive` remain the public
method identity in tasks, model IDs, listings, and cleanup. Use `--method` for
that exact identity or `--adapter statsforecast` / `--adapter sktime` for an
explicit family-wide view.

Configuration (see [ENV_VARS.md](ENV_VARS.md#async-training--model-store)):

- `MTDATA_TRAIN_WORKERS` — size of the background training thread pool (default `4`).
- `MTDATA_HEAVY_LIMIT` — concurrent heavyweight (neural / foundation) jobs (default `1`).
- `MTDATA_FORECAST_JOBS_DB` — durable SQLite task registry (default `~/.mtdata/forecast/jobs.sqlite`).
- `MTDATA_TRAIN_TIMEOUT_*_SECONDS` — per-category training timeouts for `instant`, `fast`, `moderate`, and `heavy` methods.
- `MTDATA_FORECAST_HEARTBEAT_SECONDS`, `MTDATA_FORECAST_ORPHAN_STALE_SECONDS`, `MTDATA_FORECAST_CANCEL_GRACE_SECONDS`, `MTDATA_FORECAST_SWEEPER_SECONDS` — task liveness, orphan recovery, cancellation, and cleanup tuning.
- `MTDATA_FORECAST_TASK_TTL_SECONDS` — retention for terminal task records and bounded failure diagnostics (default `86400`, or 24 hours).
- `MTDATA_MODEL_STORE` — root directory for cached models (default `~/.mtdata/models`).
- `MTDATA_MODEL_TTL_DAYS` — cache idle expiry in days since last use (default `7`); this is not a maximum model age.

`forecast_generate` auto-trains any trainable method in the background when
called with `async_mode=true`; the response includes a `task_id` to poll with
`forecast_task_status`. Without `async_mode`, it performs the same train,
persist, and predict lifecycle synchronously under the default
`model_cache=reuse` policy. Set `model_cache=ephemeral` to train and predict
without reading or writing the persistent model store, or
`model_cache=require_existing` to fail rather than train on a cache miss.

Unexpected exceptions in an isolated forecast child are logged at `ERROR` with
a bounded child traceback and captured stdout/stderr tails. These diagnostics
are kept in server logs rather than returned to API callers, because they can
contain local paths or dependency details.

Heavy background workers return a bounded exception type and message in task
payloads. Python tracebacks and captured stderr/fault output stay in operator
logs, where local paths and dependency internals do not leak into compact API or
CLI responses. Signal exits are translated to names on POSIX and native status
codes on Windows. `SIGKILL` can indicate an OOM kill or an explicit forced
termination, so system/container logs remain the authoritative way to
distinguish those causes.

---

## Quick Reference

| Task | Command |
|------|---------|
| List methods | `mtdata-cli forecast_list_methods` |
| Basic forecast | `mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta` |
| Foundation method | `mtdata-cli forecast_generate EURUSD --library pretrained --method chronos2 --horizon 24` |
| Monte Carlo with a 95% simulation band | `mtdata-cli forecast_generate EURUSD --method mc_gbm --params "n_sims=2000" --ci-alpha 0.05` |
| Backtest | `mtdata-cli forecast_backtest_run EURUSD --methods "theta analog" --steps 20` |
| Conformal intervals | `mtdata-cli forecast_conformal_intervals EURUSD --method theta --horizon 12` |
| Tune (genetic) | `mtdata-cli forecast_tune_genetic EURUSD --methods fourier_ols --metric avg_rmse` |
| Tune (Optuna) | `mtdata-cli forecast_tune_optuna EURUSD --methods fourier_ols --metric avg_rmse --n-trials 40` |

---

## See Also

- [CLI.md](CLI.md) — Full command reference
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — Barrier optimization
- [TEMPORAL.md](TEMPORAL.md) — Seasonal analysis
- [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md) — QuantLib pricing tools
