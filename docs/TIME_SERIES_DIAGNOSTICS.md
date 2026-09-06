# Time-series diagnostics

**Audience:** User

**Read-only** health checks on recent candles *before* you forecast or size risk.

Plain idea: is this series stable enough to model, does it repeat on a clock, and are there wild bars you should know about? You do not need the test names to run the first command — each section says what to do with the answer.

**Dense terms:** [Stationarity](GLOSSARY.md#stationarity) · [ADF / KPSS / PP](GLOSSARY.md#stationarity-tests-adf-kpss-phillipsperron) · [Log return](GLOSSARY.md#log-return) · [Seasonality](GLOSSARY.md#autocorrelation--seasonality-periodogram) · [Outliers (MAD/IQR)](GLOSSARY.md#outlier-scores-mad-iqr-z-score) · [Volatility](GLOSSARY.md#volatility)

**Related:** [Forecasting](FORECAST.md) · [Temporal analysis](TEMPORAL.md) · [Volatility](forecast/VOLATILITY.md) · [CLI](CLI.md) · [Glossary](GLOSSARY.md)

## Stationarity

`stationarity_test` runs [ADF and KPSS](GLOSSARY.md#stationarity-tests-adf-kpss-phillipsperron) from `statsmodels`; Phillips–Perron is also available when the `arch` extra is installed.

```bash
mtdata-cli stationarity_test EURUSD --timeframe H1 --lookback 500 --target log_return --tests adf,kpss,pp --json
```

ADF and PP use a unit-root null hypothesis, while KPSS uses a stationarity null. The combined `conclusion` is `stationary`, `non_stationary`, `mixed`, or `inconclusive`.
The minimum lookback is 20 observations after preprocessing: return and
difference targets therefore require at least 21 input bars.

## Automatic Seasonality

`seasonality_detect` combines autocorrelation peaks with a periodogram and ranks candidate periods in bars.

```bash
mtdata-cli seasonality_detect EURUSD --timeframe H1 --lookback 1000 --target log_return --min-period 2 --max-period 168 --json
```

The dominant period is exploratory evidence, not proof of a stable calendar effect. Confirm it on separate history.
Seasonality requires at least 31 input bars so its default preprocessing leaves
30 analyzed observations.

## Bar Outliers

`outliers_detect` scores returns, volume, and high-low range with MAD, IQR, or z-scores.

```bash
mtdata-cli outliers_detect EURUSD --timeframe H1 --lookback 500 --score-fields return,volume,range --method mad --threshold 3.5 --json
```

Use `--detail full` to include field-level scores and OHLC values for each flagged bar.

## Volatility Term Structure

`volatility_term_structure` compares current realized volatility with its historical distribution across multiple rolling horizons.

Its `data_as_of` is the close of the last completed input candle; `last_bar_open`
identifies that candle's opening time. With an included forming candle,
`data_as_of_basis=forming_bar_snapshot` identifies the observation time instead.
Full output carries the same availability under `meta.time`; the analysis
window's `period_end` remains a bar-open timestamp.

```bash
mtdata-cli volatility_term_structure EURUSD --timeframe H1 --lookback 1000 --horizons 1,5,10,20,60 --percentiles 10,25,50,75,90 --json
```

By default values are annualized decimal volatility. Set `--annualize false` for per-bar decimal volatility.
Lookback must be at least 30 and strictly greater than the largest requested
horizon; the default 60-bar horizon therefore requires lookback of at least 61.
For intraday data, annualization uses the median observed bars per complete UTC
session multiplied by the symbol-class calendar (365 crypto, 260 FX, or 252
other sessions). The response reports both inputs and the resulting basis.

## Typical Workflow

History defaults to bars closed by `--as-of` (or the current UTC instant).
`--include-incomplete true` includes and labels the current forming bar.
Historical partial-bar values cannot be recovered from completed OHLC history;
use completed bars for historical cutoffs inside a candle.

1. Run `outliers_detect` to identify data or event-driven anomalies.
2. Run `stationarity_test` on the exact transform intended for modeling.
3. Use `seasonality_detect` to propose candidate seasonal periods.
4. Use `volatility_term_structure` to place current risk in historical context.
