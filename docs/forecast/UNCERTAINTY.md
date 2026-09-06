# Uncertainty and confidence intervals

**Audience:** User

A point forecast without a range invites overconfidence. This page covers **model intervals** and **conformal intervals** so you can size risk and set levels with eyes open.

**Dense terms:** [Confidence interval](../GLOSSARY.md#confidence-interval) · [Conformal intervals](../GLOSSARY.md#conformal-intervals) · [CI alpha](../GLOSSARY.md#ci-alpha) · [Horizon](../GLOSSARY.md#horizon)

**Related:** [Forecasting](../FORECAST.md) · [Barriers](../BARRIER_FUNCTIONS.md) · [Volatility](VOLATILITY.md) · [Glossary](../GLOSSARY.md)

---

## Types of uncertainty

### Model confidence intervals
Intervals from the model’s assumptions (for example normal errors).

**Limitation:** Markets often have fat tails and regime shifts, so these bands can be **too narrow**.

### Conformal intervals
Bands calibrated from **historical forecast errors** — empirical residual-quantile coverage without strong distributional assumptions.

**Advantage:** More realistic bounds based on actual performance.

---

## Model Confidence Intervals

Request intervals with `--ci-alpha`:

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method analog --ci-alpha 0.1 --json
```

**Parameters:**
- `--ci-alpha 0.1` → 90% confidence interval
- `--ci-alpha 0.05` → 95% confidence interval

**Output includes:**
```json
{
  "uncertainty": {
    "status": "available",
    "mode": "interval",
    "alpha": 0.1,
    "intervals": [
      {"time": "2026-01-01T18:00Z", "forecast": 1.1755, "low": 1.1740, "high": 1.1770}
    ]
  }
}
```

**Interpretation:**
- Each `uncertainty.intervals[]` row is a forecast step. `low` and `high` are the
  model interval bounds for that row (price or return, matching the requested
  quantity).
- If model assumptions hold, the true value will fall between `low` and `high`
  about 90% (or 95%) of the time.

**Caution:** Financial markets have fat tails. Model CIs often underestimate extreme moves.

`forecast_generate` does not promote a bullish or bearish point estimate to
`direction` unless the horizon interval excludes the last observed price. If a
method cannot supply the requested interval, the response reports
`signal_status: not_actionable`, `direction_actionable: false`, and preserves
the model-only label as `point_estimate_direction` for research diagnostics.

---

## Conformal Intervals (`forecast_conformal_intervals`)

Residual-quantile calibration builds intervals from rolling backtest residuals, making no distributional assumptions.

### How It Works
1. Run a rolling-origin backtest on historical data
2. Collect actual forecast errors at each horizon step
3. Use error quantiles to set interval width for new forecasts

### Usage

```bash
mtdata-cli forecast_conformal_intervals EURUSD --timeframe H1 --method theta --horizon 12 --steps 50 --spacing 20 --json
```

**Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--method` | Forecasting method | theta |
| `--horizon` | Forecast horizon | 12 |
| `--steps` | Number of calibration anchors (default 50 for stabler quantiles) | 50 |
| `--spacing` | Bars between calibration anchors | 20 |
| `--ci-alpha` | Miscoverage rate (0.05 = 95% interval; 0.1 = 90%) | 0.05 |

> When `--steps > 1`, `--spacing` must be `>= --horizon` so calibration windows do not overlap; otherwise the request is rejected.

### Output

```json
{
  "forecast": [
    {"time": "2026-01-01T18:00Z", "value": 1.1755, "lower": 1.1740, "upper": 1.1770}
  ],
  "conformal": {
    "interval_method": "rolling_residual_quantiles",
    "calibration_anchor_tests_planned": 50,
    "calibration_anchor_tests_succeeded": 50,
    "calibration_anchor_tests_failed": 0,
    "calibration_complete": true,
    "coverage_target": 0.95,
    "empirical_coverage": 0.92
  }
}
```

**Interpretation:**
- Each `forecast[]` row has its point `value` and empirically calibrated
  `lower` / `upper` price bounds.
- With the default alpha, `nominal_confidence_level` is 95%. This is a
  calibration target, not a finite-sample guarantee. Compare it with the
  top-level `empirical_coverage` and `coverage_status` fields before using the
  interval as evidence of historical calibration quality.
- `ci_available=true` requires at least 30 calibration residuals for every
  forecast step and complete forecast paths from every planned anchor. Smaller
  samples return `ci_status=insufficient_calibration`. A failed or incomplete
  anchor returns `ci_status=incomplete_anchor_coverage`. Both cases set
  `ci_available=false` and keep any bounds diagnostic-only.
- Check `calibration_anchor_tests_failed=0` and `calibration_complete=true`.
  Empirical coverage from only the anchors where a model happened to fit can
  be biased, so mtdata does not promote such a subset to a decision-use band.

Use `--detail full` when you need the raw `lower_price` / `upper_price` arrays
or calibration diagnostics such as `conformal.per_step_q` and per-step coverage.

### When to Use
- When you don't trust model-based intervals
- For trading decisions where reliability matters
- When historical data shows fat tails or regime changes

---

## Triple-Barrier Labeling (`labels_triple_barrier`)

A different approach to uncertainty: instead of predicting *where* price goes, label *what happened* historically.

### Concept

For each historical bar, ask: "Within the next N bars, did price hit the take-profit level, stop-loss level, or neither?"

**Labels:**
- `+1` (Win): TP hit first
- `-1` (Loss): SL hit first
- `0` (Neutral): Neither hit within horizon

In `high_low` mode, `same_bar_policy` resolves a bar that touches both
barriers. The default is conservatively `sl_first`; `tp_first` and `neutral`
are explicit alternatives.

When denoising is enabled, the resolved close series anchors each barrier.
`high_low` still uses raw intrabar highs and lows so an observed tradable touch
is not smoothed away; `labeling_spec.hit_price_source` reports this as
`raw_high_low`. Choose `label_on=close` to use only the resolved close series
for both anchors and hits.

Label preprocessing is causal by default. `zero_phase` filters use future bars
and are rejected unless `--allow-noncausal-denoise` is supplied. That override
is for exploratory offline analysis only: the response sets
`denoise_lookahead_bias=true`, `suitable_as_training_target=false`, and records
the effective method, causality, parameters, and entry column under
`preprocessing.denoise`.

Every triple-barrier outcome uses later bars by design. Responses therefore set
`label_uses_future_path=true`, `suitable_as_live_feature=false`, and separately
state whether the result remains a valid historical training target. Join these
labels as targets, never as same-timestamp live features.

The label row is keyed by `entry_bar_open_time`, but its `entry_price` is the
source bar's close. Use `entry_price_available_at` as the earliest decision time
and only join features that were available by that instant. For example, an H1
row opened at `05:00Z` has a close-derived entry price available at `06:00Z`.
`tp_hit_bar_open_time` and `sl_hit_bar_open_time` identify the candle containing
the first observed touch; OHLC data cannot reveal the exact intrabar touch time.
The machine-readable `timestamp_contract` carries the same rules in compact and
full responses. Daily, weekly, and monthly availability uses broker-calendar
bar boundaries, including configured daylight-saving changes.

Compact and summary responses lead with outcome counts, rates, holding period,
and sample quality. Safety flags stay at the top level, timing rules stay in
`timestamp_contract`, and `labeling_spec` keeps the effective barrier settings.
Inactive preprocessing is omitted. `history_bars_used` is the canonical count;
requested and fetched counts appear separately when they differ. Full detail
retains the complete diagnostics and all labeled rows.

### Usage

```bash
mtdata-cli labels_triple_barrier EURUSD --timeframe H1 --horizon 12 --barrier '{"unit":"pct","take_profit":0.5,"stop_loss":0.3}' --json
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `--horizon` | Maximum bars to wait |
| `--barrier` | JSON object with `unit`, `take_profit`, and `stop_loss`. `ticks` is the broker trade tick/point, not FX pips; use `unit=pips` for forex pips. |
| `--allow-noncausal-denoise` | Explicitly permit look-ahead-contaminated zero-phase labels for offline exploration |

A conventional FX pip is not the same unit as an MT5 tick. Use `unit=pips` for
forex pip distances. `unit=ticks` remains the broker trade tick/point (for many
five-digit FX quotes, one pip is 10 ticks).

### Output

```json
{
  "data": [
    {
      "entry_bar_open_time": "2025-12-18T17:00Z",
      "entry_price_available_at": "2025-12-18T18:00Z",
      "label": 1,
      "outcome": "tp_first",
      "holding_bars": 5,
      "tp_hit_bar_open_time": "2025-12-18T22:00Z"
    }
  ],
  "summary": {
    "counts": {"tp": 45, "sl": 32, "neutral": 123}
  }
}
```

Compact output keeps at most 10 representative rows in `data`; summary counts
cover the full requested lookback. Use `--detail full` when you need the
parallel `entry_bar_open_times`, `entry_price_available_at`, `labels`, and
`holding_bars` arrays for model training.

**Interpretation:**
- Label distribution shows historical win/loss rates for these barrier levels
- Use this to evaluate signal quality or train ML models

---

## Practical Applications

### Conservative Position Sizing

Use conformal intervals instead of model CIs:

```bash
# Get conformal intervals
mtdata-cli forecast_conformal_intervals EURUSD --horizon 12 --ci-alpha 0.1

# Use the first forecast row's lower value as a stop-loss floor
# Size position so max loss (if that lower bound is hit) is within risk budget
```

### Validating Signal Quality

Use triple-barrier labels to evaluate entry signals:

```bash
# Label historical entry points
mtdata-cli labels_triple_barrier EURUSD --horizon 12 --barrier '{"unit":"pct","take_profit":0.5,"stop_loss":0.3}'

# Check resolved win rate: counts.tp / (counts.tp + counts.sl)
# If win rate < 50%, signal needs improvement
```

### Comparing Forecast Methods

Backtest with conformal intervals to compare reliability:

```bash
# Method A
mtdata-cli forecast_conformal_intervals EURUSD --method theta --horizon 12 --steps 50

# Method B
mtdata-cli forecast_conformal_intervals EURUSD --method sf_autoarima --horizon 12 --steps 50

# Compare interval widths—narrower = more precise (if coverage is similar)
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Method CI (90%) | `mtdata-cli forecast_generate EURUSD --method analog --ci-alpha 0.1` |
| Conformal intervals | `mtdata-cli forecast_conformal_intervals EURUSD --method theta --horizon 12` |
| Triple-barrier labels | `mtdata-cli labels_triple_barrier EURUSD --horizon 12 --barrier '{"unit":"pct","take_profit":0.5,"stop_loss":0.3}'` |

---

## See Also

- [FORECAST.md](../FORECAST.md) — Price forecasting
- [BARRIER_FUNCTIONS.md](../BARRIER_FUNCTIONS.md) — TP/SL probability analysis
- [GLOSSARY.md](../GLOSSARY.md) — Term definitions
