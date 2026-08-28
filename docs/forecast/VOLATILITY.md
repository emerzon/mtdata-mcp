# Volatility forecasting

**Audience:** User

How much does price usually move? Volatility answers that — and it drives **realistic TP/SL distances**, **position size**, and **barrier hit odds**. mtdata estimates and forecasts vol with EWMA, HAR-RV, GARCH-family methods, and term-structure views.

**Dense terms:** [Volatility](../GLOSSARY.md#volatility) · [EWMA](../GLOSSARY.md#ewma-exponentially-weighted-moving-average) · [GARCH](../GLOSSARY.md#garch-generalized-autoregressive-conditional-heteroskedasticity) · [HAR-RV](../GLOSSARY.md#har-rv-heterogeneous-autoregressive-realized-volatility) · [ATR](../GLOSSARY.md#atr-average-true-range)

**Related:** [Glossary](../GLOSSARY.md) · [Forecasting](../FORECAST.md) · [Barriers](../BARRIER_FUNCTIONS.md) · [Risk analytics](../TRADING_RISK.md)

---

## Quick start

```bash
# EWMA volatility (fast, reliable)
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method ewma

# With custom smoothing
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method ewma --params "lambda_=0.94"
```

---

## Understanding the Output

```
success: true
  symbol: EURUSD
  timeframe: H1
  method: ewma
  horizon: 12
  volatility_per_bar: 0.00062     # Per-bar volatility (as return)
  volatility_annualized: 0.058    # Annualized volatility
  volatility_horizon: 0.002145  # Expected volatility over horizon
```

For session-limited instruments, intraday annualization uses the median bar
count from complete observed sessions. The response reports `bars_per_year`
and `annualization_basis`; `252_trading_days_assumed_24h` means there were not
enough timestamps to infer a session and the generic 24-hour fallback was used.

`data_as_of` is the last completed bar's close in both live and replay
windows. `last_bar_open` is the open of that same bar.

**Interpretation:**
- `volatility_per_bar: 0.00062` → Expect ~0.06% moves per hour (1 standard deviation)
- `volatility_horizon: 0.002145` → Over 12 hours, expect ~0.21% total range (1 σ)
- For EURUSD at 1.1750, 0.21% ≈ 25 pips

**Rule of thumb:** Set stop-loss at 1.5-2x horizon volatility to avoid getting stopped by noise.

---

## Methods

### Fast Estimators

Use recent data to estimate current volatility. Best for quick calculations.

| Method | Description | When to Use |
|--------|-------------|-------------|
| `ewma` | Exponentially weighted MA | General purpose, fast |
| `rolling_std` | Simple rolling standard deviation | Quick baseline |
| `parkinson` | Uses high/low range (more efficient) | When H/L data is reliable |
| `gk` | Garman-Klass (uses OHLC) | More efficient than close-to-close |
| `rs` | Rogers-Satchell | Accounts for drift |
| `yang_zhang` | Combines overnight and intraday | Most efficient range-based |

**EWMA Example:**
```bash
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method ewma --params "lambda_=0.94"
```

**Parkinson Example:**
```bash
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method parkinson
```

---

### GARCH Family

Models volatility clustering—the tendency for high-volatility periods to follow high-volatility periods.

| Method | Description |
|--------|-------------|
| `garch` | Standard GARCH(1,1) — Normal innovations |
| `garch_t` | GARCH(1,1) — Student-t innovations (heavier tails) |
| `egarch` | Exponential GARCH (asymmetric) — Normal innovations |
| `egarch_t` | EGARCH — Student-t innovations |
| `gjr_garch` | GJR-GARCH (leverage effect) — Normal innovations |
| `gjr_garch_t` | GJR-GARCH — Student-t innovations |
| `figarch` | Long-memory GARCH |
| `arima` | ARIMA on volatility proxy |
| `sarima` | Seasonal ARIMA on volatility proxy |
| `ets` | Exponential smoothing state-space |
| `theta` | Theta method |
| `ensemble` | Ensemble of fast estimators |

**GARCH Example:**
```bash
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method garch
```

**When to use:** When volatility clusters are visible (big moves follow big moves). GARCH is slower but more accurate for regime-switching markets.

---

### Realized Volatility

Uses high-frequency data to compute more accurate volatility estimates.

| Method | Description |
|--------|-------------|
| `realized_kernel` | Kernel-based realized volatility |
| `har_rv` | HAR-RV model (daily/weekly/monthly components) |

**HAR-RV Example:**
```bash
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method har_rv --params "rv_timeframe=M5,days=150"
```

**Parameters:**
- `rv_timeframe`: Intraday timeframe for realized variance (for example M1,
  M5, M15, or H1). `D1`, `W1`, and `MN1` are rejected.
- `days`: Maximum trailing calendar-day span for the HAR regression. The span
  ends at the requested cutoff and is intersected with an explicit `start`.
- `window_w`: Weekly window (default: 5)
- `window_m`: Monthly window (default: 22)
- `minimum_daily_coverage_fraction`: Minimum observed-to-expected intraday
  bar coverage and exact-adjacent-return coverage for each UTC-day RV aggregate
  (default: `0.9`).
- `maximum_missing_bars_per_gap`: Largest internal gap allowed in an included
  UTC day, measured in missing `rv_timeframe` bars (default: `12`). Returns
  never bridge a gap, including one that remains within this day-level limit.

HAR-RV does not accept the requested-timeframe `lookback` option. Use
`params.days` to control its calendar span and `params.rv_timeframe` to
control the intraday observations. In full detail,
`params_used.history_cutoff`, `history_start_bound`, their exact epoch
fields, and `history_window_policy` make the fitted interval auditable;
`data_window` reports the bars actually available inside that bound. A stale
provider response can shorten the observed window, but it does not move the
requested cutoff backward and silently broaden the fit.

For sub-hour `rv_timeframe` values, HAR-RV requires every timestamp to lie on
the absolute UTC grid. Hourly and multi-hour MT5 candles can be broker phased,
so those timeframes instead establish a causal, same-weekday phase profile and
reject later phase drift. A return is computed only when two bars are in the
same UTC day and exactly one interval apart. HAR-RV therefore does not turn an
overnight boundary or an intraday history gap into a long-interval return.

Daily coverage is assessed causally. The first observed UTC day is always
excluded as a request-boundary aggregate. A complete 24-hour timestamp grid
can establish its own expected count; otherwise a session-limited day needs at
least three structurally valid prior observations of the same weekday, and its
expected counts are the largest structurally valid bootstrap profile seen for
bars and exact-adjacent returns. Leading request-boundary days do not
participate. After the weekday baseline is established, only eligible days
update it, and its high-water counts never decline inside the fetched window;
later sparse outages cannot lower the expectation and normalize themselves.
This deliberately prefers a conservative false negative if a legitimate
session permanently shortens. Three identically truncated, contiguous
nonleading bootstrap sessions remain indistinguishable from a real shorter
session without a historical symbol-session calendar; an exact full grid can
self-validate BTC/M5 instead. Count, return, and phase baselines also do not
prove the exact start/end slot shape of a session-limited weekday, so a
same-count shifted session is not detectable without that calendar. Exact
24-hour-grid schedule evidence is scoped to
the same weekday and needs three corroborating full grids before it can update
the persistent profile, so one anomalous 24-hour Friday cannot override a
legitimate shorter Friday session. Schedule evidence is timestamp-only and is
reported separately: a day with unusable RV prices can corroborate the schedule
but cannot update eligible RV count/return baselines. A current exact full grid
can still prove that day's own timestamp aggregate. Both a day's bar count and
its usable exact-adjacent-return count must meet the coverage threshold. A day
below either threshold or above the internal-gap limit is excluded without
filling or shifting bars.
Its observed-day position remains missing in the daily series, so weekly and
monthly HAR windows cannot silently compress across it. Any final observed UTC
day whose cutoff is before midnight is boundary-ineligible even when its bar
and return coverage exceed the ordinary daily threshold. It can be trimmed
before forecasting from the prior eligible day only after exact 24-hour grids
on the same weekday establish a compatible 24/7 contract and its observed
timestamps form the complete, gap-free prefix from 00:00 through the last bar
closed at the cutoff. A session-limited pre-midnight final day cannot use that
authorization and blocks the forecast, even if its normal session may already
have closed. This intentional availability cost remains until a historical
session calendar or causal slot-shape contract can prove completion. A
late-start, gapped, off-grid, duplicated, or failed completed final day also
remains missing and blocks the forecast.

Full output exposes `daily_rv_quality`, including the effective interval,
coverage, gap, and day-position policies; per-date exclusions and reasons;
return intervals rejected at gaps; and the daily-count, aligned-row, and recent
lag evidence used to decide whether the fit and forecast are ready.
`final_daily_aggregate` remains available as the focused trailing-day view.
`daily_rv_quality.final_boundary_authorization` records the exact prefix check
and authorization reason. The baseline bootstrap/update policies, retained
state per weekday, rejected updates, and weekday-scoped 24-hour-grid evidence
are reported alongside it.
Because MT5 candles do not carry a historical symbol-session calendar, HAR-RV
cannot distinguish an entirely absent trading day from a scheduled closed day
when that UTC date has no bars at all. The quality contract reports this limit
as `whole_missing_day_detection=unavailable_without_symbol_session_calendar`;
session-aware research should independently bind its expected trading calendar.
When `rv_timeframe` differs from the requested forecast timeframe, target times
are aligned to actual MT5 candle opens for the requested timeframe; `data_as_of`
is the completed-bar close of the latest high-frequency model input, while
`last_bar_open` stays the last bar's open. Live and replay windows share that
`data_as_of` meaning.

Horizon is measured in requested-timeframe bars after the last **completed**
bar at the cutoff. An intra-bar `as_of` (for example 13:55 on an H4 chart)
therefore forecasts the currently forming H4 candle as `horizon=1`, whether
the estimator is EWMA on H4 closes or HAR-RV on M5 returns. Methods that
cannot produce that common window fail instead of silently shifting the
target.

**When to use:** When you need the most accurate volatility forecasts and have access to intraday data.

---

### Volatility Proxies

Forecast a volatility proxy (like squared returns) using any forecasting method.

**Example:**
```bash
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method theta --proxy squared_return
```

**Proxies available:**
- `squared_return`: (close/prev_close - 1)²
- `abs_return`: |close/prev_close - 1|
- `log_r2`: squared log-return

---

## Practical Applications

### Setting Stop-Loss Distance

Use volatility to set stops that won't be hit by normal noise:

```bash
# Get hourly volatility
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 1 --method ewma
# Output: volatility_per_bar: 0.00062 (0.062%)

# For EURUSD at 1.1750:
# 1σ = 1.1750 × 0.00062 = 0.00073 (7.3 pips)
# Recommended SL: 2σ = 14.6 pips minimum
```

### Position Sizing

Size positions inversely to volatility:

```bash
# High volatility → smaller position
# Low volatility → larger position

# Example: Risk $100 per trade
# sigma = 0.002 (0.2%)
# SL distance = 2 × 0.002 = 0.4%
# Position size = $100 / 0.4% = $25,000 notional
```

### Barrier Optimization

Use volatility-scaled barriers instead of fixed percentages:

```bash
# Let the optimizer scale barriers to current volatility
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --grid-style volatility --params "vol_window=250"
```

---

## Comparison of Methods

| Method | Speed | Accuracy | Data Needed |
|--------|-------|----------|-------------|
| `ewma` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Close prices |
| `parkinson` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | High/Low |
| `yang_zhang` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | OHLC |
| `garch` | ⭐⭐ | ⭐⭐⭐⭐ | Close prices |
| `har_rv` | ⭐⭐ | ⭐⭐⭐⭐⭐ | Intraday data |

**Recommendations:**
- **Quick checks:** Use `ewma` or `parkinson`
- **Trading decisions:** Use `yang_zhang` or `garch`
- **Research/backtesting:** Use `har_rv`

---

## Quick Reference

| Task | Command |
|------|---------|
| EWMA volatility | `mtdata-cli forecast_volatility_estimate EURUSD --method ewma` |
| Parkinson (H/L) | `mtdata-cli forecast_volatility_estimate EURUSD --method parkinson` |
| GARCH | `mtdata-cli forecast_volatility_estimate EURUSD --method garch` |
| HAR-RV | `mtdata-cli forecast_volatility_estimate EURUSD --method har_rv --params "rv_timeframe=M5"` |

---

## See Also

- [GLOSSARY.md](../GLOSSARY.md) — Term definitions
- [FORECAST.md](../FORECAST.md) — Price forecasting
- [BARRIER_FUNCTIONS.md](../BARRIER_FUNCTIONS.md) — TP/SL probability analysis
- [REGIMES.md](REGIMES.md) — Regime detection
