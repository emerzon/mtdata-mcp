# Temporal analysis

**Audience:** User

Does this market behave differently on **Monday vs Friday**, at the **London open**, or in **certain months**? `temporal_analyze` groups returns, volatility, and volume by calendar bucket so you can spot session effects and seasonality before you lock a strategy.

**Related:** [CLI](CLI.md) · [Glossary](GLOSSARY.md) · [Volatility](forecast/VOLATILITY.md) · [Diagnostics](TIME_SERIES_DIAGNOSTICS.md)

---

## Quick start

```bash
# Average returns by day of week
mtdata-cli temporal_analyze EURUSD --group-by dow --json

# Volatility by hour of day
mtdata-cli temporal_analyze EURUSD --group-by hour --lookback 2000 --json

# Monthly seasonality
mtdata-cli temporal_analyze EURUSD --timeframe D1 --group-by month --lookback 1000 --json
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | (required) | Trading symbol |
| `--timeframe` | `H1` | Candle timeframe |
| `--lookback` | auto | Bars to analyze when `--start`/`--end` are omitted. Auto-derived per timeframe (floor 200, cap 20,000). |
| `--start` | (optional) | Start date (ISO or flexible format) |
| `--end` | (optional) | End date (ISO or flexible format) |
| `--group-by` | `dow` | Grouping: `dow` (day of week), `hour`, `month`, `session` (Asia/London/overlap/NY/off), or `all` (all four breakdowns) |
| `--session-calendar` | `auto` | Session calendar for `session` grouping: `auto`, `fx`, `equity`, or `continuous_24_7`. |
| `--limit` / `--offset` | (optional) / `0` | For a single `--group-by`, page that row list. For `all`, they page **each** of the four breakdowns independently. Compact `groups` is the concatenation; `dimension_pagination` is the per-dimension cursor; `groups_analyzed` is the unpaged total. |
| `--day-of-week` | (optional) | Filter to a specific day (0–6 or name, e.g., `Mon`, `Friday`) |
| `--month` | (optional) | Filter to a specific month (1–12 or name, e.g., `Jan`, `September`) |
| `--time-range` | (optional) | Filter by time window `HH:MM-HH:MM` using a half-open interval `[start, end)` (wraps midnight, e.g., `22:00-02:00`) |
| `--timezone` | `CLIENT_TZ`, then `UTC` | IANA clock used by `--time-range` and hour/session grouping, such as `Europe/London` |
| `--return-mode` | `pct` | Return calculation: `pct` (percentage) or `log` (logarithmic) |
| `--return-basis` | `previous_close` | `previous_close` measures from the prior available close; `bar_open` measures each candle from its own open |
| `--min-bars` | auto for DOW | Exclude grouped rows below this sample count. Explicit values apply to every breakdown under `--group-by all`; automatic filtering applies to its DOW breakdown. |

`auto` uses both symbol syntax and the broker symbol path. Currency pairs,
metals, and broker-classified index/commodity CFDs use the near-24/5 FX session
buckets; recognizable crypto pairs use `continuous_24_7` (the same
Asia/London/NY windows, with leftover hours labeled `off_hours` as a 24/7
liquidity partition, not an exchange close); stock-like symbols use equity
sessions. Responses include the resolved `session_calendar` and
`session_calendar_source`. Set `fx`, `equity`, or `continuous_24_7`
explicitly to override inference.

---

## Grouping Modes

### Day of Week (`--group-by dow`)

Shows performance by weekday. Useful for detecting day-of-week effects.

```bash
mtdata-cli temporal_analyze EURUSD --group-by dow --lookback 2000 --json
```

**Example output (simplified):**
```
group  bars  avg_return_pct  volatility_pct  win_rate_pct  avg_volume
Mon     400   -0.012%     0.065%      48.2%     1250
Tue     400    0.008%     0.071%      51.0%     1380
Wed     400    0.015%     0.078%      52.5%     1420
Thu     400   -0.003%     0.074%      49.8%     1350
Fri     400    0.005%     0.062%      50.5%     1100
```

### Hour of Day (`--group-by hour`)

Shows performance by hour. Reveals session activity patterns.
D1, W1, and MN1 cannot group by hour or session; use H1 or M15 for those, and keep calendar bars for dow/month.

```bash
mtdata-cli temporal_analyze EURUSD --group-by hour --lookback 5000 --json
```

Use `--time-range` to focus on a specific session:
```bash
# London session hours
mtdata-cli temporal_analyze EURUSD --group-by hour --time-range "08:00-16:00" --timezone Europe/London --json

# New York session (DST-aware)
mtdata-cli temporal_analyze EURUSD --group-by hour --time-range "09:30-16:00" --timezone America/New_York --json
```

`--time-range` is applied to candle open times in `--timezone`. The start time
is included and the end time is excluded, so `08:00-16:00` keeps bars stamped
`08:00` through `15:59...` and excludes a bar stamped exactly `16:00`. Supplying
the IANA timezone on the command makes the same window reproducible across
machines and follows daylight-saving changes automatically.

### Calendar Month (`--group-by month`)

Shows seasonal effects across months. Best with daily data and a long history.

```bash
mtdata-cli temporal_analyze EURUSD --timeframe D1 --group-by month --lookback 2000 --json
```

Daily, weekly, and monthly bars use the broker trading-session date for equity
weekday and month labels. The timestamp still identifies the bar-open instant,
but changing the display timezone cannot move a daily equity session into the
prior weekday or month.

Monthly rows distinguish raw bars from repeated seasonal evidence:
`distinct_period_instances` counts the number of calendar years contributing
that month, `complete_period_instances` excludes partial edge months, and
`partial_bucket` flags rows containing an incomplete edge instance. A month
needs at least two distinct yearly instances and 30 bars before it can appear
as `best`; shorter windows keep the descriptive rows but return a sample warning
instead of naming a one-off fragment as a seasonal winner.

### All Grouping Dimensions (`--group-by all`)

Returns day-of-week, hour, month, and session breakdowns in one call. With
`--detail standard` or `--detail full`, the response also includes an
`overall` block containing aggregate statistics across all analyzed bars.
An explicit `--min-bars` floor is applied independently to each breakdown;
excluded rows include their dimension in `excluded_groups`. `--limit` and
`--offset` page each of the four breakdowns independently; compact `groups`
concatenates those pages, `dimension_pagination` is the per-dimension cursor,
and `groups_analyzed` is the unpaged total. On D1/W1/MN1, hour and session
are omitted with a warning; day-of-week and month remain.

```bash
mtdata-cli temporal_analyze EURUSD --group-by all --detail standard --json
```

---

## Output Fields

Each group includes these statistics:

| Field | Description |
|-------|-------------|
| `group` | Group label (e.g., `Mon`, `14:00`, `Jan`) |
| `group_key` | Numeric group identifier |
| `bars` | Number of bars in group |
| `returns` | Count of return observations |
| `avg_return_pct` | Average return in percent (1.0 = 1%) |
| `median_return_pct` | Median return in percent (1.0 = 1%) |
| `volatility` | Standard deviation of returns |
| `avg_abs_return_pct` | Average absolute return in percentage points |
| `volatility_pct` | Per-bar return standard deviation in percentage points |
| `win_rate` | Percentage of bars with positive return |
| `avg_range` | Average high-low range |
| `avg_range_pct` | Average range as percentage of close |
| `avg_volume` | Average volume (real or tick) |
| `distinct_period_instances` | For month grouping, number of separate yearly occurrences represented |
| `complete_period_instances` | Month occurrences not cut by the analysis-window edges |
| `partial_bucket` | Whether the row includes an incomplete edge-month occurrence |

At `standard` and `full` detail, the top-level response also includes `overall`
(sample-wide summary statistics) and `volume_source` (whether `real_volume` or
`tick_volume` was used). Compact detail focuses on grouped rows and omits the
`overall` block.

Every detail mode states `return_basis`, `return_definition`, and
`session_gap_policy`. The default `previous_close` basis assigns an overnight
or market-closure gap to the destination bar. Use `--return-basis bar_open` for
an opening-hours study that should measure only the movement inside each bar.

---

## Filtering

Combine grouping with filters to drill down:

```bash
# Only Mondays, grouped by hour
mtdata-cli temporal_analyze EURUSD --group-by hour --day-of-week Mon --json

# Only January, grouped by day of week
mtdata-cli temporal_analyze EURUSD --timeframe D1 --group-by dow --month Jan --lookback 2000 --json

# London session hours, grouped by day of week
mtdata-cli temporal_analyze EURUSD --group-by dow --time-range "08:00-16:00" --timezone Europe/London --json
```

---

## Practical Applications

### Find Best Trading Days
```bash
mtdata-cli temporal_analyze EURUSD --group-by dow --lookback 5000 --json
# Look for days with highest win_rate_pct and positive avg_return_pct
```

### Find Active Trading Hours
```bash
mtdata-cli temporal_analyze EURUSD --group-by hour --lookback 5000 --json
# Look for hours with highest avg_range_pct and avg_volume
```

### Seasonal Patterns
```bash
mtdata-cli temporal_analyze SPX500 --timeframe D1 --group-by month --lookback 3000 --json
# Compare monthly avg_return_pct and volatility_pct
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Day-of-week stats | `mtdata-cli temporal_analyze EURUSD --group-by dow` |
| Hourly stats | `mtdata-cli temporal_analyze EURUSD --group-by hour` |
| Monthly seasonality | `mtdata-cli temporal_analyze EURUSD --timeframe D1 --group-by month` |
| All grouping dimensions | `mtdata-cli temporal_analyze EURUSD --group-by all` |
| Sample-wide summary | `mtdata-cli temporal_analyze EURUSD --group-by all --detail standard` (read `overall`) |
| Filter to Mondays | `mtdata-cli temporal_analyze EURUSD --group-by hour --day-of-week Mon` |
| London session only | `mtdata-cli temporal_analyze EURUSD --group-by hour --time-range "08:00-16:00" --timezone Europe/London` |

---

## See Also

- [CLI.md](CLI.md) — Command usage
- [forecast/VOLATILITY.md](forecast/VOLATILITY.md) — Volatility estimation
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
