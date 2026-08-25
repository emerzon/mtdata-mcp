# Price levels

**Audience:** User

Find **prices the market may care about** — formula pivots, retested support/resistance, multi-method confluence, and volume-profile value areas. All of these tools are **read-only** and exploratory: they map structure, they do not issue trade signals.

**Dense terms:** [Pivot points](GLOSSARY.md#pivot-points) · [Support and resistance](GLOSSARY.md#support-and-resistance) · [Confluence](GLOSSARY.md#confluence) · [Volume profile (POC/VAH/VAL)](GLOSSARY.md#volume-profile-poc-vah-val)

**Related:** [Sample trade](SAMPLE-TRADE.md) · [CLI](CLI.md) · [Glossary](GLOSSARY.md)

## Choosing a tool

| Tool | Source of levels | Use when you want… |
|------|------------------|--------------------|
| `pivot_compute_points` | Formula pivots from the last completed OHLC bar | Classic floor-trader PP/R/S targets for the session |
| `support_resistance_levels` | Historical retests and reactions | Data-driven levels the market actually respected |
| `confluence_levels` | Pivots + S/R + Fibonacci + volume profile combined | High-probability zones where several methods agree |
| `volume_profile_levels` | Traded volume distribution by price | POC / value area to frame fair-value and acceptance |

All four accept `--detail compact|standard|full`. Compact returns the nearest,
most actionable levels; `--detail full` returns the raw diagnostic
payload.

---

## `pivot_compute_points`

Computes pivot levels from the **last completed bar** on `timeframe`.

```bash
mtdata-cli pivot_compute_points EURUSD --timeframe D1 --json
mtdata-cli pivot_compute_points EURUSD --timeframe D1 --method camarilla --json
```

- `timeframe` defaults to `D1` because daily pivots are the common floor-trader convention.
- `method` selects one of `classic`, `fibonacci`, `camarilla`, `woodie`, `demark`.
  Omit it to return every method at `--detail standard`/`full`; `--detail compact`
  returns the classic pivots only.
- DeMark pivots depend on whether the bar closed above, below, or at its open.
  Its R1/S1 formulas are canonical; the returned `PP=X/4` is a common retail
  platform extension and is labeled `pivot_convention=retail_x_over_4_extension`.

Use `support_resistance_levels` for complementary data-driven levels.

---

## `support_resistance_levels`

Detects support/resistance levels around the current price from historical structure.

```bash
mtdata-cli support_resistance_levels EURUSD --timeframe H1 --lookback 200 --json
mtdata-cli support_resistance_levels EURUSD --timeframe auto --detail standard --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeframe` | `H1` | Bar timeframe, or `auto` to merge levels from M15, H1, H4, and D1. |
| `lookback` | `200` | Maximum bars used to detect levels (after any `start`/`end` window). |
| `start` / `end` | — | Optional time window (e.g. `"1 month ago"`, `"now"`). |
| `tolerance_pct` | `0.15` | Price-cluster tolerance in percentage points (`0.15` = 0.15%) when grouping touches into one level. |
| `min_touches` | `2` | Minimum tests before a price qualifies as a level. |
| `max_levels` | `4` | Maximum levels to return. |
| `max_distance_pct` | `5.0` | Keep levels within this % of current price; pass `none`/`null` (CLI) or `None` (Python) for all. |
| `volume_weighting` | `off` | Set `auto` to weight touches by traded volume. |
| `reaction_bars` | `6` | Bars after a test used to measure bounce strength. |
| `adx_period` | `14` | ADX period for pre-test trend-strength weighting. |
| `decay_half_life_bars` | — | Optional exponential time decay so recent tests matter more. |

**Score** combines repeated tests of a level, bounce strength after each test
(normalized by ATR), pre-test ADX trend strength, exponential time decay, and
ATR-filtered Fibonacci retracement/extension levels from the most relevant swing.

**Output:** ranked `levels` with `score`, a `type` reflecting current price geometry
(support below price, resistance above), and a `dominant_source` showing whether the
historical tests mostly behaved as support or resistance. `--detail standard` adds
Fibonacci swing levels; `--detail full` returns the raw
diagnostic payload.

Use `confluence_levels` to cluster these levels with pivots and Fibonacci into
scored consensus zones.

---

## `confluence_levels`

Finds nearby high-probability zones where **multiple level methods agree**. It
combines formula pivots, touch-derived support/resistance, Fibonacci swing levels,
and optional volume-profile levels.

```bash
mtdata-cli confluence_levels EURUSD --pivot-timeframe D1 --sr-timeframe auto --json
mtdata-cli confluence_levels EURUSD --min-source-families 2 --max-levels 5 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pivot_timeframe` | `D1` | Timeframe for the formula pivot inputs. |
| `sr_timeframe` | `auto` | Timeframe for support/resistance inputs (`auto` merges M15–D1). |
| `lookback` | `200` | Bars of history for the S/R component. |
| `tolerance_pct` | `0.15` | Clustering tolerance in percentage points (`0.15` = 0.15%) for merging levels into a zone. |
| `tolerance_points` | — | Broker-point count; overrides `tolerance_pct`. On five-digit EURUSD, `10 × point(0.00001) = 0.0001` price width. |
| `min_touches` | `2` | Minimum touches for the S/R component. |
| `max_levels` | `5` | Maximum confluence zones to return. |
| `max_distance_pct` | `5.0` | Keep zones within this % of current price; pass `none`/`null` (CLI) or `None` (Python) for all. |
| `min_source_families` | `1` | Require this many independent families per zone (use `2` for stricter confluence). |
| `pivot_method` | — | Restrict pivots to one method (`classic`, `fibonacci`, …). |
| `volume_weighting` | `off` | Set `auto` to volume-weight the S/R component. |
| `volume_profile_source` | `off` | Optional volume-profile input. Use `auto`, `ticks`, or `m1_bars` to enable it. |
| `volume_profile_max_tick_window_days` | `1` | Cap the raw-tick window, matching `volume_profile_levels`; longer `auto` windows use the disclosed M1 approximation. |
| `volume_profile_max_ticks` | `50000` | Cap the number of ticks pulled for volume profile. |
| `volume_profile_max_m1_bars` | `20000` | Cap M1 approximation work when volume profile is enabled. |

Single-family clusters are returned but score lower than multi-family confluence.
Set `min_source_families=2` to require independent agreement.
Volume-profile computation is opt-in because it can require a much larger M1
history request than the other level families.

**Output:** `clusters` sorted by `score`, each with a `price`, the contributing
`source_families` and `sources`, and `distance_pct` from current price.

Use `support_resistance_levels` for the structural S/R component on its own.

---

## `volume_profile_levels`

Computes the volume-profile Point of Control (POC), Value Area High (VAH), and Value
Area Low (VAL) from bounded raw ticks or an M1-bar approximation.

```bash
# Window by calendar range
mtdata-cli volume_profile_levels EURUSD --start "1 day ago" --end "now" --source auto --price-source mid --bucket-points 10 --json

# Window by lookback on a timeframe
mtdata-cli volume_profile_levels EURUSD --timeframe H1 --lookback 168 --source auto --bucket-points 10 --json
```

Choose one window mode. A calendar `start` cannot be combined with
`timeframe` or `lookback`; `end` may anchor either mode.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start` / `end` | — | Calendar window for the profile. |
| `timeframe` | — | When set without `lookback`, the window defaults to the latest 200 completed bars. |
| `lookback` | — | Completed `timeframe` bars to include. The window follows actual market bars, so closures and weekend gaps do not consume the lookback. |
| `source` | `auto` | `auto` uses bounded ticks for short windows and M1 bars for larger ones; force with `ticks` or `m1_bars`. |
| `price_source` | `mid` | Price used per tick: `mid`, `last`, `bid`, or `ask`. M1 input cannot preserve quote sides and reports `ohlc_proxy` as the effective source. |
| `volume_source` | `auto` | `auto`, candle `real_volume`/`tick_volume`, tick-snapshot `volume_real`/`volume`, or `tick_count`. Snapshot volume is counted only on MT5 trade-change flags. |
| `bucket_size` / `bucket_points` / `bucket_count` | — | Choose exactly one price-bucket control (absolute size, points, or a target bucket count). |
| `max_buckets` | `120` | Upper bound on buckets. |
| `value_area_pct` | `70.0` | Percentage of volume that defines the value area (`70` = 70%, the standard setting). |
| `reference_price` | — | Anchor for distance calculations (defaults to current price). |
| `max_tick_window_days` | `1` | Cap the tick window pulled. Natural relative windows within one second of the boundary count as in-budget; larger `auto` windows use M1 bars. |
| `max_ticks` | `50000` | Cap the number of ticks pulled. |
| `max_m1_bars` | `20000` | Cap the M1 bars pulled in approximation mode. |

**Output:** `poc`, `vah`, `val`, and a `value_area` summary. Timeframe-derived
queries also include `bar_window` with the resolved first open and last close.
`profile_source`, `source_decision`, and `volume_profile_accuracy` identify
whether `auto` used ticks or the M1 approximation and why. The top-level
`source` remains the standard structured MT5 broker-provenance object.
For latest timeframe-derived profiles, `data_stale` compares the completed
window endpoint with one requested bar period (and exposes that threshold as
`stale_after_seconds`); non-timeframe windows retain the five-minute
observation-age threshold.
`--detail full` adds the full `levels`
histogram. To request a week of ticks, explicitly set
`--max-tick-window-days 7` and an appropriate `--max-ticks` cap.

---

## Typical Workflow

1. `pivot_compute_points` — mark today's formula pivots and R/S targets.
2. `support_resistance_levels` — find the levels price has actually respected.
3. `volume_profile_levels` — frame fair value (POC) and acceptance (value area).
4. `confluence_levels` — surface zones where the methods above overlap, then trade
   plans around the highest-scoring, multi-family zones.

## Caveats

- Levels are descriptive analytics, not predictions or signals. Always confirm with
  price action and risk controls.
- Volume profile on FX usually falls back to tick count; treat `real_volume` as available
  only when your broker provides it.
- `auto` timeframe and `auto` source trade some precision for robustness; pin them
  explicitly for reproducible research.

## See Also

- [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) — Indicator reference
- [forecast/PATTERN_SEARCH.md](forecast/PATTERN_SEARCH.md) — Pattern detection (can opt in to volume-structure confluence)
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
