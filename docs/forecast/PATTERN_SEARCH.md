# Pattern detection and similarity search

**Audience:** User

Two related ideas:

1. **Pattern detection** — flag known candlestick and chart shapes
2. **Analog / similarity search** — find historical windows that look like *now*, then study what happened next

Use patterns as **context**, not automatic entry rules. Confirm with regime, volatility, and risk tools.

**Related:** [Forecasting](../FORECAST.md) · [Indicators](../TECHNICAL_INDICATORS.md) · [Glossary](../GLOSSARY.md) · [Levels](../LEVELS.md)

---

## Pattern detection (`patterns_detect`)

Identifies visual patterns traders often watch for structure and timing context.

`mode=all` runs five detector families for each selected timeframe. If only
some detector/timeframe pairs fail, the response sets `partial_failure: true`
and includes requested, succeeded, and failed counts plus `failed_items`.
Partial findings remain usable by default; pass `--allow-partial false` for a
strict scan that returns `success: false` on any omission.

### Candlestick Patterns

Single or multi-bar patterns with historical significance.

```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode candlestick --lookback 200
```

**Output:**
```json
{
  "success": true,
  "symbol": "EURUSD",
  "timeframe": "H1",
  "lookback": 200,
  "mode": "candlestick",
  "n_patterns": 29,
  "patterns_shown": 3,
  "pattern_status": "bullish",
  "pattern_confidence": 0.84,
  "top_patterns": [
    {
      "name": "Bullish Engulfing",
      "direction": "bullish",
      "match_score": 0.84,
      "time": "2025-12-22 10:00"
    }
  ]
}
```

Compact output ranks the recent candidates in `top_patterns` and reports the
aggregate `pattern_status`. The exact rows and optional status fields depend on
the detector result. Use `--detail full` when you need every surviving pattern
row rather than the compact preview.

**Filter to the curated robust pattern subset:**
```bash
mtdata-cli patterns_detect EURUSD --mode candlestick --robust-only true
```

**Common patterns detected:**
| Pattern | Meaning |
|---------|---------|
| **Engulfing** | Current candle completely covers previous (reversal) |
| **Doji** | Open ≈ Close (indecision) |
| **Hammer/Hanging Man** | Small body, long lower wick |
| **Inside** | Current bar inside previous bar's range |
| **Harami** | Small body inside previous large body |
| **Morning/Evening Star** | Three-bar reversal pattern |

### Classic Chart Patterns

Larger geometric patterns formed over multiple bars.

```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode classic --lookback 500
```

The default classic detector evaluates patterns at the right edge of the input
window and excludes pivots that do not yet have the configured right-hand
confirmation gap. Results report `available_at_index`, `available_at_time`,
`pivot_confirmation_bars`, and `detection_scope`. Set
`config.scan_historical=true` to run the slower causal prefix scan when you need
older patterns labeled at their first detection window.

**Patterns detected:**
| Pattern | Description |
|---------|-------------|
| **Head and Shoulders** | Three peaks, middle highest (bearish reversal) |
| **Inverse H&S** | Three troughs, middle lowest (bullish reversal) |
| **Double Top/Bottom** | Two peaks/troughs at similar level |
| **Triangle** | Converging trendlines (breakout setup) |
| **Wedge** | Rising or falling wedge |
| **Rectangle** | Horizontal consolidation |

### Harmonic Patterns

Fibonacci-ratio patterns built from alternating pivot legs.

The harmonic detector reports both forming and completed XABCD/ABCD candidates.
Those lifecycle states are both primary harmonic findings and are therefore
returned even when the shared `--include-completed` flag is false. That flag
continues to control historical visibility for classic, Elliott, and fractal
modes.

```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode harmonic --lookback 500
```

**Patterns detected:**
| Pattern | Description |
|---------|-------------|
| **ABCD** | Four-point measured-move completion |
| **Gartley** | XABCD retracement pattern with 0.786 XA completion |
| **Bat / Alternate Bat** | XABCD patterns with deeper D-point completion |
| **Butterfly** | XABCD extension beyond X |
| **Crab / Deep Crab** | Extended XABCD completion patterns |
| **Shark, Cypher, 5-0** | Additional Fibonacci-ratio reversal structures |

**Useful harmonic config:**
```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode harmonic --config "pattern_types=gartley,bat,crab ratio_tolerance=0.06 min_confidence=0.45"
```

**Common output fields:**
| Field | Meaning |
|-------|---------|
| `entry_price` | D-point completion price |
| `target_price`, `target_price_1`, `target_price_2` | CD retracement targets |
| `invalidation_price` | Pattern invalidation level with configured buffer |
| `lifecycle` | Post-completion state vs later OHLC: `forming`, `active`, `target_reached`, `expired`, or `historical` |
| `bias_scope` | `current` only while a completed pattern is still an active setup |
| `price_levels` | Entry, targets, invalidation, and PRZ levels |
| `details.ratios` | Measured Fibonacci ratios for the candidate |

A completed bullish harmonic is not a `long_setup` after price has already
reached the first target or broken invalidation. Compact output still keeps
`target_price` and `invalidation_price` for an active current setup.

### Fractal Patterns

Bill Williams-style bullish and bearish fractal levels with confirmation and breakout context.

```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode fractal --lookback 300
```

**Useful fractal config:**
```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode fractal --config "left_bars=2 right_bars=2 breakout_basis=high_low"
```

**Common output fields:**
| Field | Meaning |
|-------|---------|
| `level_price` | Confirmed fractal high/low level |
| `status` | Level lifecycle: `active` or `broken` (not pattern completion) |
| `level_state` | `active` when unbroken, `broken` after price breaches the level |
| `confirmation_date` | When the fractal became knowable after the right-side bars closed |
| `breakout_direction` | Direction of the later level break (`bullish` or `bearish`) |
| `breakout_date` | When the breakout occurred, if any |

Active levels are informational support/resistance context and have neutral
signal bias. A broken level takes the breakout direction as its bias. By
default, active levels are returned and historical broken levels are hidden;
use `--include-completed true` to include broken levels as well.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `candlestick` | Pattern type: all, candlestick, classic, harmonic, fractal, elliott |
| `--lookback` | 150 | Historical bars fetched for pattern analysis |
| `--robust-only` | false | Restrict detection to a curated subset of established multi-bar candlestick types. This is a name preset, not a confidence threshold. |
| `--whitelist` | — | Comma-separated list of specific patterns |
| `--min-strength` | 0.70 | Minimum OHLC-geometry and pattern-reliability strength score (0.0-1.0) |
| `--top-k` | `3` | Candidate/collision budget and compact, summary, or standard row cap; it is not a global cap for `--detail full` |
| `--last-n-bars` | — | Candlestick-only recency window for returned detections; use it to bound a full scan |
| `--config` | — | Detector-specific overrides. Fractals support `left_bars`, `right_bars`, `breakout_basis`, `min_prominence_pct`, and `confidence_prominence_cap_pct`. Harmonics support `pattern_types`, `ratio_tolerance`, `min_confidence`, and pivot controls. |

Detector config is strict. Candlestick mode accepts
`use_volume_confirmation`, `volume_confirm_breakout_bars`,
`volume_confirm_lookback_bars`, `volume_confirm_min_ratio`,
`volume_confirm_bonus`, `volume_confirm_penalty`, `use_regime_context`,
`regime_alignment_bonus`, and `regime_countertrend_penalty`; other keys return
`unknown_config_key` instead of being ignored.

Pattern names listed in this guide describe detector coverage, not a promise
that every pattern is returned at the default threshold. `robust_only=true`
restricts which candlestick detectors run based on pattern name, while
`min_strength` independently filters their conviction scores. Lower-strength
and deprioritized formations such as many dojis may be absent by default.
The score uses body/range geometry, directional close location, range expansion,
pattern span, and the curated reliability tier. Raw detector magnitudes remain
in `raw_signal` but do not alter strength because pandas-ta backends use
different native signal scales.

Named whitelists are resolved against both individual backend methods and an
aggregate pattern dispatcher. Full results report `requested_detectors`,
`detectors_evaluated`, and any `unsupported_detectors`, so a zero-hit detector
is distinguishable from one the active backend could not run.

For candlesticks, `top_k` also resolves collisions when several detector types
fire on the same bar. Compact and summary use it as a preview budget, and
standard uses it as a returned-row cap. Full detail deliberately keeps every
surviving row and reports that scope in `top_k_contract`; combine full detail
with `--last-n-bars` when the complete row set must remain small.

### Filtering Patterns

Classic detector config values `max_pattern_age_bars` and
`max_pattern_span_bars` bound all detector results, including completed
patterns. `--include-completed true` adds completed structures that remain
inside those detection bounds; it does not request an unbounded historical
scan.

**By name:**
```bash
mtdata-cli patterns_detect EURUSD --mode candlestick --whitelist "ENGULFING,HAMMER,DOJI"
```

**By confidence:**
```bash
mtdata-cli patterns_detect EURUSD --mode candlestick --min-strength 0.85
```

---

## Analog Forecasting

Finds historical windows that "look like" the current market and uses them to predict what happens next.

### Concept

"History doesn't repeat, but it rhymes."

1. Take the last N bars (the "query window")
2. Search through historical data for similar patterns
3. Look at what happened after those patterns
4. Average/aggregate those future moves into a forecast

### Basic Usage

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method analog --params "window_size=64 top_k=20"
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 64 | Length of pattern to match |
| `search_depth` | 5000 | How far back to search |
| `top_k` | 20 | Number of similar patterns to use |
| `metric` | euclidean | Distance metric |
| `scale` | zscore | Normalization: zscore, minmax, none |
| `refine_metric` | dtw | Refinement: dtw, softdtw, affine, ncc, none |
| `search_engine` | ckdtree | Search algorithm |

### Scaling Options

| Scale | Description | When to Use |
|-------|-------------|-------------|
| `zscore` | Standardize to mean=0, std=1 | Default, handles varying volatility |
| `minmax` | Scale to [0,1] | When range matters more than volatility |
| `none` | No scaling | When absolute levels matter |

### Distance Metrics

The `metric` parameter controls the initial candidate search (must be fast — Euclidean-family):

| Metric | Description |
|--------|-------------|
| `euclidean` | Standard L2 distance (default, fastest) |

The `refine_metric` parameter re-ranks candidates using a slower, more precise metric:

| Refine Metric | Description |
|---------------|-------------|
| `dtw` | Dynamic Time Warping (handles time warping) |
| `softdtw` | Differentiable DTW |
| `ncc` | Normalized cross-correlation |
| `affine` | Affine-invariant distance |

**Example with refinement:**
```bash
mtdata-cli forecast_generate EURUSD --horizon 12 --method analog --params "window_size=64 metric=euclidean refine_metric=dtw"
```

This first finds candidates with fast Euclidean distance, then refines ranking using DTW.

### Search Engines

| Engine | Description |
|--------|-------------|
| `ckdtree` | Scipy KD-tree (default, fast) |
| `hnsw` | Approximate nearest neighbor (scalable `hnswlib` backend; included in `[all]`, or add to lean installs through the source-build path in [../SETUP.md](../SETUP.md)) |
| `matrix_profile` | STUMPY-based (specialized for time series) |
| `mass` | Mueen's MASS algorithm |

---

## Practical Applications

### Pattern-Based Entry Filter

Use pattern detection as a confirmation signal:

```bash
# Check for reversal patterns at support
mtdata-cli patterns_detect EURUSD --mode candlestick --robust-only true

# If bullish pattern detected at support level → consider long entry
```

### Analog-Based Targets

Use analog forecasts to set price targets:

```bash
# Find similar historical patterns
mtdata-cli forecast_generate EURUSD --method analog --params "window_size=64 top_k=20" --json

# Use forecast percentiles for TP levels
```

### Combining with Technical Analysis

```bash
# Get patterns and indicators together
mtdata-cli data_fetch_candles EURUSD --limit 200 --indicators "ema(20),rsi(14)"

# Then check patterns
mtdata-cli patterns_detect EURUSD --mode candlestick --robust-only true

# Look for pattern + indicator confluence
```

---

## Interpreting Results

### Pattern Detection

```
data[5]{time,pattern}:
    "2025-12-19 05:00",Bearish ENGULFING
    "2025-12-22 10:00",Bearish ENGULFING
```

**Interpretation:**
- Pattern occurred at specific times
- "Bearish" suggests potential downward move
- Combine with other analysis (support/resistance, indicators)

### Analog Forecast

```json
{
  "forecast": [1.1755, 1.1758, 1.1762, ...],
  "lower": [1.1740, 1.1738, ...],
  "upper": [1.1770, 1.1778, ...],
  "analogs_found": 20
}
```

**Interpretation:**
- `forecast`: Median of analog outcomes
- `lower`/`upper`: Spread of analog outcomes
- Wide spread = diverse outcomes in similar historical patterns

---

## Quick Reference

| Task | Command |
|------|---------|
| Candlestick patterns | `mtdata-cli patterns_detect EURUSD --mode candlestick` |
| Curated candlestick subset | `mtdata-cli patterns_detect EURUSD --mode candlestick --robust-only true` |
| Chart patterns | `mtdata-cli patterns_detect EURUSD --mode classic` |
| Harmonic patterns | `mtdata-cli patterns_detect EURUSD --mode harmonic` |
| Fractal levels and breakouts | `mtdata-cli patterns_detect EURUSD --mode fractal` |
| Analog forecast | `mtdata-cli forecast_generate EURUSD --method analog --params "window_size=64 top_k=20"` |
| Analog with DTW | `mtdata-cli forecast_generate EURUSD --method analog --params "refine_metric=dtw"` |

---

## See Also

- [../FORECAST.md](../FORECAST.md) — Price forecasting overview
- [../TECHNICAL_INDICATORS.md](../TECHNICAL_INDICATORS.md) — Technical indicators
- [../GLOSSARY.md](../GLOSSARY.md) — Term definitions
