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

`available_at_time` is the close of the last candle consumed by detection;
`detection_bar_open` identifies that candle's opening time. Daily, weekly, and
monthly availability follows broker-calendar boundaries. Direct Python callers
must supply `timeframe` or `df.attrs["timeframe"]`; without a known timeframe,
the detector reports the availability index and omits the availability time.

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

For harmonic and Elliott results, `available_at_index` is the right edge of the
requested window, reported as `available_at_index_basis:
"input_window_right_edge"`. That is the only index at which the result is
guaranteed reproducible: pivot prominence, min-distance spacing and (for
Elliott under the default `scale_mode="auto"`) the swing scale itself are all
measured across the whole window, so a later bar can add, remove or move a
pivot. Each row also carries `earliest_possible_index_estimate`, a heuristic
derived from per-pivot price confirmation, alongside
`earliest_possible_index_caveat`. Do not use the estimate for backtest entry
timing; re-run over a truncated window to establish genuine point-in-time
availability.

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
default, active levels are returned while broken and stale levels are hidden;
`--include-completed true` adds both, and `--config '{"include_stale_levels":
true}'` adds only the stale ones. Whatever is withheld is counted in
`broken_levels_hidden` and `stale_levels_hidden`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `candlestick` | Pattern type: all, candlestick, classic, harmonic, fractal, elliott |
| `--lookback` | 150 | Historical bars fetched for pattern analysis. `mode=all` requires at least 150 and caps each timeframe to roughly one year of history, so the weekly leg fetches far fewer bars than requested. Detectors additionally cap their own input window (`max_bars`, 1500 for classic and harmonic); when that binds, the response reports `analyzed_bars` and warns. |
| `--robust-only` | false | Restrict detection to a curated subset of established multi-bar candlestick types. This is a name preset, not a confidence threshold. Candlestick and all modes only. |
| `--whitelist` | — | Comma-separated list of specific patterns |
| `--min-strength` | 0.70 | Minimum OHLC-geometry and pattern-reliability strength score (0.0-1.0). Candlestick and all modes only; other modes reject a non-default value rather than ignoring it, and apply their own confidence rules via `--config`. |
| `--min-gap` | 3 | Minimum bars between returned candlestick detections. Candlestick and all modes only. Spacing is applied among detections that meet `--min-strength`. |
| `--top-k` | `3` | Candidate/collision budget and compact, summary, or standard row cap; it is not a global cap for `--detail full` |
| `--last-n-bars` | — | Candlestick-only recency window for returned detections; use it to bound a full scan |
| `--config` | — | Detector-specific overrides. Fractals support `left_bars`, `right_bars`, `breakout_basis`, `min_prominence_pct`, and `confidence_prominence_cap_pct`. Harmonics support `pattern_types`, `ratio_tolerance`, `min_confidence`, and pivot controls. |

Detector config is strict. Candlestick mode accepts
`use_volume_confirmation`, `volume_confirm_breakout_bars`,
`volume_confirm_lookback_bars`, `volume_confirm_min_ratio`,
`volume_confirm_bonus`, `volume_confirm_penalty`, `use_regime_context`,
`regime_alignment_bonus`, and `regime_countertrend_penalty`; other keys return
`unknown_config_key` instead of being ignored.

In `mode=all`, un-namespaced config keys apply to every detector that has the
field. Nest under a section name to retune one detector, for example
`--config '{"harmonic": {"min_confidence": 0.7}}'`. An unknown key inside a
section is reported as `section.key` against that section alone; an unknown
top-level key is reported against every detector that could have owned it.

### Denoising

Pattern detection smooths all four OHLC columns together, because pivot
geometry is read from the open, high, low and close as a set. If you omit
`denoise.columns` the tool substitutes `ohlc`; if you pass a column list that
lacks any of them the request is rejected rather than analyzing a series that
mixes smoothed and raw prices. When denoising fails, the response reports
`denoise_applied: false` with `denoise_error` and
`preprocessing_causality: "raw_prices"` — it never echoes the requested spec as
if it had been applied. Zero-phase denoising sets `denoise_lookahead_bias: true`
and rewrites every row's `status_basis`, because such results may repaint.
See [DENOISING.md](../DENOISING.md).

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

Most detectors run through the aggregate dispatcher, so a single non-finite OHLC
value can stop nearly all of them at once. When fewer detectors run than were
selected, the response adds a `detector_coverage` block with the expected and
evaluated counts, the names that did not run, any
`aggregate_dispatcher_error`, and any names the installed backend does not
provide. Treat "no patterns found" as inconclusive whenever that block is
present.

For candlesticks, `top_k` also resolves collisions when several detector types
fire on the same bar. Compact and summary use it as a preview budget, and
standard uses it as a returned-row cap. Full detail deliberately keeps every
surviving row and reports that scope in `top_k_contract` (candlestick mode
only); combine full detail with `--last-n-bars` when the complete row set must
remain small.

`--detail` trades away per-row diagnostics, never the fields that tell you not
to trust a result: `warnings`, `data_quality`, `candles` and `effective_window`
are present at every detail level, including `summary`.

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
| `metric` | euclidean | Initial distance: euclidean, cosine, correlation |
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
| `cosine` | Cosine distance after vector normalization |
| `correlation` | Correlation distance after centering and normalization |

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

`matrix_profile` and `mass` require `metric=euclidean` with `scale=zscore`.
Unsupported values and incompatible combinations are rejected rather than
silently replaced with another similarity algorithm.

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
