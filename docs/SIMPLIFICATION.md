# Data simplification (downsampling)

**Audience:** User

Return **fewer points** so a chart or export stays light. This drops rows (or sketches the shape). It does **not** change your broker history, and it is the wrong input for a serious backtest — use full-resolution candles there.

| Approach | What it changes |
|----------|-----------------|
| **Simplify** | *How many* points you get back ([LTTB](GLOSSARY.md#lttb-largest-triangle-three-buckets)) |
| **Denoise** | *The values* (smoothing) — see [DENOISING.md](DENOISING.md) |

**Related:** [CLI](CLI.md) · [Denoising](DENOISING.md) · [Glossary (LTTB)](GLOSSARY.md#lttb-largest-triangle-three-buckets)

---

## Quick Start

```bash
# Candles: default simplification (targets ~10% of --limit)
mtdata-cli data_fetch_candles EURUSD --timeframe M1 --limit 5000 --simplify

# Candles: choose an algorithm + target points
mtdata-cli data_fetch_candles EURUSD --timeframe M1 --limit 5000 --simplify lttb --simplify-params "points=500"

# Ticks: simplify only applies when returning raw rows
mtdata-cli data_fetch_ticks EURUSD --limit 20000 --simplify rdp --simplify-params "points=2000"
```

---

## How It Works (High Level)

When `simplify` is enabled, mtdata selects a subset of rows that approximates the original series. Responses typically include:

- `simplified: true`
- `simplify: {mode, method, points, original_rows, returned_rows, ...}`

`points` and `target_points` are the overall requested row target;
`returned_rows` is the actual result size. Multi-column tick selection can also
report `per_column_target`, an internal sampling budget that is combined and
refined to the overall target.

For candles, `candle_counts.excluded.simplification` records intentionally
omitted visualization rows separately from broker/window shortfalls.

---

## Key Parameters

You can pass a method name directly:
- `--simplify lttb`
- `--simplify rdp`
- `--simplify pla`
- `--simplify apca`

And add parameters with:
- `--simplify-params "key=value,key2=value2"`
- or JSON: `--simplify '{"method":"lttb","points":500}'`

Common sizing keys:
- `points` / `target_points` / `max_points`: target number of rows
- `ratio`: fraction of rows to keep (0..1)

Notes:
- For `data_fetch_candles`, if you enable `--simplify` without `points/ratio`, mtdata defaults to ~10% of `--limit`.
- Use `mode=select` to select existing rows (recommended for most charting use-cases).

---

## Algorithms

### LTTB (`method=lttb`)
**Largest-Triangle-Three-Buckets** downsampling preserves visual shape well for plotting.
mtdata uses `tsdownsample` when available and falls back to the built-in Python implementation otherwise. The full Python 3.14 package-index install path includes `tsdownsample>=0.1.5.1`; lean installs can add it directly with `pip install "tsdownsample>=0.1.5.1"`.

Recommended parameters:
- `points` (or `ratio`)

Example:
```bash
mtdata-cli data_fetch_candles EURUSD --limit 10000 --simplify lttb --simplify-params "points=800"
```

### RDP (`method=rdp`)
**Ramer–Douglas–Peucker** polyline simplification. You can control it via:
- `epsilon` (alias: `tolerance`) — error tolerance
- or `points` / `ratio` — mtdata auto-tunes `epsilon` to approximately hit the target

Examples:
```bash
# Direct tolerance
mtdata-cli data_fetch_candles EURUSD --limit 5000 --simplify rdp --simplify-params "epsilon=0.0005"

# Auto-tune epsilon to target points
mtdata-cli data_fetch_candles EURUSD --limit 5000 --simplify rdp --simplify-params "points=500"
```

### PLA (`method=pla`)
**Piecewise Linear Approximation** (segment-based). Control with:
- `max_error` — maximum deviation per segment
- or `segments` — fixed number of segments
- or `points` / `ratio` — select a fixed segment count toward the target

Tolerance requests enforce the pointwise deviation postcondition. Fixed-size
requests report `observed_max_error`; they do not claim that a tolerance was
auto-tuned.

Example:
```bash
mtdata-cli data_fetch_candles EURUSD --limit 5000 --simplify pla --simplify-params "segments=200"
```

### APCA (`method=apca`)
**Adaptive Piecewise Constant Approximation** (step-wise). Control with:
- `max_error`
- or `segments`
- or `points` / `ratio` (fixed segment count with measured error)

Example:
```bash
mtdata-cli data_fetch_candles EURUSD --limit 5000 --simplify apca --simplify-params "points=600"
```

---

## Caveats

- Simplification is meant for **visualization and UI performance**. For quantitative analysis (e.g., volatility estimation, backtests), use full-resolution data.
- In `mode=select`, mtdata returns existing rows; this can miss intra-bar extremes if you simplify OHLC data aggressively.
- In `mode=approximate`, OHLCV uses candle aggregation, while other numeric columns are segment means. Indicator-like columns are not recomputed and the response is marked `analysis_compatible=false`.

`mode=encode` with `schema=delta` derives its scale from the median non-zero
change unless `scale` is supplied explicitly, and reports the resolved scale in
metadata. `mode=symbolic` implements [SAX / PAA](GLOSSARY.md#sax--paa) with
optional z-normalization and fixed standard-normal breakpoints, so encodings
are comparable across series.

---

## See Also

- [DENOISING.md](DENOISING.md) — Signal denoising (complementary technique)
- [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) — Indicator reference
- [CLI.md](CLI.md) — Full command reference
