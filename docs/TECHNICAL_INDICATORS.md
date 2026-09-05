# Technical indicators

**Audience:** User

Turn raw candles into **trend, momentum, volatility, and volume** context. mtdata ships **193** indicators you can attach when fetching candles or explore via the catalog tools.

Indicators are decision support — combine them with structure, volatility, and risk tools rather than treating any single reading as a trade by itself.

**Related:** [CLI](CLI.md) · [Denoising](DENOISING.md) · [Forecasting](FORECAST.md) · [Glossary](GLOSSARY.md)

---

## Quick start

**List available indicators:**
```bash
mtdata-cli indicators_list --limit 20
```

**Filter by category:**
```bash
mtdata-cli indicators_list --category momentum
mtdata-cli indicators_list --category trend
mtdata-cli indicators_list --category volatility
```

`--trading-style` filters broad discovery tags, not measured suitability or
historical performance. The response reports how many matches use curated
indicator tags versus category-level heuristics; full-detail rows expose the
same basis under `trading_context.trading_styles_basis`.

**Get indicator details:**
```bash
mtdata-cli indicators_describe rsi --json
mtdata-cli indicators_describe macd --json
```

---

## Using Indicators

### With Candle Data

Add indicators directly when fetching candles:
```powershell
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 200 --indicators "ema(20),ema(50),rsi(14),macd(12,26,9)"
```

**Syntax:**
- `indicator_name(param1,param2,...)` — With parameters
- `indicator_name` — Uses defaults
- Comma-separated list for multiple indicators

### Output Columns

Indicators add new columns to the output:
```
time,open,high,low,close,tick_volume,ema_20,ema_50,rsi_14,macd_12_26_9,macd_h_12_26_9,macd_s_12_26_9
```

Candle responses normalize indicator columns to lowercase
`indicator_param1_param2` (the same names `indicators_describe` reports), and
denoise `columns=` must use those names. Forecast feature diagnostics preserve
the indicator backend's actual names, which can be uppercase; read
`feature_usage.selected_columns` instead of assuming casing.

---

## Indicator Categories

> The tables below highlight commonly used indicators. The engine exposes **193 indicators** from `pandas_ta_classic.Category` at runtime, so the authoritative, environment-specific list comes from `mtdata-cli indicators_list` (optionally `--category <name>`). Use canonical names only (for example `bbands`, not historical nicknames like `bb`/`boll`).

### Trend / Overlap

Show direction and dynamic support/resistance levels.

| Indicator | Description | Example |
|-----------|-------------|---------|
| `ema` | Exponential Moving Average (default: 10) | `ema(20)` |
| `sma` | Simple Moving Average (default: 10) | `sma(50)` |
| `dema` | Double EMA | `dema(20)` |
| `tema` | Triple EMA | `tema(20)` |
| `wma` | Weighted Moving Average | `wma(20)` |
| `kama` | Kaufman Adaptive MA | `kama(10,2,30)` |
| `vwap` | Volume Weighted Avg Price | `vwap` |
| `bbands` | Bollinger Bands | `bbands(20,2)` |
| `kc` | Keltner Channels | `kc(20,2)` |

**Usage example:**
```powershell
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 100 --indicators "ema(20),ema(50),bbands(20,2)"
```

**Interpretation:**
- Price above EMA → Bullish bias
- Price between Bollinger Bands → Normal volatility
- Price touches upper/lower band → Potential reversal zone

### Momentum

Measure speed and strength of price changes.

| Indicator | Description | Example |
|-----------|-------------|---------|
| `rsi` | Relative Strength Index (default: 14) | `rsi(14)` |
| `macd` | Moving Average Convergence Divergence | `macd(12,26,9)` |
| `stoch` | Stochastic Oscillator | `stoch(14,3,3)` |
| `cci` | Commodity Channel Index | `cci(20)` |
| `willr` | Williams %R | `willr(14)` |
| `roc` | Rate of Change | `roc(10)` |
| `mom` | Momentum | `mom(10)` |
| `ao` | Awesome Oscillator | `ao` |

**RSI interpretation:**
- RSI > 70: Overbought (potential sell)
- RSI < 30: Oversold (potential buy)
- RSI = 50: Neutral

**MACD interpretation:**
- MACD crosses above Signal: Bullish momentum
- MACD crosses below Signal: Bearish momentum
- Histogram expanding: Momentum strengthening

### Volatility

Measure price movement magnitude.

| Indicator | Description | Example |
|-----------|-------------|---------|
| `atr` | Average True Range (default: 14) | `atr(14)` |
| `natr` | Normalized ATR (runtime default can vary; specify the period) | `natr(14)` |
| `bbands` | Bollinger Bands width | `bbands(20,2)` |
| `kc` | Keltner Channels | `kc(20,2)` |
| `donchian` | Donchian Channels | `donchian(20)` |

**ATR usage:**
- Set stop-loss: `SL = Entry ± (2 × ATR)`
- Compare volatility across timeframes
- Position sizing: Smaller size when ATR is high

### Volume

Analyze trading activity and participation.

| Indicator | Description | Example |
|-----------|-------------|---------|
| `obv` | On-Balance Volume | `obv` |
| `ad` | Accumulation/Distribution | `ad` |
| `adosc` | Chaikin A/D Oscillator | `adosc` |
| `mfi` | Money Flow Index | `mfi(14)` |
| `vwap` | Volume Weighted Avg Price | `vwap` |

**Note:** Volume indicators are most useful for instruments with reliable volume data (equities, futures). Forex volume is typically indicative only.

### Additional Categories

The indicator engine (via pandas_ta) supports additional categories beyond the four above:
- **candles** — candlestick pattern indicators (e.g. `cdl_doji`, `cdl_inside`)
- **performance** — return and cumulative performance metrics
- **statistics** — statistical measures (e.g. `zscore`, `variance`, `kurtosis`)

Use `mtdata-cli indicators_list --category <name>` to explore them.

---

## Denoising Indicators

Smooth noisy indicator outputs to reduce false signals:

**Smooth RSI after calculation:**
```powershell
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 200 --indicators "rsi(14)" --denoise ema --denoise-params "columns=rsi_14,when=post_ti,alpha=0.3"
```

**Smooth price before calculating indicators:**
```powershell
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 200 --indicators "rsi(14)" --denoise ema --denoise-params "columns=close,when=pre_ti,alpha=0.2"
```

See [DENOISING.md](DENOISING.md) for more options.

---

## Common Indicator Combinations

### Trend + Momentum
```bash
--indicators "ema(20),ema(50),rsi(14)"
```
- EMA crossover for trend direction
- RSI for entry timing (buy oversold in uptrend)

### Trend + Volatility
```bash
--indicators "ema(20),bbands(20,2),atr(14)"
```
- EMA for trend
- Bollinger Bands for volatility context
- ATR for stop-loss sizing

### Full Suite
```bash
--indicators "ema(20),ema(50),rsi(14),macd(12,26,9),atr(14)"
```

---

## Quick Reference

| Task | Command |
|------|---------|
| List indicators | `mtdata-cli indicators_list` |
| Momentum indicators | `mtdata-cli indicators_list --category momentum` |
| Indicator details | `mtdata-cli indicators_describe rsi` |
| Fetch with indicators | `mtdata-cli data_fetch_candles EURUSD --indicators "ema(20),rsi(14)"` |

---

## Deeper detail: pandas-ta vs TA-Lib

mtdata gets its indicator list from pandas-ta. That library can compute the
indicators on its own, so you do not need to install TA-Lib. If TA-Lib is
present, some functions may use it instead, and a few values (especially
candlestick pattern columns) can look slightly different. Candle responses that
include indicators attach a compact `indicator_engine` object with the
pandas-ta name/version, whether TA-Lib is available, and the effective backend.
Forecast backtests do not yet attach that provenance. For reproducible feature
studies, save `indicators_describe --detail full`, the installed backend
versions, and the backtest's actual `feature_usage.selected_columns` beside the
raw result.

---

## See Also

- [GLOSSARY.md](GLOSSARY.md) — Term definitions
- [DENOISING.md](DENOISING.md) — Smoothing techniques
- [FORECAST.md](FORECAST.md) — Using indicators in forecasts

Ichimoku candle features use the observed-time component frame. The backend's separate forward-projected frame is not appended as candles. The future-dependent Chikou column is omitted by default (`include_chikou=false`). Indicator specifications that produce the same output name are rejected; request offset variants separately to avoid ambiguous columns.

