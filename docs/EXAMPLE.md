# End-to-end research loop

**Audience:** User

A compact **command recipe** for the full mtdata research loop: discover methods → fetch data → add context → forecast → size risk → validate. Prefer this when you already know the CLI; if you want narrative “why each step,” start with [SAMPLE-TRADE.md](SAMPLE-TRADE.md).

**Related:** [Setup](SETUP.md) · [CLI](CLI.md) · [Sample trade](SAMPLE-TRADE.md) · [Forecasting](FORECAST.md)

**Assumptions**

- MetaTrader 5 is running and logged in
- You call tools via `mtdata-cli <command> ...`
- Swap `EURUSD` / `H1` for your market

Tip: add `--json` when you want structured output for scripts or agents.

---

## 1) Discover what’s available

Forecast methods + availability (optional libraries show up here):

```bash
mtdata-cli forecast_list_methods --json
```

Forecast model spaces:

```bash
mtdata-cli forecast_list_library_models native --json
mtdata-cli forecast_list_library_models statsforecast --json
mtdata-cli forecast_list_library_models sktime --json
mtdata-cli forecast_list_library_models pretrained --json
```

Indicators:

```bash
mtdata-cli indicators_list --limit 50
mtdata-cli indicators_describe rsi --json
```

---

## 2) Inspect symbol and fetch candles

```bash
mtdata-cli symbols_list --limit 20
mtdata-cli symbols_describe EURUSD --json
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 300 --json
```

Optional liquidity snapshot. This requires `MTDATA_ENABLE_MARKET_DEPTH_FETCH=1` and broker support for Level 2/DOM data:

```bash
mtdata-cli market_depth_fetch EURUSD --spread true --json
```

`--spread` is a Boolean output switch. It computes spread prices in symbol price
units and, where metadata permits, also reports points, pips, and basis points
from the DOM or fallback bid/ask quote.

---

## 3) Prepare data: indicators + denoise

Compute a few indicators:

```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 1500 --indicators "ema(20),ema(50),rsi(14),macd(12,26,9)" --json
```

Denoise before indicators (smoother inputs):

```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 1500 --indicators "rsi(14),ema(50)" --denoise ema --denoise-params "columns=close,when=pre_ti,alpha=0.2,keep_original=true" --json
```

Denoise after indicators (smoother signals):

```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 1500 --indicators "rsi(14)" --denoise ema --denoise-params "columns=rsi_14,when=post_ti,alpha=0.3,keep_original=true" --json
```

---

## 4) Baseline price forecasts

Simple baseline (Theta):

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --library native --method theta --json
```

Pattern-based “analog” forecast (nearest-neighbor style):

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --library native --method analog --params "window_size=64 search_depth=5000 top_k=20 scale=zscore" --json
```

Optional: foundation model (if available):

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 24 --library pretrained --method chronos2 --params "context_length=512" --json
```

Optional: Monte Carlo simulation forecast (range of outcomes):

```bash
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --library native --method mc_gbm --params "n_sims=3000 seed=7" --json
```

---

## 5) Volatility forecast (risk sizing)

EWMA (fast default):

```bash
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method ewma --params "lambda_=0.94" --json
```

HAR-RV (uses intraday realized volatility):

```bash
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method har_rv --params "rv_timeframe=M5,days=150,window_w=5,window_m=22" --json
```

---

## 6) Barrier analytics (TP/SL planning)

Barrier probabilities (TP/SL odds within the horizon):

```bash
mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 --method hmm_mc --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.5,"stop_loss":0.3}' --params "n_sims=5000 seed=7" --json
```

Barrier optimization (search a TP/SL grid):

```bash
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --method hmm_mc --mode pct --grid-style volatility --search-profile medium --params "n_sims=5000 seed=7" --json
```

Interpretation shortcuts:

- `edge` ≈ P(TP first) − P(SL first)
- `prob_resolve` ≈ 1 − P(no hit)
- Wider stops/targets may look “worse” on edge but can be better after accounting for time-to-hit and payoff ratio.

---

## 7) Regime detection (gatekeeper)

Change-points (BOCPD):

```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --fetch-limit 1500 --method bocpd --threshold 0.6 --lookback 300 --json
```

Regime labels (HMM):

```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --fetch-limit 1500 --method hmm --params "n_states=3" --lookback 300 --json
```

---

## 8) Quick backtest (sanity check)

Run a rolling-origin backtest to compare a few methods:

```bash
mtdata-cli forecast_backtest_run EURUSD --timeframe H1 --horizon 12 --steps 50 --spacing 5 --methods "theta sf_autoarima analog" --json
```

---

## 9) Pattern detection (optional context)

Candlestick patterns:

```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode candlestick --lookback 500 --robust-only true --json
```

Classic chart patterns:

```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode classic --lookback 800 --json
```

Harmonic Fibonacci-ratio patterns:

```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode harmonic --lookback 800 --json
```

Bill Williams fractal price-structure levels:

```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode fractal --lookback 300 --json
```

Volume-profile structure levels from bounded tick data:

```bash
mtdata-cli volume_profile_levels EURUSD --start "2026-01-07" --end "2026-01-08" --source auto --price-source mid --bucket-points 10 --json
```

Equivalent H1 lookback form:

```bash
mtdata-cli volume_profile_levels EURUSD --timeframe H1 --lookback 168 --source auto --price-source mid --bucket-points 10 --json
```

Combine the two by opting fractal detection into volume-profile confluence:

```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode fractal --lookback 300 --config '{"volume_profile":true,"volume_profile_tolerance_points":25}' --json
```

`volume_profile_levels` returns POC, VAH, and VAL. The defaults admit up to one
day and 50,000 raw ticks; `source=auto` falls back to an explicitly labeled M1
bar approximation for larger windows. Broker FX volume is usually tick
activity, not exchange-traded volume.

---

## 10) Combine signals into a decision (simple recipe)

A pragmatic decision checklist:

- Forecast direction: do multiple methods agree on the sign of the next `horizon` bars?
- Volatility: is projected volatility within your risk budget?
- Regime: are you trading in a regime your strategy is designed for?
- Barriers: does the TP/SL pair have acceptable `edge` / `prob_resolve`?
- Patterns: do detections support or contradict the idea (optional)?

---

## Appendix: Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## See Also

- [CLI.md](CLI.md) — Full command reference
- [FORECAST.md](FORECAST.md) — Forecasting methods guide
- [TEMPORAL.md](TEMPORAL.md) — Session and seasonal analysis
- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — Barrier optimization deep dive
- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) — Guided trade workflow
