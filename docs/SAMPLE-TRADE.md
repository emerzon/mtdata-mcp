# Sample trade workflow

**Audience:** User

A friendly, step-by-step research walkthrough for **short-term EURUSD analysis** using mtdata. Each step shows **which tool**, **why those inputs**, and **how to read the output** — no quant background required.

This is a **research example**, not financial advice. Numbers below are illustrative of one historical session; re-run the commands on live data for current levels.

**Terms used:** [EMA / RSI / MACD](GLOSSARY.md#moving-average) · [Pivot points](GLOSSARY.md#pivot-points) · [EWMA vol](GLOSSARY.md#ewma-exponentially-weighted-moving-average) · [Theta](GLOSSARY.md#theta-method) · [Barrier](GLOSSARY.md#barrier) — full [glossary quick find](GLOSSARY.md#quick-find).

**Related:** [Glossary](GLOSSARY.md) · [CLI](CLI.md) · [Same flow in the Web UI](SAMPLE-TRADE-WEBUI.md) · [Advanced playbook](SAMPLE-TRADE-ADVANCED.md)

Prefer clicking to typing? Use [SAMPLE-TRADE-WEBUI.md](SAMPLE-TRADE-WEBUI.md).
When you are comfortable with this flow, continue to [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) for regimes, HAR-RV, conformal intervals, Monte Carlo barriers, and tighter risk gates.

### Fast path

The same questions as this walkthrough can be asked in one preview-only command:

```bash
mtdata-cli trade_idea_compose EURUSD --timeframe H1 --horizon 12 --template quick
```

That returns a narrative, suggested direction, TP/SL geometry, a sized volume, gates, and a **dry-run** preview. It cannot place an order. Details: [TRADE_IDEAS.md](TRADE_IDEAS.md). The numbered steps below remain the explicit tool-by-tool path.

---

## 1. Pull the most recent price data (candles)

| Tool | Call | Why we used it |
|------|------|----------------|
| **`data_fetch_candles`** (H1) | `symbol=EURUSD`, `timeframe=H1`, `limit=200`, `indicators=EMA(20), EMA(50), RSI(14), MACD(12,26,9)` | <ul><li>**H1** = one‑hour bars – the natural granularity for short‑term (intraday) analysis.</li><li>200 bars give us roughly the last 8‑9 days of data, enough to see the current trend.</li><li>Adding **EMA‑20** and **EMA‑50** lets us see whether price is above or below short‑ and medium‑term moving averages (a quick trend check).</li><li>**RSI** (Relative Strength Index) shows if the market is “over‑bought” (>70) or “over‑sold” (<30).</li><li>**MACD** (Moving‑Average Convergence Divergence) tells us whether momentum is positive or negative.</li></ul> |
| **Result** | A table with columns `time,open,high,low,close,EMA_20,EMA_50,RSI_14,MACD,MACDh,MACDs`. The last few rows (the most recent hours) show: <br>‑ Price ≈ 1.1776 <br>‑ EMA‑20 ≈ 1.1754, EMA‑50 ≈ 1.1739 <br>‑ RSI ≈ 65 <br>‑ MACD line > signal (positive) | **Interpretation** <br>‑ Price is **above both EMAs** → bullish alignment. <br>‑ RSI is in the “strength” zone but not over‑bought. <br>‑ MACD histogram is near zero, meaning momentum is still positive but weakening – a possible short‑term pause. |

---

## 2. Get the daily price range for pivot‑point calculation

| Tool | Call | Why we used it |
|------|------|----------------|
| **`data_fetch_candles`** (D1) | `symbol=EURUSD`, `timeframe=D1`, `limit=30`, `ohlcv=ohlc` | <ul><li>Pivot points are traditionally calculated from the **previous day’s** high, low and close.</li><li>30 daily bars give us a recent history to confirm that the most recent day is representative.</li></ul> |
| **Result** | A table with the last 30 daily bars (open‑high‑low‑close). The most recent day (15 Sep) shows: <br>‑ High = 1.17745, Low = 1.16569, Close = 1.17608 | This daily H‑L‑C will be fed into the next step. |

---

## 3. Compute classic pivot‑point levels

| Tool | Call | Why we used it |
|------|------|----------------|
| **`pivot_compute_points`** | `symbol=EURUSD`, `timeframe=D1` | <ul><li>The compact default returns the classic pivot ladder. Use `detail=standard` or `detail=full` to compare classic, Fibonacci, Camarilla, Woodie, and DeMark tables.</li><li>Every method lists **support (S1, S2, …)** and **resistance (R1, R2, …)** tiers that traders monitor.</li></ul> |
| **Result** | JSON with: <br>‑ Pivot (PP) = 1.17505 <br>‑ R1 = 1.17848 <br>‑ S1 = 1.17264 <br>‑ R2, S2, R3, S3 also provided. | **Interpretation** <br>‑ Current price (≈ 1.1776) sits **just below R1** and **above the pivot** – a classic “test‑and‑break” situation. <br>‑ If price falls, S1 (1.17264) is the first support; if it breaks above R1, the next target is R2 (≈ 1.1809). |

Use **`confluence_levels`** when you want the pivot ladder ranked against data-driven support/resistance and Fibonacci swing levels. It highlights zones where independent methods cluster, such as a daily pivot resistance sitting within a few pips of an H1 resistance retest and a 61.8% Fibonacci retracement.

---

## 4. Estimate near‑future volatility

| Tool | Call | Why we used it |
|------|------|----------------|
| **`forecast_volatility_estimate`** | `symbol=EURUSD`, `timeframe=H1`, `horizon=12`, `method=ewma`, `params={lambda:0.94}` | <ul><li>**EWMA** (Exponentially Weighted Moving Average) gives a quick, robust estimate of recent volatility.</li><li>`lambda=0.94` is the standard smoothing factor used in many risk‑models (e.g., RiskMetrics).</li><li>`horizon=12` means we want the volatility for the next 12 hourly bars (≈ ½ day).</li></ul> |
| **Result** | <ul><li>Hourly σ (standard deviation) ≈ 0.000593 → **≈ 5.9 pips** per hour.</li><li>12‑hour σ ≈ 0.002055 → **≈ 20 pips** (≈ 0.20 %).</li></ul> | **Interpretation** <br>‑ Over the next half‑day we can expect the price to wander about **± 20 pips** (1 σ). <br>‑ This helps us size stops and targets so they are realistic relative to normal market moves. |

---

## 5. Forecast the price path for the next 12 hours

| Tool | Call | Why we used it |
|------|------|----------------|
| **`forecast_generate`** | `symbol=EURUSD`, `timeframe=H1`, `library=native`, `method=theta`, `horizon=12`, `quantity=price` | <ul><li>The **Theta** method is a fast, reliable forecasting model that works well on short‑term series.</li><li>We ask for a **price forecast** (not returns) for the next 12 hourly bars.</li></ul> |
| **Result** | JSON with: <br>‑ Forecasted price for each of the next 12 hours (≈ 1.17528 → 1.17543). <br>‑ Point-only uncertainty status for native Theta. <br>‑ Trend flag = **up**. | **Interpretation** <br>‑ The model expects a **small pull‑back** toward the pivot (1.1750) before the up‑trend resumes. <br>‑ This point forecast does not establish a risk band; use `forecast_conformal_intervals` when the workflow requires calibrated intervals. |

---

## 6. Odds for one take-profit / stop-loss pair

A **take-profit** is the price where you would bank a win. A **stop-loss** is
the price where you would cut a loss. This step does **not** search a huge
grid. It asks a simpler question: *if I pick one pair of levels, how often
would a random-looking path hit the win first, the loss first, or neither?*

| Tool | Call | Why we used it |
|------|------|----------------|
| **`forecast_barrier_prob`** | `symbol=EURUSD`, `timeframe=H1`, `horizon=12`, `direction=long`, `barrier={kind:tp_sl, unit:pct, take_profit:0.40, stop_loss:0.60}` | <ul><li>One modest pair: target about 0.40% up, stop about 0.60% down, over the next 12 hours.</li><li>The tool simulates many paths and reports three probabilities: hit TP first, hit SL first, hit neither.</li><li>Leave method at the default unless you already know you want a different simulator.</li></ul> |
| **Result** | JSON with `prob_tp_first`, `prob_sl_first`, and `prob_no_hit`. | **Interpretation** <br>- High `prob_no_hit` means the levels may be too far (or the horizon too short) for this window. <br>- This is a sketch of *path odds*, not a promise. Searching a full TP/SL grid (HMM paths, refine, Kelly / EV objectives) is in [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md). |

```bash
mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 --direction long --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.40,"stop_loss":0.60}' --json
```

---

## 7. Putting it all together – Trade ideas

| Step | How the previous outputs shaped the idea |
|------|------------------------------------------|
| **Current market picture** (Step 1 & 3) | Price is above the 20‑EMA, below R1, and near the daily pivot → likely to **pull back** to the pivot before trying to break R1. |
| **Volatility check** (Step 4) | 12‑hour σ ≈ 20 pips → a 0.40 % target is in the same neighborhood as a couple of typical hours of movement; a 0.60 % stop is wider than one typical hour but not “across the weekly range.” Re-check these distances if today’s volatility print is very different. |
| **Forecast** (Step 5) | The Theta forecast expects the price to settle around **1.1753**, i.e., near the pivot, confirming a short‑term pull‑back. |
| **Barrier odds** (Step 6) | One 0.40% / 0.60% pair shows how often a simulated path would hit the target first, the stop first, or neither. Use that to *sanity-check* distance, not to crown a “best” grid. |
| **Resulting research plan** | • **Primary long idea**: watch a pull-back toward the pivot (≈ 1.1750), with a nearby target under R1 and a stop beyond S1 — then **re-run** `forecast_barrier_prob` on *your* distances. <br>• **If price closes above R1**: the “test-and-break” picture is stale; do not keep the old pair. <br>• Full grid search, Kelly, and EV ranking stay in the [advanced playbook](SAMPLE-TRADE-ADVANCED.md). |

---

## TL;DR — the “why” in plain English

1. **Recent prices + a few indicators** (EMAs, RSI, MACD) for short-term trend and momentum.
2. **Daily high/low/close → pivot levels** so you know nearby support and resistance.
3. **Volatility** (how far price usually travels over the next half-day).
4. **A forecast** for the next 12 hours (here: a modest pull-back toward the pivot).
5. **Barrier odds** for *one* take-profit / stop-loss pair (hit target, hit stop, or neither).
6. **Combine** structure, vol, forecast, and those odds into a hypothesis you can stress-test further.

That is the full path from raw candles to research ideas you can stress-test further. It is **not** a guaranteed trade.

---

## Next steps

- [SAMPLE-TRADE-WEBUI.md](SAMPLE-TRADE-WEBUI.md) — Same questions in the chart workspace
- [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) — Regimes, conformal intervals, HAR-RV, tighter gates
- [FORECAST.md](FORECAST.md) — Methods and research stages
- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — Barrier deep dive
- [TRADING_SAFETY.md](TRADING_SAFETY.md) — If you move from ideas to orders (demo first)
- [GLOSSARY.md](GLOSSARY.md) — Terms used above
