# Sample trade workflow

**Audience:** User

A friendly, step-by-step research walkthrough for **short-term EURUSD analysis** using mtdata. Each step shows **which tool**, **why those inputs**, and **how to read the output** — no quant background required.

This is a **research example**, not financial advice. The numbers below are hypothetical worked inputs, not a captured market session or evidence of performance. Re-run the tools for current data.

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
| **`data_fetch_candles`** (H1) | `symbol=EURUSD`, `timeframe=H1`, `limit=200`, `indicators=EMA(20), EMA(50), RSI(14), MACD(12,26,9)` | <ul><li>**H1** means one-hour bars.</li><li>200 completed bars cover about eight trading days; calendar time is longer across weekends and closures.</li><li>EMA-20 and EMA-50 give short- and medium-term price context; RSI and MACD describe momentum.</li></ul> |
| **Result** | Illustrative values: <br>- Price = 1.1776 <br>- EMA-20 = 1.1754, EMA-50 = 1.1739 <br>- RSI = 65 <br>- MACD line above its signal | **Interpretation** <br>- Price is above both EMAs and momentum is positive. These observations describe the recent past; they do not establish the next move. |

---

## 2. Get the daily price range for pivot‑point calculation

| Tool | Call | Why we used it |
|------|------|----------------|
| **`data_fetch_candles`** (D1) | `symbol=EURUSD`, `timeframe=D1`, `limit=30`, `ohlcv=ohlc` | <ul><li>Pivot points are traditionally calculated from the **previous day’s** high, low and close.</li><li>30 daily bars give us a recent history to confirm that the most recent day is representative.</li></ul> |
| **Result** | Illustrative last completed daily bar: <br>- High = 1.17745, Low = 1.16569, Close = 1.17608 | These are the inputs for the classic pivot calculation below. The live pivot tool fetches its own completed daily bar. |

---

## 3. Compute classic pivot‑point levels

| Tool | Call | Why we used it |
|------|------|----------------|
| **`pivot_compute_points`** | `symbol=EURUSD`, `timeframe=D1` | <ul><li>The compact default returns the classic pivot ladder. Use `detail=standard` or `detail=full` to compare classic, Fibonacci, Camarilla, Woodie, and DeMark tables.</li><li>Every method lists **support (S1, S2, …)** and **resistance (R1, R2, …)** tiers that traders monitor.</li></ul> |
| **Result** | Classic levels from those inputs: <br>- Pivot (PP) = 1.17307 <br>- R1 = 1.18046 <br>- S1 = 1.16870 <br>- R2 = 1.18483 | **Interpretation** <br>- Price 1.1776 is between PP and R1. The levels describe reference prices; touching or crossing one is not itself a forecast. |

Use **`confluence_levels`** when you want the pivot ladder ranked against data-driven support/resistance and Fibonacci swing levels. It highlights zones where independent methods cluster, such as a daily pivot resistance sitting within a few pips of an H1 resistance retest and a 61.8% Fibonacci retracement.

---

## 4. Estimate near‑future volatility

| Tool | Call | Why we used it |
|------|------|----------------|
| **`forecast_volatility_estimate`** | `symbol=EURUSD`, `timeframe=H1`, `horizon=12`, `method=ewma`, `params={"lambda_":0.94}` | <ul><li>**EWMA** estimates recent return volatility.</li><li>`lambda_=0.94` is the smoothing factor.</li><li>`horizon=12` means 12 hourly trading bars; a session closure can extend the elapsed time.</li></ul> |
| **Result** | Illustrative return standard deviations: <br>- Hourly sigma = 0.000593 (0.0593%) <br>- 12-bar sigma = 0.002055 (0.2055%) <br>- At price 1.1776: approximately **7.0 pips** per bar and **24.2 pips** over 12 bars | **Interpretation** <br>- Convert return volatility to a price distance before comparing it with a stop: `pips ≈ return_sigma × price / pip_size`, with EURUSD `pip_size=0.0001`.<br>- One sigma is a scale estimate, not a guaranteed range or an estimate of the maximum excursion. |

---

## 5. Forecast the price path for the next 12 hours

| Tool | Call | Why we used it |
|------|------|----------------|
| **`forecast_generate`** | `symbol=EURUSD`, `timeframe=H1`, `library=native`, `method=theta`, `horizon=12`, `quantity=price` | <ul><li>The **Theta** method is a fast baseline.</li><li>We ask for prices over the next 12 trading bars and compare the terminal forecast with the last observed price.</li></ul> |
| **Result** | Illustrative forecast: first step = 1.17528, terminal step = 1.17543, last observed price = 1.1776. Native Theta supplies a point forecast here. | **Interpretation** <br>- The terminal forecast is **0.1843% below** the observed price: a bearish point estimate, even though the forecast steps slope slightly upward.<br>- Without a qualifying interval, this is an unconfirmed point estimate (`direction_actionable=false`, `signal_status=not_actionable`). Use `forecast_conformal_intervals` to examine calibrated bands. |

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
| **Current market picture** (Steps 1 & 3) | Price is above both EMAs, above PP (1.17307), and below R1 (1.18046). This locates price relative to past structure; it does not predict a pull-back or breakout. |
| **Volatility check** (Step 4) | At price 1.1776, the 0.40% target is **47.1 pips** away and the 0.60% stop is **70.7 pips** away: about **1.9** and **2.9** times the illustrative 12-bar sigma of 24.2 pips. These ratios do not determine first-hit probabilities. |
| **Forecast** (Step 5) | The terminal price of **1.17543** is about **0.1843% below** 1.1776, despite the slight rise within the forecast path. Recent bullish indicators and this bearish point estimate disagree. |
| **Barrier odds** (Step 6) | One 0.40% / 0.60% pair shows how often a simulated path would hit the target first, the stop first, or neither. Use that to *sanity-check* distance, not to crown a “best” grid. |
| **Resulting research plan** | Keep this as an unresolved research hypothesis. Check calibrated intervals and inspect the barrier result before considering a side. Any change to entry, TP, SL, direction, or horizon requires a new barrier calculation; the sample 0.40% / 0.60% pair is not a validated order. |

---

## TL;DR — the “why” in plain English

1. **Recent prices + a few indicators** (EMAs, RSI, MACD) for short-term trend and momentum.
2. **Daily high/low/close → pivot levels** so you know nearby support and resistance.
3. **Volatility** (how far price usually travels over the next half-day).
4. **A forecast** for the next 12 trading bars, measured from the last observed price (here: a lower terminal estimate without a confirmed directional signal).
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
