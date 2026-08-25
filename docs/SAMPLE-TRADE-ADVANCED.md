# Advanced forecast-to-trade playbook

**Audience:** User

Builds on the [basic sample trade](SAMPLE-TRADE.md) (one barrier pair via
`forecast_barrier_prob`) with **regime filters**, **conformal intervals**,
**HAR-RV volatility**, **Monte Carlo barrier grids**, and **disciplined
risk/execution gates**. Modular by design: run each block, inspect the output,
and only continue when your own thresholds (calibrated via backtests) say so.

**Not financial advice.** Use a demo account for any execution steps. See [TRADING_SAFETY.md](TRADING_SAFETY.md).

**Dense terms used below:** [BOCPD](GLOSSARY.md#change-point-detection-bocpd) · [HMM](GLOSSARY.md#hidden-markov-model-hmm) · [HAR-RV](GLOSSARY.md#har-rv-heterogeneous-autoregressive-realized-volatility) · [Conformal](GLOSSARY.md#conformal-intervals) · [Kelly](GLOSSARY.md#kelly-criterion) · [VaR](GLOSSARY.md#var-value-at-risk) · [Edge](GLOSSARY.md#edge)

**Related:** [Sample trade](SAMPLE-TRADE.md) · [CLI](CLI.md) · [Barriers](BARRIER_FUNCTIONS.md) · [Regimes](forecast/REGIMES.md) · [Glossary](GLOSSARY.md)

## Assumptions

| Item | Value |
|------|--------|
| Symbol / timeframe | EURUSD / H1 |
| Horizon | 12 bars (~half day) |
| Interface | `mtdata-cli <tool> ... --json` |

---

## 0) Safety and hygiene

- Skip high-impact news windows (for example ±60 minutes around CPI, NFP, FOMC).
- Enforce a daily loss cap (for example 1–2× average daily VaR) and a per-trade risk cap (for example 0.25–1.0% of equity).
- Consider a spread/liquidity filter (for example spread under 1.5× median) before entry.

---

## 1) Regime & Break Detection (Gatekeeper)

Detect structural breaks and label regimes so you avoid trading through hostile phases.

1.1 BOCPD change‑points (returns)

```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --fetch-limit 1500 --method bocpd --threshold 0.6 --lookback 24 --json
```

- Gate: if `transition_summary.max_transition_probability >= 0.6` → stand down or reduce size; retrain/recalibrate models.

1.2 HMM‑lite regimes (returns)

```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --fetch-limit 1500 --method hmm --params "n_states=3" --lookback 300 --json
```

- Derive a simple regime tag: {trend‑lowvol, trend‑highvol, range} from `state` and `state_probabilities`.
- Gate: trade only when regime in {trend‑lowvol, trend‑midvol}; reduce risk in range/highvol.

Optional: MS‑AR(1) (statsmodels)
```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --fetch-limit 1500 --method ms_ar --params "n_states=2 order=1" --json
```

---

## 2) Realized Volatility & Risk Budget (HAR‑RV)

Estimate daily realized variance from intraday returns, then map to H1.

```bash
mtdata-cli forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 --method har_rv --params "rv_timeframe=M5,days=150,window_w=5,window_m=22" --json
```

- Extract `volatility_per_bar` (per-bar sigma) and `volatility_horizon` (k-bar sigma).
- Risk budget: set per‑trade risk ≤ min(0.7× daily VaR, fixed cap). Use σ to set realistic TP/SL and lot size.

---

## 3) Denoise + Quick Technical Context

Pull data with light denoising and a few TIs for situational awareness (no heavy feature stacks in this flow).

```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 300 --indicators "ema(20),ema(50),rsi(14),macd(12,26,9)" --denoise ema --denoise-params "columns=close,when=pre_ti,alpha=0.2,keep_original=true" --json
```

- Context: price vs EMA(20/50), RSI near extremes, MACD momentum slope.
- Gate: prefer longs if price > EMA(20)>EMA(50) and regime=trend.

---

## 4) Forecast with Valid Intervals (Conformal)

Calibrate per‑step residual quantiles via rolling backtest; then get point + conformal bands.

```bash
mtdata-cli forecast_conformal_intervals EURUSD --timeframe H1 --method fourier_ols --horizon 12 --steps 25 --spacing 12 --ci-alpha 0.1 --json
```

- Use `lower_price`/`upper_price` (conformal), not model CIs, for entry gating and sizing.
- Gate: only take longs if `conformal.lower_price[h-1] > pivot` and point forecast trend=up.

---

## 5) Barrier Analytics (MC + Closed‑Form)

5.1 Optimize TP/SL grid with HMM MC paths

```bash
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --method hmm_mc --mode pct --grid-style volatility --params "n_sims=5000 seed=7 refine=true refine_radius=0.35 tp_min=0.25 tp_max=1.5 tp_steps=7 sl_min=0.25 sl_max=2.5 sl_steps=9 return_grid=false" --top-k 5 --json
```

- Choose a combo by objective (edge/kelly/ev/ev_cond/ev_per_bar/prob_resolve/profit_factor/min_loss_prob/utility) subject to constraints:
  - Use `min_prob_win`, `max_prob_no_hit`, and `max_median_time` (bars) to enforce hit-rate and timing limits.

5.2 TP/SL odds for the chosen combo

```bash
mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 --method hmm_mc --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.4,"stop_loss":0.8}' --params "n_sims=5000 seed=7" --json
```

5.3 Closed‑form GBM sanity check (fast)

```bash
mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 --method closed_form --direction long --barrier '{"kind":"single_price","level":1.1795}' --json
```

- Flag discrepancies (e.g., MC>>GBM) to reduce size or re‑check calibration.

---

## 6) Labeling & Threshold Calibration (Optional but Recommended)

Use triple‑barrier labels offline for signal evaluation and meta‑models.

```bash
mtdata-cli labels_triple_barrier EURUSD --timeframe H1 --lookback 2000 --horizon 12 --barriers '{"unit":"pct","take_profit":0.4,"stop_loss":0.8}' --label-on high_low --detail full --json
```

`--lookback` controls the labeled history; `--limit` only caps the compact or
standard sample, while `--detail full` returns the complete labeled series.

- Compute in‑sample precision/recall for your entry rules; adjust thresholds (edge, cp_prob, RSI, EMA alignment) to reach desired trade quality.

---

## 7) Position Sizing & Execution

- Position size (conservative Kelly/VaR):
  - Kelly_cap = 0.25 × Kelly (from optimizer) or ≤ 1.0% equity, whichever is smaller.
  - VaR sizing: risk_to_SL ≤ risk_budget, where risk_to_SL uses spread‑adjusted SL and conformal lower.
- Stops & targets:
  - TP/SL from optimizer; time stop at horizon if neither is hit.
  - Optional partial TP at 0.5×TP; move stop to breakeven.
- Costs: subtract spread/commission from TP; inflate SL by typical slippage.
- If you convert a plan into a live order, start with `trade_place --dry-run true` and review the CLI trade controls (`--require-sl-tp`, `--magic`, `--expiration`) before removing the preview. The unprotected-position close fail-safe is always enabled.

---

## 8) Backtest & Walk‑Forward Checks

1) Rolling backtest for chosen forecast method(s)

```bash
mtdata-cli forecast_backtest_run EURUSD --timeframe H1 --horizon 12 --steps 50 --spacing 5 --methods "theta fourier_ols" --json
```

2) Stress‑test entry thresholds
- Sweep edge minima (0.05→0.15), cp_prob caps, regime sets, and confirm Sharpe/win rate stability.

---

## 9) Trade Plans (Examples)

Plan A – Breakout with pullback filter
- Gates: regime in {trend}, cp_prob<0.6, price>EMA(20)>EMA(50), conformal.lower>pivot.
- Targets: TP=0.40%, SL=0.80% (from optimizer); time stop at 12 bars.
- Size: min(VaR_budget, 0.25×Kelly) with HAR‑RV sigma.

Plan B – Mean‑reversion in range regime (reduced size)
- Gates: regime=range & low cp_prob; RSI>70 near R1 or <30 near S1.
- Smaller size; tighter SL and 0.20–0.30% TP; MC must show positive edge.

---

## 10) Monitoring & Drift Handling

- Update BOCPD and HMM twice per session; stand down on cp spikes.
- Refresh HAR‑RV daily (intraday M5 RV aggregation).
- Re‑calibrate conformal residuals weekly; re‑grid barrier optimizer monthly.
- Track realized vs. forecast errors; trigger re‑training on degradation.

---

## TL;DR – Advanced Flow

1) Filter: regime & cp_prob gates.
2) Calibrate risk with HAR‑RV; set budget.
3) Denoise + quick TI context (EMA/RSI/MACD).
4) Forecast with conformal intervals; gate by bands.
5) Optimize TP/SL via MC; sanity‑check with GBM closed‑form.
6) (Optional) Label history with triple‑barrier; tune thresholds.
7) Execute with VaR/Kelly‑capped sizing, time stop, and costs.
8) Walk‑forward checks; adjust thresholds and monitoring cadence.

---

## See Also

- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) — Basic workflow (start here if new)
- [FORECAST.md](FORECAST.md) — Forecasting methods guide
- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — Barrier analytics reference
- [forecast/REGIMES.md](forecast/REGIMES.md) — Regime detection details
- [forecast/VOLATILITY.md](forecast/VOLATILITY.md) — Volatility estimation methods
- [forecast/UNCERTAINTY.md](forecast/UNCERTAINTY.md) — Conformal intervals
