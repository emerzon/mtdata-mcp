# Trading risk analytics

**Audience:** User

**Read-only** tools for sizing, tail risk, and stress tests. They inspect the account and open positions — they **never** place, modify, or close orders. Pair them with [TRADING_SAFETY.md](TRADING_SAFETY.md) when you move from analysis to execution.

| Tool | Answers |
|------|---------|
| `trade_risk_analyze` | “How big can this trade be?” and “How much am I risking right now?” |
| `trade_var_cvar_calculate` | “What is my statistical worst-case loss over one bar?” |
| `trade_stress_test` | “What happens to open positions under these price shocks?” |

**Dense terms:** [Fixed-fraction sizing](GLOSSARY.md#fixed-fraction-sizing) · [Kelly](GLOSSARY.md#kelly-criterion) · [VaR](GLOSSARY.md#var-value-at-risk) · [CVaR / ES](GLOSSARY.md#cvar-conditional-var--expected-shortfall) · [Drawdown](GLOSSARY.md#drawdown)

**Related:** [Sample trade](SAMPLE-TRADE.md) · [Advanced playbook](SAMPLE-TRADE-ADVANCED.md) · [Guardrails](ENV_VARS.md#trade-guardrails) · [Barriers](BARRIER_FUNCTIONS.md)

---

## `trade_risk_analyze`

Analyzes current open/pending risk and, when given a proposed entry and stop, sizes a
new trade. Supports two sizing methods: fixed-fraction and Kelly.

```bash
# Current portfolio risk only
mtdata-cli trade_risk_analyze --json

# Size a new long: risk-based volume from entry + stop
mtdata-cli trade_risk_analyze EURUSD --direction long --entry 1.0850 --stop-loss 1.0800 --sizing '{"method":"fixed_fraction","risk_pct":1.0}' --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | — | Symbol for new-trade sizing; omit for portfolio-only risk. |
| `sizing` | — | JSON object selecting `fixed_fraction` with `risk_pct`, or `kelly` with its edge inputs and cap. Risk percentages use percentage points (`1` means 1%) and must be in `(0, 100]`. |
| `strict_risk` | `true` | Return `suggested_volume=0.0` if the broker minimum lot would exceed the requested sizing risk. |
| `include_pending` | `true` | Include contingent stop-loss risk from pending orders in portfolio totals. |
| `direction` | — | `long`/`short` (aliases accepted) for the proposed trade. |
| `entry` | — | Proposed entry price. With `symbol`+`stop_loss` but no entry, it is resolved from the latest tick (ask for long, bid for short, mid otherwise). A locked, stale, closed, or otherwise non-live quote is retained only for geometry context: the candidate returns `quote_not_live_ready`, no `suggested_volume`, and must be refreshed before sizing. Provide an explicit entry only for research-only geometry. |
| `stop_loss` | — | Proposed stop (alias `sl`). Required to compute risk-based volume. |
| `take_profit` | — | Optional target (alias `tp`) for reward/risk context. |

New-trade sizing uses an incremental candidate-risk policy: `sizing.risk_pct`
limits the proposed trade's stop risk as a percentage of account equity, and
account-wide margin stress remains a hard safety gate. Existing positions on other
symbols do not prevent sizing, but the returned `sizing_risk_policy` states that the
suggestion is not an aggregate portfolio stop-risk cap.

Candidate validation splits geometry from sizing eligibility.
`geometry_valid` is true when direction, stop, and target are internally
consistent. `sizing_eligible` is true only when a proposed volume can be
executed under account safety and mtdata volume guardrails.
`candidate_valid: true` means both the geometry is valid and, when sizing was
requested, the candidate is sizeable. If direction, stop, or target is invalid,
or sizing was requested and blocked (critical margin, guardrail volume, or a
non-live quote), the response retains the account and portfolio snapshot but
returns `success: false`, `candidate_valid: false`, `candidate_status: invalid`
or `blocked`, a structured `error_code` or `position_sizing_error`, and
`portfolio_snapshot_status: available`; CLI callers receive a nonzero exit
status. Suggested volume is clamped to the same symbol volume guardrails that
`trade_place` enforces; when no compliant size exists,
`recommendation_status` is `blocked` and the binding rule is included.

### Kelly sizing

Set `sizing.method=kelly` and supply edge statistics in the same JSON object:

```bash
mtdata-cli trade_risk_analyze EURUSD --direction long --entry 1.0850 --stop-loss 1.0800 --sizing '{"method":"kelly","win_rate":0.55,"avg_win":0.012,"avg_loss":0.010,"fraction_multiplier":0.5,"max_risk_pct":2.0}' --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sizing.win_rate` | — | Win probability as a fraction in `[0, 1]`. |
| `sizing.avg_win` | — | Average winning return normalized to a consistent stake or unit of risk (for example, an R-multiple). |
| `sizing.avg_loss` | — | Average losing return magnitude on the same normalized basis. |
| `sizing.fraction_multiplier` | `0.5` | Multiplier on the raw Kelly fraction (half-Kelly = `0.5`). |
| `sizing.max_risk_pct` | `2.0` | Hard cap on account risk in percentage points (`1` means 1%); must be in `(0, 100]`. |

The Kelly fraction is `win_rate − (1 − win_rate) / (avg_win / |avg_loss|)`, then scaled
by `sizing.fraction_multiplier` and capped by `sizing.max_risk_pct`. On a non-positive edge the tool reports
`status="kelly_no_edge"` and a suggested volume of `0.0`.

`trade_journal_analyze` reports `avg_win` and `avg_loss` in account currency.
Inspect `entry_cost_coverage` before interpreting them: matched entry commission
and fees are included, while unmatched exits retain an exit-deal-only basis.
Those raw PnL averages are not Kelly inputs because deal sizes and capital at risk can vary.
Normalize each historical outcome to a consistent stake or initial risk before
computing the average return metrics supplied here. Values larger than 10 are
rejected as currency-like rather than R-multiples.

Journal side filters distinguish two directions: `side=long|short` selects the
economic position side of realized exits, while `side=buy|sell` selects the
broker fill direction. A sell fill can close a long and a buy fill can close a
short, so inspect the response's `side_filter.dimension` when automating cohorts.

Portfolio stop risk is the gross sum of each ticket's remaining loss from its
current MT5 mark to its stop. This measures equity at risk now; it does not reuse
the original entry-to-stop loss after unrealized P&L has changed. It is a
conservative path-risk measure, not a same-symbol net-exposure estimate; a path
can trigger both sides of a hedge sequentially. Pending-order stop risk is
reported separately as contingent and is included in the total only when
`include_pending=true`.

Default compact output retains a per-position exposure summary, including the
ticket, side, volume, current mark, stop/target, notional value, and stop-risk
status. When every included position and pending order has quantifiable stop
risk, `risk_total_complete=true` and the unconditional `total_risk_*` fields are
numeric. If any component has no stop, a breached stop, or unusable tick
metadata, those totals are `null`; `quantified_risk_*` remains the explicitly
labeled subtotal of components that could be measured.

`notional_value` and portfolio notional fields are linearized account-currency
exposures derived from the broker's tick value and tick size. The per-position
`contract_price_product` diagnostic preserves raw `volume × contract_size × price`
with the explicit unit `contract_size_times_price`; it must not be compared with
account equity or summed across unlike instruments.

`trade_get_open` uses the same distinction on each position row.
`notional_account` is comparable with account balance/equity and names its
currency in `notional_account_currency`; `notional_quote` is the raw contract
price product in `notional_quote_currency`. If broker tick economics are not
available for a cross-currency conversion, `notional_account` is `null` and
`notional_account_unavailable_reason` explains why. Never sum `notional_quote`
across instruments with different quote currencies.

---

## `trade_var_cvar_calculate`

Estimates Value at Risk (VaR) and Conditional VaR (CVaR, a.k.a. Expected
Shortfall) for the current open positions — either the whole portfolio or a single
symbol. The default holding period is one bar of the selected timeframe; pass
`--horizon-bars` to scale the same return sample over multiple bars.

```bash
# Portfolio VaR/CVaR at 95% over one H1 bar
mtdata-cli trade_var_cvar_calculate --timeframe H1 --lookback 500 --confidence 0.95 --json

# Symbol-scoped, parametric/Gaussian, percentage returns
mtdata-cli trade_var_cvar_calculate EURUSD --method parametric --transform pct --lookback 300 --json

# EWMA-weighted tail over a six-bar horizon, including the forming bar
mtdata-cli trade_var_cvar_calculate EURUSD --method ewma --horizon-bars 6 --include-incomplete --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | — | Restrict to one symbol's exposure; omit for the full portfolio. |
| `timeframe` | `H1` | Return interval. Combined with `horizon_bars` this is the holding period. |
| `lookback` | `500` | Historical bars used to build the return distribution. |
| `confidence` | `0.95` | Confidence fraction (`0.95`, `0.99`); must satisfy `0 < confidence < 1`. |
| `method` | `historical` | `historical` (empirical tail), `parametric` (Gaussian), `cornish_fisher` (skew/kurtosis-adjusted Gaussian), or `ewma` (exponentially weighted historical). |
| `horizon_bars` | `1` | Number of bars in the holding period. Multi-bar results scale the one-bar return sample rather than simulating a path. |
| `include_incomplete` | `false` | When true, the currently forming candle can enter the return series. Default uses only completed bars. |
| `transform` | `log_return` | Return transform: `log_return` or `pct`. |
| `min_observations` | `50` | Minimum aligned observations before estimating risk. EWMA and Cornish-Fisher need enough sample for their extra moments/weights; the tool reports the effective sample in the payload. |

**Output** includes `var` and `cvar`, position/exposure counts, method/horizon provenance, and — by detail level —
per-position and per-symbol exposure breakdowns. With no open positions, `--detail full`
returns the legacy zero-filled arrays.

VaR/CVaR converts percentage price shocks to account-currency P&L with each
symbol's broker-provided tick value and tick size (`pnl_model` is
`tick_value_linear_sensitivity`). Positions without usable tick economics are rejected
rather than mixed into a portfolio in incompatible quote-currency units. The model is
linearized and does not include gaps, spread changes, swaps, or nonlinear payoff effects.
Every open position must have usable account-currency valuation and every
included symbol must have usable history. The tool returns
`portfolio_var_incomplete` instead of silently calculating a smaller portfolio
when any position lacks valuation inputs or any symbol history is unavailable.

Method notes:

- `historical` uses the empirical tail of equally weighted sample returns.
- `parametric` assumes Gaussian returns and uses sample mean/variance.
- `cornish_fisher` starts from that Gaussian quantile and adjusts it with sample
  skewness and excess kurtosis. It is most useful when the return sample is
  clearly non-normal; small samples make the adjustment noisy.
- `ewma` reweights the same historical returns with exponential decay so recent
  observations dominate. The payload reports the effective decay/half-life used.
- `--horizon-bars N` treats the holding period as N times the selected
  timeframe. It does not simulate intra-horizon barrier hits or path dependence.
- `--include-incomplete` adds the in-progress bar to the return series. That can
  make a live reading more current, but the last observation is then a partial
  bar rather than a completed close.

---

## `trade_stress_test`

Applies deterministic percentage price shocks to open positions and reports the P&L
impact. Useful for "what if EURUSD drops 2% and everything else 3%?" scenarios.

```bash
# Per-symbol shocks
mtdata-cli trade_stress_test --shocks '{"EURUSD":-2.0,"GBPUSD":-1.5}' --json

# Wildcard shock applied to every position without an explicit entry
mtdata-cli trade_stress_test --shocks '{"*":-3.0}' --detail full --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shocks` | (required) | Per-symbol percentage shocks, e.g. `{"EURUSD": -2.0}`. Use `"*"` as a fallback for symbols without an explicit entry. |
| `include_unshocked` | `false` | Include positions that received no shock (no exact match and no `"*"` fallback). |
| `detail` | `compact` | `full` adds per-position diagnostics. |

**Output** is a list of `items` (one per evaluated position) with `ticket`, `symbol`,
`side`, `volume`, `shock_pct`, `current_price`, `shocked_price`, and `pnl_impact`, plus
totals: `total_pnl_impact`, `positions_total`/`evaluated`/`shocked`, and — when account
metadata is available — `equity_before`/`equity_after`/`impact_pct`.

---

## Caveats

- All three tools read live MT5 state; results change as positions and quotes move.
- A `None` position/order response from MT5 is a failed snapshot, not an empty
  book. Snapshot-dependent analytics return a `*_snapshot_unavailable` error;
  empty tuples/lists remain valid empty books.
- VaR/CVaR assume the recent return distribution persists over the requested
  `horizon_bars`; they are not a guarantee of maximum loss.
- Stress shocks are deterministic and linear in price; they do not model spread
  widening, gaps, swaps, or correlation breaks.
- Kelly sizing is only as good as its inputs — estimate `win_rate` and normalized
  average win/loss returns from a sufficient out-of-sample track record.

## See also

- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — TP/SL hit probabilities
- [TRADING_SAFETY.md](TRADING_SAFETY.md) — Execution preview and guardrails
- [GLOSSARY.md](GLOSSARY.md) — VaR, CVaR, Kelly
- [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) — Risk gates in a full workflow
