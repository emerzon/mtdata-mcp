# Trade idea composer

**Audience:** User

Turn one symbol, timeframe, and horizon into a **preview-only research idea**: a short narrative, exits, size, gates, and a dry-run order check. This is the sample-trade loop as a single command.

It is **not** a buy or sell instruction, and it **cannot** send a live order.

**Dense terms:** [Barrier](GLOSSARY.md#barrier) · [Dry-run](GLOSSARY.md#dry-run) · [Confluence](GLOSSARY.md#confluence) · [Fixed-fraction sizing](GLOSSARY.md#fixed-fraction-sizing)

**Related:** [Sample trade](SAMPLE-TRADE.md) · [Barriers](BARRIER_FUNCTIONS.md) · [Trading safety](TRADING_SAFETY.md) · [Reports](REPORTS.md) · [Web UI](WEBUI.md)

---

## Quick start

```bash
mtdata-cli trade_idea_compose EURUSD --timeframe H1 --horizon 12 --template quick

# Include per-side execution assumptions in the first-hit contribution gate
mtdata-cli trade_idea_compose EURUSD --direction long --commission-bps-per-side 0.25 --slippage-bps 0.5
```

Read `direction`, `narrative`, `geometry`, `sizing.suggested_volume`, and `preview.preview_ok`. If `direction` is `stand_down`, the composer is telling you the idea did not clear its gates — not that you should fade it.

The same payload is available from MCP as `trade_idea_compose` and from HTTP as `POST /api/v1/trade-ideas`. In the Web UI, use the **Idea** button.

---

## What it does

The composer **reuses existing tools**. It does not invent new forecast or barrier math.

| Step | Tool | Quick | Standard |
|------|------|-------|----------|
| Session + quote | `trade_session_context` | live only | live only |
| Structure | `confluence_levels` | no | yes (and may snap TP/SL toward nearby zones before probability analysis) |
| Price path | `forecast_conformal_intervals` (Theta) for auto; `forecast_generate` for an explicit side | yes | yes |
| Typical movement | `forecast_volatility_estimate` (EWMA) | yes | yes |
| One TP/SL pair | `forecast_barrier_prob` (0.40% / 0.60%) | yes | yes |
| Size | `trade_risk_analyze` (fixed-fraction) | live only | live only |
| Preview | `trade_place` with `dry_run=true` | live only | live only |

`--direction auto` (default) calibrates Theta residual-quantile bands over 50
rolling historical anchors, spaced by at least the requested horizon. It selects
a side only when the calibrated horizon band excludes the last-price anchor. The
result's `forecast` section identifies the method, interval method, alpha,
calibration sample, and exact interval gate basis. A neutral direction,
insufficient calibration, an interval containing the anchor, or unavailable
uncertainty stands down; the composer does not infer a side from the slope
between forecast steps. It also stands down when the TP-first and SL-first
probabilities, weighted by the final reward and risk distances, do not produce
a positive first-hit payoff contribution. Raw TP-first probability is not compared with
SL-first probability as though unequal exits had equal payoffs.
Configured commission and slippage are per fill side. The composer deducts
both on entry and exit before applying the first-hit contribution gate. They default
to zero, so provide realistic values when using the result for execution
research. The response reports the first-hit contribution before and after configured costs under `barriers`
and the normalized round-trip amount under `execution_costs`.

This contribution is `prob_tp_first * reward_pct - prob_sl_first * risk_pct`.
No-hit paths are assigned zero gross payoff, disclosed by
`no_hit_gross_payoff_assumption_pct=0` and
`timeout_mark_to_market_included=false` in both the barriers and gate metadata.
The gate checks this partial contribution after configured costs; it does not
establish full horizon profitability. A large `prob_no_hit` makes the omitted
terminal payoff particularly relevant. Use `forecast_barrier_optimize` for its
explicit timeout mark-to-market contribution. Older `expected_value_*` fields
have been replaced by the accurately named first-hit fields.

Auto mode is therefore materially slower than `--direction long` or
`--direction short`: it fits 50 rolling backtest forecasts before the current
forecast. Explicit directions use the point forecast only. Their alignment
gate can pass only when the forecast's terminal move clears its documented
effect-size threshold and agrees with the requested side; the gate reports
`basis: point_estimate_effect_size` and `uncertainty: not_available`. This is
weaker evidence than a calibrated interval, while all other quote, barrier,
sizing, and preview safety gates remain in force.
A failed forecast-alignment or barrier gate makes the overall idea ineligible,
keeps suggested volume at zero, and skips the order preview even when the side
was explicitly requested.

`--as-of` makes the idea historical and **research-only**: no live session or
quote, no live sizing, and no dry-run preview. Historical geometry uses the
barrier analysis's cutoff-bound reference price.
The response keeps the raw cutoff in `requested_as_of`. Live ideas stamp
`as_of` at assembly time and keep the last closed-bar observation on
`data_as_of`. Historical ideas keep `as_of` on that observation. Component
`lineage` records source windows and price anchors, and full detail includes
timestamped forecast points.

---

## How to read the result

| Field | Meaning |
|-------|---------|
| `direction` | `long`, `short`, or `stand_down` |
| `direction_basis` | `forecast_vs_live_quote` when a calibrated interval and live quote are available, `forecast_vs_last_price` as the closed-bar fallback, `requested` for an explicit side, or `gate_outcome` when gates force `stand_down` |
| `suggested_direction` | Forecast-based hint; may differ from `direction` |
| `forecast.calibration` | Auto mode's requested anchors, minimum usable residual sample, empirical coverage, and sufficiency status |
| `forecast.forecast_vs_last_price.direction_interval_basis` | Exact comparison used by the auto direction gate |
| `forecast.forecast_vs_live_quote` | Terminal calibrated interval compared conservatively with the current bid/ask; includes both price anchors |
| `actionability` | Always `preview_only` or `research`. Never live. |
| `idea_eligible` / `overall_gate_status` | Aggregate strategy and operational gate decision; only `true` / `pass` permits a preview-eligible idea |
| `gates` | `pass` / `fail` / `skip` for quote, session, forecast, barriers, SL/TP, sizing, preview |
| `execution_costs` | Per-side commission/slippage assumptions and their normalized round-trip total in basis points |
| `barriers.first_hit_contribution_pct` / `first_hit_contribution_after_costs_pct` | Resolved TP/SL payoff contribution before and after configured execution costs |
| `preview.preview_ok` | Local dry-run order validation. It is false whenever the aggregate idea is ineligible and is never a broker fill. |
| `as_of` / `assembled_at` / `data_as_of` | Live `as_of` is assembly time; `data_as_of` is the last closed-bar observation. Historical `as_of` follows `data_as_of`. |
| `lineage` | Per-component source cutoff, data window, price anchor, and forecast target window |
| `partial_failure` | Some sections failed; do not infer the missing ones |

Reports (`report_generate`) remain research packages. This command is the **decision artifact** that adds size, gates, and a dry-run preview.

---

## Safety

- The composer **rejects** any live send. There is no `dry_run=false` flag.
- Stale, locked, or non-tradable quotes stand down and keep `suggested_volume` at `0`.
- Prefer a demo account even for previews that you later copy into `trade_place`.
- See [TRADING_SAFETY.md](TRADING_SAFETY.md) before you ever set `--dry-run false` on `trade_place` itself.

---

## See also

- [SAMPLE-TRADE.md](SAMPLE-TRADE.md) — the same questions as separate commands
- [SAMPLE-TRADE-WEBUI.md](SAMPLE-TRADE-WEBUI.md) — run `trade_idea_compose` from **Tools** until the Idea panel ships
- [REPORTS.md](REPORTS.md) — packaged research without sizing or dry-run
