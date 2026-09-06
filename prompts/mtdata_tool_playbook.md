# mtdata Tool Playbook

This is the shared operating reference for the short-term trading profiles in
this directory.

The live tool registry, not this file, is the source of truth. At session boot
call `tools_list(detail="full", include_related=true, limit=200)`. Use the
runtime schema; do not guess at new or removed interfaces.

## Operating Principles

- A tool result is evidence, not permission to trade. Account safety, live
  tradability, price freshness, execution cost, structural invalidation, and
  quantified risk must all pass independently.
- Never claim that a strategy, forecast, pattern, or backtest guarantees a
  profit. Short-horizon evidence is market- and sample-dependent.
- Never trade from a forecast, pattern label, regime label, denoised series,
  simplified series, report summary, news headline, or indicator alone.
- Use raw completed candles for structure and a fresh bid/ask for execution.
  Buy entries use ask-side geometry; sell entries use bid-side geometry.
- Default to `detail="compact"`. Request `standard` or `full` only when the
  extra fields can change the current decision.
- Do not run optimizers, model training, broad scans, or large reports in a hot
  execution loop. Cache their conclusions and invalidate them on a new regime,
  material event, or broker-day boundary.
- Never simplify data used for risk, volatility, statistics, forecasting, or
  backtesting. Denoising used in a live workflow must be causal and compared
  with the raw series.

## Side-Effect Labels

| Label | Meaning |
|---|---|
| `R` | Read-only and normally bounded. |
| `R-heavy` | Read-only but potentially slow or computationally expensive. |
| `R-blocking` | Read-only but waits before returning. |
| `S` | Changes local task or model-store state, not broker exposure. |
| `L` | Can change live broker orders or positions. Its `dry_run` default is false. |
| `G` | Conditional tool; availability depends on configuration or provider support. |

## Common Result Parser

Apply this parser after every call, before interpreting tool-specific fields.

1. **Envelope:** if `success` is false or a non-empty `error` exists, stop using
   the payload as market evidence. Record `error_code`, `request_id`,
   `operation`, `remediation`, and `related_tools`. Retry once only when the
   failure is a corrected payload or a clearly transient read failure.
2. **Partial results:** if `partial_failure` is true, inspect `failed_sections`
   or nested errors. Never infer a missing section from the sections that did
   succeed.
3. **Freshness:** inspect `as_of`, `retrieved_at`, quote time,
   `data_age_seconds`, `data_stale`, `freshness`, and runtime timezone metadata.
   Stale quotes block new risk. Compare event times in one explicit timezone.
4. **Collections:** prefer the collection identified by `row_key` or
   `canonical_source`. Otherwise inspect, in order, `items`, `rows`, `data`,
   `series`, `groups`, or a documented tool-specific collection. Respect
   `count`, `total_count`, pagination, and empty collections.
5. **Units:** read `units`, `digits`, `point`, `trade_tick_size`, currency, and
   percentage/fraction labels before comparing numbers. A value of `0.55` is
   not interchangeable with `55` unless the contract says so.
6. **Quality:** treat `warning`, `warnings`, `sample_status`, `sample_quality`,
   `metrics_reliability`, `degraded`, incomplete risk, missing optional
   dependencies, and fallback data sources as explicit uncertainty.
7. **Probabilities:** use payoff-weighted, cost-adjusted expected value. Do not
   require `P(TP) > P(SL)` when payoff sizes differ, and do not ignore no-hit or
   unresolved probability.
8. **Execution:** a successful preview has `dry_run=true`,
   `actionability="preview_only"`, or `would_send_order=false`. A successful
   live response is still provisional until open positions, pending orders,
   and when necessary history confirm the resulting broker state.

## Shared Live-Risk Contract

The supplied profiles use these moderate defaults unless the user explicitly
lowers them:

- Maximum risk per new entry: 0.50% of current equity.
- Maximum combined risk for all profiles on one symbol: 1.00% of equity.
- Broker-day realized plus floating loss gate: 2.00% of day-start equity.
- Maximum quantified open risk across the account: 3.00% of equity.
- Pending orders count at their all-fill stop risk, not at zero risk.
- `fixed_fraction` is the default sizing method. Kelly sizing is unavailable
  until `trade_journal_analyze` has at least 30 comparable realized exits and
  returns a reliable positive edge; even then use half-Kelly or less and retain
  the limits above.
- Stop loss and take profit are required on new market orders. No martingale,
  uncapped grid, revenge trade, or add justified only by unrealized loss.
- After two consecutive losing campaigns for one profile and symbol, block new
  risk for at least 60 minutes and until a new primary-timeframe bar closes.

Magic-number coexistence assumes MT5 preserves independently ticketed
positions, as on a hedging account. On a netting account, same-symbol orders can
merge into one net position and magic ownership is not a safe mutation
boundary. If netting behavior is configured or observed, allow only one
risk-adding profile to own a symbol; the account-wide supervisor may still
protect or reduce the resulting net position.

Before every risk-increasing action: refresh account and exposure state, check
market status and quote freshness, account for spread and slippage, size with
`trade_risk_analyze`, preview with `dry_run=true`, refresh the quote, then send
the same protected payload with a stable `idempotency_key`. Make at most one
risk-increasing action in a cycle. Verify immediately afterward.

Example preview payload:

```json
{
  "symbol": "EURUSD",
  "volume": 0.01,
  "order_type": "BUY",
  "stop_loss": 1.0800,
  "take_profit": 1.0900,
  "magic": 71001,
  "dry_run": true,
  "require_sl_tp": true,
  "auto_close_on_sl_tp_fail": true,
  "idempotency_key": "scalp-eurusd-20260710-001"
}
```

Do not reuse an idempotency key for a different payload. The key is an
in-process safeguard, not broker-side idempotency and not durable across a
restart.

## Asset Context Routing

- Always start with `news(symbol=SYMBOL)`; it unifies general, calendar, and
  symbol-relevant context when providers are available. Pin `source` only when
  one adapter is required.
- US equities: add `calendar(kind="earnings")` and `equity_profile` only
  when company-specific context can alter the hold.
  Use options tools only after `options_provider_status` passes.
- FX: add `calendar(kind="economic")` and `asset_performance(universe="forex")`
  when currency or macro context is missing from `news`.
- Futures, metals, and indices: add `asset_performance(universe="futures")`
  and `calendar(kind="economic")`.
- Crypto: add `asset_performance(universe="crypto")`; treat exchange
  availability as distinct from the broker symbol's tradability and spread.
- Finviz is a context provider, not the executable quote source. Provider
  timestamps and symbol mappings can differ from MT5.

## Strategy Research Standard

Before enabling a profile for a symbol and broker day:

1. Run its specified `strategy_backtest` proxy with a nonzero slippage value
   that includes the current round-trip spread estimate.
2. Require at least 30 trades, positive net return, profit factor greater than
   1.10, and usable drawdown/risk metrics. Otherwise the profile remains in
   observe-only mode for that symbol.
3. Do not tune the proxy on the same window used for approval. A favorable
   proxy does not prove the richer live rule.
4. If a forecast affects the decision, select it using rolling-origin
   `forecast_backtest_run`; do not choose it because the current forecast agrees
   with the desired trade.
5. Revalidate after a material regime shift, cost change, contract change, or
   broker-day boundary.

The design is informed by empirical work on
[market intraday momentum](https://www.sciencedirect.com/science/article/pii/S0304405X18301351),
[short-term reversal as liquidity provision](https://academic.oup.com/rfs/article-pdf/25/7/2005/24431763/hhs066.pdf),
and [volatility-managed portfolios](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513).
Those findings do not establish universal profitability. Research on
[intraday transaction costs](https://www.sciencedirect.com/science/article/pii/S1544612316300587),
[data snooping](https://eprints.lse.ac.uk/119144/1/dp303.pdf), and
[backtest overfitting](https://escholarship.org/uc/item/4w1110bb) motivates the
cost and sample gates above. The
[SEC day-trading risk notice](https://www.sec.gov/about/reports-publications/investorpubsdaytipshtm)
is the baseline risk warning for leveraged short-horizon operation.

## Failure and Recovery

- A malformed read payload may be corrected and retried once. A malformed
  payload supplies no trading evidence.
- On a trade error or ambiguous result, do not resend immediately. Refresh open
  positions, pending orders, account readiness, symbol constraints, and quote;
  then use history if the broker outcome is still unclear.
- Retry a broker action at most once, with the same idempotency key, and only
  after identifying a correctable cause. Do not retry into a widening spread,
  stale quote, event blackout, hard blocker, or invalid geometry.
- When risk cannot be quantified, treat it as unlimited. Protect or reduce it
  before considering new exposure.
- If the registry, output schema, or provider response contradicts this guide,
  follow the runtime contract conservatively and log the discrepancy for human
  review.
