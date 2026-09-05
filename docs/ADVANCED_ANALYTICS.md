# MT5-native advanced analytics

**Audience:** User

Five **read-only** tools for when the basic chart is not enough: how tight the quote is, how your fills behaved, whether a rule held up out of sample, how portfolio risk splits, and which names led.

You do not need a second data vendor — these read the connected MetaTrader 5 terminal. Skip this page until [SAMPLE-TRADE.md](SAMPLE-TRADE.md) feels comfortable.

**Dense terms:** [Microstructure](GLOSSARY.md#microstructure) · [Execution quality](GLOSSARY.md#execution-quality) · [Relative strength](GLOSSARY.md#relative-strength) · [VaR / CVaR](GLOSSARY.md#var-value-at-risk) · [Spread](GLOSSARY.md#spread)

**Related:** [Trading risk](TRADING_RISK.md) · [CLI](CLI.md) · [Example workflow](EXAMPLE.md) · [Glossary](GLOSSARY.md)

## Tick microstructure

`market_microstructure_analyze` measures spread distributions, quote-update
intensity, gaps, mid-price volatility, and liquidity-stress windows.
Its compact spread summary marks locked, one-sided, and inverted latest quotes as
unsafe rather than treating them as unusually tight execution spreads. For the
default live window, a non-executable final stream update is reconciled with the
same current-quote policy used by `market_ticker`. The summary keeps that raw
event in `raw_update_quality` and `data_quality.latest_raw_update_quality` so a
valid carried quote does not hide feed diagnostics. Explicit historical windows
continue to report their final historical update without live reconciliation.
When a window has fewer than 20 usable ticks, the error reports the requested
window and observed count. Increase `--minutes-back` for a relative request,
or move/widen an explicit window with `--start` and `--end`; bar counts and
timeframes are not controls for this tick-based tool.

```bash
mtdata-cli market_microstructure_analyze EURUSD --minutes-back 60 --json
```

The result identifies the feed as `quote_only`, `trade_ticks`, or
`trade_volume`. Volume-impact metrics are omitted unless the broker supplies
enough non-zero real trade volume. Quote pressure is a proxy, not centralized
FX order flow.

MT5 tick rows are complete snapshots. The analyzer uses the `flags` bitmask to
identify trade events, so a quote update that repeats the last price and volume
is not counted as another trade.

Volatility fields are deliberately distinct. The summary's
`mid_log_return_realized_volatility_observed_window` is the square root of
summed squared tick-to-tick log returns over the observed window. Bucket rows'
`mid_log_return_std_per_quote_update` is the population standard deviation of
tick-to-tick log returns inside that bucket. Both are decimal log-return
statistics over irregular quote updates and are not annualized. They are not
directly comparable with each other or across different window lengths;
`mid_return_observations`, duration fields, `units`, and
`estimator_scope.volatility_metrics` state the applicable basis.

## Execution quality

`trade_execution_quality` joins MT5 deal history to order history and nearby
ticks. It reports side-aware slippage, latency, partial fills, fees, and
post-fill markouts. Choose either `--minutes-back` or an explicit `--start` /
`--end` window; supplying both is rejected so the analyzed period is never
ambiguous. Use `--magic` to isolate one strategy. Magic filters accept the MT5
unsigned 64-bit range, including zero, and responses include `magic_exact` for
clients that cannot represent large JSON integers exactly. When no window is
provided, the command analyzes the latest seven days; use an explicit longer
window for monthly reviews.

```bash
mtdata-cli trade_execution_quality --symbol EURUSD --minutes-back 43200 --markout-seconds 1,5,30 --detail full --json
```

Positive slippage is worse for the trader; positive markout is favorable. With
the default arrival-quote policy, the headline slippage distribution contains
market-order fills only. Pending fills are compared with their submitted order
price, while setup-to-fill price movement is reported separately as arrival
implementation shortfall. Unmatched or unbenchmarked fills are counted rather
than silently discarded. Compact `data_quality` reports
`eligible_symbol_count` and `analyzed_symbol_count`. Full detail still lists
`eligible_symbols` (history-filter survivors) and `analyzed_symbols` (fills that
reached the reported statistics), plus `quote_reads` cache diagnostics.
`price_improvement_pct` and `partial_fill_pct` are 0–100 percentages, same
scale as `coverage_pct`.
Commission and fee percentiles are non-negative cost magnitudes per broker lot;
signed commission and fee fields remain available on full-detail fill rows.
Each `summary.markout_bps.<seconds>` entry reports `observations`, `missing`,
`coverage_pct`, and a `sample_status` evaluated against `--min-sample`.
Markout cohorts may differ by horizon because a future tick can be available for
one horizon but not another. Root `fill_sample_quality` applies only to
fill-level metrics, not to these horizon-specific distributions.

`--limit` caps matched fills used for headline metrics and returned rows
(latest fills first). Compact output labels that sample with `summary_scope`
(for example `latest_200_of_491`), keeps `sample.selection_order`, and adds
`effective_analysis_window` beside the requested `window`. A warning is
emitted whenever the sample does not cover the full requested period. Compact
detail also returns the headline distributions, sample counts, window
provenance, data-quality counts, and warnings. Standard detail adds breakdowns,
metric definitions, units, session definitions, and full history diagnostics;
full detail also includes individual fill rows.

## Fixed-candidate chronological validation

`strategy_validate` evaluates predeclared built-in or forecast-threshold
candidates with anchored expanding chronological folds. Outcomes must finish
inside their test fold; prior calibration samples are horizon-purged and
embargo bars are excluded. Evidence uses block-bootstrap expectancy tests with
Holm correction and reports `positive`, `negative`, or `inconclusive`.

```bash
mtdata-cli strategy_validate EURUSD --strategy ema_cross --json

mtdata-cli strategy_validate EURUSD --timeframe H1 --lookback 3000 --candidates '[{"id":"fast-cross","type":"builtin_strategy","strategy":"ema_cross","params":{"fast_period":10,"slow_period":30}}]' --barrier '{"horizon":12}' --json
```

Use `--strategy` for a single built-in strategy with default parameters. Use
the JSON `--candidates` form for parameterized built-ins, forecast-threshold
candidates, or multi-candidate validation.

A forecast-threshold candidate compares each forecast's expected simple return
`(forecast_price - last_close) / last_close` with `long_above` and
`short_below`. Those thresholds are fractions: `0.005` means 0.5%. Barrier
`tp_pct`/`sl_pct` stay in percentage points (`0.5` means 0.5%), so the two
objects in the same request use different numeric conventions.

`volatility_term_structure` reports decimal fractions (`0.0465` = 4.65%).
Multiply that decimal by 100 before feeding percent-point barrier tools such as
`labels_triple_barrier` and `forecast_barrier_prob` (`unit=pct`).

| Source | Example value | Meaning | Convert for `unit=pct` barriers |
|--------|---------------|---------|----------------------------------|
| `volatility_term_structure` | `0.0465` | 4.65% annualized decimal vol | `4.65` percent-points |
| `labels_triple_barrier` / `forecast_barrier_prob` | `0.5` | 0.5% | already percent-points |
| `strategy_validate` forecast thresholds | `0.005` | 0.5% | already a fraction, not percent-points |

```bash
mtdata-cli strategy_validate EURUSD --timeframe H1 --lookback 200 --candidates '[{"id":"drift-half","type":"forecast_threshold","method":"drift","params":{"lookback":30},"horizon":1,"long_above":0.005,"short_below":-0.005}]' --barrier '{"horizon":1,"tp_pct":0.5,"sl_pct":0.5}' --json
```

If every candidate reports `evaluation_status=insufficient_data`, the command
fails with `strategy_validation_no_evaluable_candidates` instead of a top-level
success. Mixed requests stay successful and count omitted candidates.

Candidate parameters are fixed before validation; this tool does not optimize
and validate on the same sample. Candidate IDs are trimmed, case-insensitively
unique within the request, and remain the stable correlation key after ranking.
Every ranking also echoes the concrete built-in strategy or forecast method;
full detail includes the effective parameters after defaults are applied.

Built-in `sma_cross` and `ema_cross` use the same always-in state/reversal
contract as `strategy_backtest`: the position is long while the fast average
is above the slow average, short while it is below, and a reverse cross exits
then immediately enters the opposite side. Barrier `tp_pct`/`sl_pct` do not
apply to these named strategies. For the older one-bar cross event plus
horizon barrier, use `sma_cross_event` or `ema_cross_event`.
`rsi_reversion` still enters only when RSI crosses into an oversold or
overbought zone. Each ranking exposes this contract in `signal_definition`
(`state_reversal`, `cross_event`, `zone_entry_event`, or
`forecast_threshold_anchor`).
Event and forecast-threshold barrier outcomes enter at the next bar open. If a
later bar opens beyond the stop, the realized loss uses that opening fill
rather than capping the result at the requested stop percentage.

Lookback accounting reports `evaluation_bars`, `warmup_bars`,
`outcome_tail_bars`, and `fetch_bars` separately. Fetching
`lookback + horizon + 5` bars does not mean 217 evaluation bars were requested
when `lookback=200`.

Forecast-threshold candidates execute at most the latest 200 eligible forecast
anchors to keep validation bounded. Their folds partition that computed signal
window rather than empty earlier history. Each candidate reports signal range,
requested/evaluated folds, skipped-fold reasons, and fold coverage; incomplete
coverage uses `evaluation_status=partial` and cannot receive a positive evidence
classification. `evaluation_status=complete` is reserved for candidates that
evaluate every requested fold.

The default `auto` cost model uses complete historical bar spreads when
coverage is sufficient. If coverage is below 90%, `auto` substitutes a
disclosed conservative fixed estimate and still marks the cost model complete.
`historical_bar_spread` is stricter: incomplete coverage is disclosed and
prevents a positive evidence classification. Use `fixed` with an explicit
`spread_bps` for a controlled constant-cost comparison. An insufficient
forecast-threshold candidate reports the
required trade count, computed-anchor coverage, long/short/neutral counts, and
a reason distinguishing unavailable forecasts from an uncrossed threshold.

`strategy_backtest` also defaults to `cost_model=auto`. When you want a
controlled constant instead, pass `--cost-model fixed --spread-bps <value>`
to either tool. Both strategy tools accept `commission_bps_per_side`, deducted
on entry and exit, and default to zero commission plus 1 bps slippage per side.
The explicit `historical_bar_spread` policy does not run a backtest unless
historical spread coverage is complete. `strategy_backtest` reports
`cost_quality=observed`, `imputed`, or `user_assumption` beside its result
status so an auto-selected fixed spread cannot look like observed history.

Same-bar TP/SL touches default to `sl_first` and are echoed in the result.
`max_drawdown` is always the non-negative peak-to-trough return magnitude, in
the same convention used by the backtest tools.

## Portfolio risk decomposition

`portfolio_risk_decompose` maps current MT5 positions into account-currency
filtered-historical scenarios. It returns multi-horizon VaR/CVaR,
component CVaR, concentration, prescribed stresses, and optional proposed-trade
incremental CVaR and margin. `ewma_half_life` applies only to
`method=filtered_historical`; `bootstrap_historical` omits it from
`model_context` and rejects a non-default supplied value. Both portfolio
methods resample historical windows (`scenario_generation` is
`ewma_filtered_bootstrap_windows` or `bootstrap_historical_windows`) and are
not the empirical-quantile `historical` method on `trade_var_cvar_calculate`.
Multi-bar log-return paths are converted to compounded simple returns before
they are applied to account-currency position sensitivities.

When `proposed_trade` is supplied, its symbol is resolved against the broker
catalog and its volume is validated against that symbol's minimum, maximum, and
lot step before any scenarios run. Invalid requests return the constraints and
the nearest valid volume instead of modeling a trade the broker would reject.
`side` accepts `buy`/`sell` or `long`/`short` and is echoed as canonical
`buy`/`sell`. The proposed trade is marked from one frozen quote snapshot
(`ask` for buy, `bid` for sell) and the response includes `mark_price`,
`mark_price_basis`, and `quote_time`.

```bash
mtdata-cli portfolio_risk_decompose --timeframe H1 --lookback 1000 --horizon-bars 1,5 --confidence 0.95,0.99 --json
```

The default fails closed if a material position cannot be priced safely. Use
`--allow-partial true` only when an explicitly partial portfolio result is
acceptable. Fail-closed coverage includes both live sensitivity pricing and
the completed return history required by the scenario model. Partial results
list every omitted symbol and the omission stage in `data_quality`.

The perfect-positive-correlation stress applies a common one-sigma factor to
horizon marginal volatilities. Opposing sensitivities therefore offset.

## Relative strength and breadth

`market_relative_strength` ranks a bounded MT5 universe with volatility-scaled,
factor-adjusted momentum across several horizons. It also reports breadth,
temporal rank stability, live spread, per-symbol bar/alignment windows, and
data-coverage exclusions. Standardized robust z-scores and rank percentiles
require at least 10 scored symbols; smaller universes keep ordinal ranks,
withhold unbounded z-scores, and add a `universe_sensitivity` warning.
`limit` is a global output cap split between the strongest and
weakest tails; odd limits assign the extra row to leaders. Full detail exposes
the same bounded selection as `rankings`, not an unbounded universe dump.
Ranking membership is based on completed-bar history; a stale or closed-session
quote is retained as quality metadata unless an explicit spread filter cannot
be evaluated. If candidate latest-bar endpoints exceed one timeframe of
separation, the tool returns `status=incomparable` and withholds ranks and
breadth instead of publishing a misleading cross-section.
The `units` map declares live `spread_pct` as percentage points (`1.0 = 1%`).
Breadth paths are fractions from 0 to 1, except
`breadth.advance_decline_balance`, which is signed from -1 to 1, and
`breadth.dispersion`, which is a composite-score standard deviation.

```bash
mtdata-cli market_relative_strength --group "Forex\\Majors" --timeframe H1 --horizons 5,20,60 --weights 0.2,0.3,0.5 --limit 10 --json

# Pairwise comparison (no cross-sectional breadth is implied)
mtdata-cli market_relative_strength GBPUSD --benchmark EURUSD --timeframe H1 --json
```

A single explicit candidate is supported only with an external benchmark. It
returns `status=compared`, a direct volatility-scaled residual-momentum score,
and `breadth.status=not_applicable_pairwise`; it does not present the result as
a multi-symbol rank. Unknown group names fail as `symbol_group_error` before
history retrieval and include broker paths from `symbols_list` for correction.

Use homogeneous symbol groups when possible. Instruments with substantially
different trading sessions can produce less comparable cross-sectional ranks.
Omitting both `symbols` and `group` intentionally ranks the bounded visible
Market Watch universe, which may mix asset classes; use `--group` or explicit
symbols when that mixed-universe behavior is not desired.
Inspect `data_window.endpoint_alignment` before comparing mixed-session
instruments. Per-symbol windows are available in full-detail data-quality
diagnostics; compact and summary ranked rows expose concise quote/history status
fields when the endpoints are comparable.
When no symbols can be scored, `empty_reason` and `empty_reason_counts` identify
the actual exclusions. Alignment failures report available versus required
observations; spread and tick-volume messages name their filter only when that
filter was applied.

## Data caveats

- Historical tick and candle availability is controlled by the broker and the
  terminal's local history.
- FX `tick_volume` is a broker tick count, not traded lots.
- `last` and `volume_real` are commonly zero for OTC instruments.
- DOM is not required by these tools and remains a separate, gated live
  snapshot through `market_depth_fetch`.
- Volume-impact estimates describe only the connected broker's tick feed,
  even when `volume_real` is present.
- The focused FastAPI/Web UI does not expose these tools in v1; use MCP or the
  dynamic CLI.

Built-in strategy-validation candidates reject unknown parameter names. Moving
averages accept positive integer `fast_period` and `slow_period` with fast less
than slow; state-reversal variants also accept `max_hold_bars`. RSI accepts a
positive integer `rsi_length` and `0 < oversold < overbought < 100`. Parameters
reported as effective have been validated and used.

Forecast-threshold candidates validate method/parameter names before fitting.
A configuration or forecast error returns `evaluation_status=failed`, its
`failure_stage`, `first_error`, and failed-anchor count. Evaluation stops on the
first failed fit so skipped failures cannot bias a candidate's score. Other
candidates still run. `insufficient_data` is reserved for sample or signal
shortages, including thresholds that were never crossed.
