# CLI Guide

**Audience:** User

Type one command, get one result. That is the whole idea:

```bash
mtdata-cli <command> [options]
```

Try this first (read-only — it only *lists* markets):

```bash
mtdata-cli --help
mtdata-cli symbols_list --limit 5
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 20
```

Add `--json` when you want a structured result for a script or an assistant.
The default text layout is compact and human-readable ([TOON](GLOSSARY.md#toon)
— think “a small table with a header”). Stuck on an acronym (BOCPD, Kelly,
CVaR, …)? See the [glossary quick find](GLOSSARY.md#quick-find).

Prefer a website or a chat assistant? [WEBUI.md](WEBUI.md) · [MCP.md](MCP.md).

**Related:** [README](../README.md) · [Setup](SETUP.md) · [Glossary](GLOSSARY.md) · [Market discovery](MARKET.md) · [Output contract](OUTPUT.md) (Operator)

---

## Deeper detail: shell, batches, and long jobs

Each one-shot `mtdata-cli` start loads Python, runs one command family, and
exits. For a warmer loop, run `mtdata-cli shell` and type commands without the
`mtdata-cli` prefix until `exit` or `quit`.

The shell also accepts newline-delimited commands on stdin. Blank lines and
`#` comments are ignored. The process exits nonzero if any command fails.
For mixed failures, usage status `2` takes precedence over tool/provider status
`1`, regardless of command order.
Batch output is [NDJSON](GLOSSARY.md#ndjson): one JSON object per input line,
with `line`, `command`, `success`, and `status`. Parsed child JSON sits under
`result`; leftover text uses `output` / `stderr`. Shared options may follow
`shell`, for example `mtdata-cli shell --json --timeframe H4`. A shell
timeframe applies only to child commands that accept one.

Keep a long-lived process for agents and apps: `mtdata-stdio`,
`mtdata-streamable-http`, or `mtdata-webapi`.

One-shot `forecast_train` commands and stdin batches wait in the foreground so
their worker remains alive and the command returns the stored `model_id`.
Interactive shell, MCP, and Web API calls submit training in the background by
default. `forecast_generate --async-mode true` still requires one of those
persistent processes.

---

## Safety (trading commands)

`trade_*` can place, modify, or close **real** orders on the account logged into MT5.

- Prefer a **demo account** while learning.
- mtdata has no separate paper mode — demo terminal = simulated execution.
- Preview with `--dry-run true` before live sends.

| Safer research | Live execution |
|----------------|----------------|
| `symbols_*`, `market_*`, `data_fetch_*` | `trade_place` |
| `forecast_*`, `regime_detect`, `patterns_detect` | `trade_modify` |
| `report_generate`, `trade_risk_analyze`, `trade_get_*` | `trade_close` |

`trade_place`, `trade_modify`, and `trade_close` default to **preview mode** (`dry_run=true`). Set `--dry-run false` explicitly for a live request. Ticketless bulk closes require `--confirm-close-all true`; `--close-all` is needed only for an account-wide scope without a symbol or magic filter. Booleans on the CLI are `true` / `false`.

Full runbook: [TRADING_SAFETY.md](TRADING_SAFETY.md).

## Getting help

```bash
# List all commands
mtdata-cli --help

# Search by topic
mtdata-cli --help forecast
mtdata-cli --help barrier
mtdata-cli --help regime

# Help for one command
mtdata-cli forecast_generate --help
mtdata-cli regime_detect --help

# Tool catalog (filter / paginate)
mtdata-cli tools_list --category forecast --json

# Advance through the 20-tool default pages
mtdata-cli tools_list --limit 20 --offset 20 --json

# Machine-usable input schemas, constraints, defaults, and CLI forms
mtdata-cli tools_list --search portfolio_risk_decompose --detail full --json
```

Root-help headings use the same category identifiers accepted by `tools_list`:
`analysis`, `data`, `forecast`, `market`, `methods`, `news`, `options`,
`pattern_regime`, `report`, `research`, `symbols`, and `trading`. Each heading
prints its corresponding `tools_list --category ID` filter, so the visible
command group and machine-readable result stay identical.

A bare `tools_list` returns the first 20 callable tools. Use `--offset` for the
next page, or pass a sufficiently large explicit `--limit` when you need the
complete catalog in one response.

Compact catalog output points to the versioned full parameter schema. In
`--detail full`, each tool includes its canonical `input_schema` plus per-field
`parameters` metadata with requiredness, defaults, descriptions, constraints,
and positional/option CLI forms. The tool-level `cli` inventory comes from the
same completed argparse command parser used at runtime, so it also records
parser-only controls such as `--set` and `--print-config`, companion mapping
options such as `--denoise-params`, aliases such as `--days`, hidden
compatibility tokens, and value transformations. Nested request objects remain
linked through the schema's `$defs` references.

Optional positional values also expose their canonical named form in command
help—for example, both `market_status EURUSD` and
`market_status --symbol EURUSD` are discoverable. A help-keyword miss prints
only close suggestions and discovery commands; use bare `--help` for the full
catalog.

One-shot `tools_list`, `forecast_list_methods`, and
`forecast_list_library_models` results are cached on disk after a successful
build. The key includes the command arguments, mtdata source state, relevant
environment settings, and installed forecast-library versions. Responses expose
`catalog_source: rebuilt|cached`; a source edit or dependency-version change
causes an automatic rebuild. A single-library model query only discovers that
library.

---

## Output contract

JSON (`--json`) is the structured machine representation. It does not imply
`detail=full`: compact JSON and compact TOON share the same semantic fields.
CLI, MCP, and Web API share the success/error and output-shaping contract; see
[OUTPUT.md](OUTPUT.md). Scripts and agents that need JSON types instead of TOON
text must pass `--json`.

### TOON (Default)
Human-readable compact TOON output:
```bash
mtdata-cli symbols_list --limit 5
```

TOON includes a quick schema hint:
```
data[5]{symbol,group,description,currency_base,currency_profit,digits}:
    EURUSD,Forex\Majors,Euro vs US Dollar,EUR,USD,5
    ...
```
- `data[5]` is the number of rows returned
- `{symbol,...,digits}` lists the columns/keys in each row. Compact collection
  output may omit a column when every returned row carries the same default
  value; use `--detail full` when you need a guaranteed field set.

### JSON
Structured output for programmatic use:
```bash
mtdata-cli symbols_list --limit 5 --json
```

For scripts that always require JSON, set `MTDATA_OUTPUT_FORMAT=json` in the
environment or `.env` file. Accepted values are `json` and `toon`; an explicit
`--json` flag always selects JSON. A nonblank unsupported environment value is
a configuration error and exits with status 2 instead of changing formats.

JSON output always keeps numeric values unminimized. Text output uses
`--precision auto`, which compacts most tools while preserving full precision
for an explicit set of sensitive outputs, including trading, quotes, forecasts,
reports, and price-level tools.

Control display precision explicitly:
```bash
# Preserve full numeric precision in TOON text
mtdata-cli market_ticker EURUSD --precision full

# Compact a large table for token-saving display
mtdata-cli data_fetch_candles EURUSD --limit 200 --precision compact

```

`--precision raw` is accepted as an alias for `full`, and `display` is accepted
as an alias for `compact`. There is no global `--decimals` option; tools with a
domain-specific decimal control document it in their own help. Precision
controls only text presentation; internal tool processing and JSON/raw payloads
keep numeric values.

### Detail and field selection
Compact output is implicit. `--detail` is a per-command domain parameter, not
a global CLI option. When a command's `--help` lists it, use `--detail full`
for richer runtime metadata, diagnostics, request context, and supporting rows:
```bash
mtdata-cli market_status --detail full
```

Commands whose help does not list `--detail` have one output shape and reject
the option. `--json`, `--output-fields`, and `--precision` are the global
presentation options.

Use `--output-fields` to project the response without changing domain
semantics. It can retrieve a targeted full-detail path without returning the
whole full payload. Compact warnings and trading safety gates remain attached:
```bash
mtdata-cli symbols_describe EURUSD --output-fields symbol,details.digits,details.point --json
mtdata-cli market_ticker EURUSD --output-fields bid,ask,spread --json
mtdata-cli data_fetch_candles BTCUSD --output-fields symbol,source.server --json
```

Bare field names address top-level keys; dotted paths address nested keys.
Misspelled or unavailable paths are reported in `unresolved_output_fields`.
When some requested paths resolve, the usable values are retained and
`output_fields_status=partial` makes the incomplete projection explicit. When
none resolve, the response has `success=false`,
`error_code=output_fields_unresolved`, and the CLI exits `1`. Use
`valid_output_fields` to retry with available compact or targeted-rich paths.
Use canonical paths such as `meta.processing.indicators.engine` when you want
the consolidated full-detail spelling. Compact `trade_get_open` rows keep
`magic` and `comment`, so strategy-attribution projection does not require
full detail.
Stable declared row paths remain valid when their collection is empty.

When `--output-fields` is set, JSON and TOON retain the same selected keys.
Without it, JSON and TOON both use the requested detail level. Use `--json`
when you need structured types, and `--detail full` when you need the complete
metadata envelope. TOON may also apply numeric display precision.

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Command completed without a tool error |
| `1` | Tool/provider failure, unresolved output projection, invalid tool payload, interrupted command, internal CLI error, or no command selected |
| `2` | Argument parsing or command-line usage error, including a missing required symbol |

Scripts should parse JSON error fields when they need to distinguish provider,
validation, and internal failures that share exit code `1`.
Trading dry-runs with `preview_ok=false` return `success=false`, print the
retained preview body, and exit `1`; eligible previews return `success=true`,
`preview_ok=true`, and exit `0`. Always inspect `blockers` before live use.

---

## Common Patterns

### Positional Arguments
Most commands take `symbol` as the first positional argument:
```bash
mtdata-cli forecast_generate EURUSD --horizon 12
mtdata-cli regime_detect EURUSD --method hmm
mtdata-cli data_fetch_candles EURUSD --limit 100
```

FX aliases such as `EUR/USD` and `eurusd` resolve to the broker name
(`EURUSD`). When the input differs from the resolved contract, the payload
echoes both `symbol` and `symbol_input`. Equity suffixes are load-bearing:
`TSLA.NAS` and `TSLA.NAS-24` are different contracts.

### Timeframe
Specify market data granularity with `--timeframe`:
```bash
mtdata-cli data_fetch_candles EURUSD --timeframe M15 --limit 100
mtdata-cli forecast_generate EURUSD --timeframe H4 --horizon 24
```

Available timeframes: `M1`, `M2`, `M3`, `M4`, `M5`, `M6`, `M10`, `M12`, `M15`, `M20`, `M30`, `H1`, `H2`, `H3`, `H4`, `H6`, `H8`, `H12`, `D1`, `W1`, `MN1`. Broker history availability may vary.

### Parameters
Pass method-specific parameters with `--params`:
```bash
mtdata-cli forecast_volatility_estimate EURUSD --method ewma --params "lambda_=0.94"
mtdata-cli regime_detect EURUSD --method hmm --params "n_states=3"
```

Format: `key=value key2=value2` (space-separated), `key=value,key2=value2`
(comma-separated), or JSON `{"key": value}` — all three are accepted. Compact
values use JSON-like scalar types: `true`/`false`, `null`, finite numbers, arrays,
and objects become native values. Quote a numeric-looking value, such as
`code="001"`, when it must remain a string.

### Reduce Large Outputs (Simplify)
Use `--simplify` to downsample returned rows for charting or large exports.

```bash
# Default simplification (targets ~10% of --limit)
mtdata-cli data_fetch_candles EURUSD --timeframe M1 --limit 5000 --simplify

# Choose an algorithm + target points
mtdata-cli data_fetch_candles EURUSD --timeframe M1 --limit 5000 --simplify lttb --simplify-params "points=500"

# Raw ticks can also be simplified
mtdata-cli data_fetch_ticks EURUSD --limit 20000 --simplify rdp --simplify-params "points=2000"
```

Full-detail tick rows preserve MT5 snapshot fields: `bid`, `ask`, `last`,
`volume`, `volume_real`, and `flags`. Compact rows omit unavailable last-trade
and zero-volume fields, but always include `spread_valid`; the response-level
`last_unavailable` and `volume_fields` describe omissions. The volume fields
describe the current last trade, not a row-level tick count; use full-detail
`flags` to identify trade-change events. Candle rows continue to use
`tick_volume` for the broker's per-bar Bid-update count. Tick history uses all
`COPY_TICKS_ALL` records, so its `tick_count` can be larger when ask-only updates
occur. Use `bid_update_count` to compare the tape with candle `tick_volume`;
the response fields `tick_volume_event_basis` and `tick_count_event_basis`
make both definitions explicit.

See [SIMPLIFICATION.md](SIMPLIFICATION.md) for algorithms and parameters.

### Method Parameters
For forecast methods, use `--params`:
```bash
mtdata-cli forecast_generate EURUSD --method arima --params "p=2 d=1 q=2"
mtdata-cli forecast_generate EURUSD --method mc_gbm --params "n_sims=2000 seed=42" --ci-alpha 0.05
```

Free-form `--params` and detector `--config` mappings are strict: an unknown
key fails before data fetching or model execution and reports the valid keys
for the selected method, with close-name suggestions when available. Put
top-level options at the top level; for example, use `--return-grid false` on
`forecast_barrier_optimize`, not `return_grid=false` inside `--params`.

---

## Date Inputs

Commands accepting `--start` and `--end` parse flexible date strings:
```bash
# Relative dates
mtdata-cli data_fetch_candles EURUSD --start "2 days ago" --end "now"
mtdata-cli data_fetch_candles EURUSD --start "1 week ago"

# Absolute dates
mtdata-cli data_fetch_candles EURUSD --start "2025-12-01" --end "2025-12-31"
```

For intraday candle ranges, bounds are inclusive and resolved in UTC. An ISO
date-only value and natural calendar days such as `today`, `yesterday`, or
`last Friday` span the full UTC day: `--start` resolves to 00:00:00 and `--end`
to 23:59:59.999999. Week phrases span Monday through Sunday. `this month`,
`last month`, and `next month` span calendar months; `this year`, `last year`,
and `next year` span calendar years. For D1/W1/MN1, those same calendar labels
select broker session periods instead: their resolved bounds use broker-local
midnight and may therefore fall on the previous UTC date. Relative durations
such as `2 days ago` remain exact instants. Include an explicit time
and timezone when automation needs an exact instant, using either an ISO offset
(`2026-08-03T09:30-04:00`) or an IANA name
(`2026-08-03 09:30 America/New_York`). Ambiguous or nonexistent daylight-saving
local times are rejected; use an ISO offset to choose a specific instant.
Candle responses echo the resolved instants and bound modes in `query_applied`.
A date-only `--end` for the current UTC day is allowed (it has not fully
elapsed yet); `query_applied.effective_end` then records the clamp to now, and
`end_clamped_to` is `now`.
For an exact timestamp end, only candles whose close is at or before that
instant are returned; completed OHLC values from an overlapping bar are never
included early. Calendar-period ends retain their date/session-label behavior.
D1/W1/MN1 date-only bounds require `MT5_SERVER_TZ` (or a non-zero
`MT5_TIME_OFFSET_MINUTES`); they do not silently assume UTC.

Tick `--start`/`--end` date-only values and calendar phrases such as `today`
always resolve in UTC, not the broker D1 session: `--start 2026-08-14` begins
at `2026-08-14T00:00:00Z` and `--end 2026-08-14` includes through UTC
end-of-day. Pass an explicit timestamp when the window must match a broker
session open.
When `--start` and `--limit` are combined, candles are returned in ascending
order from the start bound (first-N). When `--limit` is omitted, range queries
return a 20-bar page. Follow `pagination.next_cursor` with the original range
arguments and selection to continue, or pass an explicit larger `--limit` when
one larger response is intentional. `--selection last_n` starts at the latest
end of a range; following its cursor moves backward through earlier candles.
Every page keeps its rows in ascending time order. Omit `--start` for latest-N
retrieval.

---

## Command Categories

### Data
| Command | Description |
|---------|-------------|
| `symbols_list` | List available trading symbols |
| `symbols_describe` | Get symbol details (pip size, contract, etc.) |
| `symbols_top_markets` | Rank the top MT5 markets by spread, recent volume, or recent price change |
| `market_scan` | Filter MT5 symbols by spread, price change, volume, RSI, and SMA |
| `market_radar` | Compact watchlist scan (quote, spread, change, freshness; max 20 names) |
| `data_fetch_candles` | Fetch OHLCV candles with optional indicators |
| `data_fetch_ticks` | Fetch tick data (historical window capped at 30 days before `--end`) |
| `market_depth_fetch` | Get order book (DOM) — requires `MTDATA_ENABLE_MARKET_DEPTH_FETCH=1` |
| `market_ticker` | Get current bid/ask/spread snapshot |
| `market_snapshot` | Unified pre-trade snapshot (quote, levels, patterns; optional regime/forecast sections) |
| `market_status` | Show the major-equity exchange calendar globally or by explicit venue, or MT5 tradability for a broker symbol |
| `wait_event` | **Blocking:** wait for a candle close or a timed event — see [WAIT_EVENT.md](WAIT_EVENT.md) |

### Forecasting
| Command | Description |
|---------|-------------|
| `forecast_generate` | Generate price forecasts |
| `forecast_list_methods` | List available forecasting methods |
| `forecast_list_library_models` | List models in a specific library |
| `forecast_backtest_run` | Run rolling-origin backtest |
| `strategy_backtest` | Backtest simple indicator-driven trading strategies |
| `forecast_conformal_intervals` | Generate calibrated confidence bands |
| `forecast_volatility_estimate` | Forecast volatility |
| `forecast_tune_genetic` | Optimize model parameters (genetic algorithm) |
| `forecast_tune_optuna` | Optimize model parameters (Bayesian/Optuna) |
| `forecast_optimize_hints` | Genetic search for top forecast configurations across timeframes, methods, and parameters |

### Async Training & Model Store
| Command | Description |
|---------|-------------|
| `forecast_train` | Start a background training job for heavyweight methods (returns a `task_id`) |
| `forecast_task_status` | Poll training progress for a `task_id` |
| `forecast_task_wait` | Wait for a task to finish or until a timeout is reached |
| `forecast_task_cancel` | Cancel a running training task |
| `forecast_task_cancel_all` | Cancel all active training tasks |
| `forecast_task_list` | List active and recent training tasks |
| `forecast_models_list` | List trained models cached on disk |
| `forecast_models_delete` | Preview one stored model; confirmed apply permanently deletes it |
| `forecast_models_cleanup` | Preview or delete stale/expired stored models |

Trained models are written under `~/.mtdata/models/` by default and reused by
live `forecast_generate` calls with the same method, symbol, timeframe, horizon,
seasonality, preprocessing, and training parameters. Results report model age
in bars, and live reuse is capped at one resolved seasonal cycle. Historical
`as_of` forecasts require an exact training anchor. Task
status is persisted in `~/.mtdata/forecast/jobs.sqlite` by default, so recent
task state can survive process restarts. See
[ENV_VARS.md](ENV_VARS.md#async-training--model-store) for related variables.

### Risk Analysis
| Command | Description |
|---------|-------------|
| `forecast_barrier_prob` | Calculate TP/SL hit probabilities |
| `forecast_barrier_optimize` | Find optimal TP/SL levels |
| `labels_triple_barrier` | Label data with barrier outcomes |
| `regime_detect` | Detect market regimes and change points |

### Time-Series Diagnostics
| Command | Description |
|---------|-------------|
| `stationarity_test` | Run ADF, KPSS, and optional Phillips-Perron tests |
| `seasonality_detect` | Rank dominant periods using autocorrelation and spectral peaks |
| `outliers_detect` | Detect anomalous return, volume, and range bars |
| `volatility_term_structure` | Compare realized volatility across rolling horizons and historical percentiles |

### Indicators & Patterns
| Command | Description |
|---------|-------------|
| `indicators_list` | List available indicators |
| `indicators_describe` | Get indicator details |
| `patterns_detect` | Detect candlestick, chart, harmonic, fractal, and Elliott patterns |
| `volume_profile_levels` | Compute POC, VAH, and VAL from bounded ticks or M1-bar approximation |
| `confluence_levels` | Rank multi-source consensus zones (pivots + S/R + Fibonacci + optional volume profile) |
| `pivot_compute_points` | Calculate pivot levels |
| `support_resistance_levels` | Compute single-source structural support/resistance around the current price |
| `correlation_matrix` | Pairwise correlation matrix between symbols |
| `cross_correlation` | Estimate lead/lag correlation between two symbols |
| `cointegration_test` | Engle-Granger pair tests or Johansen multivariate cointegration |
| `causal_discover_signals` | Granger predictive-link discovery between symbols |

The root `--timeframe` default maps to `--pivot-timeframe` for
`confluence_levels`; an explicit command-level `--pivot-timeframe` takes
precedence. `--sr-timeframe` remains independent and defaults to `auto`.

Volume profile example:

```bash
mtdata-cli volume_profile_levels EURUSD --start "1 day ago" --end "now" --source auto --price-source mid --bucket-points 10 --json
```

You can also derive the window from a lookback:

```bash
mtdata-cli volume_profile_levels EURUSD --timeframe H1 --lookback 168 --source auto --bucket-points 10 --json
```

The default tick window is one day and 50,000 ticks. Natural one-day relative
windows are treated as inside that budget despite sub-second parser skew. Longer
`auto` windows use the labeled M1-bar approximation unless those caps are raised
explicitly. `profile_source` and `source_decision` disclose the construction
method; `source` retains structured MT5 broker provenance.

For fractal + volume-structure confluence, opt in through pattern config:

```bash
mtdata-cli patterns_detect EURUSD --timeframe H1 --mode fractal --config '{"volume_profile":true,"volume_profile_tolerance_points":25}' --json
```

See [LEVELS.md](LEVELS.md) for the full pivots, support/resistance, confluence, and volume-profile reference.

### Denoising
| Command | Description |
|---------|-------------|
| `denoise_list_methods` | List denoise methods with their dependencies, causality support, and auto parameters |
| `denoise_describe` | Describe one denoise method and its supported options and defaults |

Denoising is applied to data via the `--denoise`/`--denoise-params` flags (see [Reduce Large Outputs](#reduce-large-outputs-simplify) and the examples below). Use `denoise_list_methods`/`denoise_describe` to discover method names and parameters first. See [DENOISING.md](DENOISING.md) for the full reference.

### Trading
| Command | Description |
|---------|-------------|
| `trade_account_info` | Get account info |
| `trade_session_context` | Snapshot of broker/session/server-time context for downstream trading prompts |
| `trade_idea_compose` | Compose a preview-only research idea (forecast, barriers, size, dry-run) |
| `trade_place` | Place orders |
| `trade_close` | Close positions |
| `trade_modify` | Modify orders |
| `trade_get_open` | Get open positions |
| `trade_get_pending` | Get pending orders |
| `trade_history` | Get trading history |
| `trade_journal_analyze` | Summarize realized exit-deal performance |
| `trade_execution_quality` | Analyze slippage, latency, partial fills, fees, and markouts |
| `trade_risk_analyze` | Analyze position risk |
| `trade_var_cvar_calculate` | Estimate portfolio VaR/CVaR from open positions |
| `trade_stress_test` | Apply deterministic percentage shocks to open positions |

See [TRADING_RISK.md](TRADING_RISK.md) for position sizing (fixed-fraction + Kelly), VaR/CVaR, and stress-test parameters and output.
`trade_journal_analyze` reports account-currency PnL per realized exit. When the
matching entry fill is present in the requested history window, entry commission
and fees are allocated by closed volume. Check `entry_cost_coverage` and
`pnl_basis`: unmatched exits remain exit-deal-only and may overstate net PnL.
These averages are useful for journal review, but they are not Kelly inputs
because they are not normalized to a consistent stake or unit of risk.

### News, calendar, and company context
| Command | Description |
|---------|-------------|
| `news` | Ranked headlines + event buckets. Pin `source` or use `view=ticker` / `view=market` for a raw provider page. See [NEWS.md](NEWS.md). |
| `calendar` | Filterable economic / earnings / dividend table (`--kind`, `--view period` for this-week earnings). |
| `equity_profile` | US-issuer dossier (`--sections` summary, description, ratings, peers, insider). |
| `screener` | Equity screen, or `--list-filters true` for the filter catalog. |
| `asset_performance` | Delayed forex/crypto/futures/insider context (`--universe`). Not a live broker quote. |

These four table/dossier commands currently use Finviz as the research adapter.
Exchange tickers such as `AAPL` are accepted; broker suffixes such as
`AAPL.NAS` are normalized and reported as `requested_symbol` / `finviz_ticker`.
Full examples: [FINVIZ.md](FINVIZ.md) (User). Everyday headlines: [NEWS.md](NEWS.md).

### Advanced MT5-native analytics

| Command | Description |
|---------|-------------|
| `market_microstructure_analyze` | Analyze tick liquidity and feed-appropriate order-flow proxies |
| `strategy_validate` | Run anchored fixed-candidate OOS validation with horizon-safe barrier outcomes and costs |
| `portfolio_risk_decompose` | Decompose filtered-historical VaR/CVaR and proposed-trade risk |
| `market_relative_strength` | Rank a bounded MT5 universe by robust factor-adjusted momentum and breadth |

See [ADVANCED_ANALYTICS.md](ADVANCED_ANALYTICS.md) for data requirements, examples, and caveats.

### Reports
| Command | Description |
|---------|-------------|
| `report_generate` | Generate a fast market-context and forecast report; use `--template basic` for broader analysis |

### Temporal Analysis
| Command | Description |
|---------|-------------|
| `temporal_analyze` | Analyze returns, volatility, and volume by time period (day of week, hour, month) |

### Options & QuantLib
| Command | Description |
|---------|-------------|
| `options_provider_status` | Report options-chain configuration and static request support without claiming live data readiness |
| `options_expirations` | List available option expiration dates |
| `options_chain` | Fetch options chain snapshot with filtering |
| `options_barrier_price` | Price a barrier option using QuantLib |
| `options_heston_calibrate` | Calibrate Heston stochastic volatility model |

See [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md) for detailed examples.

---

## Examples by Task

### Explore Available Symbols

Longer tour: [MARKET.md](MARKET.md). News tour: [NEWS.md](NEWS.md). Waits:
[WAIT_EVENT.md](WAIT_EVENT.md).
```bash
# List forex pairs
mtdata-cli symbols_list --limit 20

# Get details for a symbol
mtdata-cli symbols_describe EURUSD --json

# Rank the current watchlist by spread, volume, signed change, and absolute change
mtdata-cli symbols_top_markets --rank-by all --limit 5 --timeframe H1 --json

# Include hidden symbols within a bounded comparable category
mtdata-cli symbols_top_markets --rank-by spread --limit 10 --universe all --category forex --json

# Wait for one exact global stock leaderboard (large broker catalogs may take minutes)
mtdata-cli symbols_top_markets --rank-by spread --limit 10 --universe all --category stocks --scan-budget-seconds 0 --json

# Scan visible majors for strong RSI and price above SMA
mtdata-cli market_scan --group "Forex\\Majors" --rsi-above 60 --price-vs-sma above --sma-period 20 --timeframe H1 --lookback 120 --json

# Scan an explicit symbol basket for oversold names with tight spreads
mtdata-cli market_scan EURUSD,GBPUSD,USDJPY --rsi-below 35 --max-spread-pct 0.03 --json

# Multi-symbol selectors use the canonical `symbols` selector.
```

`market_scan` and `symbols_top_markets` keep completed-bar values such as
`close` separate from the current executable quote. When a quote is available,
rows expose `bid`, `ask`, `mid`, and `quote_as_of`; use those fields for a live
mark and the bar fields for ranking and indicator context. Price-change rows
also expose `live_price_change_pct`, measured from the previous completed close
to the current midpoint, so a forming-bar reversal is explicit. Use
`--rank-by live_price_change_pct` for gainers or
`--rank-by abs_live_price_change_pct` for two-sided live movers. Spread-ranked
and live-ranked scans plus the `tight_spread` preset exclude quotes that are not
usable for live trading before pagination; pass `--quote-usable-only false`
only when inspecting stale or otherwise non-executable snapshots intentionally.
Other rankings keep such rows by default because their scores use completed
bars; compact rows therefore include `spread_quality` and
`quote_usable_for_live_trading` so a locked or otherwise unsafe live quote
cannot look executable.

Price-change rankings compare the previous completed close with the latest
completed close over exactly one requested `timeframe` bar. Responses expose
that window in `price_change_period`, and compact rows include the per-symbol
completed-bar `time`. When returned symbols do not share that timestamp, the
response omits a single `data_as_of`, reports `data_as_of_range`, and sets
`bar_time_alignment.comparable` and `price_change_comparable` to `false`.
Treat those rows as separate session windows rather than one clock-aligned
leaderboard. `symbols_describe` reports the broker's
native `price_change` field when available. MT5 defines it as the current quote
relative to the previous trading day's close, so the response identifies that
distinct live window in `price_change_basis` and `price_change_period`.
Describe responses also include the live `bid`, `ask`, `mid`, and spread metrics
used by their freshness and execution-readiness fields.

`symbols_list` rejects non-positive limits. `symbols_top_markets` manages large
filtered universes itself. Its default 30-second sampling budget returns useful
rows with `ranking_scope=partial_global`, `ranking_complete=false`, and
`candidate_progress` if the scan does not finish. Set
`--scan-budget-seconds 0` to wait for an exact one-command global leaderboard;
large stock catalogs can take many minutes because MT5 must activate hidden
quotes serially. Candidate offset/limit controls remain available as advanced
recovery partitions, and those results are labeled `candidate_partition`.
Continue at `candidate_page.next_offset`, which reflects candidates attempted
before any timeout. Keep the same universe and filters, and merge page results
until `candidate_page.has_more` is false. Do not advance by the requested limit
after a partial page: some candidates may not have been attempted yet.

Hidden-symbol activation is temporary. MTData serializes it with visible
Market Watch snapshots across local processes, so concurrent
`symbols_list --universe visible` and default rankings do not observe
internally activated symbols. A visible-universe read may briefly wait for the
current symbol sample to finish.

### Fetch Market Data
```bash
# Basic candles
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 100

# With indicators
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 100 --indicators "ema(20),rsi(14),macd(12,26,9)"

# Compute ATR from full OHLCV, then return only close plus derived columns
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 100 --indicators "atr(14)" --ohlcv close

# With denoising
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 100 --denoise ema --denoise-params "alpha=0.2"
```

`--ohlcv` is an output projection, not an input restriction. Indicators and
denoise stages receive the full source OHLCV before the requested candle fields
are trimmed; their derived columns remain in the response. The
`processing_pipeline` and `ohlcv_filter` fields disclose that order and selector.

### Generate Forecasts
```bash
# Basic forecast
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta

# Foundation method
mtdata-cli forecast_generate EURUSD --library pretrained --method chronos2 --horizon 24

# Monte Carlo simulation
mtdata-cli forecast_generate EURUSD --method mc_gbm --params "n_sims=2000" --ci-alpha 0.05

# Search timeframes + methods + params for the best starting configuration
mtdata-cli forecast_optimize_hints EURUSD --timeframes H1 H4 D1 --methods theta ets --horizon 12 --steps 30 --top-n 5 --json
```

### Backtest Trading Rules
```bash
mtdata-cli strategy_backtest EURUSD --timeframe H1 --strategy sma_cross --fast-period 10 --slow-period 30 --lookback 300 --cost-model fixed --spread-bps 1.2 --commission-bps-per-side 0.25 --json

mtdata-cli strategy_backtest EURUSD --timeframe H1 --strategy rsi_reversion --rsi-length 14 --oversold 30 --overbought 70 --position-mode long_only --cost-model fixed --spread-bps 1.2 --commission-bps-per-side 0.25 --json

# For a controlled constant instead of historical bar spreads:
mtdata-cli strategy_backtest EURUSD --cost-model fixed --spread-bps 1.2 --json
```

The runnable examples use a fixed 1.2 bps round-trip assumption; replace it with
a defensible value for the instrument and venue. By default,
`strategy_backtest` uses `cost_model=auto`: complete historical bar spreads
when coverage is full, otherwise a disclosed conservative fixed estimate from
available spread stats or the current broker quote. It never silently uses
zero costs. `historical_bar_spread` still fails closed unless every required
bar has a usable spread. The `fixed` model requires an explicit `spread_bps`;
it never substitutes a current quote into historical trades unless you chose
`auto`.
Both `strategy_backtest` and `strategy_validate` name commission
`commission_bps_per_side` and deduct it twice per round trip. Their default
commission is zero and their default slippage is 1 bps per fill side. Barrier
and tuning searches require explicit costs when the objective uses trading
metrics, so an omitted search cost is not silently replaced with these defaults.
`strategy_validate` may evaluate with at least some historical
spread observations, but coverage below 90% prevents a positive evidence
classification. Select `fixed` with an explicit spread for controlled
comparisons.
Annualized strategy
metrics use the full evaluation duration, require at least 30 trades, and return
compact `sample_guidance` when the lookback produces too few trades. Full-detail
`drawdown_periods` are consolidated peak-to-recovery episodes rather than one
row per underwater observation.

### Analyze Risk
```bash
# Volatility estimate
mtdata-cli forecast_volatility_estimate EURUSD --horizon 12 --method ewma

# Barrier probability
mtdata-cli forecast_barrier_prob EURUSD --horizon 12 --method hmm_mc --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.5,"stop_loss":0.3}'

# Optimize TP/SL
mtdata-cli forecast_barrier_optimize EURUSD --horizon 12 --grid-style volatility --objective edge
```

### Pre-Trade Snapshot & Session Context
```bash
# One-shot pre-trade snapshot: quote + levels + patterns
mtdata-cli market_snapshot EURUSD --timeframe H1 --json

# Add the optional regime + forecast sections (sections=all)
mtdata-cli market_snapshot EURUSD --timeframe H1 --sections all --horizon 8 --json

# Global exchange status (NYSE, LSE, Tokyo, ...) or one broker symbol's tradability
mtdata-cli market_status --region all --json
mtdata-cli market_status --venue ASX --json
mtdata-cli market_status --symbol EURUSD --json

# Consolidated broker/session context (account, open/pending, quote, computed state)
mtdata-cli trade_session_context EURUSD --json
```

The US exchange view distinguishes `pre_market` (04:00–09:30 ET), regular
`open`, `after_hours` (16:00–20:00 ET), and closed `overnight` periods. Each
market row carries its exchange-local weekday; mixed-region summaries report
`day_of_week: mixed` when venue calendars are on different dates.

In symbol mode, `is_tradable` reflects the broker trade mode (including
close-only symbols), while `can_open_new_positions` additionally requires a
live-ready quote and is forced false on the standard FX weekend (unless crypto
or the inferred M1 schedule says the session is open). Inferred session is
otherwise advisory.

`--symbol` always means an exact broker instrument, even when its name matches a
venue ID. Use `--venue` for one of the static exchange calendars. The former
positional venue shorthand is not supported because it collided with valid
broker-symbol names.

### Wait for a candle close

Do not run long waits from the Web UI. See [WAIT_EVENT.md](WAIT_EVENT.md).

```bash
mtdata-cli wait_event EURUSD --timeframe H1 --watch-for '[]' --json
mtdata-cli wait_event EURUSD --timeframe M5 --watch-for order_filled --json
```

The first command is boundary-only. The second can return early when an order
fills and otherwise ends at the M5 boundary. `timeframe` is the required wait
horizon. The wait budget and polling cadence are internal: boundary-only waits
sleep directly to the boundary, while explicit event watchers are polled.

### Place Orders
`trade_place` requires `symbol`, `volume`, and `order_type`.

The examples below keep the safe default preview. Omitting `--dry-run` still
previews (`dry_run=true`); only `--dry-run false` submits the order to MT5.
See [TRADING_SAFETY.md](TRADING_SAFETY.md) for the dry-run-first workflow,
account guardrails, and broker behavior.

Public symbol aliases with separators, such as `EUR/USD`, are resolved against
the connected broker catalog before report, session, position/order-read, and
order-preview calls. When the alias is unambiguous, responses use the canonical
broker name in `symbol` and preserve the request spelling in `symbol_input`.
Use the exact broker symbol from `symbols_list` when compact aliases are
ambiguous (for example, when a broker exposes multiple suffixed variants).

Accepted `order_type` values (case-insensitive; `-` or space is normalized to `_`):
`BUY`, `SELL`, `BUY_LIMIT`, `BUY_STOP`, `BUY_STOP_LIMIT`, `SELL_LIMIT`,
`SELL_STOP`, `SELL_STOP_LIMIT`. MT5 numeric constants and `ORDER_TYPE_*` names
are **not** accepted as input — they only appear when *reading* existing
orders/positions. For stop-limit orders, `--price` is the stop trigger and
`--stop-limit-price` is the limit order price activated after that trigger.

```bash
# Preview a pending order with canonical order_type
mtdata-cli trade_place BTCUSD --volume 0.03 --order-type BUY_LIMIT --price 68750 --stop-loss 67500 --take-profit 72000 --dry-run true

# Case and separators are normalized (buy-stop -> BUY_STOP)
mtdata-cli trade_place BTCUSD --volume 0.03 --order-type buy-stop --price 70200 --stop-loss 69000 --take-profit 73000 --dry-run true

# Preview a stop-limit order with separate trigger and limit prices
mtdata-cli trade_place BTCUSD --volume 0.03 --order-type BUY_STOP_LIMIT --price 70200 --stop-limit-price 70000 --stop-loss 68000 --take-profit 74000 --dry-run true

# Preview a market order after fetching the live quote.
# BUY: stop-loss below bid, take-profit above ask.
# SELL: stop-loss above ask, take-profit below bid.
# Absolute crypto prices expire; substitute levels from market_ticker.
mtdata-cli market_ticker BTCUSD
mtdata-cli trade_place EURUSD --volume 0.01 --order-type BUY --stop-loss 1.00 --take-profit 2.00 --dry-run true
```

### Trade Execution Controls

| Flag | Applies To | Description |
|------|------------|-------------|
| `--dry-run` | `trade_place`, `trade_modify`, `trade_close` | Preview the request without sending it to MT5. |
| `--detail` | `trade_place` | Preview detail level; use `full` for execution diagnostics. |
| `--stop-limit-price` | `trade_place`, `trade_modify` | Limit leg activated by a stop-limit trigger. |
| `--magic` | `trade_place`, `trade_get_open`, `trade_get_pending`, `trade_close`, `trade_history`, `trade_journal_analyze` | MT5 unsigned 64-bit magic number (`0..18446744073709551615`); zero is a valid exact filter. History and journal filtering happens before pagination and aggregation. |
| `--require-sl-tp` | `trade_place` | Require both stop-loss and take-profit on market and pending orders. |
| `--expiration` | `trade_place`, `trade_modify` | Future expiration for pending orders (`dateparser` or positive UTC epoch seconds); use literal `GTC` for no expiration. Invalid or past values are rejected locally. |
| `--idempotency-key` | `trade_place`, `trade_modify` | Durable dedupe key shared by CLI and server processes within the configured retention window. |
| `--target` | `trade_close` | Select `positions` (default), `pending`, or `all_exposure`. |
| `--side` | `trade_get_open`, `trade_close`, `trade_history`, `trade_journal_analyze` | Filter by direction; `trade_close` accepts `BUY`/`LONG` and `SELL`/`SHORT` for open positions only. |
| `--close-all` | `trade_close` | Select the whole account when ticket, symbol, side, and magic are omitted. |
| `--confirm-close-all` | `trade_close` | Confirm any ticketless live bulk operation. |
| `--pnl-filter` | `trade_close`, `trade_get_open` | Filter positions by `all`, `profit`, or `loss`. |
| `--close-priority` | `trade_close` | When multiple positions match, close `loss_first`, `profit_first`, or `largest_first`. |

Every `trade_place`, `trade_modify`, and `trade_close` response includes a
string `request_id` and matching `correlation_id`. The same correlation value
appears in execution logs; live broker results expose MT5's numeric identifier
separately as `mt5_request_id`. Dry runs that do not reach MT5 omit that broker
field. Idempotent replays expose both the current `correlation_id` and the original invocation's
`original_correlation_id`.

For account-level safety, configure trade guardrails in [ENV_VARS.md](ENV_VARS.md#trade-guardrails) before moving from preview to live execution.

### Close or Modify Positions
Use exact tickets whenever possible. `trade_close` defaults to the `positions`
target and preview mode; set `--dry-run false` explicitly only when you intend a
live close:

```bash
mtdata-cli trade_get_open --json
mtdata-cli trade_modify --ticket 123456789 --stop-loss 60500 --take-profit 62500
mtdata-cli trade_close --ticket 123456789 --volume 0.05 --dry-run true
mtdata-cli trade_close --ticket 987654321 --target pending --dry-run false
mtdata-cli trade_close --symbol EURUSD --side BUY --dry-run true
```

`trade_close` never falls back between positions and pending orders. Use
`--target all_exposure` only for a symbol, side, magic, or account-wide bulk scope; its
response reports the position-close and pending-cancel legs separately.

### Review Trade Journal
```bash
mtdata-cli trade_journal_analyze --minutes-back 10080 --json
mtdata-cli trade_journal_analyze --symbol EURUSD --minutes-back 43200 --breakdown-limit 5 --json
mtdata-cli trade_journal_analyze --side long --minutes-back 43200 --json
mtdata-cli trade_history --history-kind deals --side buy --minutes-back 1440 --json
mtdata-cli trade_journal_analyze --magic 3001 --minutes-back 43200 --json
```

`trade_history`, `trade_journal_analyze`, and `trade_execution_quality` default
to a 7-day lookback (`--minutes-back 10080`) when you do not pass a time window
explicitly. `minutes_back` is capped at 10512000 minutes (20 years).
`trade_history` returns at most 20 rows by default. Set `--limit` for another
page size up to 500. When `pagination.has_more` is true, reuse
`pagination.next_cursor` as `--cursor` with the same history kind, filters,
time controls, and order. The cursor freezes the first page's exact UTC bounds
and expires after one hour; start a fresh query after expiration. Offset and
page-number inputs are not supported because relative windows and changing
account history make them unstable.
Use `--magic` to isolate one strategy on shared accounts. The
`trade_history --column-style humanized` option applies display labels in TOON
and other table renderers. JSON item keys, `units`, and `--output-fields`
paths stay canonical snake_case at every detail level, including `--detail full`.
`--detail summary` returns period aggregates (counts, net P&L for deals, period
bounds) without a row tape.
For deal history and journals, `--side buy|sell` filters the execution
`fill_side`, while `--side long|short` filters the economic `position_side`
after open/close direction is derived. Responses echo this choice in
`side_filter.dimension`. Order-lifecycle history has no derived position side,
so it accepts only `buy|sell`.
Order-history `order_type` values use canonical uppercase tokens such as `BUY`
and `SELL_LIMIT`; deal `fill_side` and `position_side` remain lower-case because
they are separate fill-direction and economic-position enums.
For Kelly sizing in `trade_risk_analyze`, provide a `--sizing` JSON object with
`win_rate`, `avg_win`, and `avg_loss` derived from complete trade lifecycles whose
returns are normalized consistently (for example, R-multiples). Do not map the
raw `trade_journal_analyze` PnL averages into those JSON fields.

### Estimate Portfolio Tail Risk
```bash
mtdata-cli trade_var_cvar_calculate --timeframe H1 --lookback 500 --confidence 0.95 --json
mtdata-cli trade_var_cvar_calculate --symbol EURUSD --method parametric --transform pct --lookback 300 --json
```

### Stress Open Positions
```bash
mtdata-cli trade_stress_test --shocks '{"EURUSD":-2.0,"GBPUSD":-1.5}' --json
mtdata-cli trade_stress_test --shocks '{"*":-3.0}' --detail full --json
```

Shock values are percentage price moves. `*` is a fallback for any open-position symbol without an explicit shock. The tool is read-only and reports estimated P&L and equity impact.

### Detect Patterns and Regimes
```bash
# Candlestick patterns
mtdata-cli patterns_detect EURUSD --mode candlestick --robust-only true

# Harmonic Fibonacci-ratio patterns
mtdata-cli patterns_detect EURUSD --mode harmonic --lookback 800

# Regime detection
mtdata-cli regime_detect EURUSD --method hmm --params "n_states=2"

# Change-point detection
mtdata-cli regime_detect EURUSD --method bocpd --threshold 0.5
```

### Compare Cross-Symbol Relationships (Exploratory)
```bash
# Rank co-moving symbols with transformed-return correlations
mtdata-cli correlation_matrix "EURUSD,GBPUSD,USDJPY" --timeframe H1 --window-bars 500 --method pearson --transform log_return --json

# Use an explicit MT5 group path instead of naming symbols one-by-one
mtdata-cli correlation_matrix --group "Forex\\Majors" --timeframe H1 --window-bars 500 --limit 120 --method pearson --transform log_return --detail full --json

# Find candidate mean-reverting pairs inside an MT5 group
mtdata-cli cointegration_test --group "Forex\\Majors" --timeframe H1 --window-bars 400 --transform log_level --significance 0.05 --json

# Compare a few symbols directly
mtdata-cli causal_discover_signals "EURUSD,GBPUSD,USDJPY" --timeframe H1 --window-bars 800 --max-lag 5 --transform log_return --significance 0.05

# Pass a single symbol to auto-expand its visible MT5 group (e.g., Forex\\Majors)
mtdata-cli causal_discover_signals EURUSD --timeframe H1 --window-bars 800
```

For `market_scan`, `correlation_matrix`, `cointegration_test`, and
`causal_discover_signals`, use the canonical `symbols` selector for
multi-symbol integrations. `group` remains mutually exclusive with explicit
symbol selectors.

See [CAUSAL_DISCOVERY.md](CAUSAL_DISCOVERY.md) for interpretation and caveats.

---

## Tips

### Quoting JSON on Windows PowerShell

PowerShell 5.1 strips embedded double quotes from native arguments. A documented
JSON value such as `--watch-for '{"type":"price_touch_level","symbol":"EURUSD","level":1.16}'`
arrives as `{type: price_touch_level, ...}` and fails to parse.

Use one of these forms instead:

```powershell
# Escaped inner quotes (PowerShell 5.1)
mtdata-cli wait_event EURUSD --timeframe M1 --watch-for '{\"type\":\"price_touch_level\",\"symbol\":\"EURUSD\",\"level\":1.16}'

# KV form (no JSON quotes)
mtdata-cli wait_event EURUSD --timeframe M1 --watch-for type=price_touch_level,symbol=EURUSD,level=1.16

# PowerShell 7+ / stop-parsing
mtdata-cli wait_event EURUSD --timeframe M1 --% --watch-for {"type":"price_touch_level","symbol":"EURUSD","level":1.16}
```

The same quoting applies to other JSON-or-KV parameters (`--kv-args`, `--shocks`,
`--sizing`, `--barrier`).

### Pipe Output to jq for JSON Processing
```bash
mtdata-cli forecast_generate EURUSD --json | jq '.forecast'
```

### Save Output to File
```bash
mtdata-cli data_fetch_candles EURUSD --limit 1000 --json > eurusd_data.json
```

### Debug Mode
Set the debug environment variable for verbose CLI logging:

PowerShell:
```powershell
$env:MTDATA_CLI_DEBUG = "1"
mtdata-cli forecast_generate EURUSD
$env:MTDATA_CLI_DEBUG = $null
```

Bash:
```bash
MTDATA_CLI_DEBUG=1 mtdata-cli forecast_generate EURUSD
```

---

## See Also

- [SETUP.md](SETUP.md) — Installation guide
- [OUTPUT.md](OUTPUT.md) — Response envelope, `detail`/`output_fields`, and error codes
- [TIMESTAMPS.md](TIMESTAMPS.md) — Timezone policy for inputs and output
- [TRADING_SAFETY.md](TRADING_SAFETY.md) — Dry-run-first trading runbook and guardrails
- [EXAMPLE.md](EXAMPLE.md) — Complete workflow example
- [FINVIZ.md](FINVIZ.md) — Fundamental data commands
- [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md) — Options and QuantLib commands
- [TEMPORAL.md](TEMPORAL.md) — Temporal analysis
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues
