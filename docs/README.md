# Documentation Index

**Audience:** User (this page) — some links are Operator or Contributor

Friendly guides for **mtdata** — look at MetaTrader 5 data, run forecasts, and
optionally place demo orders from a website, the command line, or an AI
assistant. For a high-level introduction, start at the [project README](../README.md).

You do **not** need to read this folder end-to-end. Pick a path, stay
**read-only** until you are comfortable, then open a deeper page only when you
need it.

How we write these pages: [STYLE.md](STYLE.md) (Contributor).

> **Safety:** `trade_*` commands can place, modify, or close **real** orders on
> the MetaTrader 5 account that is currently logged in (demo or live). Prefer a
> demo account until you are confident. The website **Tools** runner can call
> the same trading tools after you tick confirm. See [TRADING_SAFETY.md](TRADING_SAFETY.md).

## Choose your path

| Goal | Start here | Then read |
|------|------------|-----------|
| I want the website | [WEBUI.md](WEBUI.md) (User) | [SAMPLE-TRADE-WEBUI.md](SAMPLE-TRADE-WEBUI.md) |
| I want an AI assistant | [MCP.md](MCP.md) (User) | [TRADING_SAFETY.md](TRADING_SAFETY.md) |
| Install and confirm MetaTrader 5 | [SETUP.md](SETUP.md) (User) | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Learn the command line safely | [CLI.md](CLI.md) (User) | [GLOSSARY.md](GLOSSARY.md), [SAMPLE-TRADE.md](SAMPLE-TRADE.md) |
| Find a market / read a quote / news | [MARKET.md](MARKET.md) · [NEWS.md](NEWS.md) (User) | [FINVIZ.md](FINVIZ.md) (company, screens, calendars) |
| Build a research workflow | [EXAMPLE.md](EXAMPLE.md) (User) | [REPORTS.md](REPORTS.md), [FORECAST.md](FORECAST.md) |
| Prepare for trade execution | [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) (User) | [TRADING_SAFETY.md](TRADING_SAFETY.md) |
| Host or script the API | [WEB_API.md](WEB_API.md) (Operator) | [DEPLOYMENT.md](DEPLOYMENT.md), [ENV_VARS.md](ENV_VARS.md) |

## Learning path

1. [SETUP.md](SETUP.md) — Install, connect MetaTrader 5, first read-only check
2. [GLOSSARY.md](GLOSSARY.md) — Words used across the docs (start here if markets are new)
3. Pick a surface: [WEBUI.md](WEBUI.md) · [CLI.md](CLI.md) · [MCP.md](MCP.md)
4. [SAMPLE-TRADE.md](SAMPLE-TRADE.md) or [SAMPLE-TRADE-WEBUI.md](SAMPLE-TRADE-WEBUI.md)
5. [TRADE_IDEAS.md](TRADE_IDEAS.md) — one preview-only compose command
6. [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) — regimes, intervals, tighter gates
7. Deep dives as needed: [FORECAST.md](FORECAST.md), [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md), [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md)

## Getting Started

| Document | Audience | Description |
|----------|----------|-------------|
| [SETUP.md](SETUP.md) | User | Installation, MetaTrader 5 connection, first workflow |
| [WEBUI.md](WEBUI.md) | User | Chart workspace at `/app` |
| [MCP.md](MCP.md) | User | AI assistant setup (stdio / SSE) |
| [CLI.md](CLI.md) | User | Command conventions, help, output formats |
| [GLOSSARY.md](GLOSSARY.md) | User | Dense terms — BOCPD, Kelly, VaR, … ([quick find](GLOSSARY.md#quick-find)) |
| [LIMITATIONS.md](LIMITATIONS.md) | User | Practical caveats |
| [ENV_VARS.md](ENV_VARS.md) | Operator | Complete `.env` reference |
| [OUTPUT.md](OUTPUT.md) | Operator | Response envelope, `detail` / `output_fields`, errors |
| [TIMESTAMPS.md](TIMESTAMPS.md) | Operator | Broker time, UTC, and display time |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Operator | Long-lived MCP or Web API service |
| [STYLE.md](STYLE.md) | Contributor | Voice, page contract, audience tags |

## Core Topics

| Document | Audience | Description |
|----------|----------|-------------|
| [MARKET.md](MARKET.md) | User | Symbols, quotes, scans, snapshots, session status |
| [NEWS.md](NEWS.md) | User | Ranked headlines; use `calendar` for the event table |
| [WAIT_EVENT.md](WAIT_EVENT.md) | User | Pause until a candle closes or an order fills |
| [FORECAST.md](FORECAST.md) | User | Price forecasts, training, model store |
| [REPORTS.md](REPORTS.md) | User | Packaged market summaries |
| [forecast/METHODS.md](forecast/METHODS.md) | Operator | Per-method keys, defaults, dependencies |
| [forecast/FORECAST_GENERATE.md](forecast/FORECAST_GENERATE.md) | Operator | `forecast_generate` parameter reference |
| [forecast/BACKTESTING.md](forecast/BACKTESTING.md) | User | Rolling backtests |
| [forecast/VOLATILITY.md](forecast/VOLATILITY.md) | User | How much price tends to move |
| [forecast/REGIMES.md](forecast/REGIMES.md) | User | Trending / ranging / transition; HMM/BOCPD states |
| [TIME_SERIES_DIAGNOSTICS.md](TIME_SERIES_DIAGNOSTICS.md) | User | Stationarity, seasonality, outliers |
| [forecast/UNCERTAINTY.md](forecast/UNCERTAINTY.md) | User | Confidence and conformal intervals |
| [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) | User | Take-profit / stop-loss hit odds |
| [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) | User | Indicator catalog |
| [LEVELS.md](LEVELS.md) | User | Pivots, support/resistance, confluence, volume profile |
| [DENOISING.md](DENOISING.md) | User | Smoothing filters |
| [SIMPLIFICATION.md](SIMPLIFICATION.md) | User | Downsampling (`--simplify`) |
| [CAUSAL_DISCOVERY.md](CAUSAL_DISCOVERY.md) | User | Lead/lag style checks across symbols |
| [ADVANCED_ANALYTICS.md](ADVANCED_ANALYTICS.md) | User | Microstructure, execution quality, portfolio risk |
| [TEMPORAL.md](TEMPORAL.md) | User | Session, day-of-week, hour, month patterns |
| [forecast/PATTERN_SEARCH.md](forecast/PATTERN_SEARCH.md) | User | Candlestick / chart patterns and analogs |

## External Data & Options

| Document | Audience | Description |
|----------|----------|-------------|
| [FINVIZ.md](FINVIZ.md) | User | Company dossiers, screens, calendars, delayed cross-asset tables |
| [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md) | User | Options chains and local barrier pricing |
| [WEB_API.md](WEB_API.md) | Operator | HTTP routes behind the Web UI |

## Tutorials

| Document | Audience | Description |
|----------|----------|-------------|
| [SAMPLE-TRADE.md](SAMPLE-TRADE.md) | User | Beginner CLI walkthrough |
| [SAMPLE-TRADE-WEBUI.md](SAMPLE-TRADE-WEBUI.md) | User | Same questions in the chart workspace |
| [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) | User | Regimes, HAR-RV, barriers, tighter gates |
| [EXAMPLE.md](EXAMPLE.md) | User | Compact end-to-end command loop |

## Trading

| Document | Audience | Description |
|----------|----------|-------------|
| [TRADE_IDEAS.md](TRADE_IDEAS.md) | User | Preview-only compose: forecast, barriers, size, dry-run |
| [TRADING_SAFETY.md](TRADING_SAFETY.md) | User | Dry-run, guardrails, account/journal (read-only first) |
| [TRADING_RISK.md](TRADING_RISK.md) | User | Sizing, VaR/CVaR, stress tests (read-only) |
| [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) | User | Take-profit / stop-loss probabilities |

## Contributor / inventory

These are **not** tutorials. They track coverage and dependency work.

| Document | Description |
|----------|-------------|
| [WEBUI_GOAL.md](WEBUI_GOAL.md) | Chart-workspace product goal |
| [WEBUI_API_COVERAGE.md](WEBUI_API_COVERAGE.md) | Route × UI matrix |
| [WEBUI_TOOL_COVERAGE.md](WEBUI_TOOL_COVERAGE.md) | Tool → UI surface exceptions (dedicated, omitted, gated) |
| [DEPENDENCY_MIGRATION.md](DEPENDENCY_MIGRATION.md) | Python 3.14 package snapshot |
| [PUBLISHING.md](PUBLISHING.md) | PyPI + Official MCP Registry publish sequence (emerzon) |
| [STYLE.md](STYLE.md) | Docs persona and page contract |

## Common workflows (recipes)

Research-only snippets (not financial advice). For a guided narrative, use
[SAMPLE-TRADE.md](SAMPLE-TRADE.md) or [SAMPLE-TRADE-WEBUI.md](SAMPLE-TRADE-WEBUI.md).

### 1) Quick market snapshot (no trading)

```bash
mtdata-cli symbols_describe EURUSD --json
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 200 --json
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta --json
```

### 1b) Preview-only trade idea

```bash
mtdata-cli trade_idea_compose EURUSD --timeframe H1 --horizon 12 --template quick
```

### 1c) Watchlist radar

```bash
mtdata-cli market_radar --symbols EURUSD,GBPUSD,XAUUSD --timeframe H1
```

### 2) Take-profit / stop-loss odds for a trade idea

```bash
mtdata-cli forecast_barrier_prob EURUSD --timeframe H1 --horizon 12 --method mc_gbm --direction long --barrier '{"kind":"tp_sl","unit":"pct","take_profit":0.4,"stop_loss":0.6}' --json
```

### 3) Scan a small watchlist (PowerShell)

```powershell
$symbols = "EURUSD","GBPUSD","USDJPY"
$symbols | % { mtdata-cli forecast_volatility_estimate $_ --timeframe H1 --horizon 12 --method ewma --json }
```

## Troubleshooting

| Document | Audience | Description |
|----------|----------|-------------|
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | User | Common issues, including the Web UI |

## Where details live

Practical caveats: [LIMITATIONS.md](LIMITATIONS.md). Dedicated references:

- Forecast methods, defaults, and dependencies → [forecast/METHODS.md](forecast/METHODS.md)
- Response envelope, `detail` / `output_fields`, pagination, errors → [OUTPUT.md](OUTPUT.md)
- Timezones → [TIMESTAMPS.md](TIMESTAMPS.md)
- Trading dry-run, guardrails, broker behavior → [TRADING_SAFETY.md](TRADING_SAFETY.md)
- Long-running MCP / Web API service → [DEPLOYMENT.md](DEPLOYMENT.md)

## Quick reference

```bash
# List commands
mtdata-cli --help

# Search by keyword
mtdata-cli --help forecast
mtdata-cli --help barrier

# Help for one command
mtdata-cli forecast_generate --help
mtdata-cli regime_detect --help
```
