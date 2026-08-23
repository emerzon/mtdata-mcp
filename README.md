# mtdata

**Turn MetaTrader 5 into a research lab you can script, query from AI agents, or browse in a local web UI.**

mtdata is a Windows-first toolkit that sits on top of a running MT5 terminal. It gives you **80+ tools** for market data, forecasting, regime detection, patterns, risk, and trading through the command line and an optional AI-assistant plug ([MCP](docs/MCP.md)), plus a local website and HTTP API.

It is a **toolkit for exploration and automation**, not a trading strategy or financial advice.

---

## Why mtdata?

| Strength | What that means for you |
|----------|-------------------------|
| **One stack, three surfaces** | Full tool surface through `mtdata-cli` and MCP; focused HTTP workflows through a local Web API + React UI |
| **Research depth** | Classical, ML, and foundation forecasting; regimes; barriers; patterns; 100+ indicators; denoising |
| **MT5-native** | Candles, ticks, market scans, account/positions, and real order flow against your broker terminal |
| **Agent-friendly** | Designed for tool-calling workflows: structured outputs, async training, dry-run trading |
| **Safety-aware** | Demo-first guidance, optional trade guardrails, and dry-run previews on trading commands |
| **Guided learning** | Sample trade workflows, a glossary, and docs that go from “first candle fetch” to advanced playbooks |

If you already live in MT5 and want **repeatable analysis** — or you want an assistant to pull data and run forecasts without reinventing the glue — this repo is built for that.

---

## Who is this for?

- **Learners & discretionary traders** — Follow guided workflows without a quant background. Start with candles, a simple forecast, and the sample trade guide.
- **Systematic / quant-curious traders** — Prototype ideas, backtest, optimize barriers, and automate via CLI or MCP.
- **Builders & data folks** — Pull MT5 data into pipelines, agents, or a local web stack with a consistent output contract.

---

## Platform notes

| Requirement | Detail |
|-------------|--------|
| **Windows** | Required to run MetaTrader 5 (and therefore mtdata against MT5) |
| **macOS / Linux** | Run mtdata on a Windows machine or VM; connect remotely via MCP or Web API |
| **Python** | **3.14** is the supported runtime for the packaged dependency set |

---

## Safety first

`trade_*` tools can **place, modify, and close real orders** on the account logged into MT5.

- Prefer a **demo account** until you know the tools and your broker behavior.
- There is no separate paper-trading mode inside mtdata — use an MT5 demo account for simulated execution.
- When a command supports `--dry-run true`, use it to preview before anything hits the broker.
- For research only, stick to `data_*`, `forecast_*`, `regime_*`, `patterns_*`, and `report_*`.

Optional guardrails (allowed symbols, max volume, max risk % of equity) are documented in [docs/ENV_VARS.md](docs/ENV_VARS.md) and [docs/TRADING_SAFETY.md](docs/TRADING_SAFETY.md).

---

## What you can do

| Area | Highlights | Example tools |
|------|------------|---------------|
| **Data** | Candles, ticks, depth (optional), market scans | `data_fetch_candles`, `data_fetch_ticks`, `symbols_top_markets` |
| **Forecasting** | Theta → ARIMA/ML → Chronos-class foundation models; async train & model cache | `forecast_generate`, `forecast_backtest_run`, `forecast_train` |
| **Volatility & barriers** | Movement estimates; TP/SL hit probabilities via simulation | `forecast_volatility_estimate`, `forecast_barrier_prob` |
| **Regimes** | Trending / ranging / stress-style market states | `regime_detect` |
| **Patterns & levels** | Candlesticks, chart patterns, Elliott/fractals; pivots & confluence | `patterns_detect`, level tools |
| **Indicators & denoise** | 100+ technicals; smooth noise to see structure | `--indicators`, `--denoise` |
| **Multi-asset & diagnostics** | Correlation, cointegration, stationarity, outliers, seasonality | `correlation_matrix`, `stationarity_test`, `outliers_detect` |
| **Strategy & risk** | Simple rule backtests; VaR/CVaR, stress, position sizing | `strategy_backtest`, `trade_var_cvar_calculate` |
| **Trade ideas** | One preview-only research idea: forecast, barriers, size, dry-run | `trade_idea_compose` |
| **Trading** | Place/manage orders with guardrails when you opt in | `trade_place`, `trade_close` |
| **News & fundamentals** | Ranked news/calendar; equity screens and issuer dossiers | `news`, `calendar`, `screener` |
| **Options** | Chains + QuantLib barrier pricing | `options_chain`, `options_barrier_price` |
| **Reports** | [Packaged research-style summaries](docs/REPORTS.md) | `report_generate` |

**Notes**

- `market_depth_fetch` requires `MTDATA_ENABLE_MARKET_DEPTH_FETCH=1` and broker Level 2/DOM data.
- Options chains default to Yahoo Finance; Tradier is available with env config. Pure QuantLib pricing does not need a chain provider.

For method-level detail, see [docs/FORECAST.md](docs/FORECAST.md) and [docs/forecast/METHODS.md](docs/forecast/METHODS.md).

---

## Three ways to use it

```text
                    ┌─────────────────┐
                    │  MetaTrader 5   │
                    │   (Windows)     │
                    └────────┬────────┘
                             │
                      ┌──────▼──────┐
                      │   mtdata    │
                      └──────┬──────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
      mtdata-cli          MCP server        Web API
   (scripts, REPL)    (AI assistants)    (+ React UI)
```

| Surface | Entry point | Good for |
|---------|-------------|----------|
| **CLI** | `mtdata-cli` | Scripts, exploration, copy-paste workflows |
| **MCP** | `mtdata-stdio` / `mtdata-sse` / `mtdata-streamable-http` | Agent tool use (Claude, Cursor, custom clients) |
| **Web UI + API** | `mtdata-webapi` | Chart workspace at `/app`, REST for apps and notebooks |

All three are first-class delivery surfaces. Pick CLI, MCP, or Web UI for the same research stack.

---

## Quick start

**Prerequisites:** Windows + Python 3.14 + MetaTrader 5 installed and running
(**demo account recommended**). That is enough for the first candle fetch.

> Optional compiler note (you can skip this on day one): the full research
> stack may need Visual Studio Build Tools 2022 with **Desktop development
> with C++** because some packages source-build on Python 3.14. Details live
> in [Setup](docs/SETUP.md).

```bash
# Optional: isolate the environment
conda create -n mtdata python=3.14 -y
conda activate mtdata

# Lean core (data, indicators, core analysis)
pip install -e .

# Confirm MetaTrader 5 sees your terminal (read-only)
mtdata-cli symbols_list --limit 5

# Pull candles (one bar per hour, last 50 hours)
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 50

# A simple baseline forecast — an estimate, not a guarantee
mtdata-cli forecast_generate EURUSD --timeframe H1 --horizon 12 --method theta
```

Prefer a website? Jump to [Web UI](#web-ui-chart-workspace). Prefer an assistant?
See [MCP.md](docs/MCP.md).

### Web UI (chart workspace)

One path: build the SPA once, start the API, open the UI.

```bash
# Web extra if you installed lean core only
pip install -e ".[web]"

# Production frontend (once per checkout / after UI changes)
cd webui && npm ci && npm run build && cd ..

# Start API + UI
mtdata-webapi
# → open http://127.0.0.1:8000/app/
```

If `webui/dist` is missing, the API still starts and `/app` returns a clear enablement page (build steps above). Live UI development: `cd webui && npm run dev` (proxies `/api` to `:8000`). User tour: [Web UI](docs/WEBUI.md). HTTP reference: [Web API](docs/WEB_API.md) · [Setup](docs/SETUP.md).

Keep the first session **read-only** unless you are on a demo account and intentionally testing trading.

**Install flavors (summary)**

| Goal | Command |
|------|---------|
| Lean core | `pip install -e .` |
| Full docs-aligned stack | `pip install -r requirements.txt` |
| Web only | `pip install -e .[web]` |
| Classical / foundation forecast extras | `pip install -e .[forecast-classical]` / `pip install -e .[forecast-foundation]` |
| TimesFM / Git-manual add-ons | See [docs/SETUP.md](docs/SETUP.md) |

Dependency caveats (NeuralForecast optional installs, Python 3.14 exclusions, optional native accelerators) live in **[Setup](docs/SETUP.md)** so this page stays focused on getting you productive.

### Naming

The PyPI package is **`mtdata-mcp`**. The command-line entry is **`mtdata-cli`** (not `mtdata`). That avoids a clash with the unrelated [mtdata](https://pypi.org/project/mtdata/) machine-translation dataset package, which already owns the `mtdata` CLI. The importable Python package remains `mtdata` (`import mtdata`).

---

## Documentation

**Suggested path:**
[Setup](docs/SETUP.md) → [Glossary](docs/GLOSSARY.md) → pick a surface
([Web UI](docs/WEBUI.md) · [CLI](docs/CLI.md) · [AI assistant](docs/MCP.md)) →
[Sample trade](docs/SAMPLE-TRADE.md) or [the same flow in the UI](docs/SAMPLE-TRADE-WEBUI.md).

| Start here | Then explore |
|------------|--------------|
| [Setup](docs/SETUP.md) | [Troubleshooting](docs/TROUBLESHOOTING.md), [Env vars](docs/ENV_VARS.md) |
| [Web UI](docs/WEBUI.md) · [AI assistant (MCP)](docs/MCP.md) · [CLI](docs/CLI.md) | [Glossary](docs/GLOSSARY.md#quick-find), [Market discovery](docs/MARKET.md), [News](docs/NEWS.md) |
| [Sample trade](docs/SAMPLE-TRADE.md) · [Sample trade (UI)](docs/SAMPLE-TRADE-WEBUI.md) | [Trade ideas](docs/TRADE_IDEAS.md), [Advanced playbook](docs/SAMPLE-TRADE-ADVANCED.md), [Example loop](docs/EXAMPLE.md) |
| [Forecasting](docs/FORECAST.md) | [Methods](docs/forecast/METHODS.md), [Backtesting](docs/forecast/BACKTESTING.md), [Uncertainty](docs/forecast/UNCERTAINTY.md) |
| [Regimes](docs/forecast/REGIMES.md) · [Barriers](docs/BARRIER_FUNCTIONS.md) · [Patterns](docs/forecast/PATTERN_SEARCH.md) | [Indicators](docs/TECHNICAL_INDICATORS.md), [Levels](docs/LEVELS.md), [Denoising](docs/DENOISING.md) |
| [Trading safety](docs/TRADING_SAFETY.md) · [Risk analytics](docs/TRADING_RISK.md) | [Wait for events](docs/WAIT_EVENT.md), [Web API](docs/WEB_API.md), [Deployment](docs/DEPLOYMENT.md) |
| [Docs index](docs/README.md) | Full map, tagged User / Operator / Contributor |

---

## Configuration

Create a `.env` in the project root (never commit credentials):

```ini
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
MT5_SERVER_TZ=Europe/Athens   # optional broker session/calendar timezone

# Optional trade guardrails
MTDATA_TRADE_GUARDRAILS_ENABLED=1
MTDATA_TRADE_ALLOWED_SYMBOLS=EURUSD,BTCUSD,XAUUSD
MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL=EURUSD:0.50,BTCUSD:0.03
MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY=1.5
```

Full reference: **[docs/ENV_VARS.md](docs/ENV_VARS.md)**.

---

## Project layout

```text
mtdata/
├── src/mtdata/
│   ├── bootstrap/   # Runtime init, settings, tool loading
│   ├── core/        # CLI, MCP/server, trading, regime, report tools
│   ├── forecast/    # Engines, methods, model store, async tasks
│   ├── patterns/    # Chart / candlestick / structure detection
│   ├── services/    # MT5 gateway, Finviz, options, news
│   ├── shared/      # Schemas and shared constants
│   └── utils/       # Indicators, denoising, helpers
├── webui/           # React + Vite frontend
├── docs/            # User guides and references
└── tests/           # Pytest suite
```

---

## License

MIT
