# Chart workspace (Web UI)

**Audience:** User

Open a local website, pick a market, and look at candles, levels, and a simple
forecast — without using the command line.

A **candle** is one bar on the chart: open, high, low, and close for a chosen
time slice (for example one hour). This page is read-only until you open
**Tools** and run a trading command on purpose.

**Dense terms:** [Bid / ask / spread](GLOSSARY.md#bidask-and-spread) · [Timeframe](GLOSSARY.md#timeframe-and-candles-bars) · [Pivot points](GLOSSARY.md#pivot-points) · [Support and resistance](GLOSSARY.md#support-and-resistance) · [Dry-run](GLOSSARY.md#dry-run)

**Related:** [Setup](SETUP.md) · [Sample trade in the UI](SAMPLE-TRADE-WEBUI.md) · [Tools catalog](CLI.md) · [Trading safety](TRADING_SAFETY.md) · [Web API reference](WEB_API.md) (Operator)

---

## You will need

1. MetaTrader 5 running and logged in (a **demo** account is the safe start).
2. The web extra if you installed only the lean package: `pip install -e ".[web]"`.
3. A one-time frontend build, then the API:

```bash
cd webui
npm ci
npm run build
cd ..
mtdata-webapi
```

Then open [http://127.0.0.1:8000/app/](http://127.0.0.1:8000/app/).

If `webui/dist` is missing, the API still starts and `/app` shows an enablement
page with these same build steps — not a blank error.

For day-to-day UI development, `cd webui && npm run dev` serves the site on
port 5173 and forwards `/api` to the API on port 8000.

---

## First five minutes

1. **Pick a symbol** in the toolbar search (try `EURUSD`).
2. **Pick a timeframe** (start with `H1` — one bar per hour).
3. Wait for candles to draw. The status chip shows whether the API and
   MetaTrader 5 are ready.
4. Turn **Live** on if you want the latest quote to keep updating.
5. Click **Forecast**, leave the defaults (Theta is a simple baseline), and
   run a price forecast. A line appears on the chart.

You can stop there. That is already a full research glance.

Optional sixth step: click **Idea**, leave the defaults, and compose. A preview-only
research idea appears; entry, take-profit, and stop-loss lines draw on the chart.
This is not an order.

---

## What the toolbar does

| Control | What it is for |
|---------|----------------|
| Symbol search | Choose the instrument MetaTrader 5 knows (EURUSD, XAUUSD, …). |
| Timezone | How timestamps are *shown*: UTC, your computer local time, or the broker server clock. |
| Timeframe | How big each candle is (`M5`, `H1`, `D1`, …). |
| Reload | Fetch the chart again. |
| Pivot / S-R | Draw formula pivot levels and data-driven support/resistance. **Levels** also toggles confluence zones, volume-profile POC/VAH/VAL, and read-only open/pending exposure. |
| Indicators | Overlay moving averages on price and open RSI / MACD / volume panes. Start with **Sample trade** for EMA 20/50, RSI 14, and MACD. These are research overlays, not signals. |
| Bid / Ask / Last | Draw the live buy price, sell price, and last trade (when the broker sends one). |
| Denoise | Smooth the line so structure is easier to see. This changes the *display*, not your broker history. |
| Forecast | Side panel for price forecast, volatility (“how far might it move?”), and a rolling backtest. |
| Idea | Preview-only compose: narrative, TP/SL, size, gates, and dry-run. Draws entry/TP/SL on the chart. Cannot place an order. |
| Watch | Persistent watchlist with quote, spread, change, and a read-only session strip. Click a row to load the chart; Compose opens a preview-only idea. |
| Tools | Search-and-run form for the full tool list (news, reports, orders, …). |
| Auth | Paste an API token only if you started the server with `WEBAPI_AUTH_TOKEN`. It stays in this tab’s memory and clears on reload. |

On a narrow screen, extra controls move under **More**. Press Escape to close
panels.

---

## Forecast panel

Three tabs:

- **Price** — “where might the next bars go?” Start with method `theta`.
- **Volatility** — “how large are typical swings?” Start with `ewma`.
- **Backtest** — “would this method have been useful on recent history?”

Leave Advanced options closed until you need another library or a stored model.
Forecasts are estimates, not guarantees. Pair them with [barriers](BARRIER_FUNCTIONS.md)
and [risk](TRADING_RISK.md) before you size a trade.

---

## Tools runner

**Tools** lists the same catalog as the CLI and MCP. Search by name or category,
fill the form, and run.

- `trade_idea_compose` is a preview-only research idea (forecast, barriers, size, dry-run). It does not need confirm and cannot place an order.
- Research tools (candles, news, forecasts, reports) do not need a confirm tick.
- Order changes (`trade_place`, `trade_modify`, `trade_close`) and a few
  destructive model/task tools require **confirm** only for a live run
  (`dry_run=false`, or a mutating tool with no preview). A dry-run preview
  does not need the tick. Changing any parameter or running the tool clears
  confirmation, so a later live submit needs a fresh acknowledgment.
- Prefer a demo account. See [Trading safety](TRADING_SAFETY.md).

### Do not run these from the browser

| Tool | Why | Use instead |
|------|-----|-------------|
| `wait_event` | It can **block** for a long time waiting for a candle close or a fill. | [CLI / MCP wait guide](WAIT_EVENT.md) |
| `forecast_tune_optuna`, `forecast_tune_genetic` | They can run longer than an HTTP request and have no progress bar here. | CLI or MCP |

---

## If something looks empty

| What you see | What to try |
|--------------|-------------|
| Enablement page instead of a chart | Build the UI (`npm run build` in `webui/`) and restart `mtdata-webapi`. |
| Connection chip is not ready | MetaTrader 5 must be running and logged in. See [Troubleshooting](TROUBLESHOOTING.md#web-ui). |
| No candles after picking a symbol | Add the symbol to Market Watch in MetaTrader 5, then reload. |
| Forecast or overlay error banner | The rest of the chart can still work. Read the banner; retry after MT5 is ready. |
| API asks for a token | Enter the same value as `WEBAPI_AUTH_TOKEN` in **Auth**. |

More HTTP detail lives in [WEB_API.md](WEB_API.md). What the UI covers versus
omits is listed for contributors in [WEBUI_TOOL_COVERAGE.md](WEBUI_TOOL_COVERAGE.md).
