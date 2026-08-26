# Web UI ↔ Web API coverage matrix

**Audience:** Contributor

**Source of truth for route parity** with the chart workspace (`webui/`).
User tour: [WEBUI.md](WEBUI.md).  
Routes are those mounted under `/api` and `/api/v1` in `src/mtdata/core/web_api.py`.  
Root probes (`/`, `/health`, `/ready`) are listed when the UI surfaces them.

| Route | Method | UI status | Entry point | Notes / rationale |
|---|---|---|---|---|
| `/health` (app root) | GET | intentional-omit | — | Same payload as `/api/v1/health`; UI uses versioned path only. |
| `/ready` (app root) | GET | intentional-omit | — | Same readiness as `/api/v1/ready`; UI uses versioned path only. |
| `/` | GET | intentional-omit | — | JSON health for ops; SPA is at `/app`. |
| `/app`, `/app/` | GET | used (host) | FastAPI static mount | Production SPA shell; not a client fetch. |
| `/api/v1/health` | GET | intentional-omit | — | Ops liveness; UI connection chip uses `/ready` only. |
| `/api/v1/ready` | GET | used | `ConnectionStatus` → `readyCheck()` | Connection chip: 200 connected, 503 MT5 not ready. |
| `/api/v1/timeframes` | GET | used | `TimeframeSelector` | |
| `/api/v1/instruments` | GET | used | `SymbolSelector` | |
| `/api/v1/history` | GET | used | `useChartWorkspace` / history paging | Denoise, optional `indicators`, volume via `ohlcv=ohlcv`, live incomplete candles. |
| `/api/v1/tick` | GET | used | `useChartWorkspace` live quotes | Bid/ask/last price lines. |
| `/api/v1/pivots` | GET | used | Overlay controls + `usePivotLevels` | Method selectable (classic/fibonacci/…). |
| `/api/v1/support-resistance` | GET | used | Overlay controls + `useSupportResistance` | Lookback, min touches, max levels, tolerance. |
| `/api/v1/confluence` | GET | used | Levels → Confluence | Compact zones for chart price lines. |
| `/api/v1/volume-profile` | GET | used | Levels → Volume profile | POC / VAH / VAL. |
| `/api/v1/exposure` | GET | used | Levels → Open exposure | Read-only positions and pending for the chart symbol. |
| `/api/v1/radar` | GET | used | Watchlist radar | Batched watchlist quotes; cap 20. |
| `/api/v1/session-strip` | GET | used | Watchlist session strip | Account, exposure count, news; non-blocking. |
| `/api/v1/denoise/methods` | GET | used | `DenoiseModal` | |
| `/api/v1/denoise/wavelets` | GET | used | `DenoiseModal` (wavelet method) | |
| `/api/v1/dimred/methods` | GET | intentional-omit | — | Catalog for CLI / MCP / Tools runner; focused Forecast panel does not send features. |
| `/api/v1/methods` | GET | used | Forecast price tab | |
| `/api/v1/volatility/methods` | GET | used | Volatility tab | |
| `/api/v1/models` | GET | used | Forecast panel **Stored models** | Browse cached/trained models; optional method filter. |
| `/api/v1/forecast/price` | POST | used | Forecast price tab | |
| `/api/v1/forecast/volatility` | POST | used | Volatility tab | |
| `/api/v1/backtest` | POST | used | Backtest tab | |
| `/api/v1/trade-ideas` | POST | used | Idea panel | Preview-only compose; draws TP/SL/entry on the chart. |
| `/api/v1/tools` | GET | used | Tools runner catalog | Compact paginated MCP tool index + surface meta |
| `/api/v1/tools/{name}` | GET | used | Tools runner detail | Parameter field descriptors |
| `/api/v1/tools/{name}/invoke` | POST | used | Tools runner run | Mutation tools require `confirm=true` |

Full MCP tool → UI surface matrix: [WEBUI_TOOL_COVERAGE.md](WEBUI_TOOL_COVERAGE.md).

## Classification rules

- **used** — reachable from a UI control or automatic workspace poll with a clear affordance.
- **intentional-omit** — available on the server but deliberately not called from the SPA (duplicate root probes, or ops-only).

Last reviewed against `web_api.py` route list for the Web UI goal implementation.
