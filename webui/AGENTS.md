# webui/ — React Frontend

React + Vite + Tailwind single-page app for chart visualization and forecast interaction. Proxies to FastAPI backend on `:8000`. Product goal: [docs/WEBUI_GOAL.md](../docs/WEBUI_GOAL.md). REST coverage: [docs/WEBUI_API_COVERAGE.md](../docs/WEBUI_API_COVERAGE.md). Full MCP tool coverage: [docs/WEBUI_TOOL_COVERAGE.md](../docs/WEBUI_TOOL_COVERAGE.md).

## FILE MAP

| Path | Purpose |
|------|---------|
| `src/App.tsx` | Chart workspace shell — toolbar, chart, status, forecast panel |
| `src/main.tsx` | React entry (QueryClient + App) |
| `src/types.ts` | Shared TypeScript types (history, overlays, forecast, models, ready) |
| `src/styles.css` | Tailwind base + toolbar/panel primitives |
| **API** | |
| `src/api/client.ts` | Axios client for `/api/v1` (history, tick, forecast, models, health/ready, tools list/invoke) |
| **Components** | |
| `src/components/ApiAuthControl.tsx` | In-memory Bearer token (never persisted) |
| `src/components/ConnectionStatus.tsx` | Non-blocking MT5 readiness chip (`/ready` only) |
| `src/components/ChartToolbar.tsx` | Responsive toolbar (More overflow + Tools/Forecast entry) |
| `src/components/ChartWorkspaceStatus.tsx` | Empty / loading / error chart surface |
| `src/components/DenoiseModal.tsx` | Denoise config modal (Esc to close) |
| `src/components/WorkspacePanelShell.tsx` | Shared workspace drawer/sheet chrome + Escape/backdrop dismissal |
| `src/components/ForecastPanel.tsx` | Price / volatility / backtest drawer/sheet (Esc to close) |
| `src/components/IdeaPanel.tsx` | Preview-only trade-idea compose + send-to-chart |
| `src/components/RadarPanel.tsx` | Persistent watchlist + session strip |
| `src/components/ToolsRunnerPanel.tsx` | Schema-driven discovery + invoke for all MCP tools |
| `src/components/ModelsBrowser.tsx` | Discoverable `GET /models` list |
| `src/components/OHLCChart.tsx` | lightweight-charts candlestick surface |
| `src/components/OverlayControls.tsx` | Pivot method + S/R lookback and related params |
| **Features** | |
| `src/features/chart-workspace/useChartWorkspace.ts` | Workspace state: history, live, overlays, errors |
| `src/features/chart-workspace/toolbarMenus.tsx` | Symbol / TF / timezone / denoise / price-line menus |
| `src/features/chart-workspace/toolbarIcons.tsx` | Toolbar SVG icons |
| **Hooks** | |
| `src/hooks/useForecast.ts` | Methods, pivots, S/R, forecast run, chart overlays |
| `src/hooks/useGeometry.ts` | Confluence, volume-profile, and exposure overlay toggles |
| `src/lib/geometryOverlays.ts` | Price-line mapping for idea / confluence / VP / exposure |
| `src/lib/watchlist.ts` | Persist, seed, and reorder the local watchlist |
| `src/hooks/useViewportBreakpoint.ts` | mobile / tablet / desktop from window width |
| **Lib (pure + tests)** | |
| `src/lib/workspaceStatus.ts` | Chart surface status resolution |
| `src/lib/connectionStatus.ts` | `/ready` chip resolution |
| `src/lib/compactForecast.ts` | Compact forecast/backtest adapters for dedicated panels |
| `src/lib/overlayParams.ts` | Pivot / S/R query builders and clamps |
| `src/lib/indicatorSpec.ts` | Chart indicator presets, history query, overlay mapping |
| `src/lib/layout.ts` | Breakpoints + panel placement class helpers |
| `src/lib/toolCatalog.ts` | Catalog filter, param form defaults, invoke shaping, result format |
| `src/lib/time.ts` / `timeframes.ts` / `storage.ts` / `utils.ts` | Time, TF, localStorage, coerce helpers |
| `src/lib/useEscapeKey.ts` | Escape key dismiss hook |

## CONVENTIONS

- **TypeScript strict mode** — avoid `any` casts.
- **PascalCase** filenames for components, `camelCase` for hooks/lib.
- **Tailwind CSS** for styling — no CSS modules.
- **lightweight-charts** for OHLC rendering.
- **@tanstack/react-query** for server state.
- **Auth token** is memory-only via `setApiToken` — never `localStorage` / `sessionStorage`.
- Prefer pure helpers under `src/lib/` for logic that unit tests can drive without DOM.

## DEV SETUP

```bash
cd webui && npm ci && npm run dev            # Dev server on :5173 (proxies /api → :8000)
cd webui && npm run build                    # Production build → dist/ (base /app/)
cd webui && npm run typecheck                # tsc --noEmit
cd webui && npm test                         # vitest pure client unit tests
cd webui && npm run test:watch               # vitest watch
```

Vite proxies `/api/*` to `localhost:8000` (FastAPI). Production build is served by FastAPI (`web_api_runtime.py` mounts `webui/dist/` at `/app`). Missing dist → professional enablement response at `/app`, not a silent skip.
