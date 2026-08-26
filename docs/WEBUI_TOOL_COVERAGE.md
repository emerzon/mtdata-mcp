# Web UI tool coverage (by exception)

**Audience:** Contributor

User tour: [WEBUI.md](WEBUI.md).

This page maps how the SPA treats backend tools. It is **not** an exhaustive
catalog. Most tools show up in the Tools runner as a schema-driven form. Keep
the exceptions here: dedicated chart or research controls, tools the SPA will
not invoke, and environment-gated entries.

Do not copy a tool count into docs. Inspect the runtime catalog instead.

## How to inspect the runtime inventory

The live catalog is `GET /api/v1/tools` (the same bootstrap used by MCP). Detail
and invoke live at `GET /api/v1/tools/{name}` and
`POST /api/v1/tools/{name}/invoke`.

Classification helpers are in `mtdata.core.web_api_tools`. After adding a tool,
run:

```python
from mtdata.core.web_api_tools import coverage_inventory_rows

coverage_inventory_rows()
```

Each row includes `name`, `category`, `surface`, `frontend`, and
`requires_confirmation`. Gated tools stay in this inventory even when disabled.

## Surface meanings

- **dedicated_ui** — primary path is a specialized chart or research control.
  The tool remains runnable from the Tools runner.
- **generic_runner** — default. Discoverable and invocable from the SPA Tools
  runner. Mutating tools still require the confirm gate rather than being omitted.
- **intentional_omit** — listed with a rationale but not invocable from the SPA.
  Use CLI or MCP instead.

## Dedicated UI

| Tool | Frontend path | Notes |
|---|---|---|
| `confluence_levels` | chart-workspace/confluence-overlay | |
| `data_fetch_candles` | chart-workspace/history | Chart Indicators sends `indicators` / optional `ohlcv` on `/history` |
| `denoise_describe` | chart-workspace/denoise-modal | |
| `denoise_list_methods` | chart-workspace/denoise-modal | |
| `forecast_backtest_run` | forecast-panel/backtest | |
| `forecast_generate` | forecast-panel/price | |
| `forecast_list_methods` | forecast-panel/methods | |
| `forecast_models_list` | forecast-panel/models-browser | |
| `forecast_volatility_estimate` | forecast-panel/volatility | |
| `market_radar` | radar-panel/watchlist | Watchlist radar; also via Tools |
| `market_ticker` | chart-workspace/live-quotes | |
| `pivot_compute_points` | chart-workspace/pivot-overlay | |
| `support_resistance_levels` | chart-workspace/sr-overlay | |
| `tools_list` | tools-runner/catalog | |
| `trade_get_open` | chart-workspace/exposure-overlay | Read-only chart overlay |
| `trade_get_pending` | chart-workspace/exposure-overlay | Read-only chart overlay |
| `trade_idea_compose` | idea-panel/compose | Preview-only composer; also via Tools |
| `volume_profile_levels` | chart-workspace/volume-profile-overlay | |

Source of the path strings: `DEDICATED_UI_TOOLS` in
`mtdata.core.web_api_tools`.

## Intentional omit

| Tool | Rationale |
|---|---|
| `forecast_tune_genetic` | Long-running optimization has no HTTP progress or cancellation contract. Run it through CLI or MCP instead. |
| `forecast_tune_optuna` | Long-running optimization has no HTTP progress or cancellation contract. Run it through CLI or MCP instead. |
| `wait_event` | Blocking waits have no HTTP progress or cancellation contract. Run `wait_event` through CLI or MCP instead. |

Source of the rationales: `INTENTIONAL_OMIT_TOOLS` in
`mtdata.core.web_api_tools`.

## Environment-gated

`market_depth_fetch` stays in the inventory with surface `generic_runner`. It is
gated by `MTDATA_ENABLE_MARKET_DEPTH_FETCH` and is disabled by default. Enable
the env var when the Tools runner should invoke it.

Everything else is `generic_runner` unless `coverage_inventory_rows()` says
otherwise.
