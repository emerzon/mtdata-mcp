# core/ — API-Facing Layer

All 82 MCP tools (+1 conditional), CLI commands, web API endpoints, and server transport logic live here. This is the largest module.

## FILE MAP

### MCP Tools (one file per domain)

| File | Lines | Domain |
|------|-------|--------|
| `data/` | — | Data fetch tools (candles, ticks, wait events) |
| `forecast.py` | 2087 | Forecast generation/backtest tools |
| `forecast_tasks.py` | — | Forecast training/task/model-store tools |
| `analytics.py` | — | Advanced analytics, strategy validation, portfolio risk |
| `diagnostics.py` | — | Time-series diagnostics tools |
| `denoise.py` | — | Denoising catalog tools |
| `trading/` | — | Trading tool entry points |
| `patterns.py` | 1484 | Pattern detection tools |
| `regime/` | — | Regime detection: `api.py` (MCP entry), `detect.py` (method bodies), `ensemble.py`, `summarize.py`, `payload.py`, `smoothing.py` |
| `temporal.py` | 1531 | Temporal/seasonal analysis tools |
| `causal/` | — | Causal analysis tools (discover, correlation, cross, cointegration) |
| `calendar.py` | — | Provider-agnostic economic/earnings calendar |
| `equity_profile.py` | — | US-issuer dossier |
| `screener.py` | — | Equity screener |
| `asset_performance.py` | — | Delayed cross-asset performance context |
| `finviz/` | — | Internal Finviz adapters (not MCP-catalogued): news, calendar, screen, fundamentals, insider, markets |
| `indicators.py` | 743 | Indicator computation tools |
| `market_depth.py` | 914 | Market depth tools |
| `market_snapshot.py` | — | Unified market snapshot tools |
| `market_status.py` | — | Market/session status tools |
| `options.py` | — | Options and QuantLib tools |
| `symbols/` | — | Symbol listing/search/scan tools |
| `labels.py` | 883 | Labeling tools |
| `pivot.py` | 1081 | Pivot point tools |
| `volume_profile.py` | — | Volume profile tools |
| `news.py` | — | Unified news tools |
| `report/` | — | Report tool entry points |
| `tools.py` | — | Tool catalog/listing tools |

### Trading (split by concern)

| File | Lines | Purpose |
|------|-------|---------|
| `trading/use_cases/` | — | Orchestration layer (place, modify, close, history, query, risk) |
| `trading/orders.py` | 1569 | Order placement/modification |
| `trading/execution.py` | 2194 | Execution logic |
| `trading/positions.py` | 1573 | Position queries |
| `trading/risk.py` | 108 | Risk calculations |
| `trading/validation.py` | 1135 | Input validation |
| `trading/gateway.py` | 91 | MT5 trade API wrapper |
| `trading/account.py` | 944 | Account info queries |
| `trading/common.py` | 269 | Shared trading types |
| `trading/comments.py` | 209 | Order comment encoding |
| `trading/time.py` | 322 | Trading time helpers |
| `trading/requests.py` | 565 | Request models |
| `trading/ideas.py` | — | Preview-only trade-idea composer |
| `trading/ideas_requests.py` | — | Trade-idea request model |

### Reports

| File | Lines | Purpose |
|------|-------|---------|
| `report/` | — | Report tool entry points |
| `report/use_cases.py` | 1675 | Report orchestration |
| `report/utils.py` | 943 | Report helper functions |
| `report/shared.py` | 123 | Shared report types |
| `report/requests.py` | 97 | Report request models |
| `report_templates/` | 9 files | Template implementations (basic.py = 1092 lines) |

### Server, CLI, Web API

| File | Purpose |
|------|---------|
| `server.py` | MCP server entry (SSE, stdio, streamable-HTTP) |
| `cli/` | Dynamic CLI — discovers tools, builds argparse subparsers |
| `web_api.py` | Web API entry point |
| `web_api_runtime.py` | FastAPI app creation, CORS, uvicorn |
| `web_api_handlers.py` | REST route handlers |
| `web_api_geometry.py` | Compact confluence / volume-profile / exposure DTOs |
| `radar.py` | Watchlist radar compose |
| `web_api_radar.py` | Radar and session-strip HTTP routes |
| `web_api_models.py` | Request/response Pydantic models |

### Infrastructure

| File | Purpose |
|------|---------|
| `_mcp_instance.py` | Singleton MCP server instance |
| `_mcp_tools.py` | Tool registration decorator, type coercion |
| `../bootstrap/settings.py` | Environment/MT5 config loading |
| `../shared/constants.py` | SERVICE_NAME, TIMEFRAME_MAP, TIMEFRAME_SECONDS |
| `../shared/schema.py` | Shared Pydantic schemas |
| `schema_attach.py` | Schema attachment helpers |
| `../shared/error_envelope.py` | Standardized error response format |
| `../shared/execution_logging.py` | Execution/query logging |
| `../shared/output_contract.py` | Transport-independent output detail and pagination |
| `../shared/runtime_metadata.py` | Source provenance and runtime timezone metadata |
| `../shared/request_context.py` | Request correlation context |
| `../shared/symbol_catalog.py` | Pure symbol classification, matching, and ranking |
| `../analytics/requests.py` | Validated analytics inputs |
| `unified_params.py` | Shared parameter definitions |
| `features.py` | Internal rolling feature extraction helper used by forecast/regime code |
| `patterns_support.py` (2625) | Complex pattern detection helpers |
| `patterns_requests.py` | Pattern request models |
| `patterns_use_cases.py` | Pattern detection orchestration |
| `data/requests.py` | Data request models |
| `data/use_cases.py` | Data fetch orchestration |
| `data/wait_events/` | Wait-event engine (compile, loop, account, market, boundary, ticks) |
| `mt5_gateway.py` | MT5 connection management |

## CONVENTIONS

- **Domain pattern**: `{domain}.py` or `{domain}/` (tools) → `{domain}/requests.py` or `{domain}_requests.py` (input models) → `{domain}/use_cases.py` or `{domain}_use_cases.py` (orchestration).
- **`__init__.py` is intentionally minimal** — import modules directly to avoid circular deps.
- **`_` prefix** on `_mcp_instance.py` and `_mcp_tools.py` signals internal infrastructure.

## ANTI-PATTERNS

- **Never** import from `core/__init__.py`. Always `from mtdata.core.data import ...`.
- **Never** add business logic here — delegate to `forecast/`, `services/`, `utils/`, `patterns/`.
- **Never** bypass `shared/error_envelope.py` for error responses in tool functions.
