# Environment variables

**Audience:** Operator

**Plain English:** a `.env` file in the project root is a list of settings
(login, timezone, optional safety caps). You do not need most of them on day
one — [SETUP.md](SETUP.md) shows the short starter. This page is the full list.

**Related:** [Setup](SETUP.md) · [Web API](WEB_API.md) · [Timestamps](TIMESTAMPS.md) · [Trading safety](TRADING_SAFETY.md) · [Troubleshooting](TROUBLESHOOTING.md)

Never commit real credentials. Keep `.env` local.

---

## MT5 connection

| Variable | Default | Description |
|----------|---------|-------------|
| `MT5_LOGIN` | — | MetaTrader 5 account number |
| `MT5_PASSWORD` | — | Account password |
| `MT5_SERVER` | — | Broker server name |
| `MT5_TIMEOUT` | `30` | Connection timeout in seconds |

All connection settings are optional if MT5 is already logged in interactively.
For unattended/headless use, set `MT5_LOGIN`, `MT5_PASSWORD`, and `MT5_SERVER`
together; a partial credential set is rejected so mtdata cannot silently attach
to a different saved terminal account. `MT5_TIMEOUT` accepts 1–3600 seconds.

```ini
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Demo
MT5_TIMEOUT=30
```

---

## Timezone

MT5 documents UTC request datetimes and returned epochs. mtdata also supports
broker terminals that expose server-clock epochs: configure the broker setting
below so a fresh tick can verify that mode and the adapter can normalize request
bounds and results. Public payload timestamps remain UTC after normalization.

| Variable | Default | Description |
|----------|---------|-------------|
| `MT5_SERVER_TZ` | — | IANA timezone used for broker session/calendar boundaries and server-clock recognition/conversion (e.g. `Europe/Athens`). Handles DST automatically. |
| `MT5_TIME_OFFSET_MINUTES` | `0` | Fixed broker offset from UTC in minutes. A non-zero value overrides `MT5_SERVER_TZ`. |
| `MT5_CLIENT_TZ` / `CLIENT_TZ` | auto-detect | IANA timezone of the local machine. `CLIENT_TZ` takes precedence if both are set. |

Configure these when broker-local boundaries matter or when the terminal uses
broker server-clock epochs. Detection verifies a configured offset; without
one, mtdata follows MT5's native-UTC contract rather than inferring a clock
offset from a potentially stale tick.
Prefer `MT5_SERVER_TZ` because it adjusts for DST. Use
`MT5_TIME_OFFSET_MINUTES` only when you know a fixed session offset, and avoid
setting both unless you intentionally want the fixed offset to win.

```ini
# Option A — timezone name (recommended)
MT5_SERVER_TZ=Europe/Athens

# Option B — fixed offset
MT5_TIME_OFFSET_MINUTES=120
```

---

## Broker Time Verification

Optional runtime check for stale tick/bar data and implausible future UTC timestamps.

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_BROKER_TIME_CHECK` | `false` | Enable live MT5 UTC freshness verification (`1`, `true`, `yes`, or `on`) |
| `MTDATA_BROKER_TIME_CHECK_TTL_SECONDS` | `60` | Cache TTL for the check result (seconds) |

---

## MCP Server

Control how the MCP server binds and exposes endpoints.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `sse` | Transport mode: `sse`, `stdio`, or `streamable-http` |
| `FASTMCP_HOST` | `127.0.0.1` | Bind address |
| `FASTMCP_PORT` | `8000` | Listen port |
| `FASTMCP_ALLOW_REMOTE` | `false` | Set to `1` to allow non-loopback binds (e.g. `0.0.0.0`) |
| `MCP_AUTH_TOKEN` | — | Bearer / API-key token for SSE and streamable-HTTP. **Required** when binding to a non-loopback address. Optional on loopback; when set, clients must send `Authorization: Bearer <token>` or `X-API-Key: <token>`. Not used for `stdio`. |
| `MCP_ALLOWED_HOSTS` | — | Comma-separated externally used host patterns (for example `mt5.example.com:*`). Required for wildcard remote binds such as `0.0.0.0`. |
| `MCP_ALLOWED_ORIGINS` | — | Comma-separated allowed HTTP Origin patterns for remote MCP clients. |
| `FASTMCP_LOG_LEVEL` | `INFO` | Logging level |
| `FASTMCP_MOUNT_PATH` | `/` | SSE base mount path or streamable-HTTP endpoint path |
| `FASTMCP_SSE_PATH` | `/sse` | SSE event-stream path |
| `FASTMCP_MESSAGE_PATH` | `/message` | Message endpoint path |

---

## Web API

Settings for the FastAPI server that powers the React Web UI.

| Variable | Default | Description |
|----------|---------|-------------|
| `WEBAPI_HOST` | `127.0.0.1` | Bind address |
| `WEBAPI_PORT` | `8000` | Listen port |
| `WEBAPI_ALLOW_REMOTE` | `false` | Set to `1` to allow non-loopback binds |
| `WEBAPI_AUTH_TOKEN` | — | Bearer / API-key token. **Required** when binding to a non-loopback address. |
| `CORS_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | Comma-separated allowed origins. Wildcard `*` is rejected when credentials are enabled. |
| `WEBUI_DIST_DIR` | `webui/dist` | Path to the built Web UI static files (`index.html` required). Relative defaults are resolved from the process working directory, then parent directories. Run `mtdata-webapi` from the repository root, or set an absolute path. When missing, `/app` returns enablement guidance instead of a silent skip. |

```ini
# Expose the Web API on the local network with auth
WEBAPI_ALLOW_REMOTE=1
WEBAPI_HOST=0.0.0.0
WEBAPI_PORT=9000
WEBAPI_AUTH_TOKEN=my-secret-token
CORS_ORIGINS=http://192.168.1.10:5173
```

---

## News Embeddings

Configure the HuggingFace model used to rerank MT5 / external news by relevance.
Embedding reranking is disabled by default so an ordinary news request never
downloads or initializes a foundation model. Enabling it may download the
configured model on first use when it is not already cached.

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_NEWS_EMBEDDINGS_ENABLED` | `0` | Opt in to loading the embedding backend and model |
| `MTDATA_NEWS_EMBEDDINGS_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | HuggingFace model name |
| `MTDATA_NEWS_EMBEDDINGS_TOP_N` | `8` | Number of top-ranked items to keep after reranking |
| `MTDATA_NEWS_EMBEDDINGS_WEIGHT` | `1.0` | Weight for embedding-based reranking (0.0 disables) |
| `MTDATA_NEWS_EMBEDDINGS_TRUNCATE_DIM` | — | Truncate embedding vectors to this dimensionality (model must support Matryoshka) |
| `MTDATA_NEWS_EMBEDDINGS_CACHE_SIZE` | `256` | In-memory embedding vector cache size |
| `MTDATA_NEWS_EMBEDDINGS_HF_TOKEN_ENV_VAR` | `HF_TOKEN` | Name of the env var that holds the HuggingFace token |
| `HF_TOKEN` | — | HuggingFace API token for gated / private models |

---

## Finviz

| Variable | Default | Description |
|----------|---------|-------------|
| `FINVIZ_HTTP_TIMEOUT` | `15.0` | HTTP request timeout in seconds |
| `FINVIZ_SCREENER_MAX_ROWS` | `5000` | Maximum rows returned by a single screener request |
| `FINVIZ_PAGE_LIMIT_MAX` | `500` | Maximum pagination page limit |

---

## Options Data

Provider configuration for the options-chain tools (`options_expirations`, `options_chain`, `options_heston_calibrate`). The pure QuantLib calculator `options_barrier_price` does not require any of these. Run `options_provider_status` to check the effective configuration and whether mtdata is currently in authenticated or best-effort fallback mode.

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_OPTIONS_PROVIDER` | `yahoo` | Options-chain provider: `tradier`, `yahoo`, or `auto`. Yahoo negotiates an anonymous cookie/crumb but remains a best-effort fallback that may return 401/429. When `tradier` or `auto` is selected, mtdata retries Yahoo if Tradier is unavailable or misconfigured, but reliable chain access still requires a Tradier API key. Unsupported values remain visible as invalid in `options_provider_status`; Yahoo may be used only as the separately labeled effective fallback. |
| `MTDATA_OPTIONS_API_KEY` | — | Tradier API token (required for `tradier`). `TRADIER_TOKEN` and `TRADIER_API_KEY` are also accepted. |
| `MTDATA_OPTIONS_BASE_URL` | `https://api.tradier.com/v1` | Tradier API base URL. |

---

## Forecasting & GPU

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_FORECAST_PROCESS_ISOLATION` | `gpu` | Forecast subprocess isolation policy: `gpu` isolates only GPU-capable forecast calls, `all` isolates every forecast tool call, and `off` keeps in-process execution. Child exit is the reliable way to release CUDA context memory after GPU inference. |
| `MTDATA_FORECAST_PROCESS_TIMEOUT_SECONDS` | - | Optional wall-clock timeout for isolated forecast child processes. Leave unset or set `0` for no additional process timeout. |
| `MTDATA_NF_ACCEL` | auto-detect | Accelerator for NeuralForecast models: `gpu` or `cpu` |
| `CUDA_VISIBLE_DEVICES` | — | Restrict CUDA to specific GPU device(s). Auto-restricted to the first GPU when multiple are detected. |

---

## Async Training & Model Store

Concurrency caps and on-disk cache used by `forecast_train` / `forecast_task_*` / `forecast_models_*` and the auto-training path inside `forecast_generate`.

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_TRAIN_WORKERS` | `4` | Maximum number of background training threads in the shared executor. |
| `MTDATA_HEAVY_LIMIT` | `1` | Number of heavyweight (e.g. neural / foundation) training jobs allowed to run concurrently. Other jobs queue on the executor. |
| `MTDATA_FORECAST_JOBS_DB` | `~/.mtdata/forecast/jobs.sqlite` | SQLite registry for task status, progress, heartbeat, cancellation, and recovery across process restarts. |
| `MTDATA_TRAIN_TIMEOUT_INSTANT_SECONDS` | `30` | Timeout for methods categorized as instant. Values below 1 second are clamped to 1. |
| `MTDATA_TRAIN_TIMEOUT_FAST_SECONDS` | `120` | Timeout for methods categorized as fast. |
| `MTDATA_TRAIN_TIMEOUT_MODERATE_SECONDS` | `600` | Timeout for methods categorized as moderate. |
| `MTDATA_TRAIN_TIMEOUT_HEAVY_SECONDS` | `1800` | Timeout for methods categorized as heavy, such as neural / foundation training. |
| `MTDATA_FORECAST_HEARTBEAT_SECONDS` | `2` | Heartbeat interval used by heavy-process training jobs. Values below 0.5 seconds are clamped. |
| `MTDATA_FORECAST_ORPHAN_STALE_SECONDS` | `30` | Minimum heartbeat age before recovery may fail an active task whose worker PID is no longer alive. Values below 5 seconds are clamped. |
| `MTDATA_FORECAST_CANCEL_GRACE_SECONDS` | `3` | Grace period after cancellation before a still-running heavy worker is terminated. Values below 0.1 seconds are clamped. |
| `MTDATA_FORECAST_SWEEPER_SECONDS` | `60` | Interval for cleaning completed task records and expired model artifacts. Values below 5 seconds are clamped. |
| `MTDATA_FORECAST_TASK_TTL_SECONDS` | `86400` | Retention for terminal task records, including bounded failure diagnostics. Values below 60 seconds are clamped. |
| `MTDATA_MODEL_STORE` | `~/.mtdata/models` | Root directory used by the model store. Trained models are keyed by `method/data_scope/params_hash`. |
| `MTDATA_MODEL_TTL_DAYS` | `7` | Cached model idle expiry in days since last use. Idle models are evicted on access; frequently used models do not expire by age. |

---

## Market Depth

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_ENABLE_MARKET_DEPTH_FETCH` | `false` | Enable the `market_depth_fetch` tool (`1`, `true`, `yes`, or `on`). Disabled by default because it requires Level 2 data from the broker. |

---

## Trading

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_ORDER_MAGIC` | `234000` | Magic number stamped on all orders placed by mtdata. Change this to distinguish mtdata orders from orders placed by other EAs or scripts on the same account. |
| `MTDATA_TRADE_IDEMPOTENCY_DB` | `~/.mtdata/trade_idempotency.sqlite3` | SQLite database shared by CLI and server processes for durable `trade_place` and `trade_modify` retry suppression. All workers that can reach the same MT5 account should use the same path. |
| `MTDATA_TRADE_IDEMPOTENCY_TTL_SECONDS` | `86400` | Retention for completed idempotency outcomes. In-progress records are not automatically expired because the broker outcome may be ambiguous after a crash. |

---

## Trade Guardrails

Optional pre-trade controls that can block `trade_place` and risk-increasing pending-order modifications before MT5 submission.

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_TRADE_GUARDRAILS_ENABLED` | `false` | Master switch for guardrail evaluation. Guardrails also activate automatically when specific guardrail variables are configured. |
| `MTDATA_TRADE_GUARDRAILS_IGNORE_ON_DEMO` | `false` | When `true`, configured guardrails are skipped for demo accounts. Leave unset so demo rehearsals use the same caps as live. This modifies guardrail behavior but does not activate guardrails by itself. |
| `MTDATA_TRADING_ENABLED` | `true` | Set to `0` / `false` to block all new trade placements via mtdata. |
| `MTDATA_TRADE_ALLOWED_SYMBOLS` | — | Comma-separated allowlist of tradable symbols, e.g. `EURUSD,BTCUSD,XAUUSD`. When set, symbols outside the list are blocked. |
| `MTDATA_TRADE_BLOCKED_SYMBOLS` | — | Comma-separated blocklist of symbols that mtdata must refuse. |
| `MTDATA_TRADE_MAX_VOLUME` | — | Global max order volume cap. |
| `MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL` | — | Per-symbol max volume map. Format: `SYMBOL:VALUE,SYMBOL=VALUE`, e.g. `EURUSD:0.50,BTCUSD=0.03`. |
| `MTDATA_TRADE_SAFETY_MAX_VOLUME` | — | Optional max-volume cap applied through the safety-policy layer. Usually `MTDATA_TRADE_MAX_VOLUME` is easier to reason about. |
| `MTDATA_TRADE_SAFETY_REQUIRE_STOP_LOSS` | `false` | Require a stop-loss for guarded orders. Recommended when wallet-risk caps are enabled. |
| `MTDATA_TRADE_SAFETY_MAX_DEVIATION` | — | Maximum allowed MT5 deviation/slippage value in points. |
| `MTDATA_TRADE_SAFETY_REDUCE_ONLY` | `false` | Only allow orders that reduce existing exposure. |
| `MTDATA_TRADE_MIN_MARGIN_LEVEL_PCT` | — | Block orders when account margin level is below this threshold. |
| `MTDATA_TRADE_MAX_FLOATING_LOSS` | — | Block orders when current unrealized account loss exceeds this amount. |
| `MTDATA_TRADE_MAX_TOTAL_EXPOSURE_LOTS` | — | Block orders when aggregate open exposure plus the new order would exceed this lot cap. |
| `MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY` | — | Block orders when quantified portfolio risk after the new trade would exceed this percent of account equity. |
| `MTDATA_TRADE_MAX_RISK_PCT_OF_BALANCE` | — | Same as above, measured against account balance. |
| `MTDATA_TRADE_MAX_RISK_PCT_OF_FREE_MARGIN` | — | Same as above, measured against free margin. |

Notes:

- Wallet-risk caps require a quantifiable stop-loss and valid broker tick metadata.
- Existing position risk is measured from the current mark to each stop, so
  trailed stops still reserve the equity that would be lost from the current
  account state. If the position snapshot is unavailable, snapshot-dependent
  guardrails block the action.
- Wallet-risk guardrails reserve risk from open positions, existing pending
  orders, and the candidate order. They fail closed when any of those records
  lacks a quantifiable stop-loss or valid broker tick metadata.
- Leave any variable unset to disable only that rule.
- `trade_modify` guardrails apply only to pending-order changes and position SL changes that increase risk; close/reduce flows stay allowed. Risk-increasing modifications still apply symbol allowlist, blocklist, and per-symbol volume-map rules.

```ini
# Example trade guardrail setup
MTDATA_TRADE_GUARDRAILS_ENABLED=1
MTDATA_TRADING_ENABLED=1
MTDATA_TRADE_ALLOWED_SYMBOLS=EURUSD,BTCUSD,XAUUSD
MTDATA_TRADE_BLOCKED_SYMBOLS=US30
MTDATA_TRADE_MAX_VOLUME=1.0
MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL=EURUSD:0.50,BTCUSD:0.03,XAUUSD:0.10
MTDATA_TRADE_SAFETY_REQUIRE_STOP_LOSS=1
MTDATA_TRADE_SAFETY_MAX_DEVIATION=20
MTDATA_TRADE_MIN_MARGIN_LEVEL_PCT=200
MTDATA_TRADE_MAX_FLOATING_LOSS=1000
MTDATA_TRADE_MAX_TOTAL_EXPOSURE_LOTS=2.0
MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY=1.5
MTDATA_TRADE_MAX_RISK_PCT_OF_BALANCE=1.25
MTDATA_TRADE_MAX_RISK_PCT_OF_FREE_MARGIN=2.0
```

---

## CLI & Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_OUTPUT_FORMAT` | `toon` | Persistent CLI output format: `toon` or `json`. An explicit `--json` flag takes precedence; other nonblank values are rejected. |
| `MTDATA_CLI_DEBUG` | `false` | Enable verbose debug logging in the CLI (`1`, `true`, `yes`, or `on`) |
| `NO_COLOR` | — | Disable ANSI color output (any non-empty value). Follows the [no-color.org](https://no-color.org) convention. |

---

## Quick `.env` Template

A starter template with all sections. Uncomment and fill in what you need.

```ini
# ── MT5 Connection ──────────────────────────────────────
# MT5_LOGIN=12345678
# MT5_PASSWORD=your_password
# MT5_SERVER=YourBroker-Demo
# MT5_TIMEOUT=30

# ── Server timezone (pick one) + optional client TZ ─────
# MT5_SERVER_TZ=Europe/Athens
# MT5_TIME_OFFSET_MINUTES=120
# MT5_CLIENT_TZ=America/New_York

# ── Broker Time Check ──────────────────────────────────
# MTDATA_BROKER_TIME_CHECK=false
# MTDATA_BROKER_TIME_CHECK_TTL_SECONDS=60

# ── MCP Server ─────────────────────────────────────────
# MCP_TRANSPORT=sse
# FASTMCP_HOST=127.0.0.1
# FASTMCP_PORT=8000
# FASTMCP_ALLOW_REMOTE=0
# FASTMCP_LOG_LEVEL=INFO

# ── Web API ────────────────────────────────────────────
# WEBAPI_HOST=127.0.0.1
# WEBAPI_PORT=8000
# WEBAPI_ALLOW_REMOTE=0
# WEBAPI_AUTH_TOKEN=
# CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173

# ── News Embeddings ────────────────────────────────────
# MTDATA_NEWS_EMBEDDINGS_MODEL=Qwen/Qwen3-Embedding-0.6B
# MTDATA_NEWS_EMBEDDINGS_ENABLED=0
# MTDATA_NEWS_EMBEDDINGS_TOP_N=8
# MTDATA_NEWS_EMBEDDINGS_WEIGHT=1.0
# MTDATA_NEWS_EMBEDDINGS_TRUNCATE_DIM=
# MTDATA_NEWS_EMBEDDINGS_CACHE_SIZE=256
# MTDATA_NEWS_EMBEDDINGS_HF_TOKEN_ENV_VAR=HF_TOKEN
# HF_TOKEN=

# ── Finviz ─────────────────────────────────────────────
# FINVIZ_HTTP_TIMEOUT=15.0
# FINVIZ_SCREENER_MAX_ROWS=5000
# FINVIZ_PAGE_LIMIT_MAX=500

# ── Forecasting / GPU ──────────────────────────────────
# MTDATA_FORECAST_PROCESS_ISOLATION=gpu
# MTDATA_FORECAST_PROCESS_TIMEOUT_SECONDS=
# MTDATA_NF_ACCEL=cpu
# CUDA_VISIBLE_DEVICES=0

# ── Async Training & Model Store ───────────────────────
# MTDATA_TRAIN_WORKERS=4
# MTDATA_HEAVY_LIMIT=1
# MTDATA_FORECAST_JOBS_DB=~/.mtdata/forecast/jobs.sqlite
# MTDATA_TRAIN_TIMEOUT_INSTANT_SECONDS=30
# MTDATA_TRAIN_TIMEOUT_FAST_SECONDS=120
# MTDATA_TRAIN_TIMEOUT_MODERATE_SECONDS=600
# MTDATA_TRAIN_TIMEOUT_HEAVY_SECONDS=1800
# MTDATA_FORECAST_HEARTBEAT_SECONDS=2
# MTDATA_FORECAST_ORPHAN_STALE_SECONDS=30
# MTDATA_FORECAST_CANCEL_GRACE_SECONDS=3
# MTDATA_FORECAST_SWEEPER_SECONDS=60
# MTDATA_FORECAST_TASK_TTL_SECONDS=86400
# MTDATA_MODEL_STORE=~/.mtdata/models
# MTDATA_MODEL_TTL_DAYS=7

# ── Market Depth ───────────────────────────────────────
# MTDATA_ENABLE_MARKET_DEPTH_FETCH=0

# ── Trading ────────────────────────────────────────────
# MTDATA_ORDER_MAGIC=234000
# MTDATA_TRADE_IDEMPOTENCY_DB=~/.mtdata/trade_idempotency.sqlite3
# MTDATA_TRADE_IDEMPOTENCY_TTL_SECONDS=86400

# ── CLI / Debug ────────────────────────────────────────
# MTDATA_CLI_DEBUG=0
# NO_COLOR=
```
