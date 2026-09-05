# Response and output contract

**Audience:** Operator

**Plain English:** every tool answers with a small wrapper that says whether it
worked (`success`) plus the payload. This page is the contract for that wrapper
— useful when you write a script or an agent. Day-to-day CLI use only needs
`--json` and the [CLI guide](CLI.md).

CLI and [MCP](GLOSSARY.md#mcp-model-context-protocol) expose the canonical tool
payloads described here. Dedicated Web API chart routes may be more compact;
the Tools invoke path uses the same domain semantics. See
[WEB_API.md](WEB_API.md).

Presentation flags and exit codes: [CLI.md](CLI.md#output-contract).

---

## The response envelope

Successful tool responses are JSON objects that carry a `success` flag plus the tool's data:

```json
{
  "success": true,
  "symbol": "EURUSD",
  "timeframe": "H1",
  "data": [ ... ]
}
```

- `success` — `true` on success, `false` on failure. Always present on failures; most tools also set it on success.
- The remaining keys are tool-specific (`data`, `rows`, `levels`, `forecast`, etc.).
- List-style tools include a pagination block (see [Pagination](#pagination)).

> **Scripting tip:** branch on `success` first, then read the tool-specific fields. On the CLI, also check the [exit code](CLI.md#exit-codes).

### Partial composite results

Tools that combine independently requested symbols or sections use
`partial_failure: true` when at least one item succeeds and at least one fails.
Multi-symbol `market_status` and all-mode `patterns_detect` also return
`requested_count`, `succeeded_count`, `failed_count`, and machine-readable
`failed_items`; radar, snapshot, and reports identify their missing rows or
sections in their domain-specific fields (`missing_symbols` on scan and radar).
Where `allow_partial` is exposed on explicit market lists, it keeps usable
results by default and is documented as permissive; set it to false when
automation requires `success: false` and a nonzero CLI exit for any omission.
Correlation and similar statistical tools stay fail-closed. If every requested
item fails, `success` is false regardless of `allow_partial`.

### Data provenance

Compact MT5-backed output identifies the provider once:

```json
{"source": {"provider": "mt5"}}
```

`detail=full` moves the complete source context under `meta.source`:

```json
{
  "meta": {
    "source": {
      "provider": "mt5",
      "broker_company": "Broker Co",
      "server": "Broker-Demo",
      "context_id": "2d86b49c6e6c8b9e",
      "context_available": true
    }
  }
}
```

The context id is a stable digest of the non-secret broker company/server pair,
so detached results can be compared without exposing an account login or
credentials. When account context is unavailable, `provider` remains `mt5` and
`context_available` is false. Method-level lineage such as candle price basis or
tick retrieval method remains in its tool-specific field.

Payloads computed without MT5 identify their actual producer instead. For
example, venue-level `market_status` uses
`provider: mtdata_exchange_calendar` with its holiday-calendar provider.
Pure timers omit `source` because they observe only elapsed time. A narrower
legacy label may remain in `data_lineage`; it never replaces the structured
source context.

---

## Detail levels (detail)

For tools whose schema includes it, `detail` controls **how much field-level
verbosity** a response carries. It is not a universal parameter. The accepted
values for tools that expose the complete detail contract are:

| Value | Meaning |
|-------|---------|
| `compact` | **Default.** Results, safety gates, and non-nominal warnings only. Healthy telemetry and request echoes are omitted. |
| `standard` | Shared stripping is the same as `compact`; an individual tool may provide a distinct standard shape. |
| `summary` | Shared stripping is the same as `compact`; an individual tool may provide a distinct summary shape. |
| `full` | Consolidates source, time, freshness, processing, quality, request, units, and diagnostics under `meta`. |

Notes:
- `detail` changes verbosity **within** the sections a tool already returns; it does **not** add new analysis. For example, `market_snapshot` uses a separate `sections` parameter to choose analysis modules. Compact output exposes the selected structured results directly and reports only failed sections; full output retains the section-selection metadata. Exception: `data_fetch_candles --detail summary` replaces `data` rows with `summary_statistics` and `latest_candle`.
- The shared output layer has two retention modes: `full`, and the compact strip used by `compact`, `standard`, and `summary`. Tools can independently distinguish the accepted values in their own payloads.
- Use `detail=full` when you need several metadata sections. For one field,
  prefer `output_fields` so the rest remains compact.

Compact catalogs and analyses also omit default filter echoes, static formula or
unit legends, repeated counts, and instructional prose when the structured
result already carries the same information. Model inventory compact rows
contain the reusable `model_id`; an incompatible row retains its status and
reason. `detail=standard` adds method, data_scope, created_at, and horizon.
Empty task or trading collections return the collection without a second prose
explanation.

Repeated row constants may be lifted to the response root. For example,
`symbols_top_markets` reports a shared timeframe and data source once, while
each row keeps its symbol, measurements, observation times, and stale flags.
Compact risk, confluence, outlier, and microstructure results likewise keep the
computed result and non-nominal quality state while omitting derivable windows,
healthy-status legends, and duplicate aggregate values. Use `detail=full` when
auditing those calculation and quality diagnostics.

### Anomaly-first warnings

Healthy compact output does not repeat `freshness`, `data_stale=false`, age,
threshold, reason, or policy fields. A non-nominal condition appears once:

```json
{
  "warnings": [
    {
      "code": "data_stale",
      "scope": "candles",
      "message": "The latest data is outside the expected freshness window.",
      "data_as_of": "2026-08-29T04:15:00Z",
      "age_seconds": 581
    }
  ]
}
```

`code`, `scope`, and `message` are stable warning fields. Additional keys are
small condition-specific context. Identical warning messages are emitted once,
even when nested components discovered the same condition under different
scopes; the most specific code and available context are retained. Candle rows do not repeat `bar_state` or
`gap_before`: a forming row is identified once by `forming_candle_index`, and
session gaps produce one `session_gap` warning. Full output retains the
complete gap and forming-bar diagnostics under `meta.quality`.

---

## Richer output

Compact output is implicit. When the tool schema or command help lists
`detail`, set `detail=full` to retain the richer metadata and diagnostic
sections it produces. Tools with a meaningful intermediate or summary
representation expose only those detail values in their schema; tools without
`detail` reject the parameter.

```bash
# Full market-session diagnostics
mtdata-cli market_status --detail full

# Full forecast context
mtdata-cli forecast_generate EURUSD --horizon 12 --detail full --json

# One full-detail field without the complete metadata envelope
mtdata-cli data_fetch_candles BTCUSD --output-fields symbol,source.server --json
```

## Field selection (output_fields)

`output_fields` narrows a response to specific top-level keys or dotted paths.
It can select a normally-full field directly while the rest of the response
stays compact. Combine it with `--json` for token-lean machine parsing:

```bash
mtdata-cli symbols_describe EURUSD --output-fields symbol,details.digits,details.point --json
mtdata-cli data_fetch_candles BTCUSD --output-fields symbol,indicator_engine.effective_backend --json
mtdata-cli data_fetch_candles BTCUSD --output-fields symbol,meta.processing.indicators.engine --json
```

Bare names select top-level keys only. Use a dotted path such as
`general_news.title` to select nested values. Any requested path that is not
present is returned in `unresolved_output_fields`; projection never silently
searches unrelated nested objects for a matching key. A mixed projection keeps
the resolved values and sets `output_fields_status=partial`. If no requested
path resolves, the response sets `success=false`,
`error_code=output_fields_unresolved`, and `output_fields_status=failed`; the
CLI exits `1`. `valid_output_fields` lists paths available to targeted
selection, including deep canonical `meta.*` paths and compact rows. Declared
row paths remain resolvable through an empty collection, so a flat account can
return `items=[]` for `trade_get_open --output-fields items.symbol` without
reporting `items.symbol` as unresolved. Strategy-attribution fields such as
`items.magic` and `items.comment` stay in compact `trade_get_open` rows so
`--output-fields` can select them without a detail override.

Field selection is authoritative across formats. JSON and TOON retain the same
selected keys; TOON may still apply the requested numeric precision. Compact
warnings and trading safety gates are carried automatically. Projecting
`bid,ask,spread` therefore cannot drop `usable_for_live_trading=false` or a
stale-quote warning.

`json` and `output_fields` are the shared output-shaping parameters available
across tools. A domain-specific parameter named `fields` (currently used by
Finviz fundamentals) selects source data and is not response projection.

---

## Pagination

An empty market-data window is a successful query, not a provider failure.
Both candle and tick tools return `success=true`, `data=[]`, `empty=true`, and
an `empty_reason` such as `market_closed_weekend`, `forming_bar_excluded`, or
`no_ticks_in_range`. Connection, symbol, validation, and provider failures keep
the normal error envelope and a nonzero CLI exit.

For a fully bounded tick range, the latest matching ticks are returned whether
`limit` is omitted or set to the same default of 20. Full output keeps the
sample's `last_quote` diagnostics under the richer contract. Compact output
returns the tick rows and only adds a warning when freshness or quote quality
is non-nominal.
Bounded tick queries probe one event beyond the page. When `pagination.has_more`
is true, pass `pagination.next_cursor` back as `--cursor` with the same symbol,
start, and end values. The opaque cursor uses a raw-event offset, so ticks that
share a millisecond are neither skipped nor duplicated. In full output for
simplified rows, `pagination.returned` remains the number of rows actually
present while `source_events_returned` records the pre-simplification page
size.

Start-anchored candle ranges use the same continuation pattern. When
`pagination.has_more` is true, reuse `pagination.next_cursor` with the original
symbol, timeframe, start, and end values. A provider-bounded candle page reports
`total: null`, `more_available: null`, and an evidence-based
`total_lower_bound`; it never presents the fetched prefix as the full range.

Full list output returns the complete normalized pagination block. Compact
output omits pagination when the collection is complete. When more data exists,
it returns only `has_more` plus `next_cursor` or `next_offset`. Catalog tools
(`tools_list`, `indicators_list`, `denoise_list_methods`, and
`forecast_list_methods`) also keep `pagination.total` so the collection size
stays visible.

Public `limit` parameters always cap returned rows (including returned candles or
ticks). Historical samples used only for analysis are named `lookback`,
`fetch_limit`, or `window_bars` so generic paging controls cannot silently shrink
an analytical window.

```json
{
  "total": 420,
  "returned": 50,
  "offset": 0,
  "limit": 50,
  "has_more": true,
  "more_available": 370
}
```

| Field | Meaning |
|-------|---------|
| `total` | Exact rows available before paging, or `null` when the provider cannot determine it |
| `total_lower_bound` | Present only when `total` is unknown; minimum rows known to exist |
| `returned` | Rows in this response |
| `offset` | Zero-based start index of this page |
| `limit` | Page size requested (`null` when unbounded) |
| `has_more` | `true` when more rows remain after this page |
| `more_available` | Exact count of rows remaining, or `null` when `total` is unknown |

When a bounded provider can only prove that another row exists, `total` and
`more_available` remain `null`; `total_lower_bound` and `has_more` carry the
available evidence without presenting a page-size-dependent estimate as an
exact universe count.

The `pagination` object is authoritative and is the only pagination
representation in canonical payloads. Root-level `total_count`, `offset`,
`limit`, `page`, `pages`, `has_more`, and `more_available` aliases are not
emitted. Generic compact collections omit a redundant root `count`; the row
array or TOON header already carries that size. Dedicated Web UI routes may
retain `count` in their route-specific DTO.
Tools that accept a one-based `page` input convert it to the zero-based
`pagination.offset` value.

`trade_history` is the keyset-pagination exception: its default page size is
20 (capped at 500), and `pagination.next_cursor` freezes the resolved UTC window plus the last
history time/ticket key. Reuse that cursor with unchanged history filters and
time controls. This avoids skips or duplicates when relative windows move or
the account history changes between calls. Trade-history cursors expire after
one hour; `offset` and `page` are not accepted by that tool. Cursors are opaque:
store and return the exact string rather than decoding it or depending on its
representation.

Compact trade-history responses retain `item_schema`. Deal rows keep the
monetary columns `profit`, `commission`, `swap`, `fee`, and `net_pnl` at stable
paths: a known zero is `0.0`, while unavailable source data is `null`.
`net_pnl` is the sum of the four broker components on that deal, not the
journal's separately reported round-trip result with allocated entry costs.

Page through results with `--offset` and `--limit`:

```bash
mtdata-cli tools_list --category forecast --limit 20 --offset 0 --json
mtdata-cli tools_list --category forecast --limit 20 --offset 20 --json
```

---

## Market-data timestamps

For candle and tick rows, `timestamp_format` is the authoritative description
of the serialized `time` value. The default is `iso_utc` (UTC `Z` strings).
`iso` renders ISO 8601 values in `CLIENT_TZ` (`iso_offset` in the payload).
`epoch` returns Unix seconds in UTC. UTC instants use the RFC 3339 `Z`
suffix (`2026-08-28T04:15:00Z`), not `+00:00`.

Generic compact output relies on the serialized row value (`Z`, an explicit
offset, or numeric epoch) and omits repeated root timestamp metadata. The
dedicated `/api/history` DTO retains `timestamp_format` because the chart client
uses it to decode rows. Use `detail=full` for `meta.time`, including the data
anchor, retrieval time, window, timezone, and raw timestamp mode.

---

## Percentage units

Fields whose names end in `_pct` are scaled percent values: `1.0` means `1%`.
Their unit is reported as `percent`, not `percentage_points`. The latter term is
reserved for an actual difference between two percentages, such as a change in
win rate from 52% to 55% (`3 percentage points`). Decimal-return fields without
the `_pct` suffix use fractions, where `0.01` means `1%`.

---

## Error envelope

Failures return a **structured** payload (not just a string) so callers can react programmatically:

```json
{
  "success": false,
  "error": "Symbol NOTAREALPAIR not found.",
  "error_code": "symbol_not_found",
  "request_id": "b0f3…",
  "operation": "symbols_describe",
  "remediation": "Use symbols_list to browse available broker symbols.",
  "related_tools": ["symbols_list"],
  "valid_values": { ... },
  "example": "mtdata-cli symbols_describe EURUSD",
  "documentation": "docs/CLI.md",
  "details": { ... }
}
```

| Field | Always present | Meaning |
|-------|:---:|---------|
| `success` | ✅ | Always `false` on errors |
| `error` | ✅ | Human-readable message |
| `error_code` | ✅ | Stable lowercase machine-readable code (e.g. `symbol_not_found`, `invalid_date_range`). Shared failures are unprefixed; the tool name lives in `operation`. |
| `request_id` | ✅ | Correlation id for logs |
| `operation` | | The tool that failed |
| `remediation` | | Suggested fix |
| `related_tools` | | Tools that can help |
| `valid_values` | | Accepted values when the failure was a bad argument |
| `example` | | A corrected example invocation |
| `documentation` | | Relevant doc pointer |
| `details` | | Structured, tool-specific context |

Prefer `error_code` over string-matching `error` when you need to branch on failure type. On the CLI, tool/provider failures share [exit code `1`](CLI.md#exit-codes), so parse `error_code` to distinguish them.

For `symbol_not_found`, market-data tools consistently include
`details.did_you_mean`, an ordered array of broker catalog candidates with
`symbol` and optional `description`/`group` fields. The field is present as an
empty array when no candidate matches, so callers do not need to parse names
from the human-readable error string.

When a regular-session symbol has no current quote or candles but the broker
offers a live extended-session sibling, `market_ticker` and
`data_fetch_candles` include the same `details.related_live_symbols` array.
Each entry names the exact broker `symbol`, its `session_type`, and the
recommended `quote_tool`, so callers can retry without parsing remediation
text.

---

## Freshness and execution readiness

`usable_for_live_trading` is reserved for execution-oriented quote and session
outputs. Compact output keeps that single gate and emits a structured warning
only when freshness, market state, spread validity, or source agreement is
non-nominal. `detail=full` places the basis and measured ages under
`meta.freshness`:

- `quote_age_market_session_and_positive_spread` is the executable-quote
  check used by ticker, symbol status, symbol description, snapshots, and
  trading previews. Its default age threshold is 10 seconds and it also
  requires a positive two-sided bid/ask. `quote_age_and_market_session` is the
  lower-level freshness result before quote quality is applied; do not treat it
  as sufficient execution evidence.
- Historical bars, forecasts, volatility estimates, and research backtests do
  not publish this execution-sounding boolean. Use `history_policy_ok`,
  `signal_status`, and `usage` for their respective contracts, then obtain a
  current quote before execution.
- Forecast optimizers that explicitly gate candidate viability may require both
  model-history and reference-quote readiness. Probability forecasts remain
  research outputs: they expose `execution_blockers` when an input fails but do
  not publish the execution-sounding readiness boolean.

Historical detection outputs from `patterns_detect`, `regime_detect`,
`support_resistance_levels`, and `confluence_levels` disclose their repaint
boundary with `input_bar_policy: closed_bars_only`,
`latest_bar_complete: true`, and `forming_candle_status: excluded`. Level tools
may still use a live quote as their distance reference; check
`current_price_source` or `reference_price_source` separately from the
closed-bar structure policy.

For an unbounded latest-data request, historical and analytical tools evaluate
the completed-bar freshness policy. Healthy compact results omit that
telemetry. A stale or unverified result carries one warning with its anchor and
age. Full output retains the measured status, policy, basis, and threshold in
`meta.freshness`. The age starts at the latest analyzed bar's close, not its
open.

Full `meta.time` separates retrieval time from the data anchor and labels each
basis. Compact reports keep their primary `data_as_of` anchor; other compact
tools omit nominal time telemetry unless a dedicated route needs it.

---

## TOON vs JSON

The canonical payload above is what you get with `--json`. Without `--json`,
the CLI renders the same payload as compact **TOON** text and applies
`--precision auto`. Format and precision are presentation-only: they never
rewrite stored values or JSON numbers. See [CLI.md](CLI.md#output-contract)
for details, and set `MTDATA_OUTPUT_FORMAT=json` to default all output to JSON.

TOON is a display of that payload, not a second data contract. Remaining
presentation deltas:

- **Numbers.** Auto/compact TOON simplifies most values for readability
  (~0.1% relative precision unless a field rule applies). Money fields
  (`equity`, `balance`, `margin*`, `profit`, `*pnl*`, `commission`, `swap`,
  `fee`, `*_before`, `*_after`) render at 2 decimals. Price fields use
  symbol digits. Quote-summary `min`/`max` use the same decimals as `mean`.
  `*_pct` cells use per-value decimals, not a column-wide choice. A
  `p_value` of `0.0` displays as `<1e-6`. Use `--precision full` for
  unrounded TOON numbers; `--json` always keeps machine-precision values.
- **Empty collections.** Empty lists such as `items`, `data`, `rows`, and
  `tasks` stay visible (`items[0]:`). Empty objects, nulls, and blank
  strings may still be omitted.
- **Key order** follows payload insertion order and is not a public
  stability guarantee.

JSON remains the authoritative machine-readable form.

---

## See Also

- [CLI.md](CLI.md#output-contract) — TOON/JSON, `--precision`, exit codes
- [ENV_VARS.md](ENV_VARS.md) — `MTDATA_OUTPUT_FORMAT` and related settings
- [WEB_API.md](WEB_API.md) — how the same payloads are served over REST

Candle results retain `price_basis` and `price_currency` at the root in every detail profile, so OHLC values carry their quote side and currency.


Forecast tuning results carry request identity, lookback, seed, and the analysis window once at the root; the redundant `tuning_context` wrapper has been removed.

