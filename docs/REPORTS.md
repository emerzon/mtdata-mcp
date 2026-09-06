# Report generation

**Audience:** User

`report_generate` packages several read-only analysis steps into one structured
market summary. Use it when you want a repeatable overview rather than calling
context, forecast, level, pattern, barrier, news, and regime tools separately.

Reports are research output, not trade instructions. A report can also be
partial when a provider, optional dependency, or sub-analysis is unavailable;
inspect its section statuses and diagnostics before relying on it.
For a sized dry-run preview of one idea, use [trade_idea_compose](TRADE_IDEAS.md)
instead of treating a report as an order ticket.

**Related:** [CLI](CLI.md) · [Trade ideas](TRADE_IDEAS.md) · [Output contract](OUTPUT.md) · [Forecasting](FORECAST.md) · [Levels](LEVELS.md) · [Regimes](forecast/REGIMES.md) · [Barriers](BARRIER_FUNCTIONS.md) · [News](NEWS.md)

---

## Quick start

```bash
mtdata-cli report_generate EURUSD --timeframe H1
```

The command defaults to the fast `minimal` template (context and forecast only)
and compact TOON text. Use `--template basic` for confluence levels, patterns,
barriers, and broader risk context; use `--json` for a machine-readable payload or
`--detail full` for all content supported by the selected template. CLI and MCP
preserve the same canonical report payload; output format only changes its final
presentation. Compact output is a one-screen brief: last price, a short narrative,
nearest levels, forecast, and risk. Pass `--detail standard` or `--detail full`
when you need the full section dump.

## Choose a template

| Template | Typical warm runtime | Design and intended use |
|----------|----------------------|-------------------------|
| `minimal` | 3-10 seconds | Default fast path: context and direct forecast only |
| `basic` | 30-120 seconds | Research pipeline: context, daily pivots, confluence, one volatility estimate, forecast, fast barrier search, recent patterns |
| `advanced` | 60-180 seconds | Extends `basic` with regime, HAR-RV, classic/Elliott patterns, and conformal intervals when the forecast has none |
| `scalping` | 15-60 seconds | M5 path with live quote, session status, execution gates, and tick-aware barriers |
| `intraday` | 30-120 seconds | H1 path plus session status, news, and session seasonality |
| `swing` | 30-120 seconds | H4/D1 path plus volume-profile value area and news |
| `position` | 30-120 seconds | D1/W1 path plus weekly confluence, volume profile, news, and Elliott context |

Style templates still choose different default timeframes, lookbacks, and
barrier grids. They now also change which extra tools run: session and news on
intraday, volume profile and news on swing/position, and live quote/session
gates on scalping.

Compatible timeframe ranges for style templates:

| Template | Typical | Expected range | Rejected overrides |
|----------|---------|----------------|--------------------|
| `scalping` | M5 | M1-M15 | D1, W1, MN1 |
| `intraday` | H1 | M15-H4 | MN1 |
| `swing` | H4 | H1-D1 | M1 |
| `position` | D1 | H4-MN1 | M1-M5 |

Unusual but non-absurd overrides such as `scalping`+H4 or `intraday`+W1 keep
success and emit a structured `template_timeframe_warning`. Truly contradictory
pairs fail with `incompatible_template_timeframe`.

A single `--methods` value skips the ranking backtest and forecasts that method
directly. Denoising, when requested, is applied to candles, forecast, backtest,
volatility, and barrier searches.

`minimal` is the bounded interactive default. The other templates may perform several MT5
fetches and invoke pivots, patterns, backtests, barriers, or regime checks.
Runtime and dependency requirements therefore vary by template. Section
controls select the sections to execute and return, while internal
dependencies may still run when a requested section requires them.
The ranges above are guidance, not deadlines: broker history synchronization,
explicit model choices, and cold model initialization can take longer.

## Control template, scope, and output

```bash
# Fast overview
mtdata-cli report_generate EURUSD --template minimal --timeframe H1

# Basic-pipeline style preset with an explicit forecast horizon
mtdata-cli report_generate EURUSD --template swing --timeframe H4 --horizon 12

# Keep only selected computed sections
mtdata-cli report_generate EURUSD --template basic --include-sections context,forecast,barriers --max-sections 3 --json

# Run until the actual 10-second deadline and show progress
mtdata-cli report_generate EURUSD --template basic --max-runtime 10 --progress true --json

# Restrict candidate forecast methods and apply denoising
mtdata-cli report_generate EURUSD --template basic --methods theta,arima --denoise kalman --json
```

For an after-hours review, explicitly allow completed context from the latest
closed session. The result keeps the candle timestamp, age, stale status, and
warning so it cannot be mistaken for a live mark:

```bash
mtdata-cli report_generate AAPL.NAS --template minimal --allow-stale true
```

Useful controls:

- `--horizon` sets the forecast horizon in bars.
- `--timeframe`, `--start`, and `--end` constrain the requested market window.
  `--end` may be used alone for an as-of snapshot; `--start` requires `--end`
  so snapshot and range-aware sections share one historical cutoff.
  For intraday reports, context ends at the latest bar that was fully closed at
  that instant, matching the forecast training cutoff.
- When `--start` or `--end` bounds a report, sections that only support current-market
  analysis are not run. Their section payloads use `status: omitted` with reason
  `current_only_section_omitted`, and the report is marked partial rather than mixing
  current data into the bounded analysis.
  Pivots, multi-timeframe pivots, and barrier optimization support the shared
  cutoff and remain included when requested. Historical barriers use candle
  closes and retain their research-only execution restrictions.
- `--methods` supplies comma- or space-separated forecast methods.
- `--include-sections` selects the sections to execute and return; required
  internal dependencies may run but cannot independently make the request
  successful. Names must belong to the selected template:

  | Template | Valid section names |
  |----------|---------------------|
  | `minimal` | `context`, `forecast` |
  | `basic` | `context`, `pivot`, `contexts_multi`, `pivot_multi`, `volatility`, `backtest`, `forecast`, `barriers`, `patterns`, `confluence` |
  | `advanced` | Basic sections plus `regime`, `volatility_har_rv`, `forecast_conformal` |
  | `scalping` | `context`, `pivot`, `contexts_multi`, `pivot_multi`, `volatility`, `backtest`, `forecast`, `barriers`, `patterns`, `market`, `execution_gates`, `session` |
  | `intraday` | Basic sections plus `market`, `execution_gates`, `session`, `news`, `temporal` |
  | `swing` / `position` | Basic sections plus `volume_profile`, `news` |

  Unknown or unavailable names fail before any report sections run, even when
  `--allow-partial true`; `valid_sections` lists the selected template's names.
  `--max-sections` caps the selected count. The response preserves the original
  names in `requested_sections`; `capped_requested_sections` and the
  `max_sections_limited` reason distinguish capped work from work that was
  never requested.
- `--max-runtime` supplies a cooperative wall-clock budget. The runner first
  schedules the selected sections and then stops starting report sub-tools once
  the actual deadline passes. Static section estimates are advisory and never
  consume the wall-clock budget. An already-running native or MT5 call cannot
  be safely preempted, so a single call can finish just beyond the requested
  budget. `runtime_plan` separates deadline omissions from estimates and records
  elapsed time and whether the real budget was exhausted.
- `--progress true` writes sub-tool start/finish events to stderr while stdout
  remains the final structured report. Stdin shell batches preserve those lines
  in the command record's `stderr` field.
- `--allow-partial` defaults to `true`: a report with at least one usable
  section returns `success:true` and `section_run_status:partial`. Set it to
  `false` when a caller requires every selected section to complete cleanly; a
  strict rejection uses `error_code: report_partial_not_allowed` and names the
  incomplete sections.
- `--denoise` and `--denoise-params` configure optional input smoothing for
  candles and the forecast / volatility / barrier stack.
- `--params` supplies template and sub-tool overrides such as context limits,
  backtest settings, barrier grids, or additional timeframes.
- Scalping and intraday `market` sections always obtain Level 1 bid/ask/spread
  from `market_ticker`; broker DOM is optional and reports `depth_status` as
  `available`, `quote_only`, `disabled`, or `unavailable`. The `session`
  section reports whether the symbol is open. The
  `execution_gates` section always returns a gate decision. Configure an
  additional spread cap with `params.spread_max_ticks` or
  `params.spread_max_pips`; without one, the gate checks quote readiness and a
  valid positive spread.
- `--detail` controls canonical response detail; use `--detail full` for richer
  metadata and diagnostics.

Run `mtdata-cli report_generate --help` for the current parameter list and
template descriptions.

## Reading the result

Full reports contain a `sections` mapping plus summary and status information.
Section names depend on the template and may include context, forecast,
backtest, volatility, pivot, confluence, volume profile, patterns, barriers,
session, news, temporal seasonality, regime, or multi-timeframe variants. Check the report-level and per-section status before consuming a
value: a successful report envelope can still describe omitted or partial
sections.

Root `as_of` is the last completed base-timeframe bar-open anchor shared by the
context and forecast sections. `as_of_basis` names that contract, while
`oldest_section_data_as_of` preserves the oldest timestamp used by any selected
section. `generated_at` is the later assembly time. If no base-timeframe anchor
is present, `as_of` falls back to the oldest selected section timestamp and
`as_of_basis` says so. If no section exposes a trustworthy market timestamp,
`as_of` is null and `data_as_of_status` is `unavailable`. When no sections
completed and `as_of` is unavailable, the overall assessment does not claim
the report is temporally coherent; it reports `temporal_coherence=cannot_assess`
and recommends retrying.
When context, forecast, or a multi-timeframe source falls outside its
timeframe-aware session tolerance, `temporal_alignment` reports every checked
cutoff and the mismatched sections. The report is partial and the combined
narrative is omitted rather than mixing data from different cutoffs.
Multi-timeframe context and pivot entries expose `source_bar_time`,
`source_bar_timezone`, and `source_bar_state`. Use
`oldest_section_data_as_of`, rather than root `as_of`, when a workflow needs the
most conservative cross-timeframe timestamp.

Barrier sections preserve negative optimizer decisions. When neither direction
has a mathematically viable candidate, each direction retains its status,
recommendation, candidate counts, and execution blockers, and section health is
`partial` rather than an unqualified `ok`.

`section_run_status` reports whether scheduled sections completed (`complete`,
`partial`, or `failed`). `content_detail` separately reports how much content
was returned (`summary_only`, `selected_sections`, or `full_sections`). Compact
responses are therefore explicitly `content_detail: summary_only` even when all
scheduled sections ran successfully. Context trend windows are calculated from
consecutive source-timeframe candles; unavailable long windows are `null`
rather than silently shortened.

For automation, prefer `--json` and follow the stable envelope rules in
[OUTPUT.md](OUTPUT.md). Do not parse the human-oriented TOON rendering.
