# Timestamps and timezones

**Audience:** Operator

**Plain English for everyone else:** set `CLIENT_TZ=UTC` in `.env` so saved
results look the same tomorrow. That is enough for the first week. The rest of
this page explains how MetaTrader 5 clocks are normalized when a broker does
not follow the usual UTC contract.

MetaTrader5 documents UTC request datetimes and UTC Unix epochs. Most terminals
follow that contract, but some broker terminals expose Unix-shaped values on
their server-clock axis. When a broker offset is configured, mtdata verifies
that variant from a fresh live tick and normalizes it at the MT5 boundary. See MetaQuotes' [`copy_rates_from`](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py)
and [`copy_ticks_range`](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksrange_py)
documentation for the upstream UTC contract.

**Related:** [Setup](SETUP.md) · [Env vars](ENV_VARS.md) · [Output contract](OUTPUT.md)

---

## MT5 timestamp contract

The data path is:

```text
UTC request instant ──▶ MT5 adapter ──▶ terminal clock axis ──▶ UTC epoch
                                                                  │
                                                                  ▼
                                                        client display timezone
```

- Pass timezone-aware UTC datetimes to the MT5 Python API. mtdata converts
  parsed request times to that form.
- Native terminals pass UTC request bounds and returned epochs through.
- When a fresh tick is close to the configured broker offset rather than wall
  UTC, the adapter converts request bounds to the server-clock axis and
  returned `time`/`time_msc` values back to UTC exactly once.
- Epoch encoding is a terminal-wide contract. If the requested symbol is stale,
  the adapter probes a bounded set of visible symbols and reuses a confident
  live-symbol mode for closed symbols and unscoped account history.
- During a closed market, a configured positive broker offset is also applied
  when the raw tick is implausibly ahead of wall UTC but offset normalization
  places the last tick within the preceding four days. This keeps weekend
  snapshots deterministic without treating an ordinary Friday close as a
  future quote.
- Callers must not apply another broker offset to normalized payloads.
- `CLIENT_TZ` / `MT5_CLIENT_TZ` controls presentation. If neither is set,
  mtdata uses the local machine timezone when it can detect it, otherwise UTC.
  Explicit but invalid IANA timezone names are rejected at startup.

Every timestamped payload includes a `timezone` field for displayed values.
Internal filtering and range comparisons stay on the UTC epoch axis.

Live quote freshness is anchored to the wall clock after quote acquisition.
Broker ticks less than 10 seconds ahead are retained as live but disclose a
floored `data_age_seconds=0`, `timestamp_ahead_of_wall_clock=true`, and the
measured lead in `timestamp_skew_seconds`. A lead of 10 seconds or more is unsafe and sets
`timestamp_in_future=true`. Quote-reading tools reconcile the cached
`symbol_info_tick` snapshot with the latest tick stream before applying this
single policy.

The trading send path adds one bounded exception for a synchronously acquired
broker tick. When the workstation clock trails that tick by 10–30 seconds,
freshness is evaluated against the broker tick at acquisition and the quote
reports `clock_reconciled=true`, `local_clock_lag_seconds`, and
`data_age_anchor=broker_tick_reconciled_clock`. This avoids blocking an order
solely for modest workstation clock lag. Leads above 30 seconds still fail
closed as future timestamps; stale ticks and invalid quotes are never repaired
by this reconciliation.

---

## Broker session configuration

Broker wall-clock configuration is optional and is used only where a market
session, trading day, or calendar boundary needs broker context.

| Variable | Default | Purpose |
|----------|---------|---------|
| `MT5_SERVER_TZ` | — | Broker IANA timezone, such as `Europe/Athens`; required to recognize and normalize server-clock epochs with DST-aware offsets, and used for session/calendar calculations. |
| `MT5_TIME_OFFSET_MINUTES` | `0` | Fixed broker offset from UTC. A non-zero value overrides `MT5_SERVER_TZ`, including server-clock recognition and conversion. |
| `CLIENT_TZ` / `MT5_CLIENT_TZ` | auto-detect | Display timezone; `CLIENT_TZ` wins if both are set. |
| `MTDATA_BROKER_TIME_CHECK` | `false` | Optionally perform additional live tick/bar freshness verification. |

For deterministic stored output, pin the display timezone:

```ini
CLIENT_TZ=UTC
```

Add `MT5_SERVER_TZ` when broker-local session boundaries matter or when a
terminal exposes broker server-clock epochs. Without a broker offset, mtdata
uses the upstream native-UTC contract rather than guessing an offset from a
possibly stale tick:

```ini
MT5_SERVER_TZ=Europe/Athens
```

The adapter also checks fresh live ticks before caching that UTC assumption. If
a quote tracks the current minute and seconds but is in the future by an
approximately whole-hour broker offset, mtdata stops before querying history.
The error identifies the symbol and observed offset and asks you to load `.env`,
correct `MT5_SERVER_TZ` (or `MT5_TIME_OFFSET_MINUTES`), and restart the process.
This catches both a missing setting and a configured timezone whose current
offset disagrees with the live terminal clock.

---

## Time metadata

Compact candle responses retain a thin public time contract. `time_basis=utc`
describes normalized instant provenance, while `timestamp_format`,
`timestamp_mode`, `public_timestamp_mode`, and `timestamp_timezone` describe
the serialized row values. Candle and tick rows default to `timestamp_format=iso_utc`
(UTC `Z` strings) so output does not follow ambient `CLIENT_TZ`. Pass
`timestamp_format=iso` for client-local offset strings. UTC strings use
`iso_utc` / `utc`; client-local strings with an explicit offset use
`iso_offset` / `client_timezone`; numeric values use `epoch_seconds` / `utc`. Latest-N queries expose `limit_satisfied`; historical
ranges expose `range_complete`, `limit_reached`, and a `query_applied` block
that states whether the limit was anchored at the start or end. An omitted
range limit returns a 20-bar page and is reported as `default_limit`, not as a
user-requested count. Use `pagination.next_cursor` to continue the range.
Latest-N queries also default to 20 rows. A timestamp start keeps bars whose
open is at or after start. Timestamp ends use `end_filter=bar_close` and keep
bars whose close is at or before end — fully contained bars only.
Request full detail
to inspect the full normalization contract:

```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 5 --detail full --json
```

With no configured broker offset, full payloads report
`raw_time_basis=mt5_utc_epoch`, `raw_timestamp_mode=native_utc`, and
`time_normalization=mt5_utc_native`. A detected server-clock terminal instead
reports `raw_time_basis=mt5_server_clock_epoch`,
`raw_timestamp_mode=server_clock`, and
`time_normalization=server_clock_to_utc`. Compact server-clock payloads retain
`time_normalization=server_clock_to_utc` without exposing the raw mode as the
public timestamp axis. Public ISO values use the configured display timezone
and always include `Z` or an explicit numeric offset; epoch values are UTC Unix
seconds. Full detail keeps the source clock as `raw_timestamp_mode`.
Forecast generation and conformal-interval payloads normalize all displayed
datetimes, including nested diagnostics and training windows, to UTC.
Trade-history payloads expose the same `raw_time_basis`, `time_basis`,
`raw_timestamp_mode`, and `time_normalization` fields, including when no symbol
filter was supplied.

A server-clock epoch does not contain the daylight-saving `fold` bit. If such a
terminal returns a repeated fall-back wall time or a nonexistent spring-forward
wall time, mtdata fails the normalization instead of guessing or shifting the
event. Native-UTC terminal timestamps and explicitly fixed offsets are not
ambiguous in this way.

MT5 stamps candles at bar open. Daily, weekly, and monthly candle rows also
include `broker_session_date`; D1 rows include `broker_trading_day`. These
labels use the configured broker timezone and disambiguate sessions whose UTC
open falls on the preceding calendar date.

Latest-N requests exclude a forming candle by default. If that forming candle
starts after a broker session break, the response still retains the observed
discontinuity in `session_gaps` and `gap_after_last_bar`. In that case,
`bar_spacing.status=session_gaps_detected`; `spacing_matches_timeframe` may
remain true because it describes the dominant interval, while
`spacing_complete=false` describes the missing session interval.

For completed-bar analytics, `data_as_of` is the latest completed-bar close.
MT5 row timestamps stay bar-open and are exposed as `last_bar_open` (forecast
and barrier payloads) or the candle `time` column. Freshness ages are measured
from that close; forecast and volatility payloads identify this as
`latest_completed_bar_close_age_seconds`. Point-in-time `as_of` cutoffs must
not be in the future; an unfulfillable future cutoff is rejected instead of
falling back to current data.

---

## External providers

External sources do not use MT5 server time and are normalized separately:

- Finviz publish times and calendars use their provider/US-market context.
- News relative filters use the client timezone; results carry publication times.
- Options expirations and quotes follow the selected provider's convention.

Compare sources using UTC absolute instants, and retain the `timezone` or source
metadata alongside saved results.

---

## Troubleshooting

If candles appear shifted:

1. Inspect the payload's `timezone`; presentation may be client-local.
2. Set `CLIENT_TZ=UTC` and rerun the same absolute range.
3. Confirm the input included an explicit offset or `Z` when it was intended as
   an absolute instant.
4. Request `--detail full` and inspect `timestamp_mode`,
   `raw_time_basis`, and `time_normalization`.
5. Configure `MT5_SERVER_TZ` (preferred) or `MT5_TIME_OFFSET_MINUTES` to match
   the broker before relying on a terminal that exposes server-clock epochs.
6. Enable `MTDATA_BROKER_TIME_CHECK=1` for additional live freshness checks.

Do not manually shift public payload epochs. The configured broker offset is
applied inside the adapter only after server-clock mode is detected; applying it
again double-shifts the data.

---

## See also

- [ENV_VARS.md § Timezone](ENV_VARS.md#timezone)
- [OUTPUT.md](OUTPUT.md)
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

Forecast `last_observation_epoch` and `last_observation_time` both identify the input bar's close. Full output exposes its earlier open separately as `last_bar_open_epoch` and `last_bar_open`. Consumers using the former observation epoch as an open-time key must switch to the explicit bar-open field.

