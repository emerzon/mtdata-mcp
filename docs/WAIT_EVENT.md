# Wait for a candle close or a fill

**Audience:** User

Sometimes a script or assistant should **pause** until something happens —
the current hour closes, or an order fills — instead of polling forever by
hand. `wait_event` does that pause for you.

It **blocks** until it finishes or times out. That is useful in the CLI or a
Model Context Protocol (MCP) session. It is a poor fit for the Web UI Tools
runner, which has no progress bar for a long wait.

**Dense terms:** [MCP](GLOSSARY.md#mcp-model-context-protocol) · [Timeframe](GLOSSARY.md#timeframe-and-candles-bars) · [Pending vs market order](GLOSSARY.md#market-vs-pending-orders)

**Related:** [CLI](CLI.md) · [Market discovery](MARKET.md) · [Trading safety](TRADING_SAFETY.md) · [Web UI](WEBUI.md)

---

## Pick a wait target

Choose a **timeframe** as the wait horizon. The call stops at its next candle
boundary (for example the next H1 close), or earlier when an explicit watcher
matches. The engine keeps the wait budget and observation cadence internal.

Boundary-only waits do not poll: they sleep directly to the calculated boundary.
When explicit market or account watchers are present, the engine polls only as
needed to observe those events.

---

## Example 1 — wait for the next H1 close

```powershell
mtdata-cli wait_event EURUSD --timeframe H1 --watch-for '[]' --json
```

The explicit empty watch list makes this a candle-boundary-only wait, so the
next research step runs on a *completed* hour rather than returning early for
a market or account event.

A timeframe wait with no symbol and no extra watch list is a pure clock wait
(no candle payload), including on weekends. It returns `clock_boundary` and
does not infer a market session. Passing the symbol includes a best-effort closed-candle
snapshot when the boundary hits. Omitting `--watch-for` waits only for that
candle boundary. The internal one-timeframe budget prevents a symbol-bound weekend H1 wait
from blocking until Sunday reopen.

---

## Example 2 — wait for a fill within M5

```powershell
mtdata-cli wait_event EURUSD --timeframe M5 --watch-for '[{"type":"order_filled","symbol":"EURUSD"}]' --json
```

If nothing fills before the M5 boundary, the command fails (`success=false`,
`error_code=wait_event_boundary_reached`) and the CLI exits nonzero. That is
intentional: a script can decide whether to retry. If the internal budget
expires before a reachable boundary, the failure is `wait_event_timeout`.
When the symbol's market is closed — for example the FX weekend — that timeout
also reports `market_status=closed` and `assumed_closure_end`, and the
remediation points at reopen instead of asking you to wait longer.

Pass an explicit `--watch-for` when the wait should return early for an event,
as in the fill example above. Omitting `--watch-for` waits
only for the candle boundary.

---

## Web UI warning

`wait_event` is omitted from the Web UI Tools runner. Use `mtdata-cli` or an
MCP assistant instead. See [WEBUI.md](WEBUI.md#do-not-run-these-from-the-browser).

---

## Watcher contract

`--watch-for` accepts event names or JSON objects with a `type`. `--end-on`
is only for candle-close boundaries.

Account events (optional `symbol`, `order_ticket`/`position_ticket`, `magic`,
`side=buy|sell`):

| type | Extra required fields | Example |
|------|-----------------------|---------|
| `order_created`, `order_filled`, `order_cancelled` | none | `{"type":"order_filled","symbol":"EURUSD"}` |
| `position_opened`, `position_closed`, `tp_hit`, `sl_hit` | none | `{"type":"tp_hit","symbol":"EURUSD"}` |
| `pending_near_fill`, `stop_threat` | `distance` in price units | `{"type":"pending_near_fill","symbol":"EURUSD","distance":0.0005}` |

Market events (`window.kind` is `minutes` or `ticks`; `window.value` is that
unit; `threshold_mode` is `ratio_to_baseline` or `zscore` unless noted):

| type | Extra required fields | Example |
|------|-----------------------|---------|
| `price_change` | `threshold_value`; `threshold_mode` may be `fixed_pct` | `{"type":"price_change","direction":"up","threshold_mode":"fixed_pct","threshold_value":0.1}` |
| `volume_spike`, `tick_count_spike`, `spread_spike`, `range_expansion` | `threshold_value` | `{"type":"volume_spike","window":{"kind":"minutes","value":5},"threshold_value":2}` |
| `tick_count_drought` | `threshold_value` (default 0.5) | `{"type":"tick_count_drought","threshold_value":0.5}` |
| `price_touch_level` | `level` in price units | `{"type":"price_touch_level","symbol":"EURUSD","level":1.0850,"tolerance":0.0002}` |
| `price_break_level` | `level`; optional `confirm_ticks` | `{"type":"price_break_level","symbol":"EURUSD","level":1.0850,"direction":"up","confirm_ticks":2}` |
| `price_enter_zone` | `lower` and `upper` in price units | `{"type":"price_enter_zone","symbol":"EURUSD","lower":1.0800,"upper":1.0850}` |

Boundary:

```powershell
mtdata-cli wait_event EURUSD --timeframe H1 --end-on '[{"type":"candle_close","timeframe":"H1"}]' --json
```

`price_source` on price watchers is `auto`, `bid`, `ask`, `mid`, or `last`.
`direction` is `up`, `down`, or `either`.

---

## Deeper detail

- Basket: pass `--symbols` (up to 12) *or* one `--symbol`, not both. The wait
  returns on the first match.
- `accept_preexisting=true` returns immediately if a state-style watcher is
  already true at startup. The default waits for a *new* transition.
- `position_opened` is a new still-open position after the wait starts. A
  historical entry deal for a ticket that is already closed, or that was
  already open when the wait began, is ignored.
- Put candle-close boundaries in `end_on`, not in `watch_for`.
- `--detail full` adds elapsed timing diagnostics and poll counts without
  exposing internal timing controls.
- `wait_event --help` repeats this contract, including complete JSON examples.
