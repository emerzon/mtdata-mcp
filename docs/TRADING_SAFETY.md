# Trading safety runbook

**Audience:** User

If you only skim one trading doc, make it this one. The `trade_*` tools send **real requests** to the MT5 account currently logged into the terminal. This runbook covers previewing orders, validation, account guardrails, and broker quirks for `trade_place`, `trade_modify`, and `trade_close`.

> **These tools default to preview mode.** `dry_run` defaults to **`true`**. A
> request reaches MT5 only when you explicitly pass `--dry-run false` in the
> CLI or `dry_run=false` through Python/MCP. The Web API Tools path
> additionally requires `"confirm": true` only when `arguments` set
> `"dry_run": false` (an omitted `dry_run` is still a preview and does not
> need confirm). Use a **demo account** until you trust your setup — mtdata
> has no separate paper-trading mode. `trade_idea_compose` is stricter: it is
> preview-only and has no live-send flag.

**Dense terms:** [Dry-run](GLOSSARY.md#dry-run) · [Trade guardrails](GLOSSARY.md#trade-guardrails) · [Slippage](GLOSSARY.md#slippage) · [Lot size](GLOSSARY.md#lot-size) · [TP/SL](GLOSSARY.md#tpsl-take-profit--stop-loss)

**Related:** [Trade ideas](TRADE_IDEAS.md) · [Risk analytics](TRADING_RISK.md) · [Env vars (guardrails)](ENV_VARS.md#trade-guardrails) · [CLI](CLI.md) · [Sample trade](SAMPLE-TRADE.md) · [Glossary](GLOSSARY.md)

---

## Golden rules

1. **Preview first** — `--dry-run true` until the request looks right.
2. **Demo while learning** — no simulated mode except an MT5 demo.
3. **Enable guardrails** on any account that can place orders ([Account guardrails](#account-guardrails)).
4. **Exact tickets** for modify/close; treat `--close-all` as nuclear.
5. **Protective levels** — market and pending orders require SL **and** TP by default (`--require-sl-tp`). If a filled market order cannot attach those levels, mtdata always tries to close the unprotected position; that fail-safe is not a CLI flag.
6. **Confirm the account** — compact `trade_account_info` does not include the raw login. Use `account_context_id` (a hash of login + server) to tell two logins on the same broker apart. Full detail still includes `login` when you need the exact number. Compact `trade_risk_analyze` follows the same rule.

MT5 tickets and magic numbers are unsigned 64-bit identifiers. Ticket inputs
accept `1..18446744073709551615`; magic accepts
`0..18446744073709551615`, where zero is a real manual/untagged strategy scope,
not an omitted filter. Decimal input is parsed without floating-point
conversion. When an identifier exceeds JavaScript's exact integer range, JSON
responses also include a sibling such as `ticket_exact` or `magic_exact` as a
canonical decimal string and set
`identifier_encoding=decimal_string_in_exact_fields`.

---

## Preview with `--dry-run`

A dry run routes and validates the request **without sending it to MT5**.
Successful `trade_place`, `trade_modify`, and `trade_close` previews expose
`dry_run=true`, `preview_ok=true`, and `would_send_order=false`. Modify and
close previews also retain `actionability=preview_only` and a
`preview_scope_summary` that states no request was sent, including in compact
TOON output. The `trade_place` preview returns these additional markers you can
assert on:

```jsonc
{
  "dry_run": true,
  "no_action": true,
  "would_send_order": false,
  "dry_run_simulated": true,
  "preview_ok": true,
  "validation_passed": true,
  "validation_scope": "local_preview_plus_estimates",
  "preview_checks_performed": [ /* checks actually completed */ ],
  "checks_not_performed": [ /* for example, margin_estimate when unavailable */ ],
  "broker_validation_not_performed": [ /* broker acceptance/enforcement, margin reservation, fillability, SL/TP attachment */ ],
  "guardrails_preview": { /* which guardrails would apply */ }
}
```

Eligible previews return `success=true` and `preview_ok=true`. A preview that is
not eligible for live submission because of local safety failures returns
`success=false`, `error_code=preview_blocked`, and `preview_ok=false` while
retaining the preview body and its actionable `blockers`. This includes missing
required SL/TP, closed-market or stale-quote checks, and other local safety
failures. Ticketless bulk `trade_close` dry runs are different: when selection
and local validation succeed they return `success=true` and `preview_ok=true`
with `authorization_status=required`, `required_confirmation`, and
`validation.live_submission_eligible=false`. Confirmation is enforced only for
`--dry-run false`. The CLI prints blocked safety previews and exits `1`; an
eligible preview, including an unconfirmed bulk dry run, exits `0`.
Compact output always retains these gate fields and the broker-validation
limitations. Its `guardrails_preview` summary retains `enabled`, `blocked`,
`ignored_for_demo`, `would_block_live`, `live_projection`, and
`checks_not_performed`; standard/full detail includes the complete guardrail
diagnostics. When demo enforcement is disabled, `blocked` remains the decision
for the demo account while `live_projection.blocked` and `would_block_live`
show what the same request would do on a live account. A projected live block
makes the dry-run preview ineligible (`preview_ok=false`).

Trade previews and live submissions report `requested_comment` and
`applied_comment` for the outbound order tag. Live broker status text from
`MqlTradeResult.comment` is `broker_message`; `comment` keeps the applied tag
in both dry-run and live responses. MT5-bound comments use a conservative
ASCII subset and are capped at 31 characters for place and cancel requests, or
24 characters for position-close requests. `trade_modify` cannot retag an
existing ticket. When a supplied comment changes, the preview includes
structured sanitization or truncation metadata and a warning. For
`target=all_exposure`, `comment_previews` shows the separate close and
pending-cancel forms because their limits differ.

**What a dry run *does* check:** required fields, order-type validity,
market-vs-pending routing, an indicative margin estimate when MT5 exposes one,
and a guardrails preview. For pending orders, `margin_required_when_filled` is
calculated with the corresponding active BUY/SELL action at the requested entry
price; `margin_estimate_basis` records that assumption.

**What a dry run *cannot* check** (only a live send confirms these): final broker
acceptance, live price-distance/stops rules, the final margin reservation and
funds decision, fillability, and SL/TP attachment after a market fill. Treat a
clean preview as necessary, not sufficient.

---

## `trade_place`

Requires `symbol`, `volume`, and `order_type`.

| Flag | Default | Notes |
|------|---------|-------|
| `symbol` | — | Broker symbol |
| `--volume` | — | Lots (validated against broker min/max/step) |
| `--order-type` | — | See [Order types](#order-types) |
| `--price` | — | Entry for pending orders; **omit for market orders** |
| `--stop-loss` | — | Stop-loss price |
| `--take-profit` | — | Take-profit price |
| `--deviation` | `20` | Max slippage in points (market orders) |
| `--require-sl-tp` | `true` | Require both SL and TP on market and pending orders |
| `--expiration` | — | Future pending-order expiry (`dateparser` or positive UTC epoch seconds); literal `GTC` means no expiry |
| `--magic` | `MTDATA_ORDER_MAGIC` | Strategy identifier stamped on the order |
| `--comment` | — | Free-text order comment |
| `--idempotency-key` | — | Durable dedupe shared across processes/restarts (24-hour default retention) |
| `--dry-run` | `true` | Set `false` explicitly for live execution |
| `--detail` | `compact` | Use `full` for execution diagnostics |

If required SL/TP protection cannot be attached after a market fill, mtdata
always attempts to close the unprotected position. This fail-safe is not optional.

Idempotency outcomes are stored in `MTDATA_TRADE_IDEMPOTENCY_DB`. If a process
stops after reserving a key but before recording the broker outcome, retries for
that key fail closed. Reconcile the order or modification in MT5 before removing
the unresolved database row; never clear it merely to make a retry proceed.

```bash
# Preview a market buy with protective levels
mtdata-cli trade_place EURUSD --volume 0.10 --order-type BUY --stop-loss 1.0850 --take-profit 1.0950 --dry-run true

# Go live (only on the intended account)
mtdata-cli trade_place EURUSD --volume 0.10 --order-type BUY --stop-loss 1.0850 --take-profit 1.0950 --dry-run false
```

For a synchronously acquired send-path tick, mtdata tolerates a broker clock up
to 30 seconds ahead of the workstation clock by evaluating freshness on that
bounded broker reference. The preview discloses `clock_reconciled` and
`local_clock_lag_seconds`. Larger leads, stale ticks, and invalid two-sided
quotes remain hard blockers.

### Order types

`order_type` accepts these **canonical strings** (case-insensitive; `-` or space becomes `_`):

`BUY`, `SELL`, `BUY_LIMIT`, `BUY_STOP`, `BUY_STOP_LIMIT`, `SELL_LIMIT`,
`SELL_STOP`, `SELL_STOP_LIMIT`

MT5 numeric constants and `ORDER_TYPE_*` names are **rejected** as input — they
only appear when *reading* existing orders/positions. Market orders use
`BUY`/`SELL` (no `--price`). Every pending order requires `--price`. For a
stop-limit order, `--price` is the stop trigger and `--stop-limit-price` is the
limit leg activated after the trigger. A buy stop-limit's limit price must be at
or below its trigger; a sell stop-limit's limit price must be at or above it.

```bash
# Trigger above the ask, then activate a buy limit at or below that trigger
mtdata-cli trade_place EURUSD --volume 0.10 --order-type BUY_STOP_LIMIT --price 1.1050 --stop-limit-price 1.1045 --stop-loss 1.1000 --take-profit 1.1150 --dry-run true
```

---

## `trade_modify`

Modifies an existing order/position by ticket.

At least one of `price`, `stop_limit_price`, `stop_loss`, `take_profit`,
`clear_stop_loss`, `clear_take_profit`, or `expiration` must be supplied. An
explicit value that already matches the live object is a successful
idempotent no-change request; omitting every modification field is an error.
MT5 does not support changing an existing ticket's comment; set the comment
at place or close time. Passing `0` for `--stop-loss` or `--take-profit`
is rejected; use `--clear-stop-loss true` or `--clear-take-profit true` to
remove protection.

| Flag | Default | Notes |
|------|---------|-------|
| `ticket` | — | **Required** |
| `--price` | — | New pending-order price |
| `--stop-limit-price` | — | New limit leg for an existing stop-limit order |
| `--stop-loss` | — | New stop-loss price. Zero is rejected |
| `--take-profit` | — | New take-profit price. Zero is rejected |
| `--clear-stop-loss` | `false` | Explicitly remove the stop-loss |
| `--clear-take-profit` | `false` | Explicitly remove the take-profit |
| `--expiration` | — | New future pending-order expiry, or literal `GTC` |
| `--idempotency-key` | — | Durable dedupe shared across processes/restarts |
| `--dry-run` | `true` | Preview by default; set `false` explicitly for a live modification |

```bash
mtdata-cli trade_get_open --json
mtdata-cli trade_modify --ticket 123456789 --stop-loss 1.0860 --take-profit 1.0980 --dry-run true
```

Guardrails apply to `trade_modify` only for pending-order changes and SL changes that **increase** risk; risk-reducing changes stay allowed. Risk-increasing modifications still apply symbol allowlist, blocklist, and volume-map rules. Pending orders with no stop-loss still apply the kill switch and symbol rules on price or expiration changes.

---

## `trade_close`

`trade_close` acts on one explicit object class. It closes positions by default,
cancels pending orders only with `--target pending`, and flattens both classes
with `--target all_exposure`. There is no automatic position-to-order fallback.

| Flag | Default | Notes |
|------|---------|-------|
| `--ticket` | — | Act on one ticket in the selected target class |
| `--target` | `positions` | `positions`, `pending`, or `all_exposure` (bulk scopes only) |
| `--volume` | — | Partial-close size (validated against broker step) |
| `--symbol` | — | Restrict closes to a symbol |
| `--side` | — | Restrict open positions to `BUY`/`LONG` or `SELL`/`SHORT`; pending orders are unaffected |
| `--magic` | — | Restrict closes to a magic number |
| `--close-all` | `false` | Select the whole account when ticket, symbol, side, and magic are omitted |
| `--confirm-close-all` | `false` | **Required** for any ticketless live bulk operation |
| `--pnl-filter` | `all` | Close all matches, only winners (`profit`), or only losers (`loss`) |
| `--close-priority` | — | `loss_first`, `profit_first`, or `largest_first`; largest uses broker tick economics to compare approximate account-currency exposure, with lot size only as a fallback |
| `--deviation` | `20` | Max slippage in points |
| `--dry-run` | `true` | Preview by default; set `false` explicitly for a live close |

```bash
# Preview a partial close of one ticket
mtdata-cli trade_close --ticket 123456789 --volume 0.05 --dry-run true

# Cancel one pending order; default target=positions would not cancel it
mtdata-cli trade_close --ticket 987654321 --target pending --dry-run false

# Close all positions account-wide
mtdata-cli trade_close --close-all --confirm-close-all true --dry-run false

# Close positions and cancel pending orders for one strategy
mtdata-cli trade_close --magic 3001 --target all_exposure --confirm-close-all true --dry-run false

# Preview closing only long EURUSD positions in a hedged book
mtdata-cli trade_close --symbol EURUSD --side BUY --dry-run true
```

For `all_exposure`, the response keeps `closed_positions` and
`cancelled_pending_orders` as separate result legs and reports partial failures;
one failed leg does not prevent the other from being attempted. There is no
separate "confirm" token for `trade_place`/`trade_modify`; the extra
`--confirm-close-all` gate applies to every ticketless live bulk close.
Dry-run bulk previews can enumerate matching exposure without the flag. They
remain successful previews (`success=true`, `preview_ok=true`) and report
`authorization_status=required`, `required_confirmation="--confirm-close-all true"`,
and `validation.live_submission_eligible=false`. Add the confirmation only when
you intend `--dry-run false`, or include it on a preview if you want to verify
that the same request is locally eligible to go live.

---

## Account guardrails

Guardrails are optional pre-trade controls that reject risky orders **before** they reach MT5. They are evaluated when `MTDATA_TRADE_GUARDRAILS_ENABLED=1` **or** whenever any individual guardrail variable is set. They apply on demo accounts by default; set `MTDATA_TRADE_GUARDRAILS_IGNORE_ON_DEMO=true` only when you intentionally want demo to skip those caps.

Guardrails span several layers:

| Layer | Rejects when… | Key variables |
|-------|---------------|---------------|
| Kill switch | Trading is disabled | `MTDATA_TRADING_ENABLED=0` |
| Symbol rules | Symbol is blocked or not allowlisted | `MTDATA_TRADE_ALLOWED_SYMBOLS`, `MTDATA_TRADE_BLOCKED_SYMBOLS` |
| Volume caps | Order volume exceeds a global or per-symbol cap | `MTDATA_TRADE_MAX_VOLUME`, `MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL` |
| Safety policy | Missing SL, excessive deviation, or non-reducing order | `MTDATA_TRADE_SAFETY_REQUIRE_STOP_LOSS`, `MTDATA_TRADE_SAFETY_MAX_DEVIATION`, `MTDATA_TRADE_SAFETY_REDUCE_ONLY` |
| Account risk | Margin too low, floating loss or exposure too high | `MTDATA_TRADE_MIN_MARGIN_LEVEL_PCT`, `MTDATA_TRADE_MAX_FLOATING_LOSS`, `MTDATA_TRADE_MAX_TOTAL_EXPOSURE_LOTS` |
| Wallet risk | Post-trade risk exceeds a % of equity/balance/free margin | `MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY`, `MTDATA_TRADE_MAX_RISK_PCT_OF_BALANCE`, `MTDATA_TRADE_MAX_RISK_PCT_OF_FREE_MARGIN` |

> **Note:** A per-symbol volume map (`MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL`) also acts as an allowlist — a symbol missing from the map is rejected. Exposure and wallet-risk caps include both open positions and pending orders. Wallet-risk caps fail closed when any position or pending order lacks a quantifiable stop-loss or valid broker tick metadata.

Reduce-only checks the current open positions before allowing an opposite-side
order no larger than the net position. On hedging accounts, `trade_place` cannot
guarantee a reduction, so use `trade_close` with a position ticket instead.

See [ENV_VARS.md § Trade Guardrails](ENV_VARS.md#trade-guardrails) for every variable, defaults, formats, and a ready-to-copy `.env` block. A dry run returns a `guardrails_preview` so you can confirm which rules would fire before going live. On a demo account where enforcement is ignored, inspect `would_block_live` and `live_projection`; these evaluate the configured rules without changing demo enforcement.

Live market and pending placements are serialized within one mtdata process so
the portfolio snapshot, guardrail decision, and broker submission are atomic
against concurrent tool calls. Separate mtdata processes connected to the same
account do not share that lock; use a single live-trade executor per MT5 account
when exposure or wallet-risk caps must be enforced across clients.

---

## Pre-trade validation & broker behavior

Even with guardrails off, mtdata validates each order against broker constraints before submission:

- **Volume** — must be numeric, positive, finite, within the symbol's `volume_min`/`volume_max`, and aligned to `volume_step` (misaligned sizes are rejected with an aligned suggestion).
- **Pending price side** — `buy_limit` must sit below ask, `buy_stop` above ask, `sell_limit` above bid, `sell_stop` below bid.
- **Stops distance** — SL/TP and pending prices must respect the broker's minimum stops/freeze level.
- **Symbol readiness** — the symbol must be selectable and have live bid/ask.
- **Filling mode** — mtdata resolves a broker-compatible filling mode for market fills and closes.
- **Margin** — a market-order preview estimates required margin.

Because these depend on **live** broker state, they are only fully enforced on a real send — another reason to keep position sizes small when first going live.

---

## Live-trade checklist

1. Confirm the account: `mtdata-cli trade_account_info --json` (verify it's the intended demo/live account).
2. Snapshot context: `mtdata-cli trade_session_context EURUSD --json`.
3. Configure guardrails in `.env` (allowlist, volume caps, risk %). Restart mtdata.
4. Preview: run the order with `--dry-run true`; inspect `guardrails_preview`, `preview_checks_performed`, `checks_not_performed`, and `broker_validation_not_performed`. A margin estimate appears under performed checks only when MT5 returned a finite estimate.
5. Go live with a **small** size and `--dry-run false`.
6. Verify: `mtdata-cli trade_get_open --json`, then manage with `trade_modify` / `trade_close`.

---

## Account and journal (read-only)

Look at the account **without** placing an order. None of these send
`trade_place` / `trade_modify` / `trade_close`.

| Question | Tool |
|----------|------|
| Which account is logged in? Balance, equity, margin? | `trade_account_info` |
| What is open right now? | `trade_get_open` |
| What is waiting as a pending order? | `trade_get_pending` |
| What filled recently? | `trade_history` |
| How did closed trades perform? | `trade_journal_analyze` |
| Session + quote + exposure in one bundle | `trade_session_context` |

`trade_account_info` compact output keeps the two distinct gates without their
derivable aliases.
`execution_ready` is terminal/account enablement only
(`execution_ready_scope=account_and_terminal_enablement`); it does **not**
include margin policy. The canonical `new_exposure_allowed` combines broker
permission, non-critical margin, and strict execution readiness. Use it before
adding risk. Full detail retains the component aliases and basis diagnostics.
`trade_session_context` still adds
symbol/session checks on top of that account gate. Its
`execution_preconditions_allow_open` flag is that scoped execution gate; it
does not imply portfolio-risk approval (`trade_ready.portfolio_risk_assessed`
stays false until a risk tool is run). Compact output reports duplicated gates
inside `trade_ready` only; full detail retains their root aliases and account
type booleans. Compact execution-quality and journal results keep metrics,
coverage gaps, low-sample warnings, and other exceptions, but omit static unit
legends and repeated sample prose.

```bash
mtdata-cli trade_account_info --json
mtdata-cli trade_get_open --json
mtdata-cli trade_get_pending --json
mtdata-cli trade_history --history-kind deals --minutes-back 10080 --json
mtdata-cli trade_journal_analyze --minutes-back 10080 --json
mtdata-cli trade_journal_analyze --magic 3001 --minutes-back 10080 --json
mtdata-cli trade_session_context EURUSD --json
```

`trade_history` and `trade_journal_analyze` default to the last 7 days when you
omit a window.
`trade_history` returns 20 rows by default (max 500 per page). If more rows exist, pass its opaque
`pagination.next_cursor` back as `--cursor` with the same filters and time
controls. Cursor pages retain the first page's exact UTC bounds and expire
after one hour, preventing a moving relative window from skipping records.
On accounts shared by multiple strategies, pass `--magic` to either command so
history pagination and journal metrics are scoped to one MT5 strategy identifier.
For deal history, mtdata attributes every position leg to the earliest opening
deal available in the requested window. A manual exit with deal magic `0` therefore
stays with the strategy that opened the position. Each row keeps the broker's raw
`deal_magic`, the resolved `attributed_magic`, and `attribution_method`. If the
opening deal falls outside the window, attribution falls back to that row's own
deal magic; expand the window when you need complete strategy reconciliation.

**History vs journal:** history is the raw deal/order tape. The journal
summarizes *exit* deals (wins, losses, averages) for review. It matches entry
fills by position ticket and allocates their commission and fees by closed
volume. Check `entry_cost_coverage`: an entry outside the requested history
window leaves that exit on the explicitly reported exit-deal-only PnL basis.
History deal rows preserve MT5's `profit`, `commission`, `swap`, and `fee`
components and also report `net_pnl` as their sum. `profit_basis` makes clear
that the broker `profit` value excludes those separately reported cost fields.

**Do not paste journal averages into Kelly sizing.**
`trade_journal_analyze` reports profit and loss in account currency per exit,
including matched entry costs where `entry_cost_coverage` permits.
[Kelly](GLOSSARY.md#kelly-criterion) needs a win rate and average win/loss that
are normalized to a consistent stake (for example R-multiples). Build those
inputs on purpose; see [TRADING_RISK.md](TRADING_RISK.md).

**Dense terms:** [Balance / equity / free margin](GLOSSARY.md#balance-equity-and-free-margin) · [Margin](GLOSSARY.md#margin-and-leverage) · [Magic number](GLOSSARY.md#magic-number)

---

## See Also

- [CLI.md § Trading](CLI.md#trading) — Command list and execution controls
- [ENV_VARS.md § Trade Guardrails](ENV_VARS.md#trade-guardrails) — Full guardrail variable reference
- [TRADING_RISK.md](TRADING_RISK.md) — Position sizing, VaR/CVaR, and stress tests
- [SAMPLE-TRADE-ADVANCED.md](SAMPLE-TRADE-ADVANCED.md) — An end-to-end analysis-to-execution workflow
- [Account terms](GLOSSARY.md#balance-equity-and-free-margin)
- [OUTPUT.md](OUTPUT.md) — Response envelope and error codes

Quote reconciliation can supply a current analysis price while the raw submission tick remains unsafe. `usable_for_live_trading` also requires the raw tick to pass the order submission freshness policy; `send_path_tick_fresh=false` explains that veto. The broker clock tolerance and order submission checks are unchanged.
