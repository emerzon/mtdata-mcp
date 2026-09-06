# Find a market and read a quote

**Audience:** User

Before you forecast or think about a trade, answer four plain questions:

1. Does my broker list this symbol?
2. Is the market open (or is this quote stale)?
3. What is the current buy/sell price?
4. Which names on my list look active or cheap to trade right now?

**Dense terms:** [Bid / ask / spread](GLOSSARY.md#bidask-and-spread) · [Pip](GLOSSARY.md#pip) · [OHLCV](GLOSSARY.md#ohlcv) · [Support and resistance](GLOSSARY.md#support-and-resistance)

**Related:** [CLI](CLI.md) · [Web UI](WEBUI.md) · [Levels](LEVELS.md) · [Temporal sessions](TEMPORAL.md) · [Trading safety](TRADING_SAFETY.md)

---

## Quick start (read-only)

```bash
# What can MetaTrader 5 see?
mtdata-cli symbols_list --limit 10

# Details for one name (pip size, contract, live quote fields)
mtdata-cli symbols_describe EURUSD --json

# Latest bid / ask / spread
mtdata-cli market_ticker EURUSD --json

# Is a major stock exchange open, or can this FX symbol take new orders?
mtdata-cli market_status --region all --json
mtdata-cli market_status --symbol EURUSD --json
```

In the Web UI, picking a symbol and turning **Live** on is the ticker. **Watch**
opens a persisted watchlist (`market_radar`) so you can scan a few names without
retyping them. **Tools** can run any command in this page.

---

## Completed bars vs the live quote

A **completed bar** is a finished candle (yesterday’s close, last hour’s close).
A **live quote** is the current bid and ask.

Ranking tools keep those apart on purpose:

- Use bar fields (`close`, RSI, SMA) for “how did this market behave?”
- Use `bid`, `ask`, `mid`, and `quote_as_of` for “what would I pay *now*?”

If a live quote is locked, one-sided, or otherwise unsafe, spread-ranked scans
drop it unless you pass `--quote-usable-only false` to inspect it on purpose.

---

## Rank and scan a list

```bash
# Rank the current watchlist by spread, volume, and recent change
mtdata-cli symbols_top_markets --rank-by all --limit 5 --timeframe H1 --json

# Exact global stock top-N; this can take many minutes on a large broker catalog
mtdata-cli symbols_top_markets --rank-by spread --universe all --category stocks --limit 5 --scan-budget-seconds 0 --json

# Scan visible majors: strong RSI and price above its average
mtdata-cli market_scan --group "Forex\\Majors" --rsi-above 60 --price-vs-sma above --sma-period 20 --timeframe H1 --lookback 120 --json

# Compact watchlist (max 20 names; unusable quotes stay visible)
mtdata-cli market_radar --symbols EURUSD,GBPUSD,XAUUSD --timeframe H1 --json
```

`symbols_top_markets` ranks. `market_scan` **filters** (spread caps, RSI
bands, and similar). Price-change ranks compare the previous completed close
with the latest completed close over exactly one requested timeframe bar.
Each compact row includes that bar's `time` and `live_price_change_pct`, which
compares the previous completed close with the current quote midpoint. A
forming-bar leaderboard can use `--rank-by live_price_change_pct` for gainers
or `--rank-by abs_live_price_change_pct` for two-sided movers. Those live modes
exclude quotes that are not usable for live trading by default. A mixed-session
result omits a single `data_as_of` and exposes `data_as_of_range` plus
`bar_time_alignment.comparable=false`.

Large `symbols_top_markets --universe all` requests use a 30-second sampling
budget by default. A budget-limited response is successful but explicitly
partial: check `ranking_complete`, `ranking_scope`, and `candidate_progress`
before acting on it. Pass `--scan-budget-seconds 0` for an exact global result
in one invocation. Every response includes a sequential `sampling_window`;
multi-symbol scans use `atomic=false` and `comparable=false` to disclose that
quotes were not captured at one instant. Completed-bar rankings separately
report whether their bar times align. Temporary hidden-symbol activation is
coordinated across local MTData processes, so it does not leak into concurrent
visible-universe listings.

---

## One-shot pre-trade snapshot

`market_snapshot` packs a quote, nearby levels, and patterns so you do not
have to call five tools:

```bash
mtdata-cli market_snapshot EURUSD --timeframe H1 --json

# Also attach optional regime + forecast sections
mtdata-cli market_snapshot EURUSD --timeframe H1 --sections all --horizon 8 --json
```

Still read-only. It does not place an order.

---

## Session context (when you are closer to trading)

```bash
mtdata-cli trade_session_context EURUSD --json
```

This is the “what is my account and this symbol doing right now?” bundle:
session, quote, and open or pending exposure. It does not send orders.

For “is this equity venue open?” vs “can I open a *new* position on this
broker symbol?", `market_status` reports both. `can_open_new_positions` needs
a live-ready quote and, for non-crypto symbols, is forced false during the
standard FX weekend unless the inferred M1 schedule says the session is
active. Inferred session (`current_time_in_recent_session`) is otherwise
advisory; 24h CFD quotes can still report `true` off cash hours.

The symbol forms of `market_ticker`, `market_status`, and `market_snapshot`
share `quote_as_of` as the UTC tick join key. They also expose root
`data_age_seconds`, `data_stale`, and `usable_for_live_trading` whenever a tick
is available, so a pre-trade workflow can apply one freshness contract across
all three responses. Tool-specific aliases such as ticker `time`, status
`last_tick_time`, and compact snapshot `snapshot.time` remain descriptive
views of the same quote instant.

`market_status` also accepts a comma-separated symbol batch. Mixed results set
`partial_failure: true` and include requested, succeeded, and failed counts plus
`failed_items`. Explicit symbol lists on `market_status`, `market_scan`, and
`market_radar` permit partial results by default (`allow_partial=true`); radar also
warns and lists `missing_symbols`. Pass `--allow-partial false` when a
pre-trade gate must fail if any requested name is omitted. `market_scan` also
tracks history and analysis failures in `failed_symbols`, sets
`ranking_complete=false`, and reports `partial_failure=true`. With
`allow_partial=false`, any such failure fails the scan. If no symbols can be
evaluated, the scan fails regardless of this setting; ordinary filter exclusions
remain successful no-match results. Statistical tools
such as `correlation_matrix` remain fail-closed unless you opt in.

---

## Deeper detail

- Order book (depth / DOM) is off unless you set
  `MTDATA_ENABLE_MARKET_DEPTH_FETCH=1` *and* your broker supplies it.
- `wait_event` can pause until the next candle close or a fill — see
  [WAIT_EVENT.md](WAIT_EVENT.md). Do not run long waits from the Web UI.
- Quote quality and scan limits: [CLI.md](CLI.md#explore-available-symbols).
