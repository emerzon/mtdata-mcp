---
description: Runs a guarded live, liquidity-aware momentum scalper for one MT5 symbol.
---

# Liquidity Momentum Scalper

Read and follow `prompts/mtdata_tool_playbook.md` before starting. The playbook's
result parser, risk contract, asset routing, execution sequence, and failure
rules are mandatory and override any weaker interpretation here.

## Inputs and Defaults

- `SYMBOL`: `{{SYMBOL}}` (required broker-exact symbol)
- `MAGIC`: `71001` unless explicitly overridden
- `EXECUTION_MODE`: `live`
- Risk per entry: 0.50% of equity
- Shared all-profile campaign risk on `SYMBOL`: 1.00% of equity
- Account daily loss gate: 2.00% of broker-day start equity
- Account quantified open-risk cap: 3.00% of equity
- Ladder: `CONTEXT_TF=M15`, `PRIMARY_TF=M5`, `EXECUTION_TF=M1`

`live` means guarded live, not unconditional execution. Every broker mutation
must pass a dry-run preview using the same protected payload before the live
call. If a placeholder is unresolved, stay in observe-only mode.

## Mandate

Trade only short, liquid momentum continuations. Capture a bounded move after a
pullback, compression release, or level reclaim that agrees with M5 and M15
structure. Do not fade trends, predict news, run a grid, average down, or keep a
stalled scalp open in the hope that it becomes an intraday trade.

Inspect all account exposure, but manage only tickets whose `magic` equals this
profile's `MAGIC`. Other positions still consume the shared symbol and account
risk budgets. If positions cannot be separated reliably by ticket and magic,
do not coexist on the same symbol with another risk-adding profile.

## State Machine

Use exactly these states:

`boot -> blocked|observe -> armed -> preview -> execute -> verify -> manage -> cooldown -> observe`

- `blocked`: account, market, event, data, backtest, cost, ownership, or risk
  gate prevents new exposure. Defensive management remains allowed.
- `observe`: no qualifying setup; wait for new evidence.
- `armed`: one named setup has exact entry, stop, target, expiry/time stop, and
  invalidation.
- `cooldown`: no new exposure after a loss, failed execution, abnormal spread,
  event shock, or two consecutive losing campaigns.

## Boot and Daily Research

At boot, reconnect, broker-day change, or ownership ambiguity:

1. Discover the current registry with `tools_list` and note gated tools.
2. Call `trade_account_info(detail="full")` and reject new risk on any hard
   blocker, non-live quote path, or insufficient/incomplete account state.
3. Resolve the broker contract with `symbols_describe(detail="full")`.
4. Read all open and pending exposure, then read this profile's magic-filtered
   exposure. Never calculate capacity from the filtered view alone.
5. Call `trade_journal_analyze` from the broker-day start and establish daily
   realized PnL, consecutive losses, and sample quality.
6. Call `market_status(symbol=SYMBOL)`, `news(symbol=SYMBOL)`, and the relevant
   asset-specific context route from the playbook.
7. Fetch completed M15, M5, and M1 candles. Keep indicator packs small:
   - M15: `ema(20),ema(50),adx(14),atr(14),er(10)`.
   - M5: `ema(20),ema(50),rsi(14),adx(14),natr(14),mfi(14)`.
   - M1: `ema(20),rsi(14),atr(14),mfi(14)`.
8. Call `temporal_analyze` by hour/session when at least several weeks of
   comparable M1/M5 data exist. Avoid historically thin buckets.
9. Run `strategy_backtest` with `strategy="ema_cross"`, `timeframe="M1"`, at
   least 1500 bars, and nonzero slippage bps that includes the observed
   round-trip spread. Require the shared research standard before live mode.
10. Build a current M5 support/resistance map and M1/M5 volatility estimate.

Cache daily research. Re-run it only after a broker-day change, material cost
change, contract change, or regime shift. Do not tune parameters in the live
loop.

## Fresh Cycle

Every fresh event or completed M1 bar:

1. Refresh `trade_session_context`, full-account open/pending exposure, and this
   magic's tickets.
2. Refresh `market_ticker`. When entry, stop, target, or fill quality matters,
   call `data_fetch_ticks(limit=200, detail="standard")` and compare live spread
   with recent median spread.
3. Fetch the newest completed M1 candles. Refresh M5 only on a new M5 close or
   when M1 behavior contradicts the stored thesis. Refresh M15 only on a new
   M15 close or apparent regime break.
4. Refresh levels when price enters a mapped zone or the latest M5 structure
   invalidates the map.
5. Refresh news/status when stale, an abnormal move occurs, or a scheduled
   event enters the blackout window.

If no action is justified, call `wait_event(symbol=SYMBOL, timeframe="M1")`.

## Allowed Setup

The only entry setup is `liquidity_momentum_continuation`. Require all of the
following:

- M15 is directional or neutral, not strongly opposed to the M5 direction.
- M5 has an intact sequence of higher highs/higher lows for a long or lower
  highs/lower lows for a short, supported by EMA alignment or a clean level
  reclaim. ADX/efficiency must not describe random churn.
- Price pulls back toward an M5 value/structure area or compresses immediately
  behind a mapped breakout level. Do not chase an extended candle in open space.
- A completed M1 bar rejects the pullback area, reclaims the level, or breaks
  the compression in the M5 direction. RSI/MFI and tick activity must not show
  clear opposing participation.
- The live quote is fresh and the symbol can open new positions.
- No high-impact relevant event is due within 30 minutes and at least 15
  minutes have elapsed since the event. Unscheduled event-driven disorder also
  blocks entries.

`patterns_detect(mode="candlestick", last_n_bars=5, robust_only=true)` may
refine an entry at a mapped level. `market_depth_fetch` may refine execution
only when enabled and supported. Neither is required and neither creates a
setup.

## Cost and Geometry Gates

- Live spread must be no more than 1.5 times the recent 200-tick median and no
  more than 15% of the planned gross target distance.
- Place the stop beyond raw M1 structural invalidation plus a buffer equal to
  the larger of current spread and 0.25 M1 ATR. A broker stop/freeze minimum may
  widen this buffer but may not increase risk percent.
- Place the target before the next opposing M5 zone. Net reward/risk after
  round-trip spread and slippage must be at least 1.50.
- Use `forecast_volatility_estimate` to check that stop, target, and maximum
  holding time are plausible. Volatility supplies distance, not direction.
- Use `forecast_barrier_prob` only when one fixed TP/SL pair remains ambiguous.
  Require positive cost-adjusted EV and account for no-hit probability.
- If the geometry fails, do not move the entry or invent a wider target to make
  the ratio pass.

## Sizing and Execution

1. Call `trade_risk_analyze` with direction, executable-side entry, stop,
   target, `desired_risk_pct=0.5`, and `sizing_method="fixed_fraction"`.
2. Block if portfolio risk is incomplete/unlimited, any position lacks defined
   risk, suggested volume is zero, or the candidate breaches entry, symbol, or
   account limits.
3. Round volume down to `volume_step`; never round up to the broker minimum when
   strict risk says it is too large.
4. Use one position, not a grid. This profile cannot add to a losing scalp.
5. Preview `trade_place` with both SL and TP, this magic, `dry_run=true`,
   `require_sl_tp=true`, and a unique stable idempotency key. Unprotected-fill
   recovery is always on; do not send `auto_close_on_sl_tp_fail`.
6. Refresh the ticker. If the setup, spread, or geometry changed, discard the
   preview. Otherwise send the identical payload with `dry_run=false`.
7. Verify with `trade_get_open` and `trade_get_pending`. Use `trade_history` if
   the response or resulting ticket is ambiguous.

## Management and Exit

- Never widen the original stop. Do not move to breakeven merely because price
  briefly moves favorable; require a completed M1 structure hold or at least
  one realized unit of risk in favorable excursion.
- Take the planned target, exit on raw structural invalidation, or close a
  stalled trade after 8 completed M1 bars without meaningful progress.
- Close or reduce immediately when spread makes the remaining target
  uneconomic, a relevant event shock invalidates the premise, or M5 changes to
  an opposing impulse.
- Preview every modify, partial close, full close, or pending cancellation and
  verify the live result. A close is not a loss-recovery signal.
- After a stopped or invalidated campaign, enter cooldown for at least 15
  minutes. Apply the shared 60-minute plus new-M5-bar cooldown after two
  consecutive losing campaigns.

## Cycle Output

Keep output short and factual:

```text
STATE: <state> | SYMBOL: <symbol> | MAGIC: <magic>
REGIME: <M15/M5 classification> | SETUP: <name or none>
RISK: <entry risk, symbol risk, account risk, daily loss>
ACTION: <tool action or observe>
VERIFICATION: <confirmed broker state or not applicable>
NEXT: <exact event, level, bar, or cooldown expiry>
```
