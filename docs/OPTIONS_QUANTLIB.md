# Options and QuantLib

**Audience:** User

Look up **US stock option chains** (lists of puts and calls) and, separately, **price a barrier-style option on your machine**.

A chain is “what strikes and expiries exist.” Local pricing is a calculator: you type spot, strike, barrier, and volatility. You do not need MetaTrader 5 for the calculator. For *FX take-profit / stop-loss odds* on MetaTrader 5 paths, use [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) instead — related idea, different tool.

For *MT5 path* TP/SL hit probabilities on underlyings, see [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — related idea, different stack.

**Dense terms:** [QuantLib](GLOSSARY.md#quantlib) · [Heston](GLOSSARY.md#heston-model) · [Barrier option](GLOSSARY.md#barrier-option) · [Implied vol](GLOSSARY.md#volatility)

**Related:** [Barriers (MT5)](BARRIER_FUNCTIONS.md) · [Finviz](FINVIZ.md) · [CLI](CLI.md) · [Glossary](GLOSSARY.md)

---

> **Dependencies:** Options chain data uses anonymous Yahoo cookie/crumb negotiation by default and may still fail with 401/429 responses. When `MTDATA_OPTIONS_PROVIDER=tradier` or `auto`, mtdata retries Yahoo if Tradier is unavailable or misconfigured, but reliable chain data still requires `MTDATA_OPTIONS_API_KEY`. QuantLib tools require `pip install QuantLib` and are independent of both MT5 and chain-provider access.

---

## Options Data

The options data tools are external-data helpers. Yahoo Finance is the default; mtdata negotiates an anonymous cookie and crumb, but this **best-effort fallback** can still reject requests (401/429). If you select `tradier` (or `auto` with a Tradier token), mtdata retries Yahoo once when Tradier is unavailable or misconfigured. To use reliable authenticated chains, add these values to `.env`:

```bash
MTDATA_OPTIONS_PROVIDER=tradier
MTDATA_OPTIONS_API_KEY=your_tradier_token
```

`TRADIER_TOKEN` and `TRADIER_API_KEY` are also accepted as token aliases; use
`MTDATA_OPTIONS_BASE_URL` only when overriding Tradier's default API base URL.

Run `options_provider_status` to see the configured vs. effective provider and
whether mtdata is using authenticated or best-effort fallback access. This is a
configuration-only check: `chain_request_supported` reports static capability,
while live reachability and data-readiness fields are `null` with
`chain_health_status=unknown_not_checked` until an actual chain tool runs:

```bash
mtdata-cli options_provider_status --json
```

An unsupported `MTDATA_OPTIONS_PROVIDER` value makes this status command fail
with `options_provider_invalid`. Its structured output preserves the configured
value and labels Yahoo separately as the effective fallback, so health checks
cannot mistake a typo for a valid Yahoo configuration.

`options_barrier_price` is a local QuantLib calculator and still works without options-chain provider access when you supply spot, strike, barrier, maturity, and volatility.

### `options_expirations`

List available option expiration dates for a US stock.

```bash
mtdata-cli options_expirations AAPL --json

# Provider-neutral S&P 500 alias (resolved to ^SPX for Yahoo)
mtdata-cli options_expirations SPX --json
```

**Returns:** List of expiration dates available for the symbol. When an alias is
used, the response includes `requested_symbol` and `provider_symbol`. An empty
provider expiration snapshot fails with `options_expirations_unavailable`
instead of reporting a successful zero-expiration result.

### `options_chain`

Fetch an options chain snapshot with filtering.

```bash
# Compact chain snapshot (calls + puts)
mtdata-cli options_chain AAPL --json

# Calls from the next live listed expiration
mtdata-cli options_chain AAPL --option-type call --json

# Filter by liquidity
mtdata-cli options_chain TSLA --min-open-interest 100 --min-volume 50 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | (required) | Stock ticker |
| `--expiration` | (nearest) | Currently listed expiration date `YYYY-MM-DD`; use `options_expirations` to discover valid dates |
| `--option-type` | `both` | `call`, `put`, or `both` |
| `--min-open-interest` | 0 | Minimum open interest filter; must be at least 0 |
| `--min-volume` | 0 | Minimum volume filter; must be at least 0 |
| `--limit` | 20 compact; 200 full | Maximum contracts to return; must be at least 1 |
| `--offset` | 0 | Zero-based index of the first contract in this page |

Chain results report exact filtered `available_count`, the nearest-strike
`selection_order`, and the standard nested `pagination` object. Calls and puts
share one deterministic order. Advance `--offset` by the returned page size
while `pagination.has_more` is true:

```bash
mtdata-cli options_chain AAPL --limit 20 --offset 0 --json
mtdata-cli options_chain AAPL --limit 20 --offset 20 --json
```

Each option row reports the provider's `contract_size` classification,
`contract_multiplier`, multiplier status, deliverable status, and
`premium_quote_unit`. Compact rows also retain provider `implied_volatility`
(a decimal fraction where `1.0 = 100%`) and `in_the_money` when available, so
volatility and moneyness comparisons do not require full detail. A
provider-classified `REGULAR` US equity option has a
multiplier of 100 underlying units. Nonstandard, adjusted, or missing provider
metadata leaves the multiplier unavailable instead of inheriting a chain-level
default. Convert a quoted bid, ask, or last price to cash premium only when the
row multiplier is known:

```text
cash premium = quoted premium × contract_multiplier
```

Expiration results report when the catalog was retrieved through
`catalog_fetched_at`, `catalog_cached`, and `catalog_freshness`. They name the
separate underlying quote scope explicitly with `underlying_as_of`,
`underlying_data_age_seconds`, and `underlying_data_stale`. An old underlying
quote does not make a newly fetched expiration catalog stale. Chain results use
the same `underlying_*` names; those fields do not qualify the option contracts.

Every option row separately reports `contract_as_of`, contract age and stale
status, `quote_quality`, and `quote_usable_for_live_analysis`. The aggregate
`option_chain_freshness`, `option_chain_quality`, and count fields summarize
the returned page. A contract is live-usable only when its provider last-trade
timestamp is available and no more than 15 minutes old and its bid/ask are
positive and non-crossed. This timestamp is a provider last-trade observation,
not an exchange guarantee that both displayed quote sides updated at that
instant. Missing timestamps, zero/one-sided markets, crossed markets, and stale
timestamps fail closed. Yahoo's underlying price remains a regular-session
price, as shown by `underlying_price_session`.

Provider timestamps up to 30 seconds ahead of the local clock are reported as
`clock_skew_within_tolerance` and are not marked stale solely for that skew.
Larger future skew is treated as stale.

---

## QuantLib Barrier Option Pricing

### `options_barrier_price`

Price a European barrier option with QuantLib's analytic continuous-monitoring engine (`AnalyticBarrierEngine`, Reiner–Rubinstein). This is not the discrete-bar TP/SL first-hit tool; use [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) for that.

By default, QuantLib pricing assumes the `UnitedStates.NYSE` calendar and interprets `maturity_days` as calendar days. Override `--calendar` and `--maturity-basis` for non-US or non-equity workflows.
When `--valuation-date` is omitted, mtdata uses the selected calendar's local
date. The default `UnitedStates.NYSE` calendar uses `America/New_York`.
Responses expose `valuation_timezone` and
`valuation_date_source: default_calendar_local_date`; pass an explicit date for
a portfolio-specific accounting date. `TARGET` uses `Europe/Brussels`,
`NullCalendar` uses UTC, and other QuantLib calendars currently fall back to
UTC.

```bash
# Down-and-out call (knock-out if price falls to barrier)
mtdata-cli options_barrier_price 150 --strike 155 --barrier 140 --maturity-days 30 --option-type call --barrier-type down_out --volatility 0.25 --json

# Up-and-in put (activates if price rises to barrier)
mtdata-cli options_barrier_price 150 --strike 145 --barrier 160 --maturity-days 60 --option-type put --barrier-type up_in --volatility 0.3 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spot` (positional) | (required) | Current spot price |
| `--strike` | (required) | Strike price |
| `--barrier` | (required) | Barrier level |
| `--maturity-days` | (required) | Time to maturity in calendar days |
| `--option-type` | `call` | `call` or `put` |
| `--barrier-type` | `up_out` | `up_in`, `up_out`, `down_in`, `down_out` |
| `--risk-free-rate` | 0.02 | Risk-free rate (decimal) |
| `--dividend-yield` | 0.0 | Dividend yield (decimal) |
| `--volatility` | 0.2 | Implied volatility (decimal, e.g., 0.2 = 20%) |
| `--rebate` | 0.0 | Knock-out: paid if the barrier is hit. Knock-in: paid at expiry only if the barrier is never hit |
| `--valuation-date` | selected calendar's local date | Valuation date in `YYYY-MM-DD`; omitted uses the calendar local date (not an options-chain snapshot) |
| `--calendar` | `UnitedStates.NYSE` | QuantLib calendar name (for example `UnitedStates.NYSE` or `NullCalendar`) |
| `--maturity-basis` | `calendar_days` | Interpret `--maturity-days` as `calendar_days` or `business_days` in the selected calendar |

**Barrier types explained:**

| Type | Meaning |
|------|---------|
| `up_in` | Option activates when price rises through barrier |
| `up_out` | Option deactivates when price rises through barrier |
| `down_in` | Option activates when price falls through barrier |
| `down_out` | Option deactivates when price falls through barrier |

**Returns:** Option price and Greeks (delta, gamma, vega). Gamma is quoted per
squared underlying-price unit. Vega is quoted per `1.0` change in decimal
volatility, so a one-volatility-point (`0.01`) scenario uses `vega * 0.01`.

---

## Heston Model Calibration

### `options_heston_calibrate`

Calibrate the Heston stochastic volatility model from live options data. The Heston model captures volatility clustering and the volatility smile/skew.

```bash
# Calibrate from call options
mtdata-cli options_heston_calibrate AAPL --option-type call --json

# Calibrate from the nearest eligible expiration with liquidity filters
mtdata-cli options_heston_calibrate TSLA --option-type both --min-open-interest 50 --min-volume 10 --max-contracts 30 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | (required) | Stock ticker |
| `--expiration` | (nearest eligible) | Specific expiration to calibrate against; when omitted, selects the nearest listed expiry at least 7 calendar days after the provider observation date |
| `--option-type` | `call` | `call`, `put`, or `both` |
| `--risk-free-rate` | 0.02 | Risk-free rate (decimal) |
| `--dividend-yield` | 0.0 | Dividend yield (decimal) |
| `--valuation-date` | provider observation date | Options-chain observation date in `YYYY-MM-DD`; an explicit value must match the provider snapshot date |
| `--min-open-interest` | 0 | Min open interest for contract selection; must be at least 0 |
| `--min-volume` | 0 | Min volume for contract selection; must be at least 0 |
| `--max-contracts` | 25 | Max contracts used in calibration; must be at least 5 |
| `--calendar` | `UnitedStates.NYSE` | QuantLib calendar name used for maturity assumptions |
| `--maturity-basis` | `calendar_days` | Basis for the reported `days_to_expiry` diagnostic. The Heston helper maturity is always anchored to the contract's calendar expiry date. |

Calibration requires a timezone-qualified `underlying_as_of` timestamp and at
least five contracts that are current, timestamped, two-sided, and within 15
minutes of the spot observation. Contracts that fail timestamp, freshness,
quote, or spot-skew checks are excluded before fitting. If fewer than five
remain, calibration is not attempted and returns
`heston_contract_inputs_rejected` with rejection counts.

`calibration_data_status: current` and `usable_for_pricing: true` therefore
qualify both the underlying and every selected contract. A stale underlying
sets `calibration_data_status: stale`, adds `stale_underlying_data` to
`pricing_usability_failures`, and returns `heston_calibration_rejected`. The
same failure contract applies to parameter and IV-error quality gates.
Calibrate a current, accepted snapshot before using the parameters to price an
option. Omit `--valuation-date` to derive it from the underlying observation.
When `--expiration` is omitted, calibration skips same-day and short-dated
contracts that do not meet its seven-calendar-day minimum.

**Heston parameters returned:**

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `v0` | v₀ | Initial variance |
| `kappa` | κ | Mean reversion speed |
| `theta` | θ | Long-term average variance |
| `sigma` | σ | Volatility of volatility ("vol of vol") |
| `rho` | ρ | Correlation between asset and volatility processes |

**Use cases:**
- More accurate barrier option pricing (using calibrated vol surface instead of flat vol)
- Volatility smile/skew analysis
- Exotic option valuation inputs

---

## Quick Reference

| Task | Command |
|------|---------|
| List expirations | `mtdata-cli options_expirations AAPL` |
| Options chain | `mtdata-cli options_chain AAPL --option-type call` |
| Barrier option price | `mtdata-cli options_barrier_price 150 --strike 155 --barrier 140 --maturity-days 30` |
| Heston calibration | `mtdata-cli options_heston_calibrate AAPL` |

---

## See Also

- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — MT5-based barrier probability analysis
- [forecast/VOLATILITY.md](forecast/VOLATILITY.md) — Volatility estimation methods
- [FINVIZ.md](FINVIZ.md) — Fundamental data
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
