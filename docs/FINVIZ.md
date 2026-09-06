# Company and calendar context

**Audience:** User

Look up **US company facts**, stock screens, insider filings, and a delayed
forex/crypto/futures snapshot — plus a filterable **economic and earnings
calendar** — without leaving mtdata.

These commands currently use [Finviz](https://finviz.com) as the research
adapter. They **complement** MetaTrader 5; they do not replace the terminal
for live quotes. The provider states a nominal 15–20 minute delay; Finviz
does not stamp an observation time, so that range is not a measured age.
Treat it as background, not a live tape.

Everyday headlines still start at [NEWS.md](NEWS.md) (`news`). This page is
the table-and-dossier side: `equity_profile`, `screener`, `calendar`,
`asset_performance`, and the raw `news` provider pages.

**Dense terms:** [Finviz](GLOSSARY.md#finviz)

**Related:** [News](NEWS.md) · [CLI](CLI.md) · [Glossary](GLOSSARY.md) · [Setup](SETUP.md)

---

## Quick start (read-only)

```bash
# Company summary (P/E, EPS, market cap, …)
mtdata-cli equity_profile AAPL --json

# Ranked news for a ticker (preferred everyday path)
mtdata-cli news AAPL --json

# Screen US stocks
mtdata-cli screener --filters '{"Sector": "Technology", "P/E": "Under 15"}' --json

# High-impact economic calendar
mtdata-cli calendar --kind economic --impact high --json

# Delayed forex performance snapshot — not a live quote
mtdata-cli asset_performance --universe forex --json
```

In the Web UI, open **Tools** and search the same command names. From an
assistant: “Run `equity_profile` for AAPL. Do not trade.”

Company tools take exchange tickers such as `AAPL`. MetaTrader 5 symbols may
be suffixed (`AAPL.NAS`). Those suffixes are accepted and reported as
`requested_symbol` plus `finviz_ticker`. Several broker contracts can match
one ticker, so the reverse mapping is not guessed. If a MetaTrader 5 tool
rejects the bare ticker, use its `details.did_you_mean` list or
`symbols_list --search AAPL`.

---

## Which command?

| You want | Command |
|----------|---------|
| Ranked headlines + a few upcoming events | `news` — see [NEWS.md](NEWS.md) |
| One stock’s raw provider news page | `news NVDA --view ticker --source finviz` |
| Broad headlines or blogs | `news --view market --source finviz` |
| Filterable event table | `calendar` |
| This week’s earnings list | `calendar --kind earnings --view period` |
| Company summary / description / ratings / peers / insider | `equity_profile --sections …` |
| Screen US stocks or list valid filters | `screener` |
| Delayed forex, crypto, futures, or market-wide insider table | `asset_performance --universe …` |
| Live broker quote | `market_ticker` / `symbols_top_markets` — not these tools |

Pin an adapter with `--source finviz` when you want that provider only.
`equity_profile` and `screener` advertise only `auto` and `finviz`.
`--source mt5` remains on the schema for `calendar` and `asset_performance`,
but those jobs have no MetaTrader 5 table yet — the response is
`research_capability_unsupported`, not an empty fake table.
`news --source mt5` does work for the broker news feed.

Booleans on the CLI are `true` / `false`.

---

## `equity_profile`

A US-issuer dossier. Default compact output is a **fundamentals summary**.

```bash
mtdata-cli equity_profile AAPL --json
mtdata-cli equity_profile AAPL --sections valuation --detail full --json
mtdata-cli equity_profile AAPL --sections all --json
mtdata-cli equity_profile AAPL --sections all --detail full --json
mtdata-cli equity_profile TSLA --sections description --json
mtdata-cli equity_profile MSFT --sections peers --json
mtdata-cli equity_profile GOOGL --sections ratings --json
mtdata-cli equity_profile AAPL --sections insider --limit 10 --json
mtdata-cli equity_profile AAPL --sections summary,description,ratings --json
mtdata-cli equity_profile AAPL --sections valuation,ownership --json
```

| Flag | Default | What it does |
|------|---------|----------------|
| `--sections` | `summary` | Comma-separated slices. Fundamentals: `summary`, `valuation`, `performance`, `technical`, `dividends`, `ownership`, `profile`, `all`. Extra: `description`, `ratings`, `peers`, `insider`. |
| `--fields` | (none) | Optional fundamentals field list. Unknown names fail with `finviz_fundamentals_fields_invalid` and list `valid_values.fields`. |
| `--limit` | `5` | Row cap for ratings, peers, and insider. |
| `--offset` | `0` | Skip ratings/peers rows. |
| `--page` | `1` | Insider page (one-based). |
| `--source` | `auto` | Adapter pin. |

`--sections` and `--detail` are independent: `sections=all` still returns
every selected metric in compact (numbers only). `detail=full` adds
diagnostics inside the slices you asked for. Several fundamental slices
(`valuation,ownership`) return the **union** of those field groups, not the
full `category=all` dump. One slice returns that payload plus
`providers_used`. Extra slices nest under `fundamentals`, `description`,
`ratings`, `peers`, and `insider`.

Percentage metrics are JSON numbers on the `1.0 = 1%` scale and carry
`units`. Growth fields use names such as `eps_next_year_growth_pct`. A mixed
`--fields` request keeps resolved metrics, sets `partial_failure=true`, and
lists `missing_fields`.

Ratings targets are USD per share; `price_target_change_pct` is percentage
points. Insider rows include owner, role, buy/sell, shares, value, and dates.

---

## `screener`

Screen US stocks, or list the filter catalog.

```bash
# Tech on NASDAQ
mtdata-cli screener --filters '{"Exchange": "NASDAQ", "Sector": "Technology"}' --json

# Compact key=value
mtdata-cli screener --filters "exchange=NASDAQ,sector=Technology" --json

# Comparison aliases
mtdata-cli screener --filters "pe_under=15,beta_under=1" --json

# Native Finviz URL tokens
mtdata-cli screener --filters "exch_nasd,sec_technology" --json

# Valuation columns; default order is largest market cap first
mtdata-cli screener --filters '{"Dividend Yield": "Over 5%"}' --view valuation --json

# List filters, then inspect one
mtdata-cli screener --list-filters true --json
mtdata-cli screener --list-filters true --search dividend --json
mtdata-cli screener --list-filters true --filter-name "Market Cap." --json
```

| Flag | Default | What it does |
|------|---------|----------------|
| `--filters` | (none) | JSON object, `key=value` pairs, or Finviz shorthand. Names are provider-defined. |
| `--order` | `-marketcap` | Sort. Default is largest market cap first so paging is stable. Use `--order=price` for ascending price. |
| `--view` | `overview` | Column set: `overview`, `valuation`, `financial`, `ownership`, `performance`, `technical`. |
| `--list-filters` | `false` | List valid filter names instead of screening. |
| `--search` | (none) | Filter-catalog search when `--list-filters true`. Name matches rank ahead of option-value hits. Compact search returns `value_count` plus a small `matched_values` sample. |
| `--filter-name` | (none) | One filter’s accepted values when `--list-filters true`. |
| `--value-limit` / `--value-offset` | `20` / `0` (compact) | Page within one filter's accepted values. Full detail returns all values unless `--value-limit` is supplied. |
| `--limit` / `--page` | `20` / `1` | Result page. Catalog listing uses `--limit` / `--offset`. Nonzero `--offset` in results mode is rejected; use `--page`. |

**Common JSON keys:** `Exchange`, `Index`, `Sector`, `Industry`, `Country`,
`Market Cap.`, `P/E`, `Forward P/E`, `PEG`, `P/S`, `P/B`, `Dividend Yield`,
`EPS growth this year`, `Return on Equity`, `Current Ratio`,
`Analyst Recom.`, `RSI (14)`, `50-Day Simple Moving Average`,
`Average Volume`, `Price`, `Beta`.

Screener percentages are numeric points (`1.0 = 1%`), including performance,
volatility, gap, and change-from-open. Growth columns use the same
`*_growth_pct` / `*_cagr_pct` names as `equity_profile`.

---

## `calendar`

A **table** of scheduled events. `news` still surfaces a few upcoming/recent
items in ranked buckets; use `calendar` when you need filters and paging.

```bash
mtdata-cli calendar --json
mtdata-cli calendar --kind economic --impact high --currency USD --json
mtdata-cli calendar --kind earnings --start 2026-03-01 --end 2026-03-15 --json
mtdata-cli calendar --kind dividends --json
mtdata-cli calendar --kind earnings --view period --period this-week --json
mtdata-cli calendar --kind earnings --view period --period this-week --include-elapsed true --json
```

| Flag | Default | What it does |
|------|---------|----------------|
| `--kind` | `economic` | `economic`, `earnings`, or `dividends`. |
| `--view` | `range` | `range` is the date-range table. `period` is the compact earnings window and requires `--kind earnings`. |
| `--period` | `this-week` | With `--view period`: `this-week`, `next-week`, `previous-week`, `this-month`. |
| `--impact` | (all) | Economic only: `low`, `medium`, `high`, or comma-separated such as `high,medium`. |
| `--country` / `--currency` | (none) | Economic only, for example `US` / `USD`. Finviz currently covers US releases; a non-US filter can return an empty table even when that region has events. |
| `--start` / `--end` | live window | Inclusive `YYYY-MM-DD`, ISO timestamp, or relative date. Date-only values select the `America/New_York` day; timestamps keep their time-of-day and filter `scheduled_at`. |
| `--upcoming` | live default | Economic only: keep unreleased events. Defaults on when no date range is passed, off for an explicit range. |
| `--include-elapsed` | `false` | Period view: include already-released dates. `previous-week` is always an archive. |
| `--limit` / `--page` | `20` / `1` | Page size. |

Default ranges report `start`, `end`, and `calendar_timezone`. Timestamp
bounds keep ISO instants and `start_precision`/`end_precision`. Event
timestamps use the separate root `timezone` field. Successful snapshots
include UTC `data_fetched_at` and `is_realtime=false`. Compact rows keep
canonical scheduled time, event, country/currency, impact, and parsed
actual/forecast/previous values; raw provider text and parse diagnostics
are in `--detail standard` or `--detail full`.

---

## `asset_performance`

Delayed **research tables**. Not an executable quote — use `market_ticker`
or `symbols_top_markets` for the broker price. Responses set
`quote_role=research_context_not_live_broker_quote`.

```bash
mtdata-cli asset_performance --universe forex --json
mtdata-cli asset_performance --universe forex --rank-by day --json
mtdata-cli asset_performance --universe forex --symbol EURUSD --json
mtdata-cli asset_performance --universe crypto --json
mtdata-cli asset_performance --universe futures --json
mtdata-cli asset_performance GOLD --universe futures --json
mtdata-cli asset_performance --universe insider --option "top week buys" --json
```

| Flag | Default | What it does |
|------|---------|----------------|
| `--universe` | `forex` | `forex`, `crypto`, `futures`, or `insider`. |
| `--symbol` | (none) | Optional forex, crypto, or futures filter such as `EURUSD`, `BTC`/`BTCUSD`, or the provider ticker/name. Crypto is USD-quoted; `BTC/EUR` and other non-USD pairs are rejected. |
| `--option` | `latest` | Insider slice: `latest`, `latest buys`, `latest sales`, `top week`, `top week buys`, `top week sales`, `top owner trade`, `top owner buys`, `top owner sales`. |
| `--rank-by` | (none) | Forex/crypto: rank the fetched snapshot before paging (`5min`, `hour`, `day`, `week`, `month`, `quarter`, `half`, `year`, `ytd`). Futures currently only has `day`; other horizons fail before fetch with `valid_values.rank_by`. Rank keys `quarter` and `half` map to output fields `perf_quarter_pct` and `perf_half_year_pct`. Omit to keep `selection_order=provider_table_order`. |
| `--order` | `desc` with `--rank-by` | Rank direction: `desc` or `asc`. Requires `--rank-by`. |
| `--limit` / `--offset` | `20` / `0` | Forex, crypto, and futures paging. Applied after `--rank-by`. |
| `--page` | `1` | Insider paging. |

Metal futures use provider names `GOLD`/`SILVER` or contract tickers such as
`GC`. Spot broker names `XAUUSD`/`XAGUSD` identify different instruments and
return a complete command for requesting futures context explicitly.

Forex, crypto, and futures rows share one schema: delayed `price` when the
provider has one, and `perf_*_pct` as percent (`1.0 = 1%`). Quarter and
half-year horizons are `perf_quarter_pct` and `perf_half_year_pct`. Check
`units` and `performance_format`. Futures performance from this source does
not include a live price or volume — `data_limitations.price` says so.

---

## `news` provider pages

Preferred everyday path: `mtdata-cli news SYMBOL` (ranked, mixed sources).
Raw Finviz pages:

```bash
mtdata-cli news NVDA --view ticker --source finviz --limit 10 --json
mtdata-cli news --view market --source finviz --news-type news --json
mtdata-cli news --view market --source finviz --news-type blogs --json
```

`--view ticker` needs a symbol. `--page` is the provider page for ticker and
market views. Full contract: [NEWS.md](NEWS.md).

---

## Deeper detail

### Delay and throttling

Finviz states a nominal 15–20 minute delay. Rows expose
`nominal_provider_delay_minutes_min` / `nominal_provider_delay_minutes_max`;
the payload root uses `nominal_provider_delay_minutes` plus
`observation_time_status=provider_timestamp_unavailable` and
`observation_age_status=unknown`. That range is the provider’s published
window, not a measured observation age. Equity snapshots also add NYSE
`market_state`, `session_date`, and `latest_completed_session` from the
exchange calendar. Outside regular hours a structured
`observation_age_warning` makes it explicit that the 15–20 minute window
does not bound quote age. Rapid calls can return
`error_code=finviz_rate_limited` with `retryable=true` and numeric
`retry_after_seconds`.

### Company percentages and fields

`equity_profile` percentage metrics are JSON numbers on `1.0 = 1%` and carry
`units`. Growth names look like `eps_next_year_growth_pct`,
`eps_next_5y_growth_pct`, and `sales_yoy_ttm_growth_pct`.

### Screener filter formats

- JSON uses exact Finviz names: `{"Exchange":"NASDAQ"}`.
- Key-value pairs use compact keys: `country=USA,marketcap=mega`.
- Comparison aliases such as `pe_under=15` map to Finviz “Under/Over” options.
- Native shorthand uses Finviz URL tokens such as `cap_largeover,exch_nyse`.
  Invalid tokens are listed in the error details.

### Calendar rows

Economic rows use provider fields `date`, `event`, `ticker`, `importance`
(`1` low, `2` medium, `3` high), `actual`, `forecast`, `previous`,
`category`, `reference`, and `referenceDate` when present. The tool exposes
`symbol` for `ticker` and `reference_date` for `referenceDate`. Raw
`actual` / `previous` / `forecast` strings are kept; parseable prints also
expose `actual_value`, `previous_value`, `forecast_value`, plus shared
`unit` / `scale` when those fields agree (`percent` with `1.0 = 1%`,
`currency` with a `$`/`€`/`£`/`¥` marker and ISO code, or `count` with a
K/M/B/T multiplier and no currency symbol). Events are
unique by `calendar_id` before pagination. Rows without an ID use a composite
of scheduled time, event, symbol, category, reference, country, and currency.
If duplicate variants disagree on a non-empty field, the merged field is
`null` and `provider_conflicts` keeps the alternatives.

`country_attribution` is `provider`, `inferred`, or `unknown`. Country
filters warn when unknown rows were dropped.

Earnings rows distinguish exact times from session buckets. Provider `08:30`
and `16:30` New York markers become a calendar date with
`earnings_timing=before_market` or `after_market` and
`event_time_precision=session_bucket`. `is_earning_date_estimate` qualifies
the date. Period view constrains every date to the requested window; a
yearless token that cannot be reconciled is rejected with
`period_rows_rejected`, `partial`, and a warning.

Earnings EPS families stay unlabeled (`eps_basis=provider_unspecified` and
`eps_reported_basis=provider_unspecified`) because Finviz does not name the
accounting basis. Estimate/actual/surprise triples stay together in compact
output. A warning is added when the two surprise signs disagree. Monetary
earnings and dividend amounts do not invent a listing currency: the payload
sets `currency_status=unavailable` instead of `currency_basis=listing_currency`.

Dividends: if a requested range starts before the current New York date but
extends into the future, mtdata retries the current-forward portion. The
response reports effective `start`, keeps `requested_start`, and sets
`partial=true` and `range_complete=false`.

Pagination is the shared offset-based object in
[OUTPUT.md](OUTPUT.md#pagination), even when you pass one-based `--page`.

### Insider market-wide order

`universe=insider` `latest*` feeds keep provider filing-recency order
(`ordering=filed_at_descending`). Compact rows split `transaction_date` from
timezone-qualified `filed_at`. A Form 144 proposed sale is a filing, not a
completed sale, and is not added to executed-sales totals.

### Crypto rounded prices

Finviz may round very low token prices to zero. In that case the row uses
`price_status: unavailable_provider_rounded_zero`, omits `price`, and adds a
warning instead of presenting zero as tradable. After a `--symbol` filter,
warnings that only name coins not in the returned rows are dropped.

---

## See also

- [NEWS.md](NEWS.md) — Ranked headlines (preferred everyday feed)
- [CLI.md](CLI.md) — Command usage
- [MARKET.md](MARKET.md) — Live broker quotes and scans
- [GLOSSARY.md](GLOSSARY.md)
- [SAMPLE-TRADE.md](SAMPLE-TRADE.md)

Equity-profile JSON paths are stable across section combinations and partial
failures: observations are at `fundamentals.<field>` with root `units` and
projection metadata. Row sections use `ratings.items`, `peers.items`, and
`insider.items`, with pagination and provenance in that section. The business
description is at `description.text`. Consumers of older multi-section responses
should remove the repeated `fundamentals` wrapper; consumers of standalone row
sections should use the section's `items` path.
