# News and calendar

**Audience:** User

See what is happening around a market — headlines and scheduled economic
releases — without opening a separate news terminal.

The everyday command is `news`. It merges Finviz, MetaTrader 5, and CNBC when
that extra is installed, then ranks the mix. Pass `--source finviz` (or
`mt5` / `ycnbc`) to pin one adapter. Use `--view ticker` or `--view market`
for a raw provider page. For a filterable event table — impact, currency,
country, date range — use `calendar` instead of ranked news buckets.

**Dense terms:** [Finviz](GLOSSARY.md#finviz)

**Related:** [Company and calendar context](FINVIZ.md) · [CLI](CLI.md) · [Sample trade](SAMPLE-TRADE.md) · [Env vars (optional embeddings)](ENV_VARS.md#news-embeddings) (Operator)

---

## Quick start (read-only)

```bash
# Broad recent headlines
mtdata-cli news

# Focus on one instrument (FX, stocks, or crypto names your broker lists)
mtdata-cli news EURUSD --json
mtdata-cli news AAPL --json
```

In the Web UI, open **Tools**, search `news`, and run it with or without a
symbol. Same idea from an assistant: “Show me `news` for EURUSD. Do not trade.”

---

## What the buckets mean

With no symbol, you get the most important recent **general** items plus the
market-wide upcoming and recent economic calendar. Symbol requests narrow
calendar events to the instrument's currencies or macro exposure.

With a symbol, the result is split so you can scan quickly:

| Bucket | Plain idea |
|--------|------------|
| `general_news` | Market-wide headlines that are important even if they do not name your symbol. |
| `related_news` | Items that look relevant to this instrument (name, aliases, or theme). |
| `impact_news` | High-importance shocks (for example energy or geopolitical) that can move many markets. |
| `upcoming_events` | **Future** calendar releases tied to this instrument — the “what is still ahead” list. |
| `recent_events` | **Already published** calendar prints — useful for “what just came out.” |

Headline rows with a provider-observed instant use `published_at`. Some Finviz
market headlines expose only a date; those rows use `publication_date`,
`timestamp_precision=date`, and `source_timezone=America/New_York` instead of
an invented midnight timestamp. Provider order breaks ties among these
date-only rows. Economic-calendar rows use `scheduled_at`, including rows in
both event buckets, so a future release time cannot be mistaken for an article
publication time.

Broad and symbol compact output both return a global page of at most 10 rows,
with a stable `pagination` object (`total`, `returned`, `offset`, `limit`,
`scope`) and a matching top-level `returned` count. Compact also reserves the
next upcoming event (or a recent release when no future event remains) and
includes `bucket_truncation` metadata. With a symbol, a related headline is
reserved first so `--limit 1` cannot hide direct-symbol news; an event still
occupies the next slot when capacity is at least 2 or no headlines exist. Use
`--limit` for a different global page size (compact unified news defaults to
10), `--limit-per-bucket` for independent bucket caps, or `--detail full` for
the uncapped selected buckets and richer matching diagnostics. Bound
publication time with `--start`/`--end` (UTC; date-only values are UTC days)
or `--max-age` (`3600`, `60m`, `1h`). `--max-age` ends at now unless you
explicitly supply `--end`, so upcoming events cannot fill a last-hour query.
These bounds apply to headline `published_at` and event `scheduled_at`.
Rows with only a provider date are excluded whenever a time filter is used:
their unknown publication or event time cannot establish membership in that
window. The result reports applied bounds, `excluded_old_count` (outside
the window), and `excluded_untimestamped_count`, including its
`excluded_date_only_count` subset. An empty page uses
`empty_reason=no_recent_news`. Omit time filters to see date-only rows, or
use `calendar` for upcoming releases.
Scheduled calendar rows in `news` and `calendar` share `event` plus
`scheduled_at` so a timeline can merge them without renaming fields.
The related-news selector reserves up to five of the newest direct-symbol
headlines before filling the remaining internal selection by relevance. Full
detail exposes `related_selection`, including whether that selection was
truncated. For the complete provider-ordered US-equity page, continue with
`news SYMBOL --view ticker --source finviz`; public unified `news` limits
paginate the selected multi-source feed rather than the raw provider
candidate pool.
Time filters on ticker and market views apply to the requested provider page.
Their `count` and `pagination.returned` describe retained rows, with `items=[]`
when none match. `pagination.scope=provider_page` means `offset`, `limit`,
and `has_more` still navigate unfiltered provider pages. `provider_total` and
`provider_returned` preserve the provider counts; filtered `total` and
`more_available` are unknown (`null`). A later provider page may have matches
even when the current page is empty.
Calendar rows show both the absolute UTC `scheduled_at` timestamp and the
convenience `relative_time` label in the default TOON view.
When the provider supplies a reporting period, `reference_date` identifies the
period the statistic describes (for example, a month or quarter). It is not the
release instant; use `scheduled_at` to decide whether the event is ahead or has
already occurred. Full metadata keeps the same value under
`metadata.reference_date` when it can be resolved.

Full detail also adds, when available, a `market_context` quote snapshot.
Finviz snapshot performance is expressed in canonical `*_pct` metadata fields
using percentage points (`1.0 = 1%`), and summaries render a `%` sign. The
high-frequency provider fractions are never exposed as unqualified decimals.

A small `--limit` still tries to keep at least one upcoming event visible so a
tight cap does not hide the next scheduled release.

---

## When to use a raw provider page or a table

| Need | Tool |
|------|------|
| One stock’s provider news page | `news NVDA --view ticker --source finviz` |
| Broad headlines or blogs | `news --view market --source finviz` |
| Economic or earnings calendar only | `calendar` / `calendar --kind earnings --view period` |
| Fundamentals, screeners, insiders | See [FINVIZ.md](FINVIZ.md) |

Finviz US-equity data is delayed about 15–20 minutes. Treat it as research
context, not a live tape.

---

## Deeper detail

- Matching uses symbol aliases, asset-class words, MetaTrader 5 metadata, and a
  lightweight text-similarity score. It is a helper, not a guarantee that every
  headline will move the price.
- Optional embedding rerank (downloads a model on first use) is **off** by
  default. See [ENV_VARS.md](ENV_VARS.md#news-embeddings).
- CNBC via `ycnbc` is an opt-in extra (`pip install "mtdata-mcp[news-ycnbc]"`
  or, from a checkout, `pip install -e ".[news-ycnbc]"`).
  When `--source ycnbc` is pinned but that extra is unavailable, the command
  returns `source_unavailable` with install/restart guidance instead of
  presenting the missing adapter as an empty result.
- A failed Finviz endpoint does not erase successful buckets from another
  endpoint. The result is marked `partial=true` and `status=partial`; full
  detail records the affected endpoint under source diagnostics.
