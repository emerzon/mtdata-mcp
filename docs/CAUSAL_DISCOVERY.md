# Granger predictive-link discovery

**Audience:** User

Explore **who might lead whom** across symbols with pairwise Granger-style tests on recent MT5 closes. This is **exploratory feature discovery** for watchlists — not a claim of true economic causality.

| If you want… | Prefer | Concept |
|--------------|--------|---------|
| Simple co-movement | `correlation_matrix` | [Correlation](GLOSSARY.md#correlation) |
| Two-symbol lead/lag | `cross_correlation` | [Cross-correlation](GLOSSARY.md#cross-correlation-lead--lag) |
| Mean-reverting / spread pairs | `cointegration_test` | [Cointegration](GLOSSARY.md#cointegration) |
| Directed predictive links | `causal_discover_signals` | [Granger causality](GLOSSARY.md#granger-causality) |

**Related:** [CLI](CLI.md) · [Setup](SETUP.md) · [Glossary](GLOSSARY.md) · [Diagnostics](TIME_SERIES_DIAGNOSTICS.md)

---

## Quick Start

```bash
# Compare symbols by correlation strength
mtdata-cli correlation_matrix "EURUSD,GBPUSD,USDJPY" --timeframe H1 --window-bars 500 --method pearson --transform log_return --json

# Use an explicit MT5 group path for easier basket selection
mtdata-cli correlation_matrix --group "Forex\\Majors" --timeframe H1 --window-bars 500 --limit 120 --method pearson --transform log_return --json

# Test an MT5 group for candidate cointegrated pairs
mtdata-cli cointegration_test --group "Forex\\Majors" --timeframe H1 --window-bars 400 --transform log_level --significance 0.05 --json

# Estimate whether the first symbol leads the second
mtdata-cli cross_correlation "EURUSD,GBPUSD" --timeframe H1 --max-lag 20 --transform log_return --json

# Test a multivariate basket with Johansen trace and maximum-eigenvalue tests
mtdata-cli cointegration_test "EURUSD,GBPUSD,EURGBP" --timeframe H1 --method johansen --k-ar-diff 1 --transform log_level --json

# Provide an explicit list of symbols
mtdata-cli causal_discover_signals "EURUSD,GBPUSD,USDJPY" --timeframe H1 --window-bars 800 --max-lag 5 --transform log_return --significance 0.05

# Provide a single symbol to auto-expand its visible MT5 group (e.g., Forex\Majors)
mtdata-cli causal_discover_signals EURUSD --timeframe H1 --window-bars 800
```

---

## What It Does

Historical `--end` limits completed inputs by their close time, including broker
calendar boundaries for daily, weekly, and monthly bars. `--include-incomplete true`
can include a live forming candle, but cannot reconstruct a historical partial
candle from its eventual completed prices. Use the default completed-bar policy
for historical research.

### `correlation_matrix`

For each unordered pair of symbols `(A, B)`, the tool:
1. Fetches recent close-price histories
2. Applies a transform (by default `log_return`)
3. Computes pairwise correlations on overlapping transformed samples
4. Returns canonical ranked pair rows plus optional matrix/highlight views

It accepts either:

- an explicit `symbols` list (or compatibility alias `symbol`), or
- a `group` path that matches the MT5 symbol groups exposed by `symbols_list --list-mode groups`

`items` is the canonical compact payload. Each row includes the correlation,
sample count, and pairwise period window; `context` records the timeframe,
`window_bars`, transform, and `min_overlap` used. A requested `limit` records
the output page size, not the analysis sample size. Use `detail=full` when you also
need derived convenience views such as `matrix`; compact detail keeps only the
ranked pair rows plus summary highlights. Pagination applies only to `items`;
the full-detail `matrix` always represents every pair computed by the analysis.

### `cointegration_test`

For each unordered pair of symbols `(A, B)`, the tool:
1. Fetches recent price histories
2. Applies a level-style transform (`log_level` by default)
3. Checks each transformed level and first difference with ADF diagnostics
4. Runs one Engle-Granger test with the alphabetically first symbol as the
   dependent series
5. Reports that orientation's p-value, hedge ratio, spread diagnostics, and
   `orientation_policy="canonical_symbol_order"`

Engle-Granger is orientation-sensitive. The tool deliberately avoids selecting
the lower p-value after testing both directions because that would cherry-pick
the test result. Canonical symbol ordering makes results independent of request
order. A pair is classified as cointegrated or not cointegrated only when both
series are plausibly I(1): non-stationary in levels and stationary after first
differencing. Rows that fail this prerequisite retain the raw test diagnostics
but report `cointegrated=null` and `relationship="prerequisite_failed"`.

With `method=engle_granger`, it evaluates unordered pairs. With
`method=johansen`, it evaluates the aligned basket jointly and returns
trace/max-eigenvalue rank estimates plus cointegrating vectors. Johansen's
critical-value tables support `significance` values of `0.01`, `0.05`, and
`0.1` only; other values return an `invalid_input` error. Engle-Granger accepts
any significance value strictly between zero and one.

It accepts either:

- an explicit `symbols` list (or compatibility alias `symbol`), or
- a `group` path that matches the MT5 symbol groups exposed by `symbols_list --list-mode groups`

### `cross_correlation`

This tool requires exactly two symbols and evaluates lags from `-max_lag`
through `+max_lag`. A positive best lag means the first symbol leads the second;
a negative lag means the second leads the first. The result includes a
moving-block bootstrap confidence interval for the best-lag correlation.
Because the best lag is selected by maximum absolute correlation, the interval
uses a Bonferroni-adjusted per-lag confidence level to provide 95% family-wise
coverage across all evaluated lags. `best.significant` is true only when that
adjusted interval excludes zero; the context reports the number of lag tests
and both confidence levels. The adjusted bounds are exposed as
`best.ci_familywise_low` and `best.ci_familywise_high`. `correlation_matrix`
also reports `ci_familywise_low` and `ci_familywise_high`, corrected across all computed
symbol pairs. Those intervals use a moving-block bootstrap when sample size
allows; otherwise they are labeled `iid_fisher_z_approximation` because the
Fisher-z formula assumes independent observations. Use `cross_correlation` for
lead/lag inference on a specific pair.

### `causal_discover_signals`

For each ordered pair of symbols `(cause → effect)`, the tool:
1. Aligns each pair on that pair's overlapping close-price history
2. Applies a transform (by default `log_return`) to improve stationarity
3. Optionally z-scores the series for numerical conditioning (`normalize=true`);
   with the fitted intercept, this does not change the exact Granger statistic
4. Runs Granger causality tests for lags `1..max_lag`
5. Selects the **best (lowest raw p-value) lag** per pair using `ssr_ftest`
6. Applies Bonferroni correction first across the tested lags, then across all
   successfully tested directed pairs

The primary `p_value` and `significant` fields use this global family-wise
correction. `p_value_raw` is the selected lag's unadjusted value, and
`p_value_lag_adjusted` is corrected only for the lag search. The correction
becomes stricter as the basket grows; use full detail to inspect non-significant
exploratory candidates rather than interpreting an empty compact result as
proof that no predictive relationships exist.

---

## Parameters

| Parameter | Default | Description |
|----------|---------|-------------|
| `symbols` / `--group` | one required | Comma-separated MT5 symbols, or an explicit MT5 group path. If you pass **one** symbol, mtdata expands to other visible symbols in the same MT5 group. |
| `timeframe` | `H1` | Bar timeframe (`M15`, `H1`, etc.). |
| `window_bars` | `500` | Maximum overlapping transformed samples analyzed per pair. |
| `limit` | all rows | Optional maximum number of ranked result rows returned. |
| `offset` | `0` | Number of ranked result rows to skip before applying `limit`. |
| `max_lag` | `5` | Maximum lag to test (≥ 1). |
| `significance` | `0.05` | Family-wise alpha threshold after Bonferroni correction across tested lags and directed pairs. |
| `transform` | `log_return` | One of: `log_return`, `log_level`, `pct`, `diff`, `level`. |
| `normalize` | `true` | Z-score each series for numerical conditioning; this is affine-invariant with the fitted intercept. |

---

## Output

Default output is TOON (or `--json`) with ranked directed pairs in `items`.
Each row includes:

- `effect`, `cause`, `lag` — best-performing lag for the pair
- `p_value` — globally Bonferroni-adjusted p-value (`significance_basis: p_value_global_bonferroni_adjusted`)
- `p_value_raw`, `p_value_lag_adjusted` — unadjusted and lag-only adjusted values
- `significant` — `true` when `p_value < significance` (for `log_return` / `pct` / `diff`)
- `samples`, `period_start`, `period_end`

Compact detail returns significant links only; standard/full also include
non-significant rows. Top-level `summary.counts` reports `pairs_tested`,
`directed_tests`, `undirected_pairs`, and `significant_links`.
`result` is `links_found` or `no_links_found`. A `significant` flag is
evidence of incremental lagged predictability, not structural or economic
causality.

Tip: `--json` returns the same structured payload as default TOON text.

---

## Interpretation and Caveats

- `correlation_matrix`, `cointegration_test`, `causal_discover_signals`, and `market_scan` all accept `symbols` for explicit multi-symbol calls.
- `group` remains mutually exclusive with explicit symbol selectors.

- Use **`correlation_matrix`** when you want a fast view of which symbols move together or in opposite directions.
- Use **`cointegration_test`** when you want candidate pairs or baskets whose price levels may share a stable long-run relationship.
- Use **`causal_discover_signals`** when you specifically want to test whether lagged values of one symbol add predictive information for another.
- Granger causality is a **predictive** notion: “past values of A help predict B” under the model assumptions.
- Results are **pairwise** (not a full causal graph) and can be confounded by common drivers (USD strength, risk-on/off, sessions).
- Use transforms (returns/diffs) and sufficient history; non-stationary levels can produce misleading links.
- Validate any signal with out-of-sample testing before using it in a strategy.

---

## Dependencies

`causal_discover_signals` and `cointegration_test` require `statsmodels`. If it is not installed, the tools return a readable error message.

---

## See Also

- [FORECAST.md](FORECAST.md) — Forecasting methods overview
- [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) — Indicator reference
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
