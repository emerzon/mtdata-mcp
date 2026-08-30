# Regime detection

**Audience:** User

Markets switch **behavior modes** — trending, ranging, or high-volatility. A tactic that works in one mode often fails in another. `regime_detect` tries to label the current mode so you can **wait**, **shrink size**, or **pick a different tool** — not so it can place an order for you.

`rule_based` returns `trending`, `ranging`, or `transition`. HMM/GMM and similar methods use numeric or volatility-ranked states (`low_vol` / `high_vol`), not a named `stress` label. Elevated BOCPD transition probability is the usual stand-down signal when the statistical regime may have just changed.

The method names below look dense. Each one has a one-line “plain idea” in the [glossary](../GLOSSARY.md#regime). Start with `hmm` (two quiet/loud states) or `bocpd` (“did something just change?”).

**Dense terms (friendly defs):** [Regime](../GLOSSARY.md#regime) · [HMM](../GLOSSARY.md#hidden-markov-model-hmm) · [BOCPD](../GLOSSARY.md#change-point-detection-bocpd) · [PELT](../GLOSSARY.md#pelt-change-point-detection) · [MS-AR](../GLOSSARY.md#markov-switching-ar-ms-ar) · [GMM](../GLOSSARY.md#gaussian-mixture-gmm) · [Filtered vs smoothed](../GLOSSARY.md#filtered-vs-smoothed-inference)

**Related:** [Glossary](../GLOSSARY.md) · [Forecasting](../FORECAST.md) · [Advanced playbook](../SAMPLE-TRADE-ADVANCED.md) · [Sample trade](../SAMPLE-TRADE.md)

---

## Why regimes matter

Markets cycle through distinct behavioral phases:

| Regime | Characteristics | Strategy Implication |
|--------|-----------------|---------------------|
| **Low Volatility / Ranging** | Price oscillates in a band | Mean reversion works; trend-following fails |
| **Trending** | Directional momentum | Trend-following works; mean reversion fails |
| **High Volatility / Crisis** | Large unpredictable swings | Reduce size or stay out |

A strategy that profits in one regime may lose in another. Regime detection helps you:
1. **Filter trades:** Only enter when conditions match your strategy
2. **Adjust sizing:** Reduce risk in unfavorable regimes
3. **Detect breakouts:** Identify when the market is transitioning

---

## Methods

### 1. Hidden Markov Model (HMM)

**What it does:** Classifies each bar into one of N hidden "states" based on return and volatility patterns.

**How it works:**
1. Assumes the market switches between N underlying states
2. Each state has characteristic mean return and volatility
3. Uses observed data to estimate which state is currently active

The implementation is a Gaussian HMM with an estimated transition matrix.
`params.inference=filtered` is the default and uses observations through each
bar under parameters fitted on the requested window. Use
`params.inference=smoothed` only for retrospective segmentation because it uses
later observations. A state change is emitted only after the new raw state has
persisted for `min_regime_bars` consecutive observations. This is a causal
confirmation delay, not a promise that every emitted run will contain that many
rows; a still-unconfirmed terminal candidate is reported in `warnings`.
Posterior probabilities remain the model's pre-confirmation values, so their
argmax can temporarily differ from the confirmed state during that delay;
`params_used.state_probability_alignment` makes this explicit.
Model parameters are still fitted on the requested analysis window, and canonical
state IDs are ordered by full-window state means. Historical canonical IDs are
therefore retrospective labels; use rolling `--end` cutoffs for point-in-time tests.
Compact and full responses preserve this distinction in
`historical_labels_are_retrospective`, `historical_label_scope`, and
`point_in_time_guidance`.

Return direction labels use one shared statistical rule across state models,
BOCPD, and PELT: bullish or bearish requires the estimated mean to be at least
1.96 standard errors from zero. State-model labels use the state's dispersion
and observed bar count; without enough evidence the label remains neutral.

**Example:**
```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --method hmm --params "n_states=2"
```

`params` is method-specific and strict. Unknown keys fail with the selected
method's valid key list, so `n_statez=3` cannot silently fall back to the
two-state HMM default. `ensemble` and `all` accept the union needed by their
submethods, then pass each submethod only the keys it consumes.

**Output:**
```
summary:
  last_state: 0
  state_shares:
    0: 0.623
    1: 0.377
  state_sigma:
    0: 0.000303   # Low volatility state
    1: 0.000739   # High volatility state
```

**Interpretation:**
- **State 0:** Low volatility (σ = 0.0003) — ranging/quiet market
- **State 1:** High volatility (σ = 0.0007) — trending/active market
- Currently in State 0 (low volatility)

State IDs are ordered by ascending mean return, not by volatility. The example's
State 0 happens to be the low-volatility state; always use
`state_sigma[last_state]` to read the current state's fitted volatility.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_states` | 2 | Number of regimes to detect |
| `inference` | `filtered` | `filtered` for live interpretation or `smoothed` for retrospective segmentation |

**When to use:**
- Ongoing regime classification
- Filtering strategies by market type
- Multi-day to multi-week analysis

---

### 2. Change-Point Detection (BOCPD)

**What it does:** Detects moments when the market's statistical properties changed.

**How it works:**
- Bayesian Online Change Point Detection
- Estimates probability that each bar marks a regime transition
- Doesn't classify regimes—just detects when changes occur

**Example:**
```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --method bocpd --threshold 0.5
```

**Output:**
```
current_regime:
  latest_transition_probability: 0.471
  transition_risk: elevated
transition_summary:
  max_transition_probability: 0.471
  recent_change_points_count: 0
  raw_change_points_count: 0
  accepted_change_points_count: 0
  rejected_change_points_count: 0
```

**Interpretation:**
- `latest_transition_probability: 0.471` — 47% probability that a regime change just occurred
- `max_transition_probability` is the maximum probability inside `lookback`
- Below threshold (0.5), so not flagged as a change point
- Higher probabilities indicate likely structural breaks
- Invariant: `raw_change_points_count = accepted_change_points_count + rejected_change_points_count`. `rejected_change_points_count` is the number of raw candidates filtered out, not the accepted remainder.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | auto | Probability cutoff to flag a change point. Omit it for automatic asset/timeframe calibration; any supplied number, including `0.5`, is fixed. |
| `lookback` | timeframe-based | Recent window used for BOCPD summary statistics in compact/summary detail. `limit` controls the detection history. |

**When to use:**
- Detecting breakouts
- Alerting on market structure changes
- Invalidating stale forecasts

---

### 3. PELT Change-Point Detection

**What it does:** Segments the return series at structural breaks using the Pruned Exact Linear Time algorithm from `ruptures`.

```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --method pelt --params "model=rbf penalty=auto min_size=5 jump=1"
```

Supported cost models are `l1`, `l2`, `rbf`, `normal`, and `ar`. Set a numeric `penalty` for explicit sensitivity or keep `penalty=auto` for a variance-scaled default.

An explicit `--start` / `--end` range is analyzed in full by default. Add
`--fetch-limit` or `--lookback` only when you deliberately want a recent tail
of that range. Requests without dates keep the timeframe-based recent-history
default.

**When to use:**
- Retrospective structural-break detection
- Segmenting history before fitting forecasts
- Confirming breaks suggested by BOCPD

---

### 4. Markov-Switching AR (MS-AR)

**What it does:** Combines HMM with autoregressive modeling. Each regime has its own AR parameters.

**Example:**
```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --method ms_ar --params "n_states=2 order=1"
```

Use `n_states` to choose the number of regimes.
Like HMM, MS-AR defaults to filtered probabilities. Set
`params.inference=smoothed` only for retrospective analysis.
The reported probability rows are the model's pre-confirmation posterior values.
When `min_regime_bars` delays a candidate transition, their argmax can therefore
temporarily differ from the emitted confirmed `state`; `params_used.state_probability_alignment`
records this contract. Fitted `intercept`, `ar_coefficients`, and
`innovation_volatility` values are reported in canonical regime order.

**When to use:**
- When regime changes affect both mean and autocorrelation structure
- Academic research contexts

---

### 5. Clustering (`clustering`)

**What it does:** Groups bars into regimes using distance-based clustering (e.g. KMeans on return/volatility features). Does not assume a hidden Markov structure.

The scaler, optional PCA, cluster model, and canonical state ordering are fitted
on the full requested window. Clustering output is descriptive; reproduce live
behavior with rolling `--end` cutoffs. `min_regime_bars` confirmation itself is
causal and reports a terminal candidate that is still awaiting confirmation.

**Example:**
```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --method clustering --params "n_states=3"
```

Use `n_states` to choose the number of regimes; legacy aliases are ignored.

### 6. Gaussian mixture (`gmm`)

`gmm` is independent Gaussian soft clustering. It has no transition matrix or
Markov persistence and must not be interpreted as an HMM state filter.

**When to use:**
- Exploratory regime discovery without a parametric model
- When HMM convergence is unstable

---

## Practical Strategies

### Strategy 1: Regime Filter

Only enter trend-following trades when HMM detects high-volatility state:

```bash
# Check current regime
mtdata-cli regime_detect EURUSD --timeframe H1 --method hmm --params "n_states=2"

# If state_sigma shows current state is high-volatility:
#   → Enable trend-following entries
# If current state is low-volatility:
#   → Disable trend-following or switch to mean reversion
```

### Strategy 2: Change-Point Alert

Monitor for regime transitions and reduce exposure when detected:

```bash
# Check for recent change points
mtdata-cli regime_detect EURUSD --timeframe H1 --method bocpd --threshold 0.6 --lookback 24

# If transition_summary.max_transition_probability >= 0.6:
#   → Tighten stops
#   → Reduce position size
#   → Consider closing positions
```

### Strategy 3: Regime-Conditional Barrier Analysis

Run barrier optimization separately for each regime:

```bash
# In low-volatility regime: tighter barriers
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --params "tp_min=0.15,tp_max=0.5,sl_min=0.1,sl_max=0.4"

# In high-volatility regime: wider barriers
mtdata-cli forecast_barrier_optimize EURUSD --timeframe H1 --horizon 12 --params "tp_min=0.5,tp_max=2.0,sl_min=0.3,sl_max=1.5"
```

---

## Output Contract

### Compact TOON Output (Default)
```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --method hmm
```
Shows the current regime plus the last `max_regimes` segment rows (start, end,
duration, state). Standard output uses the same segment cap; `detail=full`
returns the complete segment history.

### JSON Output
```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --method hmm --json
```
Returns structured JSON for programmatic use.

Canonical fields for successful compact/full JSON responses:

| Field | Applies to | Notes |
|-------|------------|-------|
| `method` | all methods | Actual implementation method. `hmm` and `gmm` are distinct. |
| `current_regime` | all methods | Uses `regime_id`, `label`, `since`, `bars`, and `regime_confidence` when those concepts apply. Rule-based output instead marks `classification_scope=aggregate_window` and boundary/persistence as `not_estimated`. BOCPD also includes transition-oriented fields such as `status` and `transition_risk`. |
| `classification_window` | rule-based | Exact start, end, and bar count used for the aggregate classification. These bounds are not a detected regime onset or duration. |
| `transition_summary` | BOCPD | Recent-window transition maximum, accepted/raw/filtered change-point counts, activity, and calibration status. Raw `cp_prob` requires `detail=full` with `include_series=true`. |
| `regimes` | segmentation methods | Uses `start`, `end`, `bars`, and `regime_confidence` consistently where regime confidence applies. Omitted for rule-based classification. |
| `regime_info` | state/rule methods | Describes regime labels and statistics. Clustering labels are derived from return/volatility when available instead of opaque `regime_N` names. |
| `reliability` | all methods | Always includes `confidence`, `reliability_label`, and `source`; method-specific diagnostics may add more fields. |
| `historical_label_scope` | full-window fitted methods | Marks historical labels as retrospective; use `point_in_time_guidance` for rolling `--end` evaluation. |
| `warnings` | as needed | Explains accepted parameters that do not apply to the selected method. |

`current_regime.regime_confidence` and `regimes[].regime_confidence` are the canonical regime-confidence keys. Reliability diagnostics keep their own `reliability.confidence` field.

**Method-specific meaning of `regime_confidence`:**

| Method | Meaning |
|--------|---------|
| `hmm`, `gmm`, `ms_ar`, `clustering`, … | Average posterior probability of the assigned state over the segment (classification confidence). |
| `bocpd` | Segment **stability**: `1 - mean(change-point probability)` over bars in the segment. High values mean low internal CP mass (stable stretch). Boundary evidence is separate: `transition_prob_at_start` and `avg_transition_prob`. |

Do not treat BOCPD `regime_confidence` as "how sure we are of a bullish/bearish label"; BOCPD segments are change-point intervals, and bias labels (when present) come from post-change return statistics.

Volatility words in state labels (`low_vol`, `high_vol`, `quiet`, and
`volatile`) rank the fitted states within that run. They are not absolute
thresholds and should not be compared across symbols or timeframes. Use the
numeric `volatility` / `volatility_pct` fields for quantitative comparisons.

### Parameter applicability

Use `n_states` as the canonical state-count parameter for HMM/GMM, MS-AR, clustering, GARCH, wavelet, and ensemble.

`threshold` only applies to BOCPD change-point detection. If supplied for non-BOCPD methods, the tool reports a warning rather than silently changing confidence filtering.

For `rule_based`, use `params.window_bars` to choose the analysis window. The
method classifies the aggregate window; it does not detect when that label began,
how long it has persisted, historical segments, or a per-bar state series. Read
the bounds from `classification_window`, not as regime onset/duration.
`params.efficiency_threshold` is calibrated to a 160-bar reference window. The
detector scales the effective threshold by the square root of the actual window
length and reports it as `params_used.effective_efficiency_threshold`, keeping
the rule comparable when less or more history is analyzed.
`lookback`, `min_regime_bars`, `max_regimes`, and `include_series` do not add
segmentation semantics; inapplicable uses are reported in `warnings`.

### Heuristic state counts

Some methods use a heuristic to choose `n_states` when it is not explicitly supplied:

| Method | Auto-detection basis |
|--------|----------------------|
| `garch` | Realized-volatility percentile spread plus return kurtosis |
| `ensemble` | Return-distribution kurtosis |

GARCH selects four states when the 90th/10th realized-volatility percentile
ratio is above 10 or raw return kurtosis is above 6; it selects three when the
ratio is above 5 or kurtosis is above 4, and two otherwise. With 10 or fewer
usable realized-volatility observations, it defaults to three states.

The ensemble uses a separate rule based only on raw return kurtosis: above 6
selects six states, above 4.5 selects five, above 3.5 selects four, and all
other values select three.

These rules control output granularity; they are not AIC/BIC model selection and
do not estimate the true number of market regimes. Set `n_states` explicitly and
compare out-of-sample or backtest results when the state count affects a strategy.
GARCH and ensemble can choose different counts on the same data because their
heuristics use different inputs.

The ensemble defaults to `hmm`, `clustering`, and `wavelet`. Ensemble voters
are mapped into shared return-quantile centroid bins before voting, so supported overrides are
`hmm`, `gmm`, `ms_ar`, `clustering`, and `wavelet`. BOCPD/PELT change points,
rule-based labels, and GARCH volatility tiers are different concepts and are
rejected as ensemble voters instead of being mapped into the same state-ID
space. `ensemble_info` reports each voter's emitted state count, state mapping,
and the `return_quantile_centroids` alignment mode. This is a directional-return
consensus heuristic, not proof that methods discovered an identical latent
partition; use volatility-focused methods separately as transition or volatility
gates.

`method=all` fits each detector once. Its ensemble is aggregated from the
already-returned full state series for `hmm`, `gmm`, `ms_ar`, `clustering`, and
`wavelet`; it does not refit those voters. `ensemble_health` and
`runtime.ensemble_voters` list requested, used, and excluded voters. If a
successful first-pass detector cannot contribute a compatible state series,
the response sets `ensemble_degraded=true` and `partial_results=true` even
though the other per-method results remain usable.

#### Clustering reliability

The `clustering` method learns regimes in its scaled feature space (after PCA
when dimensionality reduction is active), not from the one-dimensional return
series alone. Its `reliability.source` is `feature_cluster_separation`, and
`feature_variance_ratio` reports the share of total feature-space variance
explained by the final labels. `reliability.confidence` is the corresponding
bounded score.

The scaler, PCA transform, and clusters are fitted over the full analysis
window. Treat the resulting labels and reliability as retrospective. For
live-style evaluation or backtests, roll the window and advance `--end` rather
than fitting once on the complete history.

#### Wavelet boundaries and causality

The `wavelet` method uses symmetric finite-signal extension and reports
`params_used.boundary_mode=symmetric`. This avoids circular head-to-tail
coupling at the series boundary, although ordinary endpoint effects can still
affect the first and last coefficients.

Wavelet energy features use trailing windows, but the scaler and K-means model
are fitted over the full analysis window (`model_fit_scope=full_window`). The
reported labels are therefore retrospective. Roll `--end` through history for
an honest live-style or backtest evaluation.

#### GARCH volatility tiers

The `garch` method fits a single GARCH model to estimate conditional
volatility, then bins that path into ordered volatility tiers. By default the
cut points are percentiles of the full analysis window, so labels such as
`low_vol` and `high_vol` are relative within that run and can change when the
window changes. With two states, `params.vol_threshold` selects an explicit
absolute cut point instead. Full output reports `threshold_scope` and
`volatility_thresholds` in `params_used`; compact and full output preserve the
retrospective classification contract in `historical_label_scope`.

This method is not a Markov-switching GARCH model and does not estimate latent
regime dynamics. AIC, BIC, and log likelihood are exposed as `model_fit`
diagnostics only; they are not converted into a generic regime-confidence
score.

### Richer Sections
```bash
mtdata-cli regime_detect EURUSD --timeframe H1 --method hmm --detail full --json
```
Adds supported richer sections such as metadata and diagnostics.

---

## Quick Reference

| Task | Command |
|------|---------|
| Classify regimes (2 states) | `mtdata-cli regime_detect EURUSD --method hmm --params "n_states=2"` |
| Classify regimes (3 states) | `mtdata-cli regime_detect EURUSD --method hmm --params "n_states=3"` |
| Cluster return distributions without Markov persistence | `mtdata-cli regime_detect EURUSD --method gmm --params "n_states=2"` |
| Detect change points | `mtdata-cli regime_detect EURUSD --method bocpd --threshold 0.5` |
| Segment structural breaks | `mtdata-cli regime_detect EURUSD --method pelt --params "model=rbf penalty=auto"` |
| Markov-switching AR | `mtdata-cli regime_detect EURUSD --method ms_ar --params "n_states=2"` |
| Clustering | `mtdata-cli regime_detect EURUSD --method clustering --params "n_states=3"` |

---

## See Also

- [GLOSSARY.md](../GLOSSARY.md) — Term definitions
- [FORECAST.md](../FORECAST.md) — Price forecasting
- [VOLATILITY.md](VOLATILITY.md) — Volatility estimation
- [BARRIER_FUNCTIONS.md](../BARRIER_FUNCTIONS.md) — TP/SL analysis
