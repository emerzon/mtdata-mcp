# BTCUSD Iteration 4: return-direction preregistration

**Audience:** Contributor

This document freezes the fourth development iteration before its first MT5
feature-model call. The purpose is to test whether a small, causal MLForecast
pipeline can forecast the sign of a fixed-horizon BTCUSD return with useful
abstention. It is not a promise of trading profitability or production-ready
confidence.

**Related:** [Research ledger](BTCUSD_FORECAST_RESEARCH.md) ·
[Backtesting](forecast/BACKTESTING.md) · [Uncertainty](forecast/UNCERTAINTY.md)

> **Safety:** All commands are read-only, use broker `BTCUSD` data supplied by
> MT5, and run in isolated model stores. Externally pretrained weights and all
> `trade_*` commands are forbidden.

## Evidence boundary

Iteration 3 opened model outcomes through 2024-06-30. In particular, its
2023-07 through 2024-06 baseline results influenced the decision to model
returns rather than price levels. That period is therefore a previously opened
stress window, not a fresh holdout for this iteration. Files from the denoising
B run were generated before its selection rule was registered and remain
unparsed; they cannot support a confirmatory claim.

The time roles are frozen as follows:

| Role | Dates | Permitted use |
|---|---|---|
| A1 development screen | 2022-07-01 through 2022-12-31 | Candidate screening and abstention-threshold selection |
| A2 development confirmation | 2023-01-01 through 2023-06-30 | Confirm at most three horizon finalists and select at most one global finalist |
| Previously opened stress | 2023-07-01 through 2024-06-30 | Descriptive robustness and rejection only; never promotion evidence |
| V locked validation | 2024-07-01 through 2025-06-30 | First unopened evaluation of one frozen finalist |
| C locked confirmation | 2025-07-01 through 2026-06-30 | Second unopened evaluation if V passes |
| H single-use holdout | 2026-07-01 through 2026-08-26 | Remains sealed until point, interval, and operational gates are complete |

No result after 2023-06-30 may alter the model, features, lags, horizon,
threshold, cost assumptions, or decision rule. A later window may only reject
the frozen candidate.

## Target and decision

The native target is H1 `quantity=return` at horizons 6, 12, and 24. For an
anchor and horizon H:

- forecast score: `s_H = exp(sum(forecast_return)) - 1` over all H steps;
- realized terminal return: `r_H = exp(sum(actual)) - 1` over all H steps;
- long when `s_H > tau`, short when `s_H < -tau`, otherwise abstain;
- primary statistical endpoint: balanced accuracy on called anchors;
- primary economic endpoint: mean fixed-H net return at 5 bps round trip.

The external fixed-H scorer enters at the next MT5 candle open and exits at the
H-th target candle close. Long gross return is `exit / entry - 1`; short gross
return is `entry / exit - 1`. It subtracts the declared round-trip cost once.
The built-in first-hit trading metric is reported for context only and cannot
select a candidate.

Secondary endpoints are path log-return RMSE/MAE, terminal-return MAE,
ordinary accuracy, sensitivity, specificity, coverage, predicted-long share,
win rate, compounded net return, and maximum drawdown. Economic results are
reported at 1.625, 5, and 10 bps. Swap, financing, funding, and market impact
are unavailable from the historical candle contract and remain unmodeled.

## Frozen candidates

Every model uses a 720-H1-bar rolling input window and seed 42. The adapters
already fix their estimator random state to 42.

| Adapter | Parameters |
|---|---|
| `mlf_lightgbm` | `n_estimators=50`, `learning_rate=0.05`, `num_leaves=15`, `max_depth=5`, `lags=[1,2,3,6,12,24]` |
| `mlf_rf` | `n_estimators=100`, `max_depth=8`, `lags=[1,2,3,6,12,24]` |

Each adapter/horizon is evaluated with exactly two causal feature families:

1. Momentum: `{"indicators":"rsi(14),roc(12)","observed_future_policy":"carry_forward"}`
2. Volatility/time: `{"indicators":"natr(14)","future_covariates":["hour","dow"],"observed_future_policy":"carry_forward"}`

Indicators are calculated only from MT5 OHLC history. Training indicators are
shifted one bar. Their latest observed values are carried across the forecast
horizon; they are not treated as known-future values. `hour` and `dow` produce
UTC `hr_sin`, `hr_cos`, `dow_sin`, and `dow_cos` values from target timestamps.
Indicator periods are explicit because backend defaults can vary.

No denoising, dimensionality reduction, target transform, hyperparameter
tuning, generic dotted-class adapter, or external data is permitted. Each
adapter/horizon also receives a separate raw-univariate run with identical
parameters and no `--features`. `sf_zeromodel`, always-up, always-down, and the
A-learned majority class are structural controls.

## Staged execution

1. Run one H12 anchor for each of the four adapter/feature-family pairs. Stop
   unless all consumption and artifact-isolation checks pass.
2. Screen A1 at 26 weekly origins: 12 feature candidates, six matched raw
   controls, and three zero-return controls. Threshold is zero.
3. Retain at most one feature candidate per horizon, then run it and its
   controls at daily origins on A1. Choose the threshold here only.
4. Hash and freeze the at-most-three candidate configurations and thresholds.
5. Evaluate them unchanged at daily origins on A2. Select and hash at most one
   global finalist using the ordering below.
6. Score the finalist on the previously opened stress window. It may reject the
   finalist but cannot strengthen its evidence or change it.
7. If still eligible, open V for the frozen finalist, its matched raw control,
   and the zero control. If V passes, repeat unchanged on C.
8. Keep H sealed until a timestamp-aligned interval replay verifier exists and
   the point candidate passes all earlier windows.

Weekly and daily origins are spaced 168 and 24 H1 bars respectively. This makes
all H24 terminal paths non-overlapping. Half-year dense runs are split when
needed to stay below the 200-anchor command limit; duplicate boundary anchors
are disclosed and removed before scoring.

## Consumption and integrity gates

Every feature command uses `detail=full` and must satisfy all of these checks:

- requested, planned, succeeded, and complete anchor counts agree exactly;
- method status is `complete`, with zero partial and failed anchors;
- `feature_usage.status=consumed`, both consumption booleans are true, and
  `anchors_verified` equals the planned anchor count;
- every anchor's historical feature rows equal `target_points_used`; future
  rows equal H; prepared and consumed feature counts agree;
- the momentum family resolves semantically to RSI(14) and ROC(12); the
  volatility/time family resolves to NATR(14) and all four calendar columns;
  backend-specific casing is accepted and the actual names are retained;
- `training_bars_used=720`; every forecast and actual vector has H finite
  values; candidate and controls have identical anchors and actual outcomes;
- observed lag is one bar and policy is exactly `carry_forward`;
- raw controls have no feature-usage evidence and consume zero exogenous
  columns;
- each command starts with an empty, command-specific `MTDATA_MODEL_STORE`;
  transient model files are inventoried and excluded from the evidence bundle;
- no source candle later than the registered window end is used;
- a feature forecast identical to its raw control at every matched anchor is a
  suspected no-op and stops the matrix.

The run saves the exact argv, source commit, dirty-state check, feature catalog,
`indicators_describe` output, backend versions, raw response, normalized ledger,
and SHA-256 hashes. A failed gate stops later MT5 calls until the cause is fixed
and committed.

## Screening and threshold rules

An A1 weekly candidate can be the sole dense finalist for its horizon only if:

- balanced accuracy is at least 0.52 and at least 0.02 above its matched raw
  control on the same anchors;
- it predicts both classes; and
- path RMSE is no more than 5% worse than raw.

Ties are resolved by the week-cluster-bootstrap balanced-accuracy lower bound,
then the raw-control delta, lower RMSE, and finally the lower-complexity adapter
(`mlf_rf` before `mlf_lightgbm`). All 12 screen comparisons and Holm-adjusted
p-values are reported even though A1 is a selection stage.

For each dense A1 finalist, candidate thresholds are zero and the 25th, 50th,
60th, and 65th percentiles of `abs(s_H)`. A threshold is eligible
only with at least 30% coverage, 60 calls, 20 long calls, 20 short calls, a
predicted-long share from 20% through 80%, and both realized classes. Choose the
eligible threshold with the largest week-cluster-bootstrap lower bound for
balanced accuracy; break ties by the 5-bps net-return lower bound, coverage, and
then the smaller threshold. Freeze zero if no eligible nonzero threshold
improves balanced accuracy over zero by at least 0.02.

The matched raw comparison uses its zero-threshold sign on exactly the
candidate-called anchors. Applying the candidate threshold to raw is reported
separately as a coverage diagnostic and cannot replace this paired comparison.

## Inference and promotion gates

Use 5,000 ISO-week cluster-bootstrap resamples with seed 42. Percentile 95%
intervals are reported for balanced accuracy, mean net return, and paired
deltas. A replicate without both realized classes is invalid; fewer than 95%
valid replicates invalidates the interval. Use paired McNemar tests for
candidate-versus-raw directional correctness and paired sign tests for
absolute-error and net-return deltas. Holm correction covers the 12 A1 feature
candidates and, separately, the at-most-three A2 horizon finalists.

A candidate can leave A2 only if all conditions hold:

- frozen-threshold coverage is at least 30%, with predicted-long share from 20%
  through 80%;
- balanced accuracy is at least 0.55, its bootstrap lower bound is at least
  0.50, and it exceeds matched raw by at least 0.02;
- Holm-adjusted directional `p <= 0.05`;
- path RMSE is no more than 2% worse than raw;
- the 5-bps mean-net-return lower bound is positive;
- directional and 5-bps economic results are nonnegative in both A1 and A2;
- maximum drawdown is no greater than 20%.

If more than one candidate passes, choose the greatest A2 balanced-accuracy
lower bound, then the greatest 5-bps return lower bound, lower RMSE, shorter
horizon, and lower-complexity adapter. The chosen candidate is immutable.

V and C each require the same coverage and class-balance gates, a balanced-
accuracy lower bound above 0.50, a positive 5-bps return lower bound, path RMSE
no more than 2% worse than raw, maximum drawdown no greater than 20%, and
nonnegative results in each half-year subwindow. Failure means rejection: no
threshold adjustment, fallback candidate, reinterpretation, or additional
development search in this iteration.

## Confidence limitation

Passing this iteration would establish only a repeatable point-direction and
abstention candidate on this broker feed. Production confidence intervals,
materialization, the single-use holdout, and shadow-readiness remain disabled
under BTC-R025/BTC-R027 until raw timestamped interval envelopes can be hashed
and replayed against completed MT5 outcomes.
