# BTCUSD forecast research ledger

**Audience:** Contributor

This page tracks product friction found while building a reproducible BTCUSD H1
forecasting study. It separates mtdata defects from broker/feed constraints and
research-protocol limitations so follow-up work can be prioritized without
mistaking an experimental result for a product guarantee.

**Related:** [Forecasting](FORECAST.md) · [Backtesting](forecast/BACKTESTING.md) ·
[Uncertainty](forecast/UNCERTAINTY.md) · [Known limitations](LIMITATIONS.md)

> **Safety:** This study is read-only. Its automation rejects `trade_*` commands
> and uses no live-order execution.

## Experiment contract

- Symbol/timeframe: broker `BTCUSD`, H1.
- Forecast horizons: 6, 12, and 24 completed bars.
- Runtime predictors and model-fitting data: MT5 data only. Externally
  pretrained forecasting weights are excluded from this study.
- Primary continuous-feed sample: 2022-06-01 through 2026-06-30, with
  2026-07-01 through 2026-08-26 locked as a historical holdout.
- Output: expected return/price, calibrated interval, volatility, direction,
  provenance, and abstention when reliability gates fail.
- Artifact root: ignored `backtests/btcusd_forecast/<study_id>/`; existing user
  model artifacts are not reused or modified.

## Product and workflow friction

| ID | Severity | Area | Observation and impact | Current workaround | Suggested follow-up | Status |
|---|---|---|---|---|---|---|
| BTC-R001 | High | History quality | Public candle retrieval cleans malformed, unordered, and duplicate OHLC rows, but the internal forecast history path did not. Seven BTCUSD H1 duplicate UTC groups appeared on Europe/Nicosia DST dates, sometimes with different OHLC values. Multi-year backtests could therefore fit and score duplicate timestamps. | Apply the shared OHLC cleaner inside the internal history gateway before count trimming and cadence checks. | Keep one canonical cleaning path and expose cleaning counts in forecast diagnostics. | Fixed in this change; underlying DST source remains observable |
| BTC-R002 | High | Cadence | BTCUSD H1 requests before about 2017-05 return daily/coarser observations. A wholly coarse window is rejected, but a mixed coarse/hourly range can pass the median-cadence guard. | Enforce a 2017-05-01 research cutoff; treat 2017-05 through 2022-05 as a separate session/feed regime. | Detect cadence regimes or require a minimum matching-interval share across subwindows, not only a global median. | Open |
| BTC-R003 | High | Validation contracts | `target_spec` is accepted by `forecast_generate` but not by backtest, tuning, conformal, or strategy validation. A custom target cannot be selected and validated through one native contract. | Limit the native study to price, return, and volatility quantities. | Add the same target contract to all forecast evaluation tools. | Open |
| BTC-R004 | High | Calibration | Native conformal and `strategy_validate` cannot replay feature-rich or dimensionality-reduced forecast configurations. A stronger feature model cannot receive an apples-to-apples confidence/economic check. | Prefer exactly replayable finalists; use a research-layer split-residual calibrator only if a feature model clears a predeclared materiality gate. | Accept the full forecast pipeline specification in conformal and strategy validation. | Open |
| BTC-R005 | Medium | Training | `forecast_train` cannot express denoising, features, dimensionality reduction, or a custom target. Feature-rich persistent models must be materialized through `forecast_generate --model-cache reuse`. | Store and verify the exact generation request with each artifact. | Unify train and generate pipeline specifications and compatibility hashes. | Open |
| BTC-R006 | Medium | Tuning | Native tuners search method parameters only. Feature, indicator, denoising, target, and dimensionality-reduction combinations need an outer loop. | Use the checked-in experiment driver and retain every CLI invocation/result. | Add a pipeline-level tuning/search manifest and durable experiment registry. | Open |
| BTC-R007 | Medium | Model isolation | Trainable rolling tests and tuners can write to the default model store; not every command has an ephemeral-cache flag. | Set a study-specific `MTDATA_MODEL_STORE` and Optuna database. | Add an explicit ephemeral/store-root option consistently to training and evaluation commands. | Open |
| BTC-R008 | Medium | Installation | The active editable environment exposes `python -m mtdata`, but its `mtdata-cli` executable is absent and pip reports invalid leftover distributions. Copy-paste documentation commands therefore fail in this environment. | Invoke `python -m mtdata` and record the environment warning. | Repair/reinstall the editable environment and add an install smoke check for every declared console script. | Environment issue |
| BTC-R009 | Medium | GPU setup | Both RTX 3090 GPUs are healthy, but `.env` sets `CUDA_VISIBLE_DEVICES=-1`; dependency checks can report CUDA available while the visible device count is zero. | Set process-local `CUDA_VISIBLE_DEVICES=0,1` for approved foundation benchmarks and verify device names before a run. | Report contradictory CUDA availability/device-count states in runtime diagnostics. | Environment issue |
| BTC-R010 | Medium | Backtest semantics | Built-in trade metrics enter at the next open and exit at the first close reaching the terminal forecast, otherwise at the horizon, with no stop. Those metrics do not represent a conventional fixed-horizon forecast strategy. | Use MAE/RMSE/directional metrics for screening and independently score fixed-horizon outcomes; reserve `strategy_validate` for compatible frozen candidates. | Make execution policy prominent and offer a fixed-horizon exit policy. | Open |
| BTC-R011 | Medium | Experiment tracking | General CLI results are stdout JSON/NDJSON; there is no cross-command study ledger, immutable holdout lock, or resume mechanism. | Persist manifests, raw responses, normalized results, failures, and decisions in the experiment driver. | Add a first-class read-only research-run registry. | Open |
| BTC-R012 | Low | Scale | Rolling backtests are capped at 200 anchors, which cannot cover long H1 periods uniformly at all three horizons in one command. | Use chronological monthly shards with horizon-spaced anchors and aggregate outside the CLI. | Support explicit anchor lists or streaming/windowed evaluation. | Open |
| BTC-R013 | Low | Contributor setup | Calling `python-dotenv`'s parameterless `load_dotenv()` from a stdin script raises an assertion in `find_dotenv()` in this Python 3.14 environment. This affected a direct gateway integrity probe, not normal CLI use. | Pass `dotenv_path='.env'` explicitly or use the CLI bootstrap. | Keep contributor snippets explicit about the dotenv path and track the upstream compatibility issue. | Environment issue |
| BTC-R014 | Medium | Reproducibility | Backtest JSON records method names and some resolved execution details but does not echo the requested `quantity`, denoise/features/dimensionality-reduction specs, or complete per-method parameter map. Saved output alone cannot reconstruct the run. | Hash and save the exact argument vector plus a preregistered manifest beside every raw response. | Return a canonical effective request/pipeline contract in every backtest and tuning result. | Open |
| BTC-R015 | Medium | Error diagnostics | If quality cleaning leaves too few valid rows, the gateway has structured removal counts but forecast/backtest replace them with a generic “not enough closed bars” error. The cleanup cause is lost on the failure path where it matters most. | Treat any such experiment command as failed and retain the raw gateway audit beside it. | Include `history_quality`, warnings, and a stable insufficient-valid-history error code in forecast/backtest failures. | Open |
| BTC-R016 | Low | Cross-tool diagnostics | Volatility and pattern consumers benefit from cleaned history but rebuild/reduce frames in ways that discard `DataFrame.attrs`, so they do not expose the new quality metadata. | Gate data quality in the separate audit stage before invoking those consumers. | Define a shared quality context that survives conversions and expose it consistently across analysis tools. | Open |
| BTC-R017 | High | Backtest status | All 12 expanded-screen responses reported `success=true` and `methods_failed=0`, although ARIMA failed to converge at 31 of 5,616 planned method-anchor fits. A consumer checking only root status could silently rank partial and complete samples together. | Derive failures from every method's `successful_tests` versus `num_tests`, exclude partial candidates, and compare only matched anchors. | Raw and compact contracts now expose anchor counts, method `complete\|partial\|failed` status, `complete_success`, warnings, and separate complete/partial method counts. Compact ranking, tuning fitness, report selection, and the research harness reject partial coverage unless a caller explicitly recomputes scores on a common-anchor set. | Fixed in this change |
| BTC-R018 | High | Feature contracts | Forecast preprocessing and user docs accept `features.include`, but `DataPreparationContract.uses_feature_inputs()` did not recognize it. This could bypass dimensionality and model-capability validation even though the feature matrix was built. | Use `exog` in experiment manifests and verify the effective feature columns in full-detail output. | Keep the preprocessing and compatibility vocabularies identical and cover every public alias in contract tests. | Fixed in this change |
| BTC-R019 | High | Feature consumption | The engine could build and report an exogenous matrix for a method that never consumed it; classical Holt/Theta/naive/Fourier, analog, Monte Carlo, and TimesFM paths accepted the call but ignored `exog_used`. A successful feature run could therefore be mislabeled as a feature model. | Backtests now allow features only when every selected method declares audited historical and future exogenous consumption; run univariate controls separately. Direct generation and other adapters remain source-audited only. | Extend the same fail-closed capability and runtime-attestation contract to every feature-bearing forecast, tuning, and validation surface before marking those adapters supported. | Backtest fixed for MLForecast; other surfaces open |
| BTC-R020 | Medium | Dimensionality reduction | Forecast PCA is fitted without feature standardization. Broker tick activity can dominate price-normalized indicators, so components may mostly encode scale rather than joint structure. | Treat PCA as diagnostic, skip it unless unreduced features improve first, and inspect effective columns/components. | Add an explicit train-window-fitted scaler stage and persist its configuration and diagnostics with the reducer. | Open |
| BTC-R021 | High | Interval calibration | Residual-quantile intervals silently skipped failed or path-incomplete calibration anchors. A run could retain enough residuals to report usable bounds while conditioning on an undisclosed successful-fit subset. | Require complete anchor coverage for decision-use intervals; retain any partial calibration bounds as diagnostics only. | The interval contract now reports planned/succeeded/failed anchor counts, `calibration_complete`, an explicit incomplete-coverage status, remediation, and a trust blocker. | Fixed in this change |
| BTC-R022 | Medium | Web UI | Forecast backtest UI types and summaries do not yet display the new complete/partial method status or anchor-failure counters. CLI/JSON users see the safety contract, but a UI user can miss it. | Inspect raw or compact CLI JSON and require `complete_success=true` before selection. | Add typed fields, a visible partial-result warning, and incomplete-method ranking suppression to the Web UI. | Open |
| BTC-R023 | High | Interval usability | The first review gate required near-nominal coverage and a positive mean width but no preregistered upper relative-width bound. Trivially wide, unusable intervals could therefore pass. | Require a candidate-locked dimensionless width ceiling, exact aggregate and per-window counts, and Wilson lower-bound checks before considering interval evidence. | The harness now requires an immutable `mean_relative_width`, its definition, a candidate-level ceiling capped at 10%, exact weighted aggregation, and per-window plus aggregate coverage checks. Production approval remains disabled under BTC-R025. | Fixed in harness; approval blocked by BTC-R025 |
| BTC-R024 | High | Data scope | The initial protocol text allowed disclosed generic pretrained forecasting weights, which conflicts with the stricter requirement that the model rely only on MT5-provided data. | Exclude Chronos, TimesFM, and other externally pretrained forecasting candidates. | The harness rejects externally pretrained methods in candidate, screen, tuning, and provenance inputs and records the MT5-only restriction in its manifest. | Fixed in harness |
| BTC-R025 | Critical | Interval evidence integrity | The first external interval-review path syntax-checked `raw_sha256` labels and trusted self-reported sample, coverage, and width summaries. It did not locate and hash the named raw forecast/actual envelopes or recompute those statistics, so an apparently passing review could be unsupported by MT5 output. | Production interval approval is fail-closed. Validation may write a development assessment, but review approval, freeze, locked holdout, materialization, and shadow forecasting are disabled with `BTC-INTERVAL-VERIFIER-REQUIRED`; no holdout lock is opened. | Build a first-class replay verifier that resolves the exact mtdata envelopes, checks their hashes and candidate/window identities, and causally recomputes every per-anchor and aggregate coverage/width statistic before enabling approval. | Safety guard fixed; verifier open |
| BTC-R026 | High | Trainable features | The MLForecast adapter required future exogenous values during `train()`, although the engine correctly supplies historical features to fitting and future features only to prediction. Fresh `mlforecast`, `mlf_rf`, and `mlf_lightgbm` feature forecasts therefore failed before fitting. | Pause feature-model research and reproduce the failure without MT5. | Training now accepts historical exogenous data alone; historical/future parity is checked when both legitimately meet at prediction. Future arrays remain outside model-cache identity. | Fixed in this change |
| BTC-R027 | High | Interval replay contract | Native conformal output contains one cutoff's forecast bounds and aggregate calibration diagnostics, while full backtest output contains point forecast/actual vectors but neither interval bounds nor realized target timestamps. Those outputs cannot independently prove a timestamp-aligned out-of-sample coverage ledger. | Keep interval approval disabled. A future harness replay must preregister conformal cutoffs, preserve each raw envelope, then fetch completed MT5 candles and join actual closes to exact forecast target timestamps. | Expose a first-class timestamped per-anchor interval/outcome ledger, or add realized target timestamps and interval bounds to a complete native replay contract. | Open |
| BTC-R028 | High | Feature evidence | Full backtest output discarded the engine's preparation and runtime-consumption diagnostics. The first installed-library MLForecast smoke returned finite forecasts, but the saved result could not prove that RSI/EMA reached fitting and prediction, so feature research had to stop. | Pause MT5 research until the output itself can be verified. | Full detail now reports per-anchor and invariant method-level selected columns, row/feature counts, lag/policy, effective params, and post-call fit/predict consumption. Compact output keeps bounded counts only. Missing or inconsistent evidence fails the anchor with `feature_consumption_unverified`. | Fixed in this change |
| BTC-R029 | Medium | Feature documentation | CLI help advertised unimplemented feature `lag`/`rolling` keys, while examples used a non-consuming theta method, an excluded `close` column, or H>1 observed features without the required carry-forward policy. Copy/paste runs either failed or appeared multivariate without being so. | Use explicit JSON, a compact indicator string, audited catalog flags, and `detail=full`; keep model lags in `--params`. | Generate feature examples and accepted keys from the preprocessing contract and capability registry. | Examples and help fixed in this change |
| BTC-R030 | Medium | Indicator provenance | Forecast preprocessing preserves backend-specific indicator names and values but backtest output does not identify the pandas-ta/TA-Lib engine. Candle docs previously implied lowercase names everywhere, and the documented NATR default differed from runtime metadata. | Specify every period, save `indicators_describe --detail full` plus package versions, and gate on actual `feature_usage.selected_columns` without hardcoded casing. | Propagate `indicator_engine` and resolved indicator specifications into forecast diagnostics and the model/research fingerprint. | Open |

## Broker and data constraints

| ID | Constraint | Research treatment |
|---|---|---|
| BTC-D001 | BTCUSD candles are broker bid-chart CFD data, not a consolidated exchange market. | Report broker/server provenance with every result and avoid claims about the global BTC market. |
| BTC-D002 | Candle `tick_volume` is the number of bid updates, and available ticks are quote-only with no last-trade volume. | Treat it as broker-liquidity activity, never exchange volume. |
| BTC-D003 | Tick retrieval is limited to recent history and a bounded row count. | Use ticks for recent spread/microstructure estimates, not as a long-history predictor. |
| BTC-D004 | Historical per-bar spread coverage is incomplete in some windows. | Never convert missing spread to zero; use disclosed recent p75/p95 fixed-cost scenarios when coverage is insufficient. |
| BTC-D005 | The feed changes from recurring weekend/session closures to effectively continuous 24/7 around June 2022. | Use the later period for primary selection and the earlier period only as a separate stress regime. |

## Development checkpoints

### Expanded raw baseline: 2022-06 through 2024-06

The second development-only screen used two disjoint 52-anchor blocks, weekly
origins, a 720-bar lookback, H1 horizons of 6, 12, and 24 bars, and explicit
1.625 bps round-trip known costs. It evaluated 5,616 method-anchor fits. Of
those, 5,585 completed and 31 ARIMA fits did not converge.

No candidate passed the preregistered two-block error-stability gate. The only
unadjusted directional lead was price Holt at H24: 64/104 terminal directions
correct (61.54%; Wilson 95% interval 51.94%–70.32%). It nevertheless increased
RMSE relative to naive by 2.21% in the first block and 10.06% in the second,
and its fixed-horizon average net return fell from 69.14 bps to 2.56 bps.
It also trailed the always-up majority rule in the second block (59.62% versus
63.46%). Its unadjusted direction test had p=0.0237, but Holm adjustment was
p=0.2128 within the nine price-H24 calls and p=1.0 across the full screen.
It therefore remains a development lead rather than evidence of forecast skill.

Return AutoETS at H12 reduced zero-return RMSE by only 0.15% and 0.27% in the
two blocks. The pooled improvement (0.17%) is too small to be practically
useful, and its paired error wins (53/104) do not support a skill claim.

The local evidence bundle is stored under
`backtests/btcusd_forecast/20260827_iter2_dev_expanded/`. It contains raw
full-detail results, exact invocations, source fingerprints, normalized and
fixed-horizon ledgers, the BTC-R017 reproduction record, and verified SHA-256
hashes. Transient trained-model caches are deliberately excluded.

## Reproduction notes

Use the module entry point in the current environment:

```powershell
python -m mtdata forecast_list_methods --profile all --limit 500 `
  --show-unavailable true --detail full --json

python -m mtdata forecast_backtest_run BTCUSD --timeframe H1 `
  --end 2024-06-30 --steps 3 --spacing 6 --horizon 6 --lookback 336 `
  --methods naive drift theta --quantity return --spread-bps 0.625 `
  --commission-bps-per-side 0 --slippage-bps 0.5 --detail full --json
```

The three-anchor command is only a contract smoke test. Its sample is far too
small for a performance conclusion.
