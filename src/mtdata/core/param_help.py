"""Transport-neutral per-(command, parameter) help overrides."""

from typing import Dict

COMMAND_PARAM_HELP_OVERRIDES: Dict[tuple[str, str], str] = {
    ("market_depth_fetch", "spread"): (
        "Boolean output control. Compute and include bid/ask spread metrics from "
        "broker DOM or the fallback quote; disabled by default."
    ),
    ("forecast_train", "wait"): (
        "Wait for training to finish. One-shot CLI and stdin shell batches "
        "always wait so the in-process worker stays alive (CLI default: true); "
        "the flag only applies in interactive shell, MCP, and Web API sessions "
        "(those default: false)."
    ),
    ("forecast_backtest_run", "slippage_bps"): (
        "Per-side slippage in basis points (1 bp = 0.01%). Simulated trades "
        "enter at the next bar's open and exit at the first close that reaches "
        "the terminal forecast, otherwise at the horizon. There is no stop-loss; "
        "compact results include execution_policy."
    ),
    ("forecast_tune_optuna", "n_trials"): (
        "Optuna trial count. Each trial runs --steps rolling backtests; the "
        "defaults evaluate 40*5=200 rolling backtests."
    ),
    ("forecast_tune_optuna", "timeout"): (
        "Optional wall-clock search limit in seconds."
    ),
    ("forecast_tune_genetic", "max_search_time_seconds"): (
        "Optional wall-clock search limit in seconds; returns the best completed "
        "candidate with partial-search accounting."
    ),
    ("forecast_optimize_hints", "timeframes"): (
        "Timeframes to evaluate. The default searches H1, H4, D1, and W1; "
        "pass one timeframe for a cheaper exploratory run."
    ),
    ("forecast_optimize_hints", "population"): (
        "Population size (minimum 2). With default generations and steps, "
        "the search evaluates about 190 rolling backtests."
    ),
    ("forecast_optimize_hints", "generations"): (
        "Generation count; work grows with population*generations*steps."
    ),
    ("forecast_optimize_hints", "max_search_time_seconds"): (
        "Optional wall-clock search limit in seconds."
    ),
    ("data_fetch_candles", "timestamp_format"): (
        "Format each candle's `time` value: iso for ISO in CLIENT_TZ "
        "(iso_utc / iso_offset in the payload), iso_utc for UTC Z strings, "
        "or epoch for UTC epoch seconds."
    ),
    ("data_fetch_ticks", "timestamp_format"): (
        "Format each MT5 tick event's `time` value as ISO in CLIENT_TZ "
        "(iso_utc / iso_offset in the payload), iso_utc for UTC Z strings, "
        "or UTC epoch seconds."
    ),
    ("data_fetch_ticks", "start"): (
        "Inclusive range start (dateparser). Date-only and calendar phrases "
        "are UTC midnight, not the broker D1 session open. Example: "
        "--start 2026-08-14 begins at 2026-08-14T00:00:00Z."
    ),
    ("data_fetch_ticks", "end"): (
        "Inclusive range end (dateparser). Date-only and calendar phrases are "
        "UTC end-of-day, not the broker session close."
    ),
    ("market_ticker", "price_field"): (
        "Omit for the default bid/ask/spread quote snapshot; set bid, ask, mid, "
        "last, or spread for a single-price response."
    ),
    ("patterns_detect", "timeframe"): (
        "Chart timeframe. When omitted, candlestick/classic/harmonic/fractal use "
        "H1, elliott scans H1/H4/D1, and all scans M30/H1/H4/D1/W1."
    ),
    ("patterns_detect", "include_completed"): (
        "Include completed lifecycle structures alongside forming results. "
        "Candlestick mode always scans closed-bar detections; use --last-n-bars "
        "to restrict their recency. Harmonic mode always returns both states."
    ),
    ("patterns_detect", "top_k"): (
        "Detector candidate/collision budget and compact, summary, or standard "
        "row cap. Full detail returns every surviving row; for candlesticks this "
        "still caps competing pattern types per bar. Use --last-n-bars to bound "
        "the full candlestick scan window."
    ),
    ("symbols_list", "universe"): (
        "Symbol scan universe. When omitted, unfiltered listings use visible "
        "Market Watch symbols while searches use the full broker catalog."
    ),
    ("volume_profile_levels", "source"): (
        "Profile input. auto uses bounded raw ticks when coverage is adequate, "
        "then falls back to the labeled M1-bar approximation for oversized "
        "windows, failed tick fetches, or poor tick-price coverage."
    ),
    ("forecast_list_library_models", "limit"): (
        "Maximum models to return on this page. Omitted compact output uses "
        "20; omitted full output is unbounded."
    ),
    ("forecast_models_delete", "dry_run"): (
        "Preview the exact model and its metadata without deleting it; defaults to true."
    ),
    ("forecast_models_delete", "confirm_model_id"): (
        "Exact full model ID required with --dry-run false for permanent deletion."
    ),
    ("correlation_matrix", "method"): "Correlation coefficient: pearson or spearman.",
    ("correlation_matrix", "transform"): (
        "Price transform: log_return, pct, diff, level, or log_level."
    ),
    ("cross_correlation", "method"): "Correlation coefficient: pearson or spearman.",
    ("cross_correlation", "transform"): (
        "Price transform: log_return, pct, diff, level, or log_level."
    ),
    ("stationarity_test", "tests"): (
        "Comma-separated stationarity tests: adf, kpss, pp. "
        "Example: --tests adf,kpss."
    ),
    ("stationarity_test", "trend"): (
        "ADF/KPSS/PP deterministic term: c for constant, ct for constant+trend."
    ),
    ("denoise_describe", "method"): (
        "Denoise method to describe. Run denoise_list_methods to list methods "
        "available in this installation."
    ),
    ("trade_var_cvar_calculate", "method"): (
        "Tail-risk method: historical, parametric, cornish_fisher, or ewma."
    ),
    ("trade_var_cvar_calculate", "symbol"): (
        "Optional scope: calculate VaR/CVaR for currently open positions in this "
        "symbol. Omit it for the full open portfolio."
    ),
    ("trade_var_cvar_calculate", "transform"): (
        "Return transform: log_return (aliases log_returns/log) or pct "
        "(aliases pct_return/percent/simple_return)."
    ),
    ("data_fetch_candles", "indicators"): "Technical indicators. Catalog names come from indicators_list / indicators_describe (rsi, not rsi_14). On PowerShell, quote parenthesized specs such as --indicators \"rsi(14)\", or use shell-safe fetch specs rsi_14 / sma=20. JSON arrays like '[{\"name\":\"rsi\",\"params\":[14]}]' and named params like rsi(length=14) also work. Use params syntax, not sma,20. Output columns such as rsi_14 are backend-derived.",
    ("data_fetch_candles", "limit"): (
        "Maximum returned bars. Latest queries default to 20 most-recent bars. "
        "Explicit --start/--end ranges also default to a 20-bar page and return a "
        "continuation cursor when more bars remain. Queries with --start retain the "
        "earliest matching bars (first-N); otherwise the latest "
        "bars are retained. Indicator warmup bars are fetched in addition to returned rows."
    ),
    ("data_fetch_candles", "start"): (
        "Inclusive range start. Intraday date-only and calendar phrases use UTC. "
        "For D1/W1/MN1 they select broker-session calendar periods and resolve "
        "from broker-local midnight. Adding --limit retains the first N bars."
    ),
    ("data_fetch_candles", "end"): (
        "Inclusive range end. A timestamp end uses end_filter=bar_close "
        "(bars whose close is at or before end). Intraday date-only and calendar "
        "phrases end in UTC; for D1/W1/MN1 they end at the broker-local "
        "calendar-period boundary."
    ),
    ("data_fetch_candles", "include_incomplete"): (
        "Include the latest forming candle; defaults to false. Compact responses "
        "expose forming_candle_status=skipped and an inclusion hint when a forming "
        "bar is omitted; full detail also includes counts and booleans."
    ),
    ("data_fetch_ticks", "limit"): (
        "Maximum ticks returned (maximum 50000). Latest queries default to 20. "
        "A fully bounded --start/--end range also defaults to the latest 20 "
        "matching ticks; pass an explicit limit for a larger page. Start-bounded "
        "queries keep the earliest ticks when the cap binds. Historical retrieval "
        "is limited to the 30 days ending at --end (or now); truncated responses "
        "set history_window_truncated and effective_start. Date-only start/end "
        "values are UTC midnight, not the broker session day."
    ),
    ("market_status", "symbol"): (
        "Broker symbol for MT5 session/tradability status. If omitted, the "
        "command returns a static major-equity-exchange calendar, not the "
        "connected broker book."
    ),
    ("market_status", "venue"): (
        "Static major-equity venue calendar: NYSE, NASDAQ, LSE, XETRA, "
        "EURONEXT, TSE, HKEX, SSE, or ASX. Mutually exclusive with --symbol."
    ),
    ("forecast_task_wait", "timeout_seconds"): (
        "Seconds to wait for a terminal task state. Default 30 is a short poll, "
        "not a training budget. Maximum 86400 (24 hours)."
    ),
    ("forecast_tune_genetic", "population"): (
        "Population size (minimum 2). Defaults evaluate about 12*10*5=600 "
        "rolling backtests."
    ),
    ("forecast_tune_genetic", "generations"): (
        "Generation count. Work is about population*generations*steps rolling "
        "backtests (600 at the defaults)."
    ),
    ("forecast_tune_genetic", "steps"): (
        "Rolling-origin backtest anchors per candidate. Combined with "
        "population and generations, defaults evaluate about 600 backtests."
    ),
    ("forecast_task_cancel_all", "status_filter"): (
        "Cancelable task status: all, pending, or running. Defaults to all active tasks."
    ),
    ("indicators_list", "trading_style"): (
        "Filter by broad workflow tags (intraday, swing, or position). Many tags "
        "are category heuristics, not indicator-specific recommendations."
    ),
    ("trade_place", "magic"): "MT5 magic number: integer strategy/order identifier used to group EA or strategy trades. Defaults to configured order_magic when omitted.",
    ("trade_get_open", "magic"): "MT5 magic number filter for positions from one strategy or EA. Omit for all magic numbers.",
    ("trade_get_pending", "magic"): "MT5 magic number filter for pending orders from one strategy or EA. Omit for all magic numbers.",
    ("trade_close", "magic"): "Standalone strategy scope for matching objects in the selected target class. Omit for all magic numbers.",
    ("wait_event", "magic"): "MT5 magic number filter for account events from one strategy or EA. Omit for all magic numbers.",
    ("screener", "filters"): "Filter key=value pairs, operator aliases like beta_under=1, Finviz shorthand, or JSON object. Examples: 'country=USA,marketcap=mega', 'pe_under=15,beta_under=1', 'cap_largeover,exch_nyse', '{\"Exchange\":\"NASDAQ\",\"Sector\":\"Technology\"}'. Common keys include Exchange, Index, Sector, Industry, Country, Market Cap., P/E, Dividend Yield, RSI (14), Average Volume, and Price.",
    ("screener", "limit"): "Max screener results to return on this page.",
    ("screener", "order"): "Sort key. Use --order=-marketcap for descending or --order=price for ascending.",
    ("equity_profile", "limit"): "Max insider, ratings, or peer rows to return.",
    ("asset_performance", "option"): (
        "Insider activity view when universe=insider: latest, latest buys/sales, "
        "top week buys/sales, or top owner trade/buys/sales."
    ),
    ("calendar", "start"): (
        "Inclusive start. YYYY-MM-DD or a relative phrase such as today or "
        "2 days ago."
    ),
    ("calendar", "end"): (
        "Inclusive end. YYYY-MM-DD or a relative phrase such as today or "
        "2 days ago."
    ),
    ("calendar", "upcoming"): (
        "When omitted with no start/end, economic calendar defaults to upcoming "
        "unreleased events. Pass false to include already-printed releases."
    ),
    ("calendar", "period"): (
        "Earnings window when view=period: this-week, next-week, previous-week, "
        "or this-month. Defaults to this-week when omitted."
    ),
    ("calendar", "include_elapsed"): (
        "Include earnings already released in the selected period. Defaults to "
        "false; after the US cash close this can empty this-week results."
    ),
    ("forecast_barrier_optimize", "method"): "Barrier simulation method: mc_gbm, mc_gbm_bb, hmm_mc, garch, bootstrap, heston, jump_diffusion, auto, or ensemble.",
    ("forecast_barrier_prob", "barrier"): (
        "Barrier object. Prefer the shell-safe form "
        "kind=tp_sl,unit=pct,take_profit=0.5,stop_loss=0.5 or "
        "kind=single_price,level=1.1000. JSON objects and "
        "--set barrier.kind=tp_sl --set barrier.unit=pct ... also work. "
        "The kind may be omitted from a complete TP/SL or single-price object."
    ),
    ("forecast_barrier_prob", "mu"): (
        "Annual log-return drift override (decimal fraction) on the shared "
        "symbol/timeframe annualization basis."
    ),
    ("forecast_barrier_prob", "sigma"): (
        "Annual return-volatility override (decimal fraction) on the shared "
        "symbol/timeframe annualization basis."
    ),
    ("forecast_volatility_estimate", "method"): (
        "Volatility estimator, such as ewma, rolling_std, har_rv, garch, "
        "arima, theta, or ensemble. Run forecast_list_methods --detail full "
        "--search-term NAME to inspect parameters for volatility and barrier "
        "methods as well as general forecast methods."
    ),
    ("volatility_term_structure", "horizons"): (
        "Comma-separated realized-volatility horizons in bars, for example 1,5,20."
    ),
    ("market_relative_strength", "horizons"): (
        "Comma-separated ranking horizons in bars; values align one-to-one with --weights."
    ),
    ("market_relative_strength", "weights"): (
        "Comma-separated non-negative ranking weights matching --horizons; normalized "
        "to sum to 1."
    ),
    ("market_relative_strength", "limit"): (
        "Maximum distinct ranked symbols returned across strongest and weakest "
        "tails; the stronger tail receives the extra row for odd limits."
    ),
    ("options_chain", "limit"): (
        "Max option contracts to return. Omitted compact output uses 20 nearest "
        "strikes; omitted full output uses 200."
    ),
    ("options_heston_calibrate", "valuation_date"): (
        "Options-chain observation date in YYYY-MM-DD format; omit to use the "
        "provider chain snapshot date. A different calendar date is rejected."
    ),
    ("options_barrier_price", "valuation_date"): (
        "Valuation date in YYYY-MM-DD format; omit for the selected calendar's local date."
    ),
    ("volume_profile_levels", "lookback"): (
        "Historical bar count for a timeframe-based profile; requires --timeframe."
    ),
    ("outliers_detect", "limit"): "Max anomalous bars to return.",
    ("temporal_analyze", "limit"): (
        "Max grouped time buckets to return; pagination only, not the analysis window."
    ),
    ("temporal_analyze", "session_calendar"): "Session calendar: auto, fx, or equity.",
    ("temporal_analyze", "time_range"): (
        "Half-open clock filter HH:MM-HH:MM in --timezone; defaults to CLIENT_TZ, "
        "then UTC, and wraps midnight."
    ),
    ("temporal_analyze", "timezone"): (
        "IANA timezone for hour/session grouping and --time-range (for example, "
        "Europe/London). Defaults to CLIENT_TZ, then UTC."
    ),
    ("temporal_analyze", "return_basis"): (
        "Return basis: previous_close includes overnight/session gaps in the current "
        "bar; bar_open measures same-bar open-to-close movement."
    ),
    ("seasonality_detect", "max_period"): (
        "Maximum candidate seasonal period in bars; defaults from available samples and "
        "--min-cycles."
    ),
    ("causal_discover_signals", "symbols"): (
        "Comma- or space-separated MT5 symbols (e.g. EURUSD,GBPUSD or "
        "EURUSD GBPUSD); one symbol auto-expands to its MT5 group. Optional "
        "with --group."
    ),
    ("trade_execution_quality", "side"): "Execution fill side filter: buy or sell.",
    ("trade_history", "side"): (
        "For deals, buy/sell filters fill_side and long/short filters "
        "position_side. Order history accepts buy/sell only."
    ),
    ("trade_journal_analyze", "side"): (
        "buy/sell filters exit-fill direction; long/short filters realized "
        "position direction."
    ),
    ("trade_execution_quality", "min_sample"): (
        "Minimum eligible fills required for sufficient execution-quality evidence."
    ),
    ("trade_history", "column_style"): (
        "Trade-history field naming: snake_case or humanized."
    ),
    ("market_microstructure_analyze", "max_ticks"): (
        "Maximum raw ticks retained for microstructure analysis."
    ),
    ("options_barrier_price", "option_type"): "Option side: call or put.",
    ("options_barrier_price", "calendar"): (
        "QuantLib calendar name, such as UnitedStates.NYSE, TARGET, or NullCalendar."
    ),
    ("options_barrier_price", "maturity_basis"): (
        "Interpret maturity_days as calendar_days or business_days in the selected QuantLib calendar."
    ),
    ("options_barrier_price", "barrier"): (
        "Option knock-in/knock-out barrier price level, in the same units as spot "
        "and strike. This is a numeric parametric pricer; it does not fetch a symbol quote."
    ),
    ("strategy_validate", "candidates"): (
        "JSON strategy candidate list. Built-in example: "
        "'[{\"id\":\"cross\",\"type\":\"builtin_strategy\","
        "\"strategy\":\"ema_cross\"}]'. Forecast-threshold example: "
        "'[{\"id\":\"drift-half\",\"type\":\"forecast_threshold\","
        "\"method\":\"drift\",\"params\":{\"lookback\":30},\"horizon\":1,"
        "\"long_above\":0.005,\"short_below\":-0.005}]'. long_above and "
        "short_below are simple-return fractions (0.005 = 0.5%), not "
        "percentage points. Candidate types are builtin_strategy and "
        "forecast_threshold."
    ),
    ("strategy_validate", "barrier"): (
        "JSON next-open execution barrier for strategy P&L (not labels_triple_barrier). "
        "Uses horizon, tp_pct, sl_pct, and same_bar_policy; tp_pct/sl_pct are "
        "percent values (0.5 means 0.5%). Entry is the next bar's open; timeout "
        "is mark-to-market return, not a 0 label."
    ),
    ("options_chain", "symbol"): (
        "Underlying symbol for listed options, e.g. AAPL or SPX. SPX resolves "
        "to Yahoo's ^SPX identifier when Yahoo is effective."
    ),
    ("options_expirations", "symbol"): (
        "Underlying symbol for listed options, e.g. AAPL or SPX. SPX resolves "
        "to Yahoo's ^SPX identifier when Yahoo is effective."
    ),
    ("options_heston_calibrate", "symbol"): (
        "Underlying symbol for listed options, e.g. AAPL or SPX. SPX resolves "
        "to Yahoo's ^SPX identifier when Yahoo is effective."
    ),
    ("equity_profile", "symbol"): "US equity ticker, e.g. AAPL or TSLA.",
    ("options_heston_calibrate", "calendar"): (
        "QuantLib calendar name used by calibration helpers, such as UnitedStates.NYSE or NullCalendar."
    ),
    ("options_heston_calibrate", "maturity_basis"): (
        "Basis for the reported days_to_expiry diagnostic; calibration remains anchored to the contract expiry date."
    ),
    ("options_chain", "expiration"): (
        "Listed option expiration date in YYYY-MM-DD format, e.g. 2026-07-17. "
        "Omit to use the next live listed expiration (skips the same-day weekly "
        "after the regular US cash close). A listed date is labeled live or "
        "expired; an unlisted date returns the provider's current expiration list."
    ),
    ("options_heston_calibrate", "expiration"): (
        "Listed option expiration date in YYYY-MM-DD format, e.g. 2026-07-17. "
        "Omit to use the provider's nearest listed expiration at least 7 "
        "calendar days after the observation date."
    ),
    ("forecast_tune_optuna", "search_space"): "Optuna search space (JSON or k=v).",
    ("indicators_list", "detail"): "Output detail: compact table or full rows with aliases and descriptions.",
    ("market_snapshot", "sections"): (
        "Analysis modules to include: quote, status, levels, patterns, regime, "
        "forecast, or all. Defaults to quote,status,levels,patterns."
    ),
    ("market_snapshot", "detail"): (
        "Field verbosity inside selected sections; full does not add sections. "
        "Use --sections all for every snapshot module."
    ),
    ("causal_discover_signals", "limit"): "Max causal link rows to return.",
    ("causal_discover_signals", "window_bars"): (
        "Historical bars per symbol used for causal tests."
    ),
    ("cointegration_test", "symbols"): (
        "Comma- or space-separated MT5 symbols (e.g. EURUSD,GBPUSD or EURUSD GBPUSD); one symbol auto-expands "
        "to its MT5 group. Optional with --group."
    ),
    ("cointegration_test", "limit"): "Max cointegration pair rows to return.",
    ("cointegration_test", "window_bars"): (
        "Historical bars per symbol used for the cointegration test window."
    ),
    ("correlation_matrix", "limit"): "Max correlation pair rows to return.",
    ("correlation_matrix", "window_bars"): (
        "Historical bars per symbol used for the correlation window."
    ),
    ("correlation_matrix", "symbols"): (
        "Comma- or space-separated MT5 symbols (e.g. EURUSD,GBPUSD or EURUSD GBPUSD); one symbol auto-expands "
        "to its MT5 group. Optional with --group."
    ),
    ("cross_correlation", "symbols"): (
        "Comma- or space-separated MT5 symbols (e.g. EURUSD,GBPUSD or EURUSD GBPUSD)."
    ),
    ("market_scan", "symbols"): (
        "Comma-separated MT5 symbols to scan. Optional with --group."
    ),
    ("market_relative_strength", "symbols"): (
        "Comma- or space-separated MT5 symbols to rank (e.g. EURUSD,GBPUSD "
        "or EURUSD GBPUSD). Provide at least two symbols, use --group to rank "
        "an MT5 group, or omit both to rank the visible Market Watch universe. "
        "Use a homogeneous group when comparable peers are required."
    ),
    ("market_scan", "preset"): (
        "Built-in scan preset: oversold, overbought, high-volume, tight-spread, "
        "gap-up, or gap-down. Explicit filter flags override preset defaults."
    ),
    ("market_scan", "rank_by"): (
        "Ranking metric. Default abs_price_change_pct uses completed-bar closes, "
        "not live bid/ask. Use abs_live_price_change_pct to rank by executable quotes."
    ),
    ("market_scan", "rank_order"): (
        "Sort direction for ranked rows: auto, asc/ascending, or desc/descending. "
        "Auto keeps tight spreads and oversold RSI ascending; most other ranks descending."
    ),
    ("market_scan", "quote_usable_only"): (
        "Exclude quotes that are stale, future-dated, locked, inverted, or one-sided. "
        "Defaults to true for spread rankings and the tight-spread preset."
    ),
    ("outliers_detect", "score_fields"): (
        "Comma-separated candle features to score: return, volume, and/or range."
    ),
    ("outliers_detect", "threshold"): (
        "Positive robust-deviation cutoff; 3.5 is a common MAD threshold."
    ),
    ("labels_triple_barrier", "detail"): (
        "Detail level: compact (small outcome sample), standard (recent lookback rows), "
        "summary, or full."
    ),
    ("labels_triple_barrier", "limit"): (
        "Maximum labeled rows for compact/standard output. Compact is capped at "
        "10 and normally shows the recent tail; when that tail is entirely neutral, "
        "it reserves up to two rows for recent resolved TP/SL examples; full returns "
        "the complete labeled series."
    ),
    ("labels_triple_barrier", "lookback"): (
        "Number of labeled entries to calculate; the tool fetches lookback plus "
        "horizon bars."
    ),
    ("labels_triple_barrier", "barriers"): (
        "Barrier pair as KV or JSON. Prefer the shell-safe form "
        "'unit=pct take_profit=0.5 stop_loss=0.5'. JSON objects also work: "
        "'{\"kind\":\"tp_sl\",\"unit\":\"pct\",\"take_profit\":0.5,\"stop_loss\":0.5}'. "
        "kind='tp_sl' is optional, so forecast_barrier_prob TP/SL objects can be reused. "
        "pct/ticks are distances from entry; price values are absolute levels."
    ),
    ("labels_triple_barrier", "allow_noncausal_denoise"): (
        "Allow explicitly requested zero-phase denoising. This uses future bars, "
        "sets denoise_lookahead_bias=true, and makes labels unsuitable as training targets."
    ),
    ("market_scan", "limit"): "Max matching symbols to return.",
    ("news", "limit"): (
        "Global maximum across all news/event buckets. One upcoming event is "
        "reserved when available; use --limit-per-bucket to cap each family separately."
    ),
    ("news", "limit_per_bucket"): (
        "Maximum rows in each news/event family while preserving the separate buckets."
    ),
    ("market_depth_fetch", "require_dom"): "Fail if DOM is unavailable instead of falling back to a quote snapshot.",
    ("patterns_detect", "mode"): "Pattern mode: all, candlestick, classic, harmonic, fractal, or elliott.",
    ("patterns_detect", "engine"): (
        "Classic-mode engine: native or stock_pattern. Omitted classic calls "
        "use native; invalid for other modes."
    ),
    ("report_generate", "template"): (
        "Report template: minimal fast context+forecast (default), basic research with confluence, "
        "advanced regimes/HAR/conformal, scalping M5, intraday H1, swing H4/D1, "
        "or position D1/W1. Typical warm runtimes: minimal 3-10s, scalping "
        "15-60s, basic/style templates 30-120s, advanced 60-180s; broker "
        "history and enabled methods can increase them. Use --max-runtime, "
        "--include-sections, or --max-sections to bound work."
    ),
    ("report_generate", "max_runtime"): (
        "Cooperative runtime budget in seconds (1-3600). Sections whose "
        "estimated cost does not fit are omitted, and new sub-tools stop after "
        "the deadline; an active native/MT5 call is allowed to finish safely."
    ),
    ("report_generate", "allow_partial"): (
        "Return success=true when at least one report section is usable while "
        "retaining section_run_status=partial; set false for strict completion."
    ),
    ("report_generate", "progress"): (
        "Write report sub-tool start/finish progress lines to stderr."
    ),
    ("temporal_analyze", "lookback"): (
        "Historical bars used when start/end are omitted. Defaults to a "
        "timeframe-aware seasonal window: 210 days for day-of-week, 60 days "
        "for hour/session, 730 days for month, and 365 days for overall "
        "analysis, bounded to 200-20000 bars (H1 session: 1440 bars)."
    ),
    ("regime_detect", "fetch_limit"): (
        "Historical bars fetched for regime detection. Defaults to the effective "
        "lookback plus warmup bars; use max_regimes for compact output count."
    ),
    ("symbols_list", "limit"): "Max symbols or groups to return.",
    ("symbols_top_markets", "rank_by"): (
        "Leaderboard to compute: abs_price_change_pct (default), all, "
        "spread/spread_pct, tick_volume, price_change/price_change_pct, "
        "or abs_price_change/abs_price_change_pct."
    ),
    ("symbols_top_markets", "limit"): (
        "Max symbols for the selected ranking; per leaderboard when rank_by=all."
    ),
    ("symbols_top_markets", "candidate_limit"): (
        "Advanced deterministic recovery partition size (1-250). Omit it for "
        "the managed global scan."
    ),
    ("patterns_detect", "allow_partial"): (
        "Keep usable mode=all detector/timeframe results after partial failure; "
        "set false for strict completion."
    ),
    ("market_status", "allow_partial"): (
        "Keep usable comma-separated symbol results after partial failure; set "
        "false for strict completion."
    ),
    ("market_scan", "allow_partial"): (
        "Keep usable rows after unknown requested symbols are dropped; set "
        "false to fail closed when any requested name is missing."
    ),
    ("symbols_top_markets", "candidate_offset"): (
        "Zero-based offset into the deterministic sorted candidate universe. Increment "
        "by candidate_limit until candidate_page.has_more is false."
    ),
    ("symbols_top_markets", "scan_budget_seconds"): (
        "Wall-clock budget for global candidate sampling (default 30 seconds). "
        "Use 0 to wait for the exact full-universe leaderboard."
    ),
    ("trade_close", "close_all"): (
        "Select the whole account when ticket, symbol, and magic are omitted."
    ),
    ("trade_close", "target"): (
        "Object class: positions, pending, or all_exposure. The default positions "
        "target never cancels pending orders."
    ),
    ("trade_close", "confirm_close_all"): (
        "Confirm any live ticketless bulk operation, including symbol and magic scopes."
    ),
    ("trade_close", "dry_run"): (
        "Preview the close request without sending it to the broker."
    ),
    ("trade_close", "pnl_filter"): (
        "Position P&L filter: all, profit, or loss."
    ),
    ("trade_close", "close_priority"): (
        "When multiple positions match, close loss_first, profit_first, or largest_first."
    ),
    ("trade_modify", "dry_run"): (
        "Preview the modification without sending it to the broker."
    ),
    ("trade_modify", "price"): (
        "New pending-order trigger or entry price. Omit when only "
        "stop_loss/take_profit change."
    ),
    ("trade_modify", "stop_limit_price"): (
        "New limit leg for a stop-limit pending order; omitted values preserve the "
        "existing broker price."
    ),
    ("trade_modify", "clear_stop_loss"): (
        "Explicitly remove stop-loss protection from the ticket."
    ),
    ("trade_modify", "clear_take_profit"): (
        "Explicitly remove take-profit protection from the ticket."
    ),
    ("trade_get_pending", "order_type"): (
        "Pending-order filter: BUY_LIMIT, BUY_STOP, BUY_STOP_LIMIT, SELL_LIMIT, "
        "SELL_STOP, or SELL_STOP_LIMIT."
    ),
    ("trade_modify", "idempotency_key"): (
        "Durable dedupe key shared by CLI and server processes. Reusing the same "
        "key and payload within the retention window replays the prior outcome."
    ),
    ("trade_place", "idempotency_key"): (
        "Durable dedupe key for live submissions shared by CLI and server "
        "processes. Reusing the same key and payload within the retention window "
        "replays the prior live outcome. Dry-run previews are not stored."
    ),
    ("trade_place", "dry_run"): (
        "Preview the order without sending it to the broker."
    ),
    ("trade_place", "detail"): (
        "Dry-run preview detail: compact for key checks, full for execution diagnostics."
    ),
    ("trade_place", "stop_limit_price"): (
        "Limit price activated by a BUY_STOP_LIMIT or SELL_STOP_LIMIT trigger."
    ),
    ("trade_stress_test", "shocks"): (
        "JSON object mapping symbols to percentage shocks. Examples: "
        "'{\"*\":-2}' or '{\"EURUSD\":-1,\"XAUUSD\":-3}'."
    ),
    ("trade_place", "require_sl_tp"): (
        "Require both stop_loss and take_profit for market and pending orders."
    ),
    ("trade_history", "minutes_back"): (
        "History lookback in minutes. Defaults to 10080 minutes (7 days) when "
        "start/end and minutes_back are omitted."
    ),
    ("trade_journal_analyze", "minutes_back"): (
        "Journal history lookback in minutes. Defaults to 10080 minutes (7 days) "
        "when start/end and minutes_back are omitted."
    ),
    ("trade_journal_analyze", "limit"): (
        "Maximum per-trade rows returned in full detail (default 50). Period "
        "statistics always analyze all realized exit deals in the resolved window."
    ),
    ("trade_execution_quality", "minutes_back"): (
        "Execution-history lookback in minutes (default 10080 = 7 days)."
    ),
    ("trade_execution_quality", "limit"): (
        "Maximum eligible fills to analyze (default 200)."
    ),
    ("trade_modify", "expiration"): "Future pending-order expiration (dateparser string or positive UTC epoch seconds); use the literal GTC token for no expiration.",
    ("trade_place", "expiration"): "Future pending-order expiration (dateparser string or positive UTC epoch seconds); use the literal GTC token for no expiration.",
    ("wait_event", "symbol"): (
        "Single trading symbol (e.g. EURUSD). Cannot be combined with symbols. "
        "Requires --timeframe or --watch-for; a symbol plus --max-wait-seconds "
        "alone is rejected because duration mode ignores the symbol. "
        "Timer-only duration and clock-only timeframe waits may omit both."
    ),
    ("wait_event", "symbols"): (
        "Basket of 1-12 trading symbols. Cannot be combined with symbol; omitted-symbol "
        "watchers apply to every basket member."
    ),
    ("wait_event", "timeframe"): (
        "Candle-boundary wait mode. Set max_wait_seconds for an optional safety cap. "
        "With inferred watchers, reaching the boundary is a successful completion."
    ),
    ("wait_event", "max_wait_seconds"): (
        "Maximum wait in seconds (alias: --timeout). With timeframe, defaults to "
        "the timeframe length plus 60 seconds. Without timeframe, omit the symbol "
        "and watch_for for a timer, or pass watchers to return early."
    ),
    ("wait_event", "poll_interval_seconds"): (
        "Seconds between polls; must be at least 0.1. Omit to use 0.5."
    ),
    ("wait_event", "watch_for"): (
        "Event names or event objects. Examples: order_filled, "
        "'{\"type\":\"order_filled\",\"symbol\":\"EURUSD\"}'. "
        "Put candle_close boundaries in end_on. In timeframe mode, omit for a "
        "candle-boundary wait only. In duration mode, omit for a pure timer. "
        "Explicit watchers make an unmatched timeout or boundary a failed wait."
    ),
    ("wait_event", "end_on"): (
        "Optional timeframe-mode boundaries. Explicit boundary timeframes must "
        "match the top-level timeframe."
    ),
    ("causal_discover_signals", "allow_partial"): (
        "Keep symbols with usable history and report excluded symbols; false "
        "fails the analysis when any requested symbol is unusable."
    ),
    ("cointegration_test", "k_ar_diff"): (
        "Number of lagged differences in the Johansen test; ignored by the "
        "Engle-Granger method."
    ),
    ("cointegration_test", "allow_partial"): (
        "Keep symbols with usable aligned history and report exclusions; false "
        "fails when any requested symbol is unusable."
    ),
    ("confluence_levels", "volume_profile_max_m1_bars"): (
        "Maximum M1 bars used by the volume-profile component; limits work for "
        "large confluence windows."
    ),
    ("correlation_matrix", "allow_partial"): (
        "Keep symbols with sufficient aligned observations and report exclusions; "
        "false fails when any requested symbol is unusable."
    ),
    ("cross_correlation", "window_bars"): (
        "Historical bars per symbol used for the lagged cross-correlation window."
    ),
    ("cross_correlation", "bootstrap_samples"): (
        "Bootstrap resample count used to estimate cross-correlation uncertainty."
    ),
    ("denoise_list_methods", "causality"): (
        "Filter methods by causal real-time support or zero-phase offline support."
    ),
    ("screener", "filter_name"): (
        "Exact screener filter name to describe; omit it to list filters."
    ),
    ("equity_profile", "fields"): (
        "Comma-separated fundamental fields to return; this selects domain "
        "data and is distinct from output_fields projection."
    ),
    ("forecast_barrier_optimize", "same_bar_policy"): (
        "Resolve a candle that touches TP and SL: sl_first counts a loss, tp_first "
        "counts a win, and neutral leaves the outcome unresolved."
    ),
    ("forecast_barrier_prob", "same_bar_policy"): (
        "Resolve simulated TP/SL ties: sl_first assigns tie probability to SL, "
        "tp_first assigns it to TP, and neutral reports it as unresolved."
    ),
    ("forecast_generate", "async_mode"): (
        "Queue supported trainable forecasts in the in-process worker and return a "
        "task ID; one-shot CLI calls must run synchronously."
    ),
    ("forecast_generate", "model_cache"): (
        "Model policy: reuse loads or trains a compatible model, ephemeral never "
        "persists one, and require_existing fails unless a compatible model exists."
    ),
    ("forecast_list_methods", "profile"): (
        "Filter methods by workflow profile: quickstart is a small native baseline "
        "set, core is the recommended general catalog, and all disables profile "
        "filtering."
    ),
    ("labels_triple_barrier", "same_bar_policy"): (
        "Resolve a bar that touches TP and SL: sl_first labels -1, tp_first labels "
        "+1, and neutral labels 0."
    ),
    ("market_relative_strength", "volatility_lookback"): (
        "Return bars used to estimate volatility for risk-adjusted relative-strength "
        "scores."
    ),
    ("market_relative_strength", "benchmark"): (
        "Optional comparison symbol; each candidate's horizon return is measured "
        "relative to this symbol before ranking."
    ),
    ("market_relative_strength", "max_symbols"): (
        "Maximum symbols admitted from the selected broker universe before ranking."
    ),
    ("portfolio_risk_decompose", "horizon_bars"): (
        "Forecast horizons expressed in bars of the requested timeframe."
    ),
    ("portfolio_risk_decompose", "ewma_half_life"): (
        "EWMA volatility half-life in bars of the requested timeframe."
    ),
    ("portfolio_risk_decompose", "simulations"): (
        "Monte Carlo scenario count used for portfolio tail-risk estimates."
    ),
    ("portfolio_risk_decompose", "proposed_trade"): (
        "Optional JSON trade object with symbol, buy/sell side, and volume in lots "
        "for incremental-risk analysis."
    ),
    ("portfolio_risk_decompose", "allow_partial"): (
        "Omit positions without safe marks or sufficient history and report them; "
        "false fails closed when any position is unusable."
    ),
    ("seasonality_detect", "min_period"): (
        "Smallest candidate seasonal period, measured in observed bars."
    ),
    ("seasonality_detect", "min_cycles"): (
        "Minimum complete cycles required for a candidate seasonal period."
    ),
    ("strategy_backtest", "cost_model"): (
        "Spread source: historical_bar_spread is the fail-closed default and "
        "requires complete completed-bar coverage; fixed requires spread_bps."
    ),
    ("strategy_backtest", "spread_bps"): (
        "Fixed round-trip spread cost in basis points; required when "
        "cost_model=fixed and invalid with historical_bar_spread."
    ),
    ("strategy_validate", "strategy"): (
        "Single built-in strategy shortcut. Use candidates for parameterized or "
        "mixed validation sets."
    ),
    ("strategy_validate", "n_splits"): (
        "Number of chronological walk-forward validation folds."
    ),
    ("strategy_validate", "purge_bars"): (
        "Bars removed between training and validation folds to prevent outcome-window "
        "leakage; omit to derive it from the barrier horizon."
    ),
    ("strategy_validate", "embargo_bars"): (
        "Bars withheld after each validation fold before later training data may be "
        "used; omit to derive it from the barrier horizon."
    ),
    ("strategy_validate", "cost_model"): (
        "Spread source: historical_bar_spread uses completed validation bars; fixed "
        "requires spread_bps."
    ),
    ("strategy_validate", "spread_bps"): (
        "Fixed round-trip spread cost in basis points, required only when "
        "cost_model=fixed."
    ),
    ("strategy_validate", "commission_bps"): (
        "Commission per fill side in basis points; validation applies it twice per "
        "round trip."
    ),
    ("strategy_validate", "bootstrap_samples"): (
        "Bootstrap resample count used for expectancy and win-rate confidence intervals."
    ),
    ("strategy_validate", "significance_alpha"): (
        "Maximum multiple-testing-adjusted p-value accepted as statistically significant."
    ),
    ("strategy_validate", "min_positive_fold_share"): (
        "Minimum fraction from 0 to 1 of walk-forward folds that must have positive "
        "net expectancy."
    ),
    ("tools_list", "include_related"): (
        "Include related-tool recommendations in each catalog row."
    ),
    ("trade_execution_quality", "benchmark"): (
        "Slippage reference: arrival_quote uses the executable quote at order setup; "
        "order_price uses the submitted price. Positive slippage bps is adverse."
    ),
    ("trade_execution_quality", "benchmark_fallback"): (
        "When an arrival quote is unavailable, skip the fill or use its submitted "
        "order price as the fallback benchmark."
    ),
    ("trade_execution_quality", "quote_window_seconds"): (
        "Seconds searched backward from order setup or fill time for the latest "
        "eligible quote."
    ),
    ("trade_execution_quality", "markout_seconds"): (
        "Post-fill horizons in seconds. Markout bps is direction-adjusted price change; "
        "positive values favor the trade."
    ),
    ("trade_history", "order"): (
        "Sort history by event time: desc returns newest activity first; asc returns "
        "oldest first."
    ),
    ("trade_stress_test", "include_unshocked"): (
        "Include open positions whose symbol has no matching explicit or wildcard shock."
    ),
    ("volatility_term_structure", "percentiles"): (
        "Comma-separated percentile levels strictly between 0 and 100, such as "
        "10,25,50,75,90."
    ),
    ("volatility_term_structure", "annualize"): (
        "Annualize realized volatility using observed bars per session and 365 crypto, "
        "260 FX, or 252 other trading sessions per year."
    ),
    ("wait_event", "accept_preexisting"): (
        "Return immediately when a state-style watcher is already satisfied at setup; "
        "false waits for a new transition after startup."
    ),
}

