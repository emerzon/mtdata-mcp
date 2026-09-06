"""Transport-neutral per-(command, parameter) help overrides."""

from typing import Dict

COMMAND_PARAM_HELP_OVERRIDES: Dict[tuple[str, str], str] = {
    ("market_depth_fetch", "spread"): (
        "Boolean output control. Compute and include bid/ask spread metrics from "
        "broker DOM or the fallback quote; disabled by default."
    ),
    ("forecast_train", "quantity"): (
        "Train a price-level or return target. Volatility uses the dedicated "
        "forecast_volatility_estimate tool and is not separately trainable."
    ),
    ("forecast_train", "wait"): (
        "Wait for training to finish. One-shot CLI and stdin shell batches "
        "always wait so the in-process worker stays alive (CLI default: true); "
        "--wait false is rejected there. The flag only applies in interactive "
        "shell, MCP, and Web API sessions (those default: false)."
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
        "(Z when CLIENT_TZ is UTC; the envelope then labels timestamp_format "
        "iso_utc, otherwise iso_offset), iso_utc for UTC Z strings, "
        "or epoch for UTC epoch seconds."
    ),
    ("data_fetch_candles", "selection"): (
        "Range pagination anchor: first_n returns the earliest matching candles; "
        "last_n returns the latest matching candles. Omit for first_n when a "
        "range is provided, otherwise the latest candles are returned."
    ),
    ("data_fetch_ticks", "timestamp_format"): (
        "Format each MT5 tick event's `time` value: iso for ISO in CLIENT_TZ "
        "(Z when CLIENT_TZ is UTC; the envelope then labels timestamp_format "
        "iso_utc, otherwise iso_offset), iso_utc for UTC Z strings, "
        "or UTC epoch seconds."
    ),
    ("data_fetch_ticks", "selection"): (
        "Range snapshot anchor: last_n returns the newest matching ticks; first_n "
        "returns the oldest. Omit for last_n."
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
    ("support_resistance_levels", "timeframe"): (
        "Price-history timeframe. Choose M1, M2, M3, M4, M5, M6, M10, M12, "
        "M15, M20, M30, H1, H2, H3, H4, H6, H8, H12, D1, W1, MN1, or auto. "
        "Auto selects a timeframe from the requested lookback."
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
        "Tail-risk method: historical (empirical observed P&L quantile), "
        "parametric (Gaussian VaR/CVaR), cornish_fisher (skew/kurtosis-adjusted "
        "parametric VaR and expected shortfall), or ewma. Not the bootstrap "
        "scenarios used by portfolio_risk_decompose method=bootstrap_historical."
    ),
    ("trade_var_cvar_calculate", "confidence"): (
        "VaR/CVaR tail confidence as a fraction such as 0.95 or 0.99. "
        "Must satisfy 0.5 < confidence < 1."
    ),
    ("portfolio_risk_decompose", "method"): (
        "Scenario generator: filtered_historical or bootstrap_historical. "
        "bootstrap_historical resamples historical windows; it is not the "
        "empirical-quantile historical method on trade_var_cvar_calculate."
    ),
    ("portfolio_risk_decompose", "confidence"): (
        "VaR/CVaR tail confidence levels. Each value must satisfy "
        "0.5 < confidence < 1."
    ),
    ("trade_var_cvar_calculate", "symbol"): (
        "Optional scope: calculate VaR/CVaR for currently open positions in this "
        "symbol. Omit it for the full open portfolio."
    ),
    ("trade_var_cvar_calculate", "transform"): (
        "Return transform: log_return (aliases log_returns/log) or pct "
        "(aliases pct_return/percent/simple_return)."
    ),
    ("trade_var_cvar_calculate", "horizon_bars"): (
        "Holding period in bars of the requested timeframe. Default 1 is a "
        "one-bar VaR; pass 5 to match portfolio_risk_decompose's 5-bar horizon."
    ),
    ("trade_var_cvar_calculate", "include_incomplete"): (
        "Include the current forming candle in return history. Defaults to false "
        "so VaR/CVaR uses completed bars only."
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
        "Inclusive range start. A timestamp start keeps bars whose open is at "
        "or after start. Intraday date-only and calendar phrases use UTC. "
        "For D1/W1/MN1 they select broker-session calendar periods and resolve "
        "from broker-local midnight. Adding --limit retains the first N bars."
    ),
    ("data_fetch_candles", "end"): (
        "Inclusive range end. A timestamp end keeps bars whose close is at or "
        "before end (end_filter=bar_close). Intraday date-only and calendar "
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
        "A fully bounded --start/--end range also returns the latest matching "
        "ticks (default 20), whether --limit is omitted or set explicitly. "
        "Start-bounded queries keep the earliest ticks when the cap binds. "
        "Historical retrieval is limited to the 30 days ending at --end (or now); "
        "truncated responses set history_window_truncated and effective_start. "
        "Date-only start/end values are UTC midnight, not the broker session day."
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
    ("screener", "order"): (
        "Sort key. Default -marketcap (largest first). Use --order=price for "
        "ascending price. Pagination follows this provider order."
    ),
    ("asset_performance", "universe"): (
        "Context table: forex, crypto, futures, or market-wide insider."
    ),
    ("asset_performance", "source"): (
        "Adapter pin. auto uses Finviz; mt5 is unsupported for this delayed "
        "performance table."
    ),
    ("asset_performance", "rank_by"): (
        "Rank the fetched forex/crypto/futures snapshot by a performance horizon "
        "before paging: 5min, hour, day, week, month, quarter, half, year, or ytd."
    ),
    ("asset_performance", "order"): (
        "Rank direction when --rank-by is set: desc (default) or asc. Ignored "
        "unless --rank-by is provided."
    ),
    ("equity_profile", "limit"): "Max insider, ratings, or peer rows to return.",
    ("asset_performance", "option"): (
        "Insider activity view when universe=insider: latest, latest buys/sales, "
        "top week buys/sales, or top owner trade/buys/sales."
    ),
    ("calendar", "start"): (
        "Inclusive range start (YYYY-MM-DD, ISO timestamp, or relative such as "
        "today). Timestamps keep their time-of-day and filter scheduled_at; "
        "date-only values select the America/New_York calendar day."
    ),
    ("calendar", "end"): (
        "Inclusive range end (YYYY-MM-DD, ISO timestamp, or relative such as "
        "today). Timestamps keep their time-of-day and filter scheduled_at; "
        "date-only values select the America/New_York calendar day."
    ),
    ("calendar", "impact"): (
        "Economic impact filter: low, medium, or high. Comma-separate levels "
        "such as high,medium. Omit to include every impact level."
    ),
    ("calendar", "country"): (
        "Economic country filter such as US. Finviz currently covers US "
        "releases; a non-US code is valid but an empty table does not mean "
        "that region has no events."
    ),
    ("calendar", "currency"): (
        "Economic currency filter such as USD. Finviz currently covers US "
        "releases; a non-USD code is valid but an empty table does not mean "
        "that currency has a clear calendar."
    ),
    ("calendar", "upcoming"): (
        "When omitted with no start/end, economic calendar defaults to upcoming "
        "unreleased events. Pass false to include already-printed releases."
    ),
    ("calendar", "period"): (
        "Earnings window when view=period: this-week, next-week, previous-week, "
        "or this-month. Defaults to this-week when omitted."
    ),
    ("calendar", "source"): (
        "Adapter pin. auto uses Finviz; mt5 is not a calendar provider and "
        "returns a capability error instead of a fake table."
    ),
    ("calendar", "include_elapsed"): (
        "Include earnings already released in the selected period. Defaults to "
        "false; after the US cash close this can empty this-week results."
    ),
    ("forecast_barrier_optimize", "method"): "Barrier simulation method: mc_gbm, mc_gbm_bb, hmm_mc, garch, bootstrap, heston, jump_diffusion, auto, or ensemble. Default mc_gbm_bb, same as forecast_barrier_prob.",
    ("forecast_barrier_optimize", "params"): (
        "Optimizer extras as JSON or k=v. Grid bounds: tp_min, tp_max, sl_min, "
        "sl_max (percent points when --mode pct, ticks when --mode ticks, pips "
        "when --mode pips), plus tp_steps and sl_steps. Tick/pips-mode "
        "fixed/ratio defaults convert the implicit 0.25/1.5/0.25/2.5 percent "
        "(intraday) grid into that distance unit. ticks is the broker trade "
        "tick/point, not FX pips. "
        'Example: --params "tp_min=20 tp_max=80 sl_min=20 sl_max=80".'
    ),
    ("forecast_barrier_prob", "barrier"): (
        "Barrier object. Prefer the shell-safe form "
        "kind=tp_sl,unit=pct,take_profit=0.5,stop_loss=0.5 or "
        "kind=single_price,level=1.1000. JSON objects and "
        "--set barrier.kind=tp_sl --set barrier.unit=pct ... also work. "
        "The kind may be omitted from a complete TP/SL or single-price object. "
        "ticks uses broker trade tick/point (0.1 pip on typical 5-digit FX), "
        "not FX pips; use unit=pips for conventional forex pips."
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
    ("options_expirations", "limit"): (
        "Max expirations to return. Compact output defaults to the nearest 12; "
        "omitted full output returns the complete calendar. Pagination includes "
        "has_more when more expirations remain."
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
    ("patterns_detect", "lookback"): (
        "Historical bars to scan for patterns after applying any start/end window."
    ),
    ("regime_detect", "lookback"): (
        "Number of recent observations to analyze when fetch_limit is omitted, "
        "and the summary window when fetch_limit is provided. Extra history may "
        "be fetched for feature warmup but is excluded from model fitting."
    ),
    ("regime_detect", "threshold"): (
        "BOCPD-only change-point probability threshold (0-1). Rejected for hmm, "
        "pelt, and other non-BOCPD methods; omit it or pass --method bocpd."
    ),
    ("regime_detect", "target"): (
        "Series used for detection. Default auto resolves to price for "
        "rule_based and return for other methods."
    ),
    ("seasonality_detect", "lookback"): (
        "Historical bars used to detect seasonal periods; must be at least 31."
    ),
    ("outliers_detect", "lookback"): (
        "Historical bars scored for anomalous return, volume, and range."
    ),
    ("strategy_validate", "lookback"): (
        "Evaluation bars for strategy candidates. Fetch also includes the "
        "barrier outcome tail plus 5 warmup bars; see fetch_bars and "
        "evaluation_bars in the result."
    ),
    ("forecast_generate", "horizon"): (
        "Bars to forecast. Counts from the open of the current forming bar "
        "unless --as-of or a historical range is set."
    ),
    ("forecast_generate", "lookback"): (
        "Historical bars to train on. Omit for the method default "
        "(native theta/fourier_ols: 300 bars)."
    ),
    ("forecast_backtest_run", "lookback"): (
        "Training bars per anchor; omit for an expanding window or pass N for a "
        "fixed window. HAR-RV rejects lookback; use params.days and optionally "
        "params.rv_timeframe."
    ),
    ("forecast_backtest_run", "horizon"): (
        "Bars forecast after each backtest anchor. In rolling mode, --spacing "
        "must be at least horizon when --steps is greater than 1."
    ),
    ("forecast_backtest_run", "steps"): (
        "Number of rolling-origin anchors when --anchors is omitted. This value "
        "does not select, truncate, or otherwise alter explicit anchors."
    ),
    ("forecast_backtest_run", "spacing"): (
        "Bars between rolling-origin anchors when --anchors is omitted. This "
        "value does not select or alter explicit anchors."
    ),
    ("forecast_backtest_run", "anchors"): (
        "Optional explicit backtest anchors (1-200) in strictly increasing UTC "
        "ISO order. Pass multiple values or a JSON array; values normalize to "
        "second-precision ...Z form. When set, --steps and --spacing remain "
        "rolling-mode settings and do not select or alter these anchors."
    ),
    ("forecast_train", "lookback"): (
        "Historical bars to train on. Omit for the method default "
        "(native theta/fourier_ols: 300 bars)."
    ),
    ("forecast_tune_optuna", "lookback"): (
        "Fixed training bars for each trial. Omit for the method default "
        "(native theta/fourier_ols: 300 bars)."
    ),
    ("forecast_tune_genetic", "lookback"): (
        "Fixed training bars for each candidate. Omit for the method default "
        "(native theta/fourier_ols: 300 bars)."
    ),
    ("forecast_optimize_hints", "lookback"): (
        "Fixed training bars matching forecast_generate. Omit for the method default "
        "(native theta/fourier_ols: 300 bars)."
    ),
    ("forecast_conformal_intervals", "lookback"): (
        "Historical bars used to fit conformal intervals. Omit for the method default "
        "(native theta/fourier_ols: 300 bars)."
    ),
    ("forecast_volatility_estimate", "lookback"): (
        "Historical bars used where applicable; HAR-RV rejects lookback, so set "
        "params.days and optionally params.rv_timeframe for its intraday fit "
        "window."
    ),
    ("outliers_detect", "limit"): "Max anomalous bars to return.",
    ("temporal_analyze", "limit"): (
        "For a single group_by, max grouped time buckets to return. For "
        "group_by=all, limit and offset page each of the four breakdowns "
        "independently. Compact groups concatenates those pages; "
        "dimension_pagination is the per-dimension cursor; groups_analyzed "
        "is the unpaged total. Pagination only, not the analysis window."
    ),
    ("temporal_analyze", "session_calendar"): (
        "Session calendar: auto, fx, equity, or continuous_24_7. auto selects "
        "fx, equity, or continuous_24_7 from the symbol."
    ),
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
        "Display labels for TOON/table output: snake_case or humanized. JSON "
        "item keys, units, and output_fields paths stay snake_case."
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
    ("options_barrier_price", "barrier_already_hit"): (
        "Set true when an existing monitored contract touched its barrier before "
        "valuation, even if current spot has returned to the unbreached side."
    ),
    ("options_barrier_price", "model"): (
        "Pricing model: black_scholes_merton (default analytic barrier with "
        "flat --volatility) or heston (FdHestonBarrierEngine using the five "
        "calibrated Heston parameters)."
    ),
    ("options_barrier_price", "heston_v0"): (
        "Heston initial variance v0. Required with --model heston; from "
        "options_heston_calibrate params.v0."
    ),
    ("options_barrier_price", "heston_kappa"): (
        "Heston mean-reversion speed kappa. Required with --model heston."
    ),
    ("options_barrier_price", "heston_theta"): (
        "Heston long-run variance theta. Required with --model heston."
    ),
    ("options_barrier_price", "heston_sigma"): (
        "Heston volatility of variance sigma. Required with --model heston."
    ),
    ("options_barrier_price", "heston_rho"): (
        "Heston spot-variance correlation rho in [-1, 1]. Required with --model heston."
    ),
    ("options_barrier_price", "volatility"): (
        "Annualized Black volatility as a decimal fraction; 0.20 = 20%. Used "
        "only with --model black_scholes_merton. For a smile-consistent price, "
        "use --model heston with calibrated parameters."
    ),
    ("options_barrier_price", "risk_free_rate"): (
        "Annual domestic/quote-currency risk-free rate r as a decimal fraction; "
        "0.05 = 5%. Equity/index Black-Scholes-Merton uses this as r."
    ),
    ("options_barrier_price", "dividend_yield"): (
        "Annual dividend yield q as a decimal fraction; 0.01 = 1%. For FX, q is "
        "approximately the foreign/base-currency rate (Garman-Kohlhagen)."
    ),
    ("options_chain", "quote_usable_only"): (
        "Keep only contracts with a provider option-quote timestamp and a "
        "two-sided live quote. Yahoo and Tradier do not supply quote timestamps, "
        "so this filter is rejected with capability_unavailable before querying. "
        "Use last_trade_recent_and_market_two_sided or options_heston_calibrate."
    ),
    ("options_chain", "max_quote_age_seconds"): (
        "Maximum age in seconds for a provider option-quote timestamp. Yahoo "
        "and Tradier do not supply quote timestamps, so this filter is rejected "
        "with capability_unavailable before querying."
    ),
    ("volume_profile_levels", "volume_source"): (
        "Volume weight: auto, real_volume, tick_volume, volume_real, volume, or "
        "tick_count. tick_count is broker/provider tick rows and is rejected "
        "with source=m1_bars; use tick_volume for M1 bars or source=ticks for "
        "true tick counts."
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
        "is mark-to-market return, not a 0 label. sma_cross/ema_cross are "
        "position-reversal strategies and reject tp_pct/sl_pct; use "
        "sma_cross_event/ema_cross_event for barrier outcomes."
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
        "QuantLib calendar for valuation timezone and the reported business-day "
        "days_to_expiry diagnostic, such as UnitedStates.NYSE or NullCalendar. "
        "Calibration helper maturity is fixed to calendar days ending on the "
        "contract expiry (NullCalendar); this flag does not change helper dates."
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
        "to its MT5 group. Optional with --group. Pair order is ignored: each unordered pair uses the "
        "alphabetically first symbol as the Engle-Granger dependent (orientation_policy=canonical_symbol_order)."
    ),
    ("cointegration_test", "limit"): (
        "Max ranked pair rows to return. Omitted compact/summary output uses 10; "
        "omitted full/standard output is unbounded."
    ),
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
        "not live bid/ask. Use abs_live_price_change_pct to rank by executable quotes. "
        "market_scan is the filtered screener; symbols_top_markets is the unfiltered "
        "leaderboard; market_radar is a named-watchlist quote board."
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
        "Positive robust-z cutoff applied after MAD, IQR, or mean/std scaling; "
        "3.5 is the default for all methods."
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
        "Optional tail cap of labeled entries. When start/end are omitted the "
        "default is 50 and the tool fetches lookback plus horizon bars. An "
        "explicit date range is analyzed in full unless lookback is also set."
    ),
    ("labels_triple_barrier", "barrier"): (
        "Barrier pair as KV or JSON. Prefer the shell-safe form "
        "kind=tp_sl,unit=pct,take_profit=0.5,stop_loss=0.5. JSON objects also work: "
        "'{\"kind\":\"tp_sl\",\"unit\":\"pct\",\"take_profit\":0.5,\"stop_loss\":0.5}'. "
        "Spaces or commas may separate key=value pairs. kind='tp_sl' is optional, so "
        "forecast_barrier_prob TP/SL objects can be reused. pct/ticks/pips are "
        "distances from entry; price values are absolute levels. ticks uses the "
        "broker trade tick/point, not FX pips; use unit=pips for forex pips."
    ),
    ("labels_triple_barrier", "allow_noncausal_denoise"): (
        "Allow explicitly requested zero-phase denoising. This uses future bars, "
        "sets denoise_lookahead_bias=true, and makes labels unsuitable as training targets."
    ),
    ("market_scan", "limit"): "Max matching symbols to return.",
    ("news", "limit"): (
        "Global maximum across all news/event buckets. Compact unified view "
        "defaults to 10; otherwise unbounded. One upcoming event is reserved "
        "when available; use --limit-per-bucket to cap each family separately."
    ),
    ("news", "start"): (
        "Inclusive UTC publication start. Date-only values start at 00:00 UTC."
    ),
    ("news", "end"): (
        "Inclusive UTC publication end. Date-only values include the full UTC day."
    ),
    ("news", "max_age"): (
        "Keep items published within this age. Seconds or a duration such as "
        "3600, 60m, or 1h."
    ),
    ("market_microstructure_analyze", "minutes_back"): (
        "Look back this many minutes from end/now instead of using start. "
        "Defaults to 60 when start/end are omitted."
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
    ("trade_idea_compose", "template"): (
        "Idea template: quick runs session, forecast, volatility, one barrier "
        "pair, and sizing; standard also adds confluence and snaps exits "
        "toward nearby structure."
    ),
    ("trade_idea_compose", "commission_bps_per_side"): (
        "Commission for one fill side in basis points. The expected-value gate "
        "deducts twice this amount for entry plus exit."
    ),
    ("trade_idea_compose", "slippage_bps"): (
        "Slippage for one fill side in basis points. The expected-value gate "
        "deducts twice this amount for entry plus exit."
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
        "Cooperative wall-clock budget in seconds (1-3600). Static section "
        "estimates are advisory and never consume the budget; new sub-tools "
        "stop after the actual deadline, and an active native/MT5 call is "
        "allowed to finish safely."
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
        "Bars fetched for regime detection. Rejected if negative or below the "
        "method minimum (20 for rule_based, 10 otherwise). For non-rule methods "
        "this also becomes the model-fit window when provided; lookback is then "
        "only the summary window. Defaults to the effective lookback plus warmup "
        "bars. Use max_regimes to cap compact and standard regime segment rows."
    ),
    ("symbols_list", "limit"): "Max symbols or groups to return.",
    ("symbols_top_markets", "rank_by"): (
        "Leaderboard to compute: abs_price_change_pct (default), all, "
        "spread/spread_pct, tick_volume, price_change/price_change_pct, "
        "or abs_price_change/abs_price_change_pct. Row time follows data_source: "
        "live_tick (quote time) for spread rankings, otherwise the selected "
        "timeframe's completed-bar time. Use market_scan to filter the universe; "
        "use market_radar for a named watchlist."
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
        "Keep usable comma-separated symbol results after partial failure. "
        "Explicit lists default permissive; set false for strict completion."
    ),
    ("market_scan", "allow_partial"): (
        "Keep partial results when some symbols cannot be evaluated. "
        "Set false to fail if any symbol is missing or its analysis fails. "
        "If every evaluation fails, the scan fails regardless of this setting."
    ),
    ("market_radar", "allow_partial"): (
        "Keep usable rows after unknown requested symbols are dropped. "
        "Explicit watchlists default permissive; set false to fail closed when "
        "any requested name is missing."
    ),
    ("symbols_top_markets", "candidate_offset"): (
        "Zero-based offset into the deterministic sorted candidate universe. Resume "
        "at candidate_page.next_offset until candidate_page.has_more is false; "
        "timeouts can stop before candidate_limit. Keep the same universe and filters."
    ),
    ("symbols_top_markets", "scan_budget_seconds"): (
        "Wall-clock budget for global candidate sampling (default 30 seconds). "
        "Use 0 to wait for the exact full-universe leaderboard."
    ),
    ("trade_modify", "stop_loss"): (
        "New stop-loss price. Omit to leave the existing stop unchanged; use "
        "--clear-stop-loss to remove it."
    ),
    ("trade_modify", "take_profit"): (
        "New take-profit price. Omit to leave the existing target unchanged; use "
        "--clear-take-profit to remove it."
    ),
    ("trade_risk_analyze", "sizing"): (
        "Sizing spec as JSON or key=value pairs. method: fixed_fraction|kelly. "
        "Keys: risk_pct (percent of equity), kelly_fraction. Example: "
        "method=fixed_fraction,risk_pct=1"
    ),
    ("trade_risk_analyze", "sizing_params"): (
        "Extra sizing keys as key=value pairs. Prefer putting method and "
        "risk_pct in --sizing; --sizing-params only adds leftover keys."
    ),
    ("trade_risk_analyze", "stop_loss"): (
        "Stop-loss price required to compute risk-based position size when "
        "--sizing is supplied."
    ),
    ("trade_risk_analyze", "take_profit"): (
        "Optional take-profit price used for reward-to-risk when sizing a "
        "candidate trade."
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
    ("trade_place", "order_type"): (
        "Order type: BUY/SELL for market orders, or pending types such as "
        "BUY_LIMIT and SELL_STOP. --side buy/sell is accepted as a market-order "
        "alias for --order-type."
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
        "Symbol-to-percent mapping, as JSON or KV. Examples: '{\"*\":-2}', "
        "'EURUSD=-2%', or '*=-2'. Prefer --shock-pct for a uniform shock. "
        "-100 is rejected because it would imply a zero or negative price; "
        "use a near-total shock such as -99.99 to model almost complete loss."
    ),
    ("trade_stress_test", "shock_pct"): (
        "Uniform percentage shock for every open position, for example -2. "
        "Equivalent to --shocks '*=-2'."
    ),
    ("trade_place", "require_sl_tp"): (
        "Require both stop_loss and take_profit for market and pending orders."
    ),
    ("trade_history", "minutes_back"): (
        "History lookback in minutes. Defaults to 10080 minutes (7 days) when "
        "start/end and minutes_back are omitted. Maximum is 10512000 minutes "
        "(20 years)."
    ),
    ("trade_journal_analyze", "minutes_back"): (
        "Journal history lookback in minutes. Defaults to 10080 minutes (7 days) "
        "when start/end and minutes_back are omitted. Maximum is 10512000 minutes "
        "(20 years)."
    ),
    ("trade_journal_analyze", "limit"): (
        "Maximum unique per-trade rows returned in full detail (default 50), "
        "including items plus ranked best/worst lists. Period statistics always "
        "analyze all realized exit deals in the resolved window."
    ),
    ("trade_history", "limit"): (
        "Maximum rows returned per page. Defaults to 20; the safety cap is 500. "
        "Use --cursor for additional pages."
    ),
    ("trade_execution_quality", "minutes_back"): (
        "Execution-history lookback in minutes (default 10080 = 7 days). "
        "Maximum is 10512000 minutes (20 years)."
    ),
    ("trade_execution_quality", "limit"): (
        "Maximum eligible fills to analyze (default 200)."
    ),
    ("trade_modify", "expiration"): "Future pending-order expiration (dateparser string or positive UTC epoch seconds); use the literal GTC token for no expiration.",
    ("trade_place", "expiration"): "Future pending-order expiration (dateparser string or positive UTC epoch seconds); use the literal GTC token for no expiration.",
    ("wait_event", "symbol"): (
        "Single trading symbol (e.g. EURUSD). Cannot be combined with symbols. "
        "Omit symbol and symbols for a clock-only timeframe-boundary wait."
    ),
    ("wait_event", "symbols"): (
        "Basket of 1-12 trading symbols. Cannot be combined with symbol; omitted-symbol "
        "watchers apply to every basket member, and explicitly named watcher symbols "
        "must belong to the basket."
    ),
    ("wait_event", "timeframe"): (
        "Required wait horizon. The engine derives the wait budget internally, "
        "sleeps directly for boundary-only waits, and polls only when explicit "
        "event watchers need observation."
    ),
    ("wait_event", "watch_for"): (
        "Event names or JSON event objects. Supported types (required fields): "
        "order_created/order_filled/order_cancelled (optional symbol, order_ticket, "
        "magic, side=buy|sell); position_opened/position_closed/tp_hit/sl_hit "
        "(optional symbol, position_ticket, magic, side); pending_near_fill/"
        "stop_threat (distance in price units, optional symbol/ticket/magic/"
        "price_source=auto|bid|ask|mid|last); price_change (threshold_value; "
        "threshold_mode=fixed_pct|ratio_to_baseline|zscore; direction=up|down|either; "
        "window.kind=minutes|ticks and window.value); volume_spike/"
        "tick_count_spike/spread_spike/tick_count_drought/range_expansion "
        "(threshold_value; threshold_mode=ratio_to_baseline|zscore; window in "
        "minutes or ticks); price_touch_level/price_break_level (level in price "
        "units; optional direction, tolerance, confirm_ticks for breaks); "
        "price_enter_zone (lower and upper in price units). Put candle_close "
        "boundaries in end_on. Omit for a boundary-only wait. Explicit watchers "
        "make an unmatched timeout or boundary a failed wait. Examples: order_filled; "
        "'{\"type\":\"order_filled\",\"symbol\":\"EURUSD\"}'; "
        "'{\"type\":\"price_change\",\"direction\":\"up\",\"threshold_mode\":"
        "\"fixed_pct\",\"threshold_value\":0.1}'; "
        "'{\"type\":\"price_touch_level\",\"symbol\":\"EURUSD\",\"level\":1.0850,"
        "\"tolerance\":0.0002}'; "
        "'{\"type\":\"volume_spike\",\"window\":{\"kind\":\"minutes\",\"value\":5},"
        "\"threshold_mode\":\"ratio_to_baseline\",\"threshold_value\":2}'."
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
    ("pivot_compute_points", "end"): (
        "Historical cutoff. Uses the last completed bar at or before this "
        "instant; later bars are not used. Alias: as_of; do not pass both."
    ),
    ("pivot_compute_points", "as_of"): (
        "Alias for end. Historical cutoff for the last completed source bar; "
        "do not pass both aliases."
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
        "EWMA volatility half-life in bars of the requested timeframe. "
        "Used only by method=filtered_historical; omit it for bootstrap_historical."
    ),
    ("portfolio_risk_decompose", "simulations"): (
        "Monte Carlo scenario count used for portfolio tail-risk estimates."
    ),
    ("portfolio_risk_decompose", "proposed_trade"): (
        "Optional JSON trade object with symbol, buy/sell (or long/short) side, "
        "and volume in lots for incremental-risk analysis. Output side is buy/sell."
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
        "Spread source: auto (default) uses complete historical bar spreads "
        "when coverage is full, otherwise a disclosed conservative fixed "
        "estimate. historical_bar_spread fails closed unless coverage is "
        "complete; fixed requires spread_bps."
    ),
    ("strategy_backtest", "spread_bps"): (
        "Fixed round-trip spread cost in basis points; required when "
        "cost_model=fixed and invalid with auto or historical_bar_spread."
    ),
    ("strategy_backtest", "commission_bps_per_side"): (
        "Commission per fill side in basis points, deducted twice per simulated "
        "round trip; defaults to zero."
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
        "Spread source: auto (default) uses complete historical bar spreads when "
        "coverage is sufficient, otherwise a disclosed conservative fixed estimate. "
        "historical_bar_spread uses completed validation bars and disables positive "
        "evidence below 90% coverage; fixed requires spread_bps."
    ),
    ("strategy_validate", "spread_bps"): (
        "Fixed round-trip spread cost in basis points; required when "
        "cost_model=fixed and invalid with auto or historical_bar_spread."
    ),
    ("labels_triple_barrier", "start"): (
        "Optional UTC start of the labeled history window. Combine with end; "
        "cannot be used with as_of."
    ),
    ("labels_triple_barrier", "end"): (
        "Optional UTC cutoff. Combined with start, this is the analysis window "
        "unless an explicit lookback tail-cap is also set. Cannot be combined "
        "with as_of."
    ),
    ("labels_triple_barrier", "as_of"): (
        "Point-in-time cutoff for labeled history. Cannot be combined with start/end."
    ),
    ("strategy_validate", "commission_bps_per_side"): (
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

