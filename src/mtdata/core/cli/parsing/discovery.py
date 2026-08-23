import argparse
import inspect
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from ....utils.coercion import UNPARSED_BOOL, parse_bool_like

ToolInfo = Dict[str, Any]


_OPTIONAL_POSITIONAL_PARAMS: set[tuple[str, str]] = {
    ("asset_performance", "symbol"),
    ("news", "symbol"),
    ("equity_profile", "symbol"),
    ("correlation_matrix", "symbols"),
    ("cointegration_test", "symbols"),
    ("market_relative_strength", "symbols"),
    ("market_radar", "symbols"),
    ("market_scan", "symbols"),
    ("causal_discover_signals", "symbols"),
    ("market_status", "symbol"),
    ("trade_close", "symbol"),
    ("trade_execution_quality", "symbol"),
    ("trade_get_open", "symbol"),
    ("trade_get_pending", "symbol"),
    ("trade_place", "symbol"),
    ("trade_risk_analyze", "symbol"),
    ("trade_var_cvar_calculate", "symbol"),
    ("forecast_list_library_models", "library"),
    ("wait_event", "symbol"),
}

# Choice discovery comes from the same Literal/Pydantic annotations used to
# build public MCP schemas. Keep this map only for exceptional transport-only
# compatibility cases.
_COMMAND_PARAM_CHOICE_OVERRIDES: Dict[tuple[str, str], list[str]] = {
    ("temporal_analyze", "group_by"): [
        "dow",
        "day_of_week",
        "hour",
        "month",
        "session",
        "all",
    ],
}

_POSITIONAL_ONLY_OPTIONAL_PARAMS: set[tuple[str, str]] = set()

_SEARCH_ALIAS_COMMANDS = frozenset(
    {
        "screener",
        "forecast_list_methods",
        "indicators_list",
        "symbols_list",
        "tools_list",
    }
)

_OPTION_ALIAS_DEST_PREFIX = "_cli_option_"

_MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS = frozenset(
    {
        "causal_discover_signals",
        "correlation_matrix",
        "cointegration_test",
        "cross_correlation",
        "market_relative_strength",
    }
)

_COMMAND_REQUIRED_OPTIONS: set[tuple[str, str]] = {
    ("trade_modify", "ticket"),
    ("trade_stress_test", "shocks"),
}

_NAMED_ONLY_REQUIRED_PARAMS: set[tuple[str, str]] = {
    ("trade_modify", "ticket"),
    ("trade_stress_test", "shocks"),
}

_PRESERVE_OMITTED_DEFAULT_PARAMS: set[tuple[str, str]] = {
    ("data_fetch_candles", "limit"),
    ("data_fetch_ticks", "limit"),
    ("market_microstructure_analyze", "minutes_back"),
    ("trade_execution_quality", "minutes_back"),
}

_COMMAND_PARAM_HELP_OVERRIDES: Dict[tuple[str, str], str] = {
    ("market_depth_fetch", "spread"): (
        "Boolean output control. Compute and include bid/ask spread metrics from "
        "broker DOM or the fallback quote; disabled by default."
    ),
    ("forecast_train", "wait"): (
        "Wait for training to finish. One-shot CLI and stdin shell batches "
        "always wait so the in-process worker stays alive; the flag only "
        "applies in interactive shell, MCP, and Web API sessions."
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
    ("wait_event", "watch_tick_count_spike"): (
        "Include the inferred timeframe tick-count-spike watcher. Ignored in timer-only "
        "duration mode and with explicit watch_for."
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

_VOLATILITY_METHOD_LITERAL_MARKERS = {
    "ewma",
    "parkinson",
    "gk",
    "rs",
    "yang_zhang",
    "rolling_std",
    "realized_kernel",
    "har_rv",
    "garch_t",
    "egarch_t",
    "gjr_garch_t",
    "figarch",
}

_FORECAST_METHOD_LITERAL_MARKERS = {
    "theta",
    "naive",
    "arima",
    "chronos2",
    "statsforecast",
}


def _parse_cli_bool_value(value: Any) -> str:
    """Accept the shared bool vocabulary and return argparse's canonical token."""
    parsed = parse_bool_like(value)
    if parsed is UNPARSED_BOOL:
        raise argparse.ArgumentTypeError(
            "expected true/false, 1/0, yes/no, or on/off"
        )
    return "true" if bool(parsed) else "false"


def _case_insensitive_choice_parser(choices: Sequence[str]) -> Callable[[Any], str]:
    canonical = [str(choice) for choice in choices]
    folded: Dict[str, Optional[str]] = {}
    for choice in canonical:
        key = choice.casefold()
        folded[key] = choice if key not in folded else None

    def _parse(value: Any) -> str:
        text = str(value or "").strip()
        if text in canonical:
            return text
        return folded.get(text.casefold()) or text

    return _parse


def _comma_aware_choice_parser(choices: Sequence[str]) -> Callable[[Any], str]:
    """Validate and canonicalize one CLI token containing one or more choices."""
    parse_choice = _case_insensitive_choice_parser(choices)
    canonical = {str(choice) for choice in choices}

    def _parse(value: Any) -> str:
        parts = [part.strip() for part in str(value or "").split(",")]
        if not parts or any(not part for part in parts):
            raise argparse.ArgumentTypeError("expected one or more non-empty values")
        parsed = [parse_choice(part) for part in parts]
        invalid = [part for part in parsed if part not in canonical]
        if invalid:
            raise argparse.ArgumentTypeError(
                f"invalid choice: {invalid[0]!r} (choose from {', '.join(choices)})"
            )
        return ",".join(parsed)

    return _parse


def _is_forecast_method_literal(
    ptype: Any,
    *,
    is_literal_origin: Callable[[Any], bool],
    get_origin_func: Callable[[Any], Any],
    get_args_func: Callable[[Any], Tuple[Any, ...]],
) -> bool:
    try:
        origin = get_origin_func(ptype)
        if not is_literal_origin(origin):
            return False
        args = {str(v) for v in get_args_func(ptype) if v is not None}
        if args.intersection(_VOLATILITY_METHOD_LITERAL_MARKERS):
            return False
        return bool(args.intersection(_FORECAST_METHOD_LITERAL_MARKERS))
    except Exception:
        return False


def _dedupe_flags(*flags: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(flag for flag in flags if flag))


def _canonicalize_long_option(flag: str) -> str:
    text = str(flag or "").strip()
    if not text.startswith("--"):
        return text
    if "=" in text:
        option, value = text.split("=", 1)
        return f"{option.replace('_', '-')}={value}"
    return text.replace("_", "-")


def _split_visible_and_hidden_flags(*flags: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    visible: list[str] = []
    hidden: list[str] = []
    for flag in _dedupe_flags(*flags):
        canonical = _canonicalize_long_option(flag)
        if canonical and canonical not in visible:
            visible.append(canonical)
        if flag != canonical and flag not in hidden:
            hidden.append(flag)
    return tuple(visible), tuple(hidden)


def should_expose_cli_param(*, cmd_name: Optional[str], param_name: str) -> bool:
    """Return whether a function parameter should surface as a user CLI argument."""
    if str(cmd_name or "") == "calendar" and str(param_name or "") in {"date_from", "date_to"}:
        return False
    if str(cmd_name or "") == "wait_event" and str(param_name or "") == "instrument":
        return False
    return True


def get_function_info(
    func: Any,
    *,
    schema_get_function_info: Callable[[Any], Dict[str, Any]],
    flatten_request_model_param: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Attach the underlying callable to schema introspection data."""
    info = schema_get_function_info(func)
    info["func"] = func
    info = flatten_request_model_param(info)
    if not info.get("doc"):
        info["doc"] = f"Execute {info.get('name') or getattr(func, '__name__', 'function')}"
    for param in info.get("params", []):
        if param.get("type") is None:
            param["type"] = str
        if "required" not in param:
            param["required"] = param.get("default") is None
    return info


def apply_schema_overrides(
    tool: ToolInfo,
    func_info: Dict[str, Any],
    *,
    enrich_schema_with_shared_defs: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply JSON schema defaults and required flags to CLI parameter metadata."""
    meta = tool.setdefault("meta", {})
    schema = meta.get("schema") or {}
    schema = enrich_schema_with_shared_defs(schema, func_info)
    meta["schema"] = schema
    params_obj = schema.get("parameters") if isinstance(schema.get("parameters"), dict) else schema
    schema_props = params_obj.get("properties") if isinstance(params_obj, dict) else {}
    schema_required = set(params_obj.get("required", [])) if isinstance(params_obj, dict) else set()
    for param in func_info.get("params", []):
        prop = schema_props.get(param["name"]) if isinstance(schema_props, dict) else None
        if isinstance(prop, dict) and "default" in prop and param.get("default") is None:
            param["default"] = prop["default"]
        if param["name"] in schema_required:
            param["required"] = True
    return schema


def extract_function_from_tool_obj(tool_obj: Any) -> Any:
    """Best-effort extraction of the underlying function from an MCP tool object."""
    for attr in ("func", "function", "callable", "handler", "wrapped", "_func"):
        if hasattr(tool_obj, attr) and callable(getattr(tool_obj, attr)):
            return getattr(tool_obj, attr)
    if callable(tool_obj):
        return tool_obj
    return None


def extract_metadata_from_tool_obj(tool_obj: Any) -> Dict[str, Any]:
    """Extract tool descriptions and per-parameter docs from registry objects."""
    meta: Dict[str, Any] = {"description": None, "param_docs": {}, "schema": None}

    for attr in ("description", "doc", "docs"):
        val = getattr(tool_obj, attr, None)
        if isinstance(val, str) and val.strip():
            meta["description"] = val.strip()
            break

    schema = None
    for attr in ("schema", "input_schema", "parameters", "spec"):
        val = getattr(tool_obj, attr, None)
        if isinstance(val, dict) and val:
            schema = val
            break

    if schema:
        meta["schema"] = schema
        if not meta["description"] and isinstance(schema.get("description"), str):
            meta["description"] = schema.get("description")
        params_obj = schema.get("parameters") if isinstance(schema.get("parameters"), dict) else schema
        props = params_obj.get("properties") if isinstance(params_obj, dict) else None
        if isinstance(props, dict):
            for pname, pdef in props.items():
                desc = pdef.get("description") if isinstance(pdef, dict) else None
                if isinstance(desc, str) and desc.strip():
                    meta["param_docs"][pname] = desc.strip()

    return meta


def discover_tools(
    *,
    bootstrap_tools: Callable[[], Tuple[Any, ...]],
    get_registered_tools: Callable[[], Any],
    mcp: Any,
    get_mcp_registry: Callable[[Any], Any],
    debug: Callable[[str], None],
    extract_function_from_tool_obj: Callable[[Any], Any],
    extract_metadata_from_tool_obj: Callable[[Any], Dict[str, Any]],
    errors: Optional[list[str]] = None,
) -> Dict[str, ToolInfo]:
    """Discover CLI-visible tools from the bootstrap and MCP registries."""
    tools: Dict[str, ToolInfo] = {}

    def _module_is_visible(module_name: Any, allowed_modules: set[str], allowed_prefixes: tuple[str, ...]) -> bool:
        if not isinstance(module_name, str):
            return False
        if module_name in allowed_modules:
            return True
        return any(module_name.startswith(prefix) for prefix in allowed_prefixes)

    registry = None
    bootstrapped_modules: Tuple[Any, ...] = ()
    try:
        bootstrapped_modules = tuple(bootstrap_tools())
    except Exception as exc:
        message = f"bootstrap_tools failed: {exc}"
        debug(message)
        if errors is not None:
            errors.append(message)
    try:
        reg = get_registered_tools()
        if reg and hasattr(reg, "items"):
            registry = reg
    except Exception as exc:
        message = f"get_registered_tools failed: {exc}"
        debug(message)
        if errors is not None:
            errors.append(message)
    if mcp is not None:
        try:
            registry = get_mcp_registry(mcp) or registry
        except Exception as exc:
            message = f"get_mcp_registry failed: {exc}"
            debug(message)
            if errors is not None:
                errors.append(message)

    module_names = {
        str(getattr(module, "__name__", "")).strip()
        for module in bootstrapped_modules
        if getattr(module, "__name__", None)
    }
    module_prefixes = tuple(
        f"{module_name.rsplit('.', 1)[0]}."
        for module_name in module_names
        if "." in module_name
    )
    if registry and hasattr(registry, "items"):
        for name, obj in registry.items():
            if not str(name or "").strip():
                continue
            func = extract_function_from_tool_obj(obj)
            mod = getattr(func, "__module__", None) if func else None
            if func and (not module_names or _module_is_visible(mod, module_names, module_prefixes)):
                meta = extract_metadata_from_tool_obj(obj)
                tools[name] = {"func": func, "meta": meta}

    if tools:
        return tools

    for module in bootstrapped_modules:
        module_name = getattr(module, "__name__", None)
        if not isinstance(module_name, str):
            continue
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if callable(obj) and getattr(obj, "__module__", None) == module_name:
                try:
                    inspect.signature(obj)
                except (TypeError, ValueError):
                    continue
                if isinstance(obj, type):
                    continue
                if name.endswith(("_wrapper",)):
                    continue
                tools[name] = {"func": obj, "meta": {"description": None, "param_docs": {}}}

    return tools


def resolve_param_kwargs(
    param: Dict[str, Any],
    param_docs: Optional[Dict[str, str]],
    *,
    cmd_name: Optional[str],
    param_names: Optional[set],
    param_hints: Dict[str, str],
    debug: Callable[[str], None],
    is_literal_origin: Callable[[Any], bool],
    unwrap_optional_type: Callable[[Any], Tuple[Any, Any]],
    get_origin: Callable[[Any], Any],
    get_args: Callable[[Any], Tuple[Any, ...]],
    is_mapping_annotation: Callable[[Any], bool],
) -> Tuple[Dict[str, Any], bool]:
    """Resolve argparse kwargs for a single CLI parameter."""

    def _is_model_type(value: Any) -> bool:
        return isinstance(value, type) and (
            callable(getattr(value, "model_validate", None))
            or callable(getattr(value, "parse_obj", None))
        )

    def _escape_argparse_help(text: Optional[str]) -> Optional[str]:
        return text.replace("%", "%%") if isinstance(text, str) else text

    desc = None
    if param_docs and param["name"] in param_docs:
        desc = param_docs[param["name"]]
    hint = desc or param_hints.get(param["name"])
    override_help = _COMMAND_PARAM_HELP_OVERRIDES.get((str(cmd_name or ""), str(param["name"])))
    if override_help:
        hint = override_help
    fallback_help = (
        f"Input parameter --{str(param['name']).replace('_', '-')} for this command."
    )
    kwargs = {"help": _escape_argparse_help(hint) or fallback_help, "dest": param["name"]}
    is_mapping_type = False

    if param["name"] == "method" and (
        (cmd_name in {"forecast_generate", "forecast_conformal_intervals", "forecast_tune_genetic", "forecast_tune_optuna"})
        or _is_forecast_method_literal(
            param.get("type"),
            is_literal_origin=is_literal_origin,
            get_origin_func=get_origin,
            get_args_func=get_args,
        )
    ):
        if not (param_names and "library" in param_names):
            help_suffix = " Use forecast_list_methods to browse available methods."
            if "forecast_list_methods" not in kwargs["help"]:
                kwargs["help"] = f"{kwargs['help']}{help_suffix}"
            kwargs["metavar"] = "METHOD"
    else:
        try:
            ptype = param.get("type")
            base_type, origin = unwrap_optional_type(ptype)

            is_mapping_type = is_mapping_annotation(ptype)

            kwargs["type"] = str

            if base_type in (int, float, str):
                kwargs["type"] = base_type
            elif base_type is bool:
                kwargs["type"] = _parse_cli_bool_value
                kwargs["choices"] = ["true", "false"]

            if origin in (list, tuple):
                inner = get_args(ptype)[0] if get_args(ptype) else None
                inner_origin = get_origin(inner)
                if is_literal_origin(inner_origin):
                    choices = [str(v) for v in get_args(inner)]
                    if choices:
                        kwargs["type"] = _comma_aware_choice_parser(choices)
                        kwargs["metavar"] = "{" + ",".join(choices) + "}"
                    else:
                        kwargs["type"] = str
                    kwargs["nargs"] = "+"
                else:
                    kwargs["type"] = str
                    kwargs["nargs"] = "+"
            elif is_literal_origin(origin):
                choices = [str(v) for v in get_args(base_type)]
                if choices:
                    kwargs["choices"] = choices
                    kwargs["type"] = _case_insensitive_choice_parser(choices)
                else:
                    kwargs["type"] = str
        except Exception as exc:
            debug(f"Type resolution failed for param '{param['name']}': {exc}")
            kwargs["type"] = str

    if not param["required"] and not (param["type"] is bool and param["default"] is None):
        if (str(cmd_name or ""), str(param["name"])) in _PRESERVE_OMITTED_DEFAULT_PARAMS:
            kwargs["default"] = argparse.SUPPRESS
        else:
            kwargs["default"] = param["default"]

    choice_override_key = (str(cmd_name or ""), str(param["name"]))
    choice_override = _COMMAND_PARAM_CHOICE_OVERRIDES.get(choice_override_key)
    if choice_override:
        choices = list(choice_override)
        kwargs["choices"] = choices
        kwargs["type"] = _case_insensitive_choice_parser(choices)

    if choice_override_key == ("temporal_analyze", "group_by"):
        parse_group_by = kwargs["type"]

        def _parse_temporal_group(value: Any) -> str:
            parsed = parse_group_by(value)
            return "dow" if parsed == "day_of_week" else parsed

        kwargs["type"] = _parse_temporal_group

    if choice_override_key == ("trade_place", "order_type") and kwargs.get("choices"):
        parse_choice = _case_insensitive_choice_parser(kwargs["choices"])

        def _parse_order_type(value: Any) -> str:
            normalized = str(value or "").strip().replace("-", "_").replace(" ", "_")
            return parse_choice(normalized)

        kwargs["type"] = _parse_order_type

    if (str(cmd_name or ""), str(param["name"])) == ("indicators_list", "category"):
        kwargs["type"] = lambda value: str(value or "").strip().lower()

    return kwargs, is_mapping_type


def add_dynamic_arguments(  # noqa: C901
    parser: Any,
    param_info: Dict[str, Any],
    *,
    resolve_param_kwargs: Callable[..., Tuple[Dict[str, Any], bool]],
    param_docs: Optional[Dict[str, str]] = None,
    cmd_name: Optional[str] = None,
) -> None:
    """Add CLI arguments for an introspected function schema."""
    has_mapping_param = False

    def _extra_option_flags(param_name: str, cmd_name_value: Optional[str]) -> tuple[str, ...]:
        extras: list[str] = []
        if cmd_name_value == "trade_history" and param_name == "position_ticket":
            extras.append("--ticket")
        if cmd_name_value == "trade_history" and param_name == "history_kind":
            extras.append("--kind")
        if cmd_name_value in {
            "forecast_backtest_run",
            "forecast_tune_genetic",
            "forecast_tune_optuna",
        } and param_name == "methods":
            extras.append("--method")
        if cmd_name_value in _SEARCH_ALIAS_COMMANDS and param_name == "search":
            extras.append("--search-term")
        elif cmd_name_value in _SEARCH_ALIAS_COMMANDS and param_name == "search_term":
            extras.append("--search")
        if cmd_name_value == "temporal_analyze" and param_name == "group_by":
            extras.append("--by")
        if cmd_name_value in {
            "causal_discover_signals",
            "cointegration_test",
            "correlation_matrix",
            "cross_correlation",
        } and param_name == "window_bars":
            extras.append("--lookback")
        if cmd_name_value == "wait_event" and param_name == "max_wait_seconds":
            extras.append("--timeout")
        return tuple(extras)

    for param in param_info["params"]:
        if not should_expose_cli_param(cmd_name=cmd_name, param_name=str(param.get("name") or "")):
            continue
        hyph = f"--{param['name'].replace('_', '-')}"
        uscr = f"--{param['name']}"
        option_flags, hidden_option_flags = _split_visible_and_hidden_flags(
            hyph,
            uscr,
            *_extra_option_flags(param["name"], cmd_name),
        )

        param_names = {p.get("name") for p in (param_info.get("params") or []) if isinstance(p, dict)}
        kwargs, is_mapping_type = resolve_param_kwargs(
            param,
            param_docs,
            cmd_name=cmd_name,
            param_names=param_names,
        )
        is_required_option = (
            param["required"] and param != param_info["params"][0]
        ) or (str(cmd_name or ""), str(param["name"])) in _COMMAND_REQUIRED_OPTIONS
        if is_required_option:
            kwargs["required"] = True
            kwargs["default"] = argparse.SUPPRESS
            kwargs["help"] = f"{kwargs.get('help') or param['name']} (required)"

        is_optional_bool = param.get("type") is bool and not param.get("required", False)
        allow_optional_positional = (
            str(cmd_name or ""),
            str(param["name"]),
        ) in _OPTIONAL_POSITIONAL_PARAMS

        required_symbol_alias = (
            param["required"]
            and param == param_info["params"][0]
            and str(param["name"]) in {"symbol", "symbols"}
        )
        if required_symbol_alias:
            parser.usage = (
                "%(prog)s (SYMBOL | --symbol SYMBOL) [options]"
                if str(param["name"]) == "symbol"
                else "%(prog)s (SYMBOL [SYMBOL ...] | --symbols SYMBOLS) [options]"
            )
            positional_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ("help", "type", "choices", "metavar")
            }
            positional_kwargs["nargs"] = (
                "*"
                if (
                    str(cmd_name or "") in _MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                    and str(param["name"]) == "symbols"
                )
                else "?"
            )
            positional_kwargs["default"] = argparse.SUPPRESS
            positional_kwargs["help"] = (
                f"{positional_kwargs.get('help') or param['name']} (required)"
            )
            parser.add_argument(param["name"], **positional_kwargs)
            option_kwargs = dict(kwargs)
            option_kwargs["dest"] = f"{_OPTION_ALIAS_DEST_PREFIX}{param['name']}"
            option_kwargs.setdefault("metavar", str(param["name"]).upper())
            option_kwargs["default"] = argparse.SUPPRESS
            option_kwargs["required"] = False
            if (
                str(cmd_name or "") in _MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                and str(param["name"]) == "symbols"
            ):
                option_kwargs["nargs"] = "+"
            if option_flags:
                parser.add_argument(*option_flags, **option_kwargs)
            if hidden_option_flags:
                hidden_option_kwargs = dict(option_kwargs)
                hidden_option_kwargs["help"] = argparse.SUPPRESS
                parser.add_argument(*hidden_option_flags, **hidden_option_kwargs)
        elif (
            param["required"]
            and param == param_info["params"][0]
            and (str(cmd_name or ""), str(param["name"]))
            not in _NAMED_ONLY_REQUIRED_PARAMS
        ):
            positional_kwargs = {k: v for k, v in kwargs.items() if k in ("help", "type", "choices", "metavar")}
            if (
                str(cmd_name or "") in _MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                and str(param["name"]) == "symbols"
            ):
                positional_kwargs["nargs"] = "+"
            positional_kwargs["help"] = f"{positional_kwargs.get('help') or param['name']} (required)"
            parser.add_argument(param["name"], **positional_kwargs)
        elif allow_optional_positional:
            positional_kwargs = {k: v for k, v in kwargs.items() if k in ("help", "type", "choices", "metavar")}
            positional_kwargs["nargs"] = (
                "*"
                if (
                    str(cmd_name or "") in _MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                    and str(param["name"]) == "symbols"
                )
                else "?"
            )
            positional_kwargs["default"] = argparse.SUPPRESS
            parser.add_argument(param["name"], **positional_kwargs)
            option_kwargs = dict(kwargs)
            option_kwargs["dest"] = f"{_OPTION_ALIAS_DEST_PREFIX}{param['name']}"
            option_kwargs.setdefault("metavar", str(param["name"]).upper())
            option_kwargs["default"] = argparse.SUPPRESS
            if (
                str(cmd_name or "") in _MULTI_VALUE_SYMBOL_POSITIONAL_COMMANDS
                and str(param["name"]) == "symbols"
            ):
                option_kwargs["nargs"] = "+"
            positional_key = (str(cmd_name or ""), str(param["name"]))
            if option_flags and positional_key not in _POSITIONAL_ONLY_OPTIONAL_PARAMS:
                parser.add_argument(*option_flags, **option_kwargs)
            if hidden_option_flags and positional_key not in _POSITIONAL_ONLY_OPTIONAL_PARAMS:
                hidden_option_kwargs = dict(option_kwargs)
                hidden_option_kwargs["help"] = argparse.SUPPRESS
                parser.add_argument(*hidden_option_flags, **hidden_option_kwargs)
        else:
            if is_optional_bool:
                local_kwargs = dict(kwargs)
                local_kwargs["nargs"] = "?"
                local_kwargs["const"] = "true"
                if option_flags:
                    parser.add_argument(*option_flags, **local_kwargs)
                if hidden_option_flags:
                    hidden_kwargs = dict(local_kwargs)
                    hidden_kwargs["help"] = argparse.SUPPRESS
                    parser.add_argument(*hidden_option_flags, **hidden_kwargs)
                if param["name"] not in {"dry_run", "require_sl_tp"}:
                    no_flags, no_hidden_flags = _split_visible_and_hidden_flags(
                        f"--no-{param['name'].replace('_', '-')}",
                        f"--no_{param['name']}",
                    )
                    no_default = kwargs.get("default", argparse.SUPPRESS)
                    if no_flags:
                        parser.add_argument(
                            *no_flags,
                            dest=param["name"],
                            action="store_const",
                            const="false",
                            default=no_default,
                            help=argparse.SUPPRESS,
                        )
                    if no_hidden_flags:
                        hidden_no_kwargs = {
                            "dest": param["name"],
                            "action": "store_const",
                            "const": "false",
                            "default": no_default,
                            "help": argparse.SUPPRESS,
                        }
                        parser.add_argument(*no_hidden_flags, **hidden_no_kwargs)
            elif is_mapping_type:
                local_kwargs = dict(kwargs)
                if not is_required_option:
                    local_kwargs["nargs"] = "?"
                    local_kwargs["const"] = "__PRESENT__"
                if option_flags:
                    parser.add_argument(*option_flags, **local_kwargs)
                if hidden_option_flags:
                    hidden_kwargs = dict(local_kwargs)
                    hidden_kwargs["help"] = argparse.SUPPRESS
                    parser.add_argument(*hidden_option_flags, **hidden_kwargs)
            else:
                if option_flags:
                    parser.add_argument(*option_flags, **kwargs)
                if hidden_option_flags:
                    hidden_kwargs = dict(kwargs)
                    hidden_kwargs["help"] = argparse.SUPPRESS
                    hidden_kwargs["required"] = False
                    parser.add_argument(*hidden_option_flags, **hidden_kwargs)
        if str(param["name"]) == "minutes_back" and str(cmd_name or "").startswith("trade_"):
            parser.add_argument(
                "--days",
                dest="_trade_days",
                type=float,
                default=argparse.SUPPRESS,
                metavar="DAYS",
                help=(
                    "Alias for --minutes-back expressed in days; choose one "
                    "lookback spelling."
                ),
        )

        if is_mapping_type:
            has_mapping_param = True
            if param["name"] == "params":
                continue
            params_flags = _dedupe_flags(
                f"--{param['name'].replace('_', '-')}-params",
                f"--{param['name']}_params",
            )
            parser.add_argument(
                *params_flags,
                dest=f"{param['name']}_params",
                type=str,
                default=None,
                help=f"Extra params for {param['name']} (key=value[,key=value])",
            )
    if has_mapping_param:
        parser.add_argument(
            "--set",
            dest="set_overrides",
            action="append",
            default=None,
            metavar="PARAM.KEY=VALUE",
            help=(
                "Override nested mapping params, e.g. --set denoise.params.lookback=50."
            ),
        )
